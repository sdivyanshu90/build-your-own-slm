"""Authentication and rate limiting primitives.

* **Authentication** — an opaque API key supplied via the ``X-API-Key`` header
  (or ``Authorization: Bearer <key>``). Keys are compared in constant time.
  Auth is enforced only when the deployment configures at least one key.
* **Rate limiting** — an in-process token bucket per identity. For multi-replica
  deployments swap :class:`RateLimiter` for a Redis-backed implementation
  exposing the same ``allow`` method (see docs/security.md).
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from slm.config import Settings

__all__ = ["RateLimiter", "api_key_dependency", "identify_api_key", "rate_limit_dependency"]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def identify_api_key(key: str | None) -> str:
    """Return a short, non-reversible identifier for logging/metrics."""
    if not key:
        return "anonymous"
    return "key_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _extract_key(request: Request, header_key: str | None) -> str | None:
    if header_key:
        return header_key
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def _matches_any(candidate: str, keys: frozenset[str]) -> bool:
    """Constant-time membership test to avoid timing side channels."""
    result = False
    for key in keys:
        if hmac.compare_digest(candidate, key):
            result = True
    return result


async def api_key_dependency(
    request: Request,
    header_key: str | None = Depends(_api_key_header),
) -> str:
    """Authenticate the request and return the caller's identity string.

    Returns ``"anonymous"`` when authentication is disabled. Raises ``401`` when
    a key is required but missing/invalid.
    """
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return "anonymous"
    candidate = _extract_key(request, header_key)
    if candidate is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not _matches_any(candidate, settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return identify_api_key(candidate)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


@dataclass
class RateLimiter:
    """Thread-safe token-bucket limiter keyed by caller identity."""

    rate_per_minute: int
    burst: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def _refill_per_sec(self) -> float:
        return self.rate_per_minute / 60.0

    def allow(self, identity: str, *, now: float | None = None) -> tuple[bool, float]:
        """Attempt to consume one token.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is 0
        when allowed.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last_refill=now)
                self._buckets[identity] = bucket
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self._refill_per_sec)
            bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            deficit = 1.0 - bucket.tokens
            retry_after = deficit / self._refill_per_sec if self._refill_per_sec > 0 else 60.0
            return False, retry_after


async def rate_limit_dependency(
    request: Request,
    identity: str = Depends(api_key_dependency),
) -> str:
    """Enforce the per-identity rate limit; raises ``429`` when exceeded."""
    limiter: RateLimiter = request.app.state.rate_limiter
    # Fall back to client IP when unauthenticated so anonymous callers are also
    # bounded.
    key = (
        identity
        if identity != "anonymous"
        else f"ip_{request.client.host if request.client else 'unknown'}"
    )
    allowed, retry_after = limiter.allow(key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return identity
