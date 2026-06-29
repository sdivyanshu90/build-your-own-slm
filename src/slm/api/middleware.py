"""ASGI middleware: request id, structured access logs, metrics, secure headers."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from slm.api.metrics import Metrics

__all__ = ["RequestContextMiddleware", "SecurityHeadersMiddleware"]

log = structlog.get_logger("slm.api.access")

# Conservative security headers applied to every response. HSTS is only sent
# when the request arrived over HTTPS so local HTTP development is unaffected.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


def _route_template(request: Request) -> str:
    """Use the matched route path (low cardinality) for metric labels."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind logging context, record metrics and access logs."""

    def __init__(self, app: object, metrics: Metrics) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        request.state.request_id = request_id

        start = time.perf_counter()
        self.metrics.in_flight.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            log.exception("request.unhandled_error")
            raise
        finally:
            self.metrics.in_flight.dec()
            elapsed = time.perf_counter() - start
            path = _route_template(request)
            self.metrics.request_latency.labels(request.method, path).observe(elapsed)
            self.metrics.requests_total.labels(request.method, path, str(status_code)).inc()
            log.info(
                "request.completed",
                status=status_code,
                duration_ms=round(elapsed * 1000, 2),
            )

        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach hardening headers to every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response
