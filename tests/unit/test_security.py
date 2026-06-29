"""Tests for authentication and rate-limiting primitives."""

from __future__ import annotations

from slm.api.security import (
    RateLimiter,
    _matches_any,
    identify_api_key,
)


def test_identify_api_key():
    assert identify_api_key(None) == "anonymous"
    assert identify_api_key("") == "anonymous"
    ident = identify_api_key("super-secret")
    assert ident.startswith("key_")
    assert "super-secret" not in ident  # never leaks the raw key


def test_matches_any_constant_time():
    keys = frozenset({"alpha", "beta"})
    assert _matches_any("alpha", keys) is True
    assert _matches_any("beta", keys) is True
    assert _matches_any("gamma", keys) is False


def test_rate_limiter_allows_burst_then_blocks():
    limiter = RateLimiter(rate_per_minute=60, burst=3)
    results = [limiter.allow("id", now=0.0)[0] for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_rate_limiter_refills_over_time():
    limiter = RateLimiter(rate_per_minute=60, burst=1)  # 1 token/sec
    assert limiter.allow("id", now=0.0)[0] is True
    assert limiter.allow("id", now=0.0)[0] is False
    # After one second, one token has refilled.
    assert limiter.allow("id", now=1.0)[0] is True


def test_rate_limiter_retry_after_positive():
    limiter = RateLimiter(rate_per_minute=60, burst=1)
    limiter.allow("id", now=0.0)
    allowed, retry_after = limiter.allow("id", now=0.0)
    assert allowed is False
    assert retry_after > 0


def test_rate_limiter_isolated_identities():
    limiter = RateLimiter(rate_per_minute=60, burst=1)
    assert limiter.allow("a", now=0.0)[0] is True
    assert limiter.allow("b", now=0.0)[0] is True  # different identity, own bucket
