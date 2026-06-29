"""Prometheus metrics, bound to a per-application registry.

Creating collectors against an app-owned :class:`CollectorRegistry` (rather than
the global default) means multiple app instances — notably in the test suite —
can coexist without ``Duplicated timeseries`` registration errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["Metrics"]

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


@dataclass
class Metrics:
    """Container for all application metrics over a single registry."""

    registry: CollectorRegistry
    requests_total: Counter
    request_latency: Histogram
    in_flight: Gauge
    tokens_generated: Counter

    @classmethod
    def create(cls) -> Metrics:
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests_total=Counter(
                "slm_http_requests_total",
                "Total HTTP requests.",
                labelnames=("method", "path", "status"),
                registry=registry,
            ),
            request_latency=Histogram(
                "slm_http_request_duration_seconds",
                "HTTP request latency in seconds.",
                labelnames=("method", "path"),
                buckets=_LATENCY_BUCKETS,
                registry=registry,
            ),
            in_flight=Gauge(
                "slm_http_in_flight_requests",
                "In-flight HTTP requests.",
                registry=registry,
            ),
            tokens_generated=Counter(
                "slm_tokens_generated_total",
                "Total completion tokens generated.",
                registry=registry,
            ),
        )
