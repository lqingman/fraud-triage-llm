"""Phase 4 — Prometheus instrumentation for the serving layer.

Kept as a thin, dependency-isolated module so src.serve.guardrails (pure,
unit-tested without prometheus_client) stays decoupled: app.py wires
guardrails' `on_invalid` hook to `record_invalid_output` here, rather than
guardrails importing a metrics library directly.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests handled",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

INVALID_OUTPUT_COUNT = Counter(
    "triage_invalid_output_total",
    "Count of model outputs that failed schema validation (per guardrail attempt)",
)


def observe_request(endpoint: str, status: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_invalid_output() -> None:
    INVALID_OUTPUT_COUNT.inc()


def render() -> bytes:
    return generate_latest()
