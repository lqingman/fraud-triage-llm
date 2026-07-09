"""Phase 4 — Prometheus instrumentation for the serving layer.

Kept as a thin, dependency-isolated module so src.serve.guardrails (pure,
unit-tested without prometheus_client) stays decoupled: app.py wires
guardrails' `on_invalid` hook to `record_invalid_output` here, rather than
guardrails importing a metrics library directly.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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

FRAUD_RATE_GAUGE = Gauge(
    "triage_fraud_rate_window",
    "Fraud rate (is_fraud proportion) over the rolling prediction window (src.serve.drift)",
)

DRIFT_ALERT_GAUGE = Gauge(
    "triage_drift_alert",
    "1 if the rolling fraud-rate distribution has drifted from the training baseline "
    "(PSI over threshold), else 0",
)


def observe_request(endpoint: str, status: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_invalid_output() -> None:
    INVALID_OUTPUT_COUNT.inc()


def observe_drift(fraud_rate: float, drifting: bool) -> None:
    FRAUD_RATE_GAUGE.set(fraud_rate)
    DRIFT_ALERT_GAUGE.set(1.0 if drifting else 0.0)


def render() -> bytes:
    return generate_latest()
