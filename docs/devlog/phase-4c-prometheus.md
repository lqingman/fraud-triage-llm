# Phase 4c — Prometheus instrumentation

**Date:** 2026-07-09
**Status:** Done.

## Goal

No Prometheus code existed anywhere in the repo before this. Add real
request/latency/invalid-output instrumentation to the FastAPI app from
[phase-4-serving.md](phase-4-serving.md), verifiable locally with no GPU and
no running vLLM server.

## What I did

### `src/serve/metrics.py` (new)
Kept in its own module, deliberately separate from
`src/serve/guardrails.py`, so guardrails stays free of the
`prometheus_client` dependency and fast/pure to unit-test — `app.py` wires
guardrails' existing `on_invalid` hook (added in phase-4-serving) to
`metrics.record_invalid_output` instead. Three instruments:
- `http_requests_total{endpoint,status}` (Counter) and
  `http_request_latency_seconds{endpoint}` (Histogram), both populated by the
  same request-logging middleware that already existed for structured logs —
  renamed `_request_logging` → `_request_logging_and_metrics`.
- `triage_invalid_output_total` (Counter), incremented once per failed parse
  attempt inside `safe_generate` — so it reflects retry attempts, not just
  final outcomes. A backend that needs 2 retries per request shows up here
  even when every request ultimately returns a valid verdict.

### `src/serve/app.py`
`GET /metrics` returns `prometheus_client.generate_latest()`. The
`/triage/text` handler now passes `on_invalid=metrics.record_invalid_output`
to `safe_generate`.

### Tests (`tests/test_serve_app.py`, extended)
Added three cases using `prometheus_client.REGISTRY.get_sample_value` to read
counter values before/after a request (rather than asserting exact totals,
which would be order-dependent across the test file):
- `/metrics` returns 200 and contains the expected metric names.
- A successful `/triage/text` call increments `http_requests_total` for that
  endpoint/status pair by exactly 1.
- A malformed-output stub increments `triage_invalid_output_total`.

## Verification

- `pytest -q` → 70 passed (was 67 after the MLflow commit).
- Manual smoke test: hit the running server (no backend, no vLLM) with
  `/health`, `/ready`, `/triage/text` (happy + malformed + oversized), then
  `curl .../metrics`:
  ```
  http_requests_total{endpoint="/health",status="200"} 1.0
  http_requests_total{endpoint="/ready",status="503"} 1.0
  http_requests_total{endpoint="/triage/text",status="200"} 1.0
  http_requests_total{endpoint="/triage/text",status="422"} 1.0
  triage_invalid_output_total 3.0
  ```
  `triage_invalid_output_total == 3` matches `MAX_RETRIES + 1` failed parse
  attempts for the one request that had no backend to talk to — confirms the
  counter reflects retry attempts as designed, not just final request outcomes.

## Scope / honesty note

Grafana dashboards, Locust/k6 load testing, and drift monitoring are
explicitly **not** part of this — they need a running deployment to be
meaningful and aren't verifiable in this sandbox. The accurate resume phrase
is "Prometheus instrumentation" (request/latency/invalid-output metrics +
`/metrics` endpoint), not "Prometheus/Grafana observability, load testing,
and drift monitoring."

## Follow-ups

- [ ] Grafana dashboard + alerts once there's a real deployment to point it at.
- [ ] Locust/k6 load test once a real backend (or at least a load-bearing stub)
      exists to generate meaningful latency/throughput numbers against.
- [ ] Input/output drift + fraud-distribution monitoring — needs production
      traffic, not attempted here.
