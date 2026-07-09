# Phase 4g — Grafana dashboard + Prometheus scrape config

**Date:** 2026-07-09
**Status:** Done. Verified via real Prometheus scraping and real queries
through Grafana's own datasource proxy; no browser screenshot (see note).

## Goal

Zero Grafana/Prometheus-server code existed — Phase 4c only added
instrumentation (`/metrics`), with nothing scraping or visualizing it. Stand
up a real local observability stack: Prometheus scraping the serving image's
`/metrics`, Grafana pre-provisioned with a dashboard querying the actual
metric names from `src/serve/metrics.py`.

## What I did

`monitoring/` (new):
- `docker-compose.yml`: three services — `app` (built from
  `src/serve/Dockerfile`, the real serving image from Phase 4d),
  `prometheus` (scrapes `app:8000`), `grafana` (anonymous viewer access for
  local dev, provisioned from files — no manual dashboard setup).
- `prometheus.yml`: a single scrape job, 5s interval (fast enough to see
  changes quickly in a demo).
- `grafana/provisioning/datasources/prometheus.yml`: auto-registers
  Prometheus as the default datasource.
- `grafana/provisioning/dashboards/dashboards.yml` +
  `grafana/dashboards/fraud-triage.json`: a dashboard with 5 panels —
  request rate by status, p95 latency by endpoint
  (`histogram_quantile` over the `http_request_latency_seconds` histogram),
  invalid-output rate, a fraud-rate-window gauge, and a drift-alert stat
  panel (green "OK" / red "DRIFTING").

## Verification — real containers, real scraping, real query results

Docker Compose brought up all three services for real
(`docker compose -f monitoring/docker-compose.yml up -d --build`):

1. **App healthy**: `GET /health` → 200.
2. **Generated real traffic**: 30 `POST /triage/text` requests (no
   `VLLM_BASE_URL` configured — guardrails fall back to the safe verdict
   every time, same as every other manual test in this project).
3. **Prometheus target health**: `GET /api/v1/targets` → `up`.
4. **Prometheus has real values**: `GET /api/v1/query?query=triage_fraud_rate_window`
   → `1` (matches the all-fallback-verdict traffic just sent) and
   `triage_drift_alert` → `1` — the exact same outage-detection behavior
   verified manually in Phase 4f, now observed through a real Prometheus
   server rather than curling `/metrics` directly.
5. **Grafana actually provisioned the dashboard, not just accepted the
   file**: `GET /api/dashboards/uid/fraud-triage-llm` →
   `"provisioned": true, "provisionedExternalId": "fraud-triage.json"`.
6. **Grafana's datasource proxy returns correct live data for the exact
   dashboard panel expressions** — ran the p95-latency panel's literal
   PromQL (`histogram_quantile(0.95, sum(rate(http_request_latency_seconds_bucket[5m])) by (le, endpoint))`)
   through `GET /api/datasources/proxy/uid/<uid>/api/v1/query` (Grafana's
   own query path, not a direct Prometheus call) and got real per-endpoint
   latencies back (e.g. `/triage/text` → ~49ms, matching the guardrails'
   3-retry-then-fallback cost seen in the Phase 4e load test).

Torn down cleanly afterward (`docker compose down`, confirmed no leftover
containers/networks).

## Scope / honesty note — no visual screenshot

The Chrome browser tool wasn't reachable in this sandbox (extension
disconnected), so this wasn't confirmed with an actual rendered screenshot
of the dashboard in a browser. What's verified instead is arguably stronger
for correctness (Grafana's own API confirms the dashboard is provisioned and
its exact panel queries return correct live data) but it isn't the same as
someone visually seeing the graphs. Anyone can reproduce the visual check in
under a minute: `docker compose -f monitoring/docker-compose.yml up --build`,
generate a few requests, open `http://localhost:3000` (anonymous viewer
access is on for this dev stack).

## Follow-ups

- [ ] Real screenshot/visual confirmation once a browser tool is available.
- [ ] Alerting rules (Grafana alerting or Prometheus Alertmanager) on
      `triage_drift_alert == 1` and elevated `triage_invalid_output_total`
      rate — the dashboard shows the signal, but nothing pages anyone yet.
- [ ] Pin `prom/prometheus`/`grafana/grafana` to specific versions if this
      stack ever needs to be reproducible beyond local dev.
