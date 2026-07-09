# Phase 4e — real load testing (Locust)

**Date:** 2026-07-09
**Status:** Done. Real 30s run against a live server, not a synthetic estimate.

## Goal

No load-testing code existed anywhere in the repo. Add a real Locust load
test against the FastAPI serving layer and report actual numbers — not a
guess at what throughput "should" look like.

## What I did

- `tests/load/locustfile.py`: a `TriageUser` hitting `/triage/text` (5x
  weight, random transcript from a small pool), `/health`, `/ready`,
  `/metrics` (1x weight each) with a 0.1–0.5s wait between tasks.
- `requirements-loadtest.txt`: `locust`, isolated from the app's own
  dependencies since it's a dev-only tool, not something the service needs
  to run or be tested.

## Verification — a real run, not a dry read of the locustfile

Started a real `uvicorn src.serve.app:app` process locally (no `VLLM_BASE_URL`
configured — no GPU/backend exists in this environment) and ran:
```
locust -f tests/load/locustfile.py --host http://127.0.0.1:8131 --headless \
  -u 20 -r 5 -t 30s --csv reports/load_test
```
20 concurrent users, ramped 5/s, sustained 30s. Results (full CSV committed
at `reports/load_test_stats.csv`, summary at `reports/load_test.md`):

| endpoint | requests | p50 | p95 | p99 | req/s |
|---|---:|---:|---:|---:|---:|
| `POST /triage/text` | 1,116 | 12ms | 17ms | 20ms | 38.4 |
| `GET /health` | 226 | 1ms | 2ms | 2ms | 7.8 |
| `GET /metrics` | 223 | 1ms | 2ms | 4ms | 7.7 |
| `GET /ready` | 236 | 5ms | 7ms | 10ms | 8.1 |

Zero failures on every endpoint except `/ready`, which "failed" 236/236 —
correctly: Locust counts non-2xx as a failure, and with no backend attached
`/ready` is supposed to return 503. That's the probe working, not a bug.

## What this number actually means (and the caveat that matters)

There is no real LLM backend in this environment, so `/triage/text`'s 38.4
req/s / 17ms-p95 is the cost of HTTP handling + 3 failed connection attempts
(guardrails' retry budget) + fallback verdict construction + Prometheus
recording + structured logging — **not** LLM inference latency, which
doesn't exist to measure here. This is still a meaningful number (it's the
floor latency/overhead the serving layer itself adds, and proves it stays
fast and error-free under concurrent load even with the backend completely
down), but it would be dishonest to present it as "the API's real-world
latency" without that caveat. Framed the report explicitly around this.

## Follow-ups

- [ ] Rerun once a real vLLM backend exists — the interesting comparison is
      backend-down overhead vs. backend-up total latency, not either number
      alone.
- [ ] Multi-worker / reverse-proxy topology test for a more realistic
      throughput ceiling (this was one uvicorn process, one core).
