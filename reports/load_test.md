# Load test — FastAPI serving layer

**Date:** 2026-07-09
**Tool:** Locust 2.45.0, `tests/load/locustfile.py`
**Target:** `uvicorn src.serve.app:app`, local (127.0.0.1), **no LLM backend
configured** — `VLLM_BASE_URL` unset, so every `/triage/text` request hits a
connection-refused error 3 times (guardrails' retry budget) before degrading
to the safe fallback verdict.
**Load:** 20 concurrent users, ramped at 5/s, sustained 30 seconds.

## What this measures (and doesn't)

This measures the **HTTP + guardrails + Prometheus overhead of this service**
under load with the backend completely down — a real, relevant scenario
(does the service stay responsive and correct when the LLM is unreachable?),
not real LLM inference latency. There's no GPU in this environment to run a
real vLLM server, so an actual model-inference number isn't available here.
Once a real backend exists, rerun the same locustfile against it — the
comparison between "backend down" and "backend up" numbers would itself be
informative.

## Results (raw CSV: `reports/load_test_stats.csv`)

| endpoint | requests | failures | p50 | p95 | p99 | max | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POST /triage/text` | 1,116 | 0 | 12ms | 17ms | 20ms | 51ms | 38.4 |
| `GET /health` | 226 | 0 | 1ms | 2ms | 2ms | 3ms | 7.8 |
| `GET /metrics` | 223 | 0 | 1ms | 2ms | 4ms | 5ms | 7.7 |
| `GET /ready` | 236 | 236 (100%) | 5ms | 7ms | 10ms | 42ms | 8.1 |
| **Aggregate** | 1,801 | 236 (13%) | 11ms | 16ms | 20ms | 51ms | 62.0 |

**`/ready`'s "100% failure" is expected, not a problem**: Locust counts any
non-2xx as a failure by default, and `/ready` correctly returns 503 with no
backend configured — that IS the probe working as designed. Zero failures
on every other endpoint, including `/triage/text` (which always returns 200
with either a real verdict or the guardrail's safe fallback — it's designed
to never propagate a backend outage as a client-facing error).

## Takeaways

- **38.4 req/s sustained on `/triage/text`** with 20 concurrent users, p95
  17ms, zero failures — this is the guardrails/logging/metrics overhead
  alone (3 failed connection attempts + fallback construction), since
  there's no model call happening. A real backend would add its own
  inference latency on top of this, not replace it.
- **p99 20ms / max 51ms** — the tail is still fast; no evidence of
  connection-pool exhaustion or blocking I/O issues at this concurrency on a
  single process.
- **`/metrics` stayed cheap under load** (p95 2ms) — the Prometheus
  instrumentation itself isn't adding meaningful overhead.
- Single uvicorn worker process, no load balancer, no real backend: this is
  a lower bound on latency (nothing to wait on) and an upper bound on
  meaningful throughput conclusions (one process, one CPU core doing the
  work) — see Follow-ups.

## Follow-ups

- [ ] Rerun against a real vLLM backend once one exists, to get an actual
      model-inference-included latency number.
- [ ] Rerun with multiple uvicorn workers / behind a reverse proxy to get a
      more realistic production-topology throughput ceiling.
- [ ] Longer duration (minutes, not 30s) to catch any slow memory growth or
      connection leaks that a short burst wouldn't reveal.
