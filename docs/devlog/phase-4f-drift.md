# Phase 4f — drift + fraud-distribution monitoring

**Date:** 2026-07-09
**Status:** Done.

## Goal

Zero drift-monitoring code existed. Add real monitoring of the predicted
fraud-rate distribution vs. the training-time baseline, exposed as
Prometheus metrics — and specifically make it catch the one scenario this
repo can actually produce without a live model: a backend outage, where
`safe_generate`'s fallback verdict (`risk=medium` → `is_fraud=True`) fires on
every request.

## What I did

### `src/serve/drift.py` (new)
`DriftTracker`: a fixed-size rolling window (`collections.deque`) of recent
`is_fraud` booleans, compared against a configured baseline fraud rate via
the **Population Stability Index (PSI)**, collapsed to two buckets (fraud /
not) since the model's output is binary at the point drift is measured here.
PSI is the standard industry heuristic for this: < 0.1 no significant shift,
0.1–0.25 moderate, > 0.25 significant — used 0.25 as the default alert
threshold. Proportions are clipped away from exactly 0/1 so `log()` never
blows up on a window that's briefly all-fraud or all-legitimate (a real
failure mode during a backend outage, see below).

Config-driven (`config.monitoring`): `drift_window` (default 200),
`baseline_fraud_rate` (default 0.5, matching the ~50% fraud training
corpus), `psi_alert_threshold` (default 0.25). A module-level singleton
(`TRACKER`, plus `record`/`fraud_rate`/`is_drifting` wrapper functions)
mirrors `src.serve.metrics`'s existing pattern of one shared instance per
process.

### `src/serve/metrics.py` (extended)
Two new Gauges: `triage_fraud_rate_window` and `triage_drift_alert` (0/1),
via `observe_drift(fraud_rate, drifting)`.

### `src/serve/app.py`
`_triage()` — the shared helper both `/triage/text` and `/triage/audio` call
— now records every verdict's `is_fraud` into the tracker and updates the
gauges immediately after `safe_generate` returns, so both paths are covered
by one wire-up.

## Tests

- `tests/test_drift.py`: empty tracker, exact-baseline-match doesn't alert,
  sustained all-fraud/all-legit both alert, a mild 55%-vs-50% wobble does
  *not* alert (distinguishing real shift from sampling noise), the window is
  bounded and drops the oldest entry, `reset()` clears state, and an
  explicit test named for the outage scenario the module docstring
  describes (all-fallback-verdicts triggers drift).
- `tests/test_serve_app.py`: a new case asserts the Prometheus gauges after
  a real request exactly match what `drift.fraud_rate()`/`is_drifting()`
  compute — an integration check of the wiring, not a re-derivation of the
  PSI math (already covered in `test_drift.py`).

## Verification — a real demonstrated alert, not just unit tests

Started a real `uvicorn src.serve.app:app` (no backend configured — none
exists in this environment) and fired 15 real requests at `/triage/text`:

```
before:  triage_fraud_rate_window 0.0   triage_drift_alert 0.0
after:   triage_fraud_rate_window 1.0   triage_drift_alert 1.0
```

This is the exact scenario the module was built for: a downed LLM backend
makes every request return the guardrails' safe-fallback verdict
(`risk=medium`), which silently look like "the model is now flagging 100% of
calls as fraud" unless something is watching the distribution — and here,
something is.

`pytest -q` → 93 passed (was 84 after Phase 0e).

## Scope / honesty note

This is real, working drift detection on the *fraud-rate distribution* of
served verdicts. It does not (yet) monitor input-side drift (e.g. transcript
length/vocabulary shift) or per-fraud-type distribution — only the binary
fraud/not-fraud proportion the PSI formula needs. It has also only ever been
exercised against the guardrails'-fallback-during-outage scenario, since
there's no live model in this environment to produce a more nuanced,
real-model-driven distribution shift to test against.

## Follow-ups

- [ ] Per-fraud-type distribution monitoring (not just binary fraud/not),
      once there's a real model producing varied `fraud_type` values to
      compare against a training-time reference distribution.
- [ ] Input-side drift (transcript length, vocabulary) as a second signal
      independent of the model's own output.
- [ ] Wire `triage_drift_alert`/`triage_fraud_rate_window` into the Grafana
      dashboard (Phase 4g) and an actual alerting rule.
