# Phase 2 — Eval harness: metrics + XGBoost baseline

**Date:** 2026-06-09
**Status:** Done (metrics, baseline, gate, harness wiring). LLM eval branch wired
but unexercised end-to-end (no trained model yet — Phase 1). DIFRAUD cross-domain
loader still pending.

## Goal

Make the credibility claim from the data-strategy note concrete: run a model on
the held-out test set and report fraud-detection metrics, plus a classical
XGBoost control to answer "why not just use XGBoost?". This is the regression
gate the rest of the project is measured against, so it had to actually run —
not stay a stub — before Phase 1 produces a model to grade.

## What I did

### 1. Split loader — `load_split` (`src/eval/evaluate.py`)
Reads a processed `{prompt, completion}` jsonl and returns
`(transcripts, gold_is_fraud, completions)`. Two deliberate choices:
- **Gold label comes from the completion verdict's `is_fraud`**, parsed through
  the same `schema.parse_verdict` the model is graded against — so gold and
  prediction share one definition of "fraud" (`risk in {medium, high}`).
- **Transcript is recovered** from the formatted prompt via the
  `Transcript:\n … \n\nVerdict:` markers that `load.format_example` writes, so
  the baseline trains on the same text the LLM sees, with no second trip to the
  raw dataset. A gold completion that fails schema parse raises loudly.

### 2. Metrics — `evaluate_llm` + `_binary_metrics`
- precision / recall / F1 (`average="binary"`, `zero_division=0`) and **PR-AUC**
  (`average_precision_score`) — PR-AUC is the headline number because fraud is
  imbalanced and accuracy is misleading.
- **PR-AUC uses a graded score**, not the flat 0/1 prediction: risk maps
  `low→0.0, medium→0.5, high→1.0`, so a "medium" hedge ranks between low and
  high and the ranking metric is meaningful.
- **Unparseable output is a miss, not a dropped row**: it counts against
  `json_validity` *and* is scored as non-fraud (score 0), so a garbage-emitting
  model is penalised on recall rather than silently shrinking the test set.

### 3. Classical baseline — `train_baseline` / `evaluate_baseline`
TF-IDF (1–2 grams) → XGBoost in an sklearn `Pipeline`. It predicts the binary
label only — by construction it has **no `reason` / `flagged_spans`**, which is
the whole point: it makes the explainability gap concrete instead of asserted.

### 4. Harness wiring + gate — `main()`
- Trains the baseline on `train.jsonl`, evaluates on `test.jsonl`, prints a table.
- **Baseline is a control group, reported but never gates.** A weak baseline is
  evidence, not a build failure.
- **The `config.eval` thresholds gate only the LLM**, and only when
  `--predictions` (raw outputs aligned to the test split) is supplied — that is
  the CI regression target. Until Phase 1 exists there are no predictions, so the
  gate is skipped and the harness still runs end to end on the baseline.
- `--crossdomain <dir>` runs the baseline against a DIFRAUD `test.jsonl` when one
  is produced.

### 5. Offline tests — `tests/test_evaluate.py`
Metric logic only (no network/GPU/model): transcript round-trip, perfect-pred
metrics, unparseable-as-miss, length-mismatch guard, and a `tmp_path` round-trip
of `load_split` deriving gold from completions.

## Verification

- `pytest` → **17 passed** (4 schema + 7 load + 6 new eval).
- `python -m src.eval.evaluate --data data/processed` →
  XGBoost baseline `f1=1.000 pr_auc=1.000 (n=240)`; LLM eval skipped (no preds).
- Gold-as-predictions smoke test → LLM `f1=1.000 json_validity=1.000`, **exit 0**
  (gate passes). Garbage-as-predictions → `f1=0.000 json_validity=0.000`,
  `GATE FAILED`, **exit 1** (gate fails as intended).

## Decisions & trade-offs

- **Baseline scores a perfect 1.000 on BothBosu — and that's expected, not a
  bug.** BothBosu is the synthetic scaffold set: scam types are lexically
  separable and the `reason` field is templated, so TF-IDF alone nails it. It
  confirms the harness is wired correctly; the *meaningful* baseline-vs-LLM gap
  only shows up on TeleAntiFraud/DIFRAUD, where neither set is trivially
  separable. Reporting it honestly beats hiding it.
- **Baseline doesn't gate; LLM does.** The thresholds exist to catch LLM
  regressions, not to fail CI because the control is weak.
- **PR-AUC from graded risk** rather than calibrated probabilities — the LLM
  emits a category, not a probability; the 3-level ordinal is the best honest
  score available without a separate calibration step.

## Follow-ups

- [ ] TeleAntiFraud-28k loader (real training signal) — unblocks a meaningful
      baseline-vs-LLM comparison on the primary set.
- [ ] DIFRAUD loader → run `--crossdomain` for the out-of-distribution number.
- [ ] Phase 1 produces a model → generate a predictions file and exercise the
      LLM eval branch + gate against real outputs.
- [ ] Wire `python -m src.eval.evaluate` into `.github/workflows/eval.yml` once a
      committed predictions artifact exists.
