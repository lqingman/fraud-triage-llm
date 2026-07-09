# Phase 5 — CI eval gate: fixing a gate that never ran

**Date:** 2026-07-09
**Status:** Done.

## Goal

`.github/workflows/eval.yml` was supposed to be the project's headline
credibility signal: an automated regression gate that fails the build if F1 /
PR-AUC / JSON-validity drop below `config/config.yaml` thresholds. A repo audit
found it had never actually executed the gate.

## What was broken

Three separate bugs, each independently fatal:

1. **Wrong path.** The workflow checked `data/eval/predictions.jsonl` — a path
   that has never existed anywhere in the repo. The real committed artifact is
   `reports/predictions.jsonl` (produced by Phase 1's Kaggle run).
2. **Missing dependency.** The `pip install` line was
   `pydantic pyyaml pytest scikit-learn pandas numpy` — no `xgboost`, which
   `src.eval.evaluate.train_baseline` imports. Even with the path fixed, the
   gate step would `ImportError` before it could evaluate anything.
3. **Missing data.** `src.eval.evaluate.main()` also needs `train.jsonl` /
   `test.jsonl` under `--data` (default `data/processed/`) to fit the baseline
   and derive gold labels — and `data/processed/` is gitignored. There was
   nothing for the gate to read even with (1) and (2) fixed.

Net effect: every CI run since the workflow was added took the `else` branch
and printed "No eval artifacts yet — skipping gate," silently, on every push.

## What I did

Considered regenerating the real split from HuggingFace in CI
(`python -m src.data.load --dataset calls --out data/processed`), but rejected
it: `reports/predictions.jsonl` is aligned 1:1, in row order, to the exact
`test.jsonl` the original Kaggle run produced. A fresh CI download re-splits
with the same seed but from whatever the upstream HF datasets currently
contain — if any upstream row changed, `evaluate.main()`'s only structural
check is `len(predictions) == len(gold)` (`evaluate_llm`, evaluate.py:100-101),
which would silently pass while grading against misaligned rows. That's worse
than not running at all.

Instead, took the project's own already-documented roadmap item at face value
(`README.md` Priority 3): commit a small, synthetic fixture so the gate
mechanism runs unconditionally, every push.

- `tests/fixtures/ci_eval/{train.jsonl,test.jsonl,predictions.jsonl}` — 30
  invented rows (8 fraud / 22 legit; not scraped from any real dataset, so no
  license question), built through the real `format_example` /
  `FraudVerdict` contract so they're indistinguishable in shape from real
  data. Disjoint 22-row train / 8-row test split (both classes in both).
  `predictions.jsonl` is the test split's gold completions **verbatim** —
  the same "gold-as-predictions" trick Phase 2's devlog already used as a
  smoke test — so the gate is deterministic and always passes on unchanged
  code.
- `tests/test_ci_fixture.py` — asserts the fixture files exist, parse through
  `load_split`, contain both classes, and that `predictions.jsonl` really is
  the test split's completions verbatim, so a future edit that breaks that
  contract fails loudly instead of quietly changing the gate's behavior.
- Rewrote `eval.yml`: `pip install -r requirements-base.txt` (now includes
  `xgboost`, plus the new `fastapi`/`httpx`/`prometheus-client`/`mlflow` deps
  the serving/MLflow tests need — see phase-4-serving.md / phase-4b-mlflow.md),
  full `pytest -q`, then an **unconditional** eval-gate step against the
  fixture, writing outputs to `/tmp/` so CI doesn't dirty tracked `reports/` /
  `models/` paths.

## Verification

Ran locally (Python 3.14.6, no GPU):
```
python -m src.eval.evaluate --data tests/fixtures/ci_eval \
  --predictions tests/fixtures/ci_eval/predictions.jsonl \
  --metrics-out /tmp/ci_metrics.json --baseline-out /tmp/ci_baseline.joblib
```
→ `Fine-tuned LLM: precision=1.000 recall=1.000 f1=1.000 pr_auc=1.000
json_validity=1.000 (n=8)`, exit 0. (Baseline XGBoost scores f1=0 on this tiny
fixture — expected and irrelevant, since the baseline never gates; 8 test rows
isn't enough signal for TF-IDF to generalize, which is exactly why it's a
reported control, not a quality bar.)

`pytest -q` passes locally including the new fixture tests.

## Scope / honesty note

This gate proves **the CI mechanism actually runs and would fail the build on
a real regression** — it does not re-derive the project's headline numbers
(0.941 F1 / 0.937 PR-AUC on 1,350 real held-out calls). That real result
requires a GPU and the trained QLoRA adapter, neither of which exist in GitHub
Actions; it stays a manually-verified, SHA-tracked Kaggle artifact
(`reports/metrics.json`, `reports/predictions.jsonl`). Say this precisely if
asked — "CI enforces the gate mechanism on a synthetic fixture; the real model
number is a manually verified artifact" — rather than implying CI reproduces
the 0.941 figure on every push.

## Follow-ups

- [ ] If a second real Kaggle run ever happens, consider also committing that
      run's exact `data/processed/test.jsonl` (not just predictions) so a
      *second*, real-data gate step could run in CI alongside the synthetic
      one — today only the synthetic fixture is exercised in CI.
