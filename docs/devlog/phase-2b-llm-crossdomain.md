# Phase 2b — LLM cross-domain eval on CLAIR (the "why LLM, not XGBoost" payoff)

**Date:** 2026-06-21
**Status:** Code + tests done and wired; awaiting the real Kaggle predict run to
fill in the LLM CLAIR numbers (table below has a placeholder row).

## Goal

Close the loop the earlier phases set up. Phase 0d measured the classical control
collapsing off-domain: the call-trained TF-IDF→XGBoost baseline drops from
in-distribution **f1 0.989 / pr_auc 0.999** to **f1 0.572 / pr_auc 0.381** on
CLAIR (fraud emails). That gap is the project's core argument — *use an LLM
because it generalizes across channels*. To make the argument instead of
asserting it, we need the fine-tuned LLM's number on the **same** CLAIR test set,
reported next to 0.572.

This needs **no retraining**: `predict.py` and `evaluate_llm` are both
domain-agnostic. We only had to let the eval harness ingest a cross-domain
predictions file.

## What I did

### 1. `evaluate_crossdomain()` (`src/eval/evaluate.py`)
Pure function `(pipeline, xd_texts, xd_gold, xd_preds=None) -> {"baseline", "llm"}`.
Always scores the baseline (the call-trained control reused on the new channel);
adds an LLM block via the existing `evaluate_llm` when predictions are supplied,
else `llm=None`. Extracted as a standalone function specifically so it's
unit-testable with a stub pipeline — no XGBoost, no network.

### 2. `--crossdomain-predictions` (CLI)
New flag carrying raw LLM outputs aligned 1:1 with the cross-domain split,
consumed by the same `_load_predictions` loader as in-distribution. `main()`'s
cross-domain branch now prints a baseline row and an LLM row side by side and
folds both into `report["crossdomain"]`. Guard: `--crossdomain-predictions`
without `--crossdomain` is a hard arg error (they must align to a split).

### 3. Reported, never gated
The `config.eval` thresholds keep gating **only** the in-distribution LLM. OOD
F1 is expected to sit below the in-distribution floor (it's a harder, off-channel
test), so gating it would convert an honest generalization check into a false
regression. The cross-domain LLM row is informational; `gate_failed` ignores it.

### 4. Offline tests (`tests/test_evaluate.py`)
- `test_evaluate_crossdomain_with_llm_preds`: stub pipeline that misses one of two
  frauds vs. LLM preds that catch both → asserts baseline recall 0.5, LLM recall
  1.0, json_validity 1.0.
- `test_evaluate_crossdomain_baseline_only`: no preds → `llm is None`.
- Stub returns a numpy array from `predict_proba` so it matches the `[:, 1]`
  slice `evaluate_baseline` does.

## Verification

- `pytest -q` → **36 passed** (was 34 + 2 new cross-domain tests).
- Wiring smoke test (synthesized CLAIR predictions = gold completions, to exercise
  the branch without a model):
  `python -m src.eval.evaluate --data data/processed --crossdomain data/processed/clair --crossdomain-predictions <preds>`
  - In-distribution baseline: **f1 0.989 / pr_auc 0.999 (n=1350)** — matches Phase 0d.
  - Cross-domain baseline: **f1 0.572 / pr_auc 0.381 (n=1926)** — reproduces the
    documented collapse.
  - Cross-domain LLM (preds=gold): f1 1.000 / json_validity 1.000 — confirms the
    branch parses, scores, and prints. (Real numbers from Kaggle replace this.)
- Arg guard: `--crossdomain-predictions` without `--crossdomain` → exit 2 with the
  intended message.

## Kaggle run steps (real LLM predictions)

```bash
# 1) (one-time) generate the CLAIR cross-domain split — already at data/processed/clair
python -m src.data.load --dataset clair --out data/processed/clair

# 2) Kaggle T4: run the fine-tuned adapter over CLAIR (~3-4h; 12h session cap)
python -m src.train.predict \
  --split data/processed/clair/test.jsonl \
  --adapter models/qwen7b-fraud-qlora \
  --out reports/predictions_clair.jsonl

# 3) local: report LLM vs baseline side by side (in-distribution gate still applies)
python -m src.eval.evaluate \
  --data data/processed \
  --predictions reports/predictions.jsonl \
  --crossdomain data/processed/clair \
  --crossdomain-predictions reports/predictions_clair.jsonl
```

## Results — CLAIR cross-domain (n=1926)

| model            | precision | recall | f1        | pr_auc    | json_validity |
|------------------|-----------|--------|-----------|-----------|---------------|
| XGBoost baseline | 0.431     | 0.850  | **0.572** | 0.381     | —             |
| Fine-tuned LLM   | _TBD_     | _TBD_  | **_TBD_** | _TBD_     | _TBD_         |

> Fill the LLM row from the Kaggle run's printout / `reports/metrics.json`
> (`crossdomain.llm`). The thesis holds if LLM f1 ≫ 0.572.

## Caveats / trade-offs

- **CLAIR is email, training is phone calls** — a channel shift, by design (the
  point is cross-channel generalization). The label is unambiguously *fraud*,
  which is why CLAIR replaced DIFRAUD (see phase-0d).
- **`max_seq_len=1024`** (single-T4 config) left-truncates long emails;
  `truncation_side="left"` keeps the trailing `Verdict:` marker so the model still
  emits a verdict rather than continuing the text. Long-email truncation is a known
  limitation of the free-tier config, not a harness bug.
- **json_validity off-domain may differ** from the 0.947 in-distribution figure;
  it's reported but does not gate cross-domain.

## Follow-ups

- [ ] Run Kaggle predict on CLAIR, fill the LLM row, commit `reports/metrics.json`.
- [ ] Add the LLM-vs-baseline cross-domain comparison to the README once real
      numbers land.
- [ ] (Optional) CI: cross-domain predictions are Kaggle artifacts not committed
      to the repo, so `eval.yml` stays in-distribution-only for now.
</content>
</invoke>
