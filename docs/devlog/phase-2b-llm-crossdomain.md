# Phase 2b — LLM cross-domain eval on CLAIR (the "why LLM, not XGBoost" payoff)

**Date:** 2026-06-23
**Status:** Complete — the real Kaggle cross-domain run is reported below.

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
    branch parses, scores, and prints.
- Arg guard: `--crossdomain-predictions` without `--crossdomain` → exit 2 with the
  intended message.
- Real Kaggle run (2026-06-23): Qwen2.5-7B plus the trained QLoRA adapter produced
  1,926 predictions on the CLAIR test split. The evaluation completed without a
  runtime error and produced the metrics in the results table below.

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
| XGBoost baseline | 0.435     | 0.861  | **0.578** | 0.414     | —             |
| Fine-tuned LLM   | **0.750** | **0.864** | **0.803** | **0.708** | 0.933       |

The fine-tuned LLM improves cross-domain F1 by **0.225 absolute** over the
baseline (0.803 vs. 0.578), a **38.9% relative gain**. Recall is nearly unchanged
(0.864 vs. 0.861), while precision rises from 0.435 to 0.750; the gain therefore
comes mainly from substantially fewer false positives. PR-AUC also improves by
0.294 (0.708 vs. 0.414). These results support the central hypothesis: compared
with the call-trained TF-IDF/XGBoost control, the fine-tuned LLM transfers much
better from phone-call transcripts to fraud emails.

> These baseline values differ slightly from the earlier Phase 0d snapshot
> (F1 0.572 / PR-AUC 0.381). For a controlled comparison, this table uses the
> baseline and LLM values emitted by the **same 2026-06-23 Kaggle evaluation run**
> on the same regenerated CLAIR split (`n=1926`).

### Saved artifacts and reproduction check

The original Kaggle artifacts are preserved as
`reports/metrics_crossdomain_clair.json` and
`reports/predictions_clair.jsonl`. The prediction file contains exactly 1,926
non-empty JSONL records, each with a `prediction` field.

- metrics SHA-256:
  `0c3414ec2b9eecc2e00937cefb4bb6fc34742167787b46a9e910eae280baf645`
- predictions SHA-256:
  `1d53b1365076ff3cdaf60bf015f483f05160188f16633e011e2807ac825c6d9a`
- exact LLM metrics: precision 0.7497467072, recall 0.8644859813,
  F1 0.8030385241, PR-AUC 0.7083739707, JSON validity 0.9330218069.

A local replay with the saved predictions reproduced all LLM metrics exactly.
It did **not** reproduce the same-run XGBoost row: the local environment yields
F1 0.5723270440 / PR-AUC 0.3813189027, versus Kaggle's F1 0.5775862069 /
PR-AUC 0.4138429832. A subsequent comparison against the supplied Kaggle splits
confirmed that extracted train/test transcripts and labels are identical in
content and order, ruling out data membership or alignment as the cause. The
remaining difference is environment-dependent: the historical Kaggle XGBoost
and scikit-learn versions were not captured, while the requirements specify
only lower bounds. LLM prediction alignment is exactly reproducible; the saved
same-run baseline remains valid evidence, but cannot currently be regenerated
bit-for-bit. Artifact, data, and replay details are recorded in
`reports/crossdomain_clair_manifest.json`.

## Caveats / trade-offs

- **CLAIR is email, training is phone calls** — a channel shift, by design (the
  point is cross-channel generalization). The label is unambiguously *fraud*,
  which is why CLAIR replaced DIFRAUD (see phase-0d).
- **`max_seq_len=1024`** (single-T4 config) left-truncates long emails;
  `truncation_side="left"` keeps the trailing `Verdict:` marker so the model still
  emits a verdict rather than continuing the text. Long-email truncation is a known
  limitation of the free-tier config, not a harness bug.
- **json_validity off-domain may differ** from the 0.947 in-distribution figure;
  the measured value is 0.933 (about 6.7% of outputs are not valid JSON). It is
  reported but does not gate cross-domain, and should be treated as a deployment
  reliability limitation even though classification quality is strong.

## Follow-ups

- [x] Run Kaggle predict on CLAIR and fill the LLM row.
- [ ] Download and commit the Kaggle `reports/metrics.json` and, if artifact size
      permits, `reports/predictions_clair.jsonl` for exact reproducibility.
- [ ] Add the LLM-vs-baseline cross-domain comparison to the README once real
      numbers land.
- [ ] (Optional) CI: cross-domain predictions are Kaggle artifacts not committed
      to the repo, so `eval.yml` stays in-distribution-only for now.
</content>
</invoke>
