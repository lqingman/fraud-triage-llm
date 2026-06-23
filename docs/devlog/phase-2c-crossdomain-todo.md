# Phase 2c — Cross-domain evaluation follow-up TODO

**Started:** 2026-06-23  
**Status:** In progress  
**Current result:** Fine-tuned LLM F1 0.803 vs. XGBoost F1 0.578 on CLAIR
(`n=1926`), with JSON validity 0.933.

## Objective

Turn the promising single-run CLAIR result into a reproducible and defensible
cross-domain evaluation. Work through the tasks in order; do not retrain until
the existing predictions have been preserved and analysed.

## Priority 0 — Preserve the Kaggle run

- [x] Download the original Kaggle artifacts:
  - `metrics.json`
  - `predictions_clair.jsonl`
- [x] Copy them into the repo as:
  - `reports/metrics_crossdomain_clair.json`
  - `reports/predictions_clair.jsonl`
- [x] Verify that the predictions file contains exactly 1,926 non-empty rows.
      All 1,926 rows are valid JSONL records with a `prediction` key.
- [x] Re-run the local evaluator against the saved predictions. The LLM metrics
      match the Kaggle artifact exactly (including full-precision values).
- [x] Verify the supplied Kaggle call splits. Their serialized files differ from
      the repo copies, but extracted train/test transcripts and labels are
      identical in content and order; data differences are ruled out.
- [ ] Resolve baseline environment provenance: the local XGBoost 3.2.0 /
      scikit-learn 1.9.0 replay is F1 0.572 / PR-AUC 0.381, while the historical
      Kaggle run is F1 0.578 / PR-AUC 0.414. The Kaggle package versions were not
      captured, and the requirements only specify lower bounds. Pin versions
      and record an environment manifest for future runs; treat the old baseline
      row as a preserved same-run result rather than an exactly reproducible one.
- [x] Record artifact hashes and exact metrics in the Phase 2b report.

**Done when:** another person can reproduce F1 0.803 from committed inputs
without opening the Kaggle notebook.

## Priority 1 — Error analysis

- [ ] Generate a per-example analysis table containing the gold label, parsed
      prediction, validity flag, and error category.
- [ ] Produce the LLM confusion matrix.
- [ ] Count invalid JSON outputs and split them by fraud/non-fraud gold label.
- [ ] Sample and review at least 20 examples from each available category:
  - false positives;
  - false negatives;
  - invalid JSON;
  - XGBoost wrong / LLM correct.
- [ ] Summarise 4–6 recurring error patterns in a short Markdown report.

**Done when:** the aggregate score is accompanied by concrete failure modes and
actionable examples.

## Priority 2 — Statistical confidence

- [ ] Add a paired bootstrap script for the LLM-vs-XGBoost F1 difference.
- [ ] Report 95% bootstrap confidence intervals for precision, recall, and F1.
- [ ] Run McNemar's test on paired binary correctness.
- [ ] Add the intervals and significance result to the Phase 2b report.

**Done when:** the report quantifies uncertainty and tests whether the observed
0.225 absolute F1 gain is robust on this test set.

## Priority 3 — Base-model ablation

- [ ] Generate CLAIR predictions with unmodified `Qwen/Qwen2.5-7B-Instruct`
      using the same prompt, decoding settings, and test ordering.
- [ ] Evaluate three systems side by side:
  - call-trained TF-IDF/XGBoost;
  - zero-shot base Qwen2.5-7B-Instruct;
  - QLoRA fine-tuned Qwen2.5-7B-Instruct.
- [ ] Attribute the measured gain carefully: base-model prior vs. QLoRA effect.

**Done when:** the project can say how much cross-domain performance comes from
the foundation model and how much comes from fraud fine-tuning.

## Priority 4 — Output reliability

- [ ] Categorise the current invalid outputs (malformed JSON, missing fields,
      extra prose, invalid enum values, or other).
- [ ] Implement constrained decoding or a bounded repair/retry path.
- [ ] Measure both raw-model JSON validity and end-to-end served validity.
- [ ] Target end-to-end JSON validity >= 0.98 without reducing fraud F1.
- [ ] Raise the regression gate only after the improvement is measured.

**Done when:** structured-output reliability is production-appropriate and the
reported metric distinguishes raw generation from repaired serving output.

## Priority 5 — Additional OOD validation

- [ ] Select one fraud-labelled dataset from a third channel (SMS, chat, or
      support conversations), checking label semantics before use.
- [ ] Freeze the model and evaluation procedure before running the new dataset.
- [ ] Compare all three systems and document dataset limitations.

**Done when:** the cross-domain claim is supported on more than one external
corpus and does not depend only on CLAIR email characteristics.

## Reporting guardrails

- Prefer F1, precision, and recall for direct LLM-vs-XGBoost comparisons.
- Note that LLM PR-AUC currently uses ordinal risk scores `{0, 0.5, 1}`, while
  XGBoost uses continuous probabilities; their PR-AUC values are informative
  but not perfectly like-for-like.
- Do not claim full fraud-triage quality from the binary test: CLAIR validates
  `is_fraud`, not rationale quality, evidence spans, or fine-grained fraud type.
- Keep the CLAIR test split evaluation-only; never tune on its labels.

## Progress log

- 2026-06-23: Real Kaggle CLAIR run completed. LLM precision 0.750, recall
  0.864, F1 0.803, PR-AUC 0.708, JSON validity 0.933. Same-run XGBoost F1 0.578.
- 2026-06-23: Follow-up plan created; artifact preservation is the active step.
- 2026-06-23: Preserved both Kaggle artifacts. Local replay exactly reproduced
  the LLM metrics, but exposed an XGBoost baseline mismatch.
- 2026-06-23: Supplied Kaggle train/test transcripts and labels match the local
  inputs exactly. The remaining baseline difference is environment-dependent;
  Kaggle package versions were not captured. Added a hash/version manifest.
