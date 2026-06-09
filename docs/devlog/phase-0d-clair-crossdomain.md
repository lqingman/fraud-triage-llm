# Phase 0d — Cross-domain eval: DIFRAUD → CLAIR

**Date:** 2026-06-09
**Status:** Done. CLAIR replaces DIFRAUD as the cross-domain (OOD) test.

## Why switch off DIFRAUD

DIFRAUD (the original cross-domain set) had a semantic flaw as a *fraud* test: its label means
**"deceptive", not "fraud"**. Four of its seven domains — fake_news,
political_statements, product_reviews, twitter_rumours — are deception but not
fraud. Projecting `deceptive=1` onto our `is_fraud` would penalize the model for
correctly judging e.g. a fake-news article as non-fraud. Even restricted to its 3
scam domains it's still "deception-corpus labels," so I replaced it with a corpus
whose positive class *is* fraud.

## CLAIR — the replacement

`tasksource/CLAIR_email_fraud`, verified from HF before wiring:

- **Label is explicit `FRAUD` / `NOT_FRAUD`** — no deception-vs-fraud ambiguity.
- Schema `{text, label}`; ships train/val/test; English; ungated.
- Content: the classic CLAIR corpus — advance-fee ("419") scam emails vs.
  legitimate email.

It's a different *channel* (email) and *source* from the phone-call training set,
so it's a clean "same task (scam a victim), different channel" generalization
test. The modality caveat (email, not call) is acceptable: the LLM reads text
either way, and the label being unambiguously fraud was the point.

## What I did

- `clair_row_to_verdict` + `_load_clair` (`src/data/load.py`): FRAUD → high/`other`
  (no fine-grained type or gold rationale in CLAIR), NOT_FRAUD → low/`none`,
  templated reason. Pulls only the canonical `test` split (eval-only).
- Registered `--dataset clair`; added to `EVAL_ONLY`.
- `config.data.crossdomain_eval` → `tasksource/CLAIR_email_fraud`.
- DIFRAUD loader **kept but retired** (unused; comment explains why, and that a
  revival should restrict to its genuinely-fraud domains).
- Tests: 2 CLAIR mapping tests → **28 passed**.

## Verification

- `python -m src.data.load --dataset clair --out data/processed/clair` →
  **1,926 rows, fraud_ratio 0.444**.
- `python -m src.eval.evaluate --data data/processed --crossdomain data/processed/clair`:
  - In-distribution (calls): baseline f1 **0.989**, pr_auc **0.999**.
  - Cross-domain (CLAIR): baseline precision **0.431**, recall **0.850**,
    f1 **0.572**, pr_auc **0.381**.

  The call-trained TF-IDF baseline over-flags emails (high recall, low precision)
  and its ranking quality collapses (pr_auc 0.999 → 0.381) — an honest
  generalization gap measured against text that is *actually labeled fraud*.
  Persisted to `models/baseline_xgb.joblib` + `reports/metrics.json`.

## Follow-ups

- [ ] Update README datasets table (DIFRAUD → CLAIR). *(done in this change)*
- [ ] Phase 1: after the LLM is trained, generate CLAIR predictions and add an
      LLM cross-domain row to compare against this baseline.
