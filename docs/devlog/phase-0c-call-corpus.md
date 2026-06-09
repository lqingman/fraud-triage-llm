# Phase 0c — Primary dataset pivot: English call corpus

**Date:** 2026-06-09
**Status:** Done. This replaces TeleAntiFraud-28k as the primary train/val set.

## Why the pivot

The original plan used **TeleAntiFraud-28k** as the primary train/test set. On
inspection it turned out to be unworkable for this project:

- **It's Chinese** (`language: zh`) — the whole point is an English fraud-triage
  demo, and the default base model (Mistral-7B) is weak at Chinese.
- **It's gated** on HF (`gated: auto`) — every data file (`sft.zip`,
  `binary_classification.zip`, viewer parquet) returns HTTP 401 without accepting
  terms + a token.
- The large English fraud datasets that *do* exist are the wrong shape: either
  **tabular transaction fraud** (millions of rows, no text — pure XGBoost
  territory) or **scam emails/SMS** (messages, not phone calls). There is **no
  large English phone-call-transcript fraud dataset** — that's a real constraint.

Decision (with the user): keep the project on-domain (phone calls), build the
**primary train/val set from the combined English scam-call corpus**, and keep
DIFRAUD strictly as the out-of-distribution test — never mixed into training.
~9k is comfortably enough for QLoRA on a narrow structured-output task.

## What I did

### 1. Combined call-corpus loader — `_load_calls` (`src/data/load.py`)
Unions four English scam-call datasets (verified schemas/sizes from HF), all
`{dialogue, label[, type]}` with label 1 = scam:

| Source | Rows | Fields |
|---|---|---|
| `menaattia/phone-scam-dataset` | 4,000 | dialogue, label |
| `shakeleoatmeal/phone-scam-detection-synthetic` | 1,800 | dialogue, type, label |
| `BothBosu/multi-agent-scam-conversation` | 1,600 | dialogue, type, labels |
| `BothBosu/single-agent-scam-conversations` | 1,600 | dialogue, type, labels |

De-dupes on normalized dialogue text (the BothBosu-family sets can overlap).

### 2. Shared mapping — `call_row_to_verdict` + `_scam_type_to_fraudtype`
Generalized the old BothBosu-specific mapping into a shared one used by all call
sources (`bothbosu_row_to_verdict` now delegates to it):
- `SCAM_TYPE_MAP` expanded (ssn / tech_support / refund / reward / impersonation
  synonyms) with exact-then-substring matching; unknown scam types → `other`.
- Handles sources with **no `type` field** (menaattia) cleanly — no literal
  "None" leaking into the templated reason.
- No gold reason/spans exist → templated reason, empty spans (same honest
  limitation as the prototype).

### 3. Registered `--dataset calls`; TeleAntiFraud stub now documents the drop.
`calls` goes through the standard stratified 75/10/15 re-split in `main()`.
Updated `config.data.primary` → `english-call-corpus`.

### 4. Tests — `tests/test_load.py`
Added type-taxonomy parametrization + a no-`type` round-trip test. **26 passed.**

## Verification

- `python -m src.data.load --dataset calls --out data/processed` →
  **9,000 rows → 6749 / 901 / 1350** train/val/test, **fraud_ratio 0.500** in all
  three (label polarity consistent across all four sources; stratification holds).
- `fraud_type` distribution (not collapsed): none 4500, other 2000, ssn 700,
  tech_support 700, refund 700, reward 400.
- End-to-end eval (`--data data/processed --crossdomain data/processed/difraud`):
  - **In-distribution** call test: XGBoost f1 **0.989**, pr_auc **0.999**.
  - **Cross-domain** DIFRAUD: XGBoost f1 **0.527**, pr_auc **0.450**.

  That ~0.99 → ~0.45 PR-AUC drop is the honest in-distribution-vs-OOD
  generalization gap the data-strategy note is about — now a *meaningful* number
  because there's a real primary training set behind it. It's the bar the
  fine-tuned LLM has to beat in Phase 2.

## Decisions & trade-offs

- **DIFRAUD never enters training.** Calls (train) vs messages/news (DIFRAUD test)
  share no source, so DIFRAUD stays a genuine OOD exam — no leakage, no
  spam-label noise.
- **Synthetic, no gold rationale.** Same limitation as BothBosu; reasons stay
  templated. Recovering real rationales would require LLM-synthesized reasons
  (a possible later step).
- **~9k is enough for QLoRA.** Narrow structured task + rank-16 adapter; the
  val split + few epochs guard overfitting.

## Follow-ups

- [ ] Update README datasets table + the data-strategy devlog to reflect the
      TeleAntiFraud → English-call-corpus pivot.
- [ ] (Optional) LLM-synthesized gold reasons/flagged_spans to enrich the
      explanation training signal.
- [ ] Phase 1: QLoRA fine-tune on these splits (prefer a multilingual-capable
      base only if Chinese is ever reintroduced; English-only here).
