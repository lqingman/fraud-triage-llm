# Phase 0b — DIFRAUD loader (cross-domain eval set)

**Date:** 2026-06-09
**Status:** Done. TeleAntiFraud-28k loader still pending (next).

## Goal

Implement the second Phase 0 follow-up: load `redasers/difraud` as the
**out-of-distribution exam** described in the data-strategy note — the
"completely different workbook" that tells overfitting apart from real
generalization. It is used for evaluation only and never trained on.

## Verifying the schema first (per the BothBosu precedent)

Before writing any mapping I confirmed the real layout from the HF dataset card
+ raw files (BothBosu's fields were likewise "verified from the dataset card"):

- **Repo id** is `redasers/difraud` (matches `config.data.crossdomain_eval`).
- **7 domains**, each a config with `train/validation/test.jsonl`:
  `fake_news, job_scams, phishing, political_statements, product_reviews, sms,
  twitter_rumours`.
- **Every row is `{text: str, label: int}`** — confirmed identical across
  domains. `label` 1 = deceptive/fraud, 0 = non-deceptive. 95,854 total
  (37,282 deceptive ≈ 0.389).
- The repo ships a legacy `difraud.py` loader script that **modern `datasets`
  refuses** ("Dataset scripts are no longer supported"), so we read the raw
  jsonl directly via `load_dataset("json", data_files=<resolve-url>)`.

## What I did

### 1. `difraud_row_to_verdict(row, domain)` — `src/data/load.py`
Maps `{text, label}` + domain onto a `FraudVerdict`:
- deceptive → `risk=high, fraud_type=other`; non-deceptive → `risk=low,
  fraud_type=none`.
- **`fraud_type=other` is deliberate**: DIFRAUD's domains (phishing, fake_news,
  …) don't map onto our phone-scam taxonomy, and it's eval-only, so only the
  binary `is_fraud` projection is ever scored. The domain is kept in the
  `reason` ("Cross-domain phishing text labeled deceptive.") for traceability.
- No gold `reason`/`flagged_spans` exist (like BothBosu) → templated reason,
  empty spans. Every row validated through `FraudVerdict`.

### 2. `_load_difraud()` — eval-only, canonical test split
Pulls **only each domain's `test.jsonl`** (the benchmark's intended eval
protocol) and never train/val — DIFRAUD is never trained on. Loads per-domain so
each row carries its domain into the verdict reason.

### 3. `EVAL_ONLY` branch in `main()`
Added `EVAL_ONLY = {"difraud"}`. For these datasets `main()` **skips the
stratified train/val/test re-split** and writes a single held-out `test.jsonl` —
the file the eval harness reads via `--crossdomain`. (Re-splitting DIFRAUD would
both manufacture a "train" set that contradicts "never trained on" and destroy
its canonical splits.)

### 4. Offline tests — `tests/test_load.py`
Two mapping tests (deceptive→high/other, non-deceptive→low/none, domain in
reason), no network — same pattern as the BothBosu tests.

## Verification

- `pytest` → **19 passed** (added 2 DIFRAUD mapping tests).
- `python -m src.data.load --dataset difraud --out data/processed/difraud` →
  **9,589 rows, fraud_ratio 0.389** (matches DIFRAUD's documented deceptive
  fraction and ~10% test size; per-domain counts sum exactly to 9,589).
- End-to-end with the eval harness:
  `python -m src.eval.evaluate --data data/processed --crossdomain data/processed/difraud`
  → cross-domain XGBoost **f1=0.020, recall=0.010, pr_auc=0.407 (n=9,589)**.

  That collapse is the point, not a bug: a TF-IDF baseline trained on BothBosu
  phone-call dialogue cannot transfer to DIFRAUD emails/SMS/news — the exact
  "memorized the workbook" failure the data-strategy note predicts. It confirms
  the cross-domain machinery works and that this exam is genuinely OOD. The
  number becomes meaningful once the *real* training set (TeleAntiFraud) feeds
  the baseline + LLM.

## Decisions & trade-offs

- **Canonical test split (~9.6k), not all 95k.** Using the benchmark's own test
  split is the standard, tractable eval protocol; we never touch train/val. The
  "95k" in the README is the dataset's size, not the eval size.
- **Output under `data/processed/difraud/`** so it stays gitignored (held-out
  eval data is not committed) and never clobbers BothBosu's `test.jsonl`.
- **`fraud_type=other` for all deceptive rows** — honest about the taxonomy
  mismatch rather than forcing a wrong fine-grained label that nothing scores.

## Follow-ups

- [ ] TeleAntiFraud-28k loader — the real training signal (text in `sft.zip`,
      repo id `JimmyMa99/TeleAntiFraud`). Once it lands, retrain the baseline on
      it and the DIFRAUD number becomes a real generalization measure.
- [ ] After Phase 1: generate DIFRAUD LLM predictions and add an LLM cross-domain
      row to the harness.
