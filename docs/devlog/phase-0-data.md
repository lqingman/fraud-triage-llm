# Phase 0 — Data: load, format, split

**Date:** 2026-06-08
**Status:** Done (BothBosu prototype set). TeleAntiFraud / DIFRAUD loaders deferred.

## Goal

Produce formatted instruction data so every downstream phase has something to consume.
Each example is a `{prompt, completion}` pair where the prompt is the system instruction +
transcript and the completion is a verdict JSON validated against `src/data/schema.py`.
Without this, training (Phase 1) and eval (Phase 2) have no inputs.

## What I did

### 1. Fixed the dependency install (root cause)
`pip install -r requirements.txt` was failing because the single file mixed light,
cross-platform deps with heavy GPU/serving deps — `vllm` (no Windows wheel) and
`bitsandbytes` (no reliable Windows wheel) — so the install aborted before the light deps
landed, leaving the `.venv` empty. Phase 0 needs neither.

Split deps by environment:
- `requirements-base.txt` — light, cross-platform; everything needed locally for Phase 0
  (data) and Phase 2 (eval). Installs cleanly on Windows.
- `requirements-train.txt` — Phase 1, Kaggle/Colab GPU only.
- `requirements-serve.txt` — Phase 4, Docker only.
- `requirements.txt` kept as the full reference with a header explaining the split.

Installed `requirements-base.txt` into `.venv` and confirmed.

### 2. Implemented the BothBosu loader — `src/data/load.py`
BothBosu/scam-dialogue schema (verified from the HF dataset card):

| field | type | values |
|---|---|---|
| `dialogue` | str | transcript |
| `type` | str | `ssn`/`refund`/`support`/`reward` (label 1); `delivery`/`insurance`/`telemarketing`/`wrong` (label 0) |
| `label` | int | 1 = scam, 0 = non-scam |

Mapping onto `FraudVerdict`:
- `type` → `FraudType` (`support` → `tech_support_scam`, etc.); unknown scam types → `other`.
- `label` → `risk`: 1 → `high`, 0 → `low` (consistent with `FraudVerdict.is_fraud`).
- `reason`: templated from `type`/`label` — **BothBosu has no gold rationale**.
- `flagged_spans`: empty — no gold spans in this dataset.
- Every mapped row is validated through `FraudVerdict(...)` so bad rows fail loudly.

### 3. Split + write — `main()`
- Concatenate the dataset's own train/test, then re-split deterministically.
- `train_test_split` with `random_state=seed` (42) and `stratify` on the binary fraud label;
  carve out test first, then val from the remainder. Sizes from `config.data` (test 0.15,
  val 0.10).
- Write `train/val/test.jsonl` to `data/processed/` (gitignored — held-out test never committed).

### 4. Offline tests — `tests/test_load.py`
Mapping/formatting tests with hand-built rows; no network, no GPU (CI-safe). Includes a
round-trip check that `format_example` output parses back through `parse_verdict`, tying
Phase 0 output to the schema contract.

## Verification

- `pytest` → **11 passed** (4 schema + 7 new).
- `python -m src.data.load --dataset bothbosu --out data/processed` →
  1,600 rows → **1199 / 161 / 240** train/val/test, fraud ratio **0.500 / 0.497 / 0.500**
  (stratification holds).
- Output inspected: `prompt` ends with `Verdict:` and contains the transcript; `completion`
  is strict JSON the schema parser accepts.
- `git check-ignore data/processed/test.jsonl` confirms the split stays local.

## Decisions & trade-offs

- **Started with BothBosu, not the primary TeleAntiFraud-28k.** Text-only, tiny (1,600 rows),
  CPU-friendly — fast to get a working pipeline end to end. It's the README's designated
  prototype set.
- **Templated reasons.** BothBosu has no gold `reason`/`flagged_spans`, so the model can't
  learn rich explanations from this set alone. Acceptable for prototyping; real rationales
  come from TeleAntiFraud.

## Follow-ups

- [ ] TeleAntiFraud-28k loader with gold reasons/spans (the real training signal).
- [ ] DIFRAUD loader as cross-domain eval only.
- [ ] Phase 2: XGBoost baseline + metrics over these splits.
