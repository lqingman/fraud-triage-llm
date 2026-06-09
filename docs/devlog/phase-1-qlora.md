# Phase 1 — QLoRA fine-tune + predictions

**Date:** 2026-06-09
**Status:** Code + offline tests done; the GPU training run is pending (executed
on Kaggle, not local — Windows has no GPU). The eval gate against real model
outputs runs there.

## Goal

Turn the `NotImplementedError` stub into a working QLoRA fine-tune of a 7B base
model on the English call corpus, and — crucially — produce the predictions
artifact the Phase 2 harness already gates on. Phase 2's `--predictions` branch
has been wired but unexercised since there was no model; Phase 1 closes the loop
**train → predict → evaluate**.

## What I did

### 1. Trainer — `src/train/qlora_train.py`
- `train(cfg)`: 4-bit base load (`BitsAndBytesConfig`, nf4 + double-quant) →
  `prepare_model_for_kbit_training` + gradient checkpointing → LoRA on
  q/k/v/o_proj (r=16, α=32) → TRL `SFTTrainer`/`SFTConfig` (3 epochs, eff. batch
  16 = 4×4, lr 2e-4, cosine, paged_adamw_8bit, eval+save per epoch) → save
  adapter + tokenizer to `config.train.output_dir`.
- **Completion-only loss** via `DataCollatorForCompletionOnlyLM` keyed on
  `RESPONSE_TEMPLATE = "\n\nVerdict:"` — the exact suffix `load.format_example`
  appends — so loss falls only on the verdict JSON, not the long templated prompt.
- Pure helpers `build_sft_text` / `read_jsonl` carry no torch import.

### 2. Inference — `src/train/predict.py` (new)
- Loads base+adapter in 4-bit, greedy-decodes a verdict per test prompt, writes
  `reports/predictions.jsonl` with a `"prediction"` key aligned 1:1 (in order) to
  the test split — exactly what `evaluate._load_predictions` consumes.
- Decodes **only the newly generated tokens** so the string is just the
  completion (keeps json_validity high). Asserts `len(preds)==len(prompts)`.

### 3. Offline tests — `tests/test_train.py` (new)
Pure helpers only (no GPU/network/torch): `build_sft_text` content + eos
placement, `RESPONSE_TEMPLATE` uniqueness, marker-matches-Phase-0 guard,
`format_prediction_record` key, and a **producer↔consumer round-trip**
(`write_predictions` → `evaluate._load_predictions`).

### 4. Driver + config
- `notebooks/kaggle_train.py`: real cell-by-cell driver (install → repo → data →
  optional W&B → train → predict → eval gate → persist).
- `config/config.yaml`: `base_model` → `Qwen/Qwen2.5-7B-Instruct`, output_dir →
  `models/qwen7b-fraud-qlora`.

## Verification

- `pytest` → **34 passed** (full suite green, incl. 7 new train tests).
- `python -c "import src.train.qlora_train, src.train.predict"` imports cleanly
  with no torch/transformers installed (heavy imports deferred into functions).
- **Pending (Kaggle):** run `notebooks/kaggle_train.py`, then
  `python -m src.eval.evaluate --data data/processed --predictions reports/predictions.jsonl`
  and record f1 / pr_auc / json_validity vs the gate (≥0.85 / ≥0.85 / ≥0.98).

## Decisions & trade-offs

- **Qwen2.5-7B-Instruct over Mistral-7B-Instruct-v0.3.** Qwen is ungated — no HF
  token/license dance on Kaggle. The code is model-agnostic via
  `config.model.base_model`, so switching back to Mistral is a one-line config
  change (plus an HF login cell).
- **Raw concatenation, not the chat template.** Phase 0 already bakes the system
  prompt into `prompt` as prose. Re-wrapping in `[INST]…`/`<|im_start|>` would
  double-instruct and make the masking marker tokenizer-dependent. Raw concat
  keeps training text == eval prompt == inference prompt — one string contract,
  consistent with `evaluate._extract_transcript`'s literal markers.
- **bf16 → fp16 fallback.** Config asks for bf16 compute, but T4/P100 lack bf16;
  `_resolve_dtype` downgrades to fp16 (compute dtype *and* the SFTConfig flag)
  when `torch.cuda.is_bf16_supported()` is False.
- **Greedy decoding** at inference so the gated metrics are deterministic.

## Follow-ups

- [ ] Run the Kaggle training pass; commit `reports/predictions.jsonl` +
      `reports/metrics.json` and record the numbers here.
- [ ] Wire the committed `reports/predictions.jsonl` into
      `.github/workflows/eval.yml` so CI grades the model on every push.
- [ ] If eff. batch 16 @ seq 2048 OOMs on T4, drop batch_size to 2 or
      max_seq_len to 1536 (noted in the driver).
- [ ] Cross-domain LLM eval: generate a separate predictions file for the CLAIR
      split and compare against the baseline's 0.572 F1.
