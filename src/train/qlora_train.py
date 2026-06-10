"""Phase 1 — QLoRA fine-tune the base LLM. Designed to run on a single
Kaggle/Colab GPU (T4/P100). The heavy run lives in notebooks/kaggle_train.py;
this module holds the reusable logic.

Style note (matches src/data/load.py and src/eval/evaluate.py): torch /
transformers / peft / trl imports live INSIDE function bodies, so this module
imports cleanly on Windows-without-GPU and the pure helpers below are
unit-testable offline (see tests/test_train.py).

Training contract: the data is already `{prompt, completion}` pairs from Phase 0,
where prompt ends with the literal "\\n\\nVerdict:" marker. We do raw
concatenation (NOT a chat template) so that training text == eval prompt ==
inference prompt — one string contract that works for Qwen or Mistral with no
per-model branching. Loss is masked to the completion only via TRL's
DataCollatorForCompletionOnlyLM keyed on RESPONSE_TEMPLATE.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

# The exact suffix src.data.load.format_example appends to every prompt
# (f"...\n\nVerdict:"). Used as TRL's response template so the loss falls only on
# the verdict JSON, not the long templated prompt. tests/test_train.py guards
# this against drifting out of sync with Phase 0.
RESPONSE_TEMPLATE = "\n\nVerdict:"


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a {prompt, completion} jsonl split into a list of dicts (in order)."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sft_text(row: dict, eos: str = "") -> str:
    """Render one training string from a {prompt, completion} row.

    Raw concatenation with a single space between the prompt's trailing
    "Verdict:" and the JSON. The explicit `eos` (caller passes
    tokenizer.eos_token) is what teaches the model to stop after the JSON — batch
    generation relies on that learned EOS. Pure; no tokenizer required.
    """
    return f"{row['prompt']} {row['completion']}{eos}"


def _resolve_dtype(name: str):
    """Map a config dtype string to a torch dtype, downgrading bf16 -> fp16 when
    the GPU can't do bf16 (T4 / P100 lack it)."""
    import torch

    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16  # T4/P100 fallback
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _build_bnb_config(qcfg: dict):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=qcfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_resolve_dtype(qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


def _load_model_and_tokenizer(cfg: dict):
    """Load the 4-bit base model + tokenizer, prepped for k-bit LoRA training."""
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right for training (left for generation)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_build_bnb_config(cfg["qlora"]),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def _build_lora_config(qcfg: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=qcfg.get("lora_r", 16),
        lora_alpha=qcfg.get("lora_alpha", 32),
        lora_dropout=qcfg.get("lora_dropout", 0.05),
        target_modules=list(qcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _wandb_available() -> bool:
    """True only if wandb is importable AND an API key is configured, so a run
    without W&B credentials doesn't crash on init."""
    import importlib.util
    import os

    if importlib.util.find_spec("wandb") is None:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def train(cfg: dict) -> str:
    """QLoRA SFT on data/processed/{train,val}.jsonl. Returns the adapter dir.

    Runs on a single GPU (Kaggle/Colab). Saves the LoRA adapter + tokenizer to
    config.train.output_dir, ready for src.train.predict.
    """
    import torch
    from datasets import load_dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg["train"]
    set_seed(tcfg.get("seed", 42))

    model, tokenizer = _load_model_and_tokenizer(cfg)

    # The splits already have `prompt` and `completion` columns. Modern TRL
    # (>=0.20, which dropped DataCollatorForCompletionOnlyLM) detects this
    # prompt-completion format and, with completion_only_loss=True, masks the
    # prompt and trains on the verdict JSON only — no manual collator needed. It
    # also appends EOS to the completion, so the model learns to stop. (RESPONSE_
    # TEMPLATE / build_sft_text remain the documented boundary contract, asserted
    # in tests/test_train.py.)
    ds = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/val.jsonl",
        },
    )

    bf16 = _resolve_dtype(cfg["qlora"].get("bnb_4bit_compute_dtype", "bfloat16")) == torch.bfloat16
    sft_config = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("epochs", 3),
        per_device_train_batch_size=tcfg.get("batch_size", 4),
        per_device_eval_batch_size=tcfg.get("batch_size", 4),
        gradient_accumulation_steps=tcfg.get("grad_accum", 4),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        lr_scheduler_type="cosine",
        max_length=cfg["model"].get("max_seq_len", 2048),
        completion_only_loss=True,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        optim="paged_adamw_8bit",
        report_to=["wandb"] if _wandb_available() else [],
        seed=tcfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=_build_lora_config(cfg["qlora"]),
    )
    trainer.train()

    out_dir = tcfg["output_dir"]
    trainer.save_model(out_dir)  # adapter weights only (small)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter + tokenizer -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    train(load_config())
