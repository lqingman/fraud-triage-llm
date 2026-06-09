"""Phase 1 — QLoRA fine-tune the base LLM. Designed to run on a single
Kaggle/Colab GPU (T4/P100). The heavy run lives in notebooks/kaggle_train.py;
this module holds the reusable logic.

TODO(Phase 1):
  - build BitsAndBytesConfig from config.qlora
  - load 4-bit base model + tokenizer
  - attach LoRA adapters (peft.LoraConfig)
  - train with trl.SFTTrainer on data/processed/train.jsonl
  - log loss / sample generations to Weights & Biases
  - save adapter to config.train.output_dir
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def train(cfg: dict) -> None:
    # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # from peft import LoraConfig, get_peft_model
    # from trl import SFTTrainer, SFTConfig
    raise NotImplementedError(
        "Phase 1: implement QLoRA SFT. Use notebooks/kaggle_train.py for the GPU run."
    )


if __name__ == "__main__":
    train(load_config())
