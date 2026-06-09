# Kaggle/Colab training driver (Phase 1) — QLoRA fine-tune + predictions.
# Paste cell-by-cell into a Kaggle Notebook with a GPU (T4 x2 or P100) enabled.
# Heavy compute lives here; reusable logic lives in src/train/qlora_train.py
# (training) and src/train/predict.py (inference). Base model defaults to
# Qwen/Qwen2.5-7B-Instruct (ungated — no HF token needed).
#
# Kaggle free-GPU budget: ~30 hrs/week. A 7B QLoRA run on ~6.7k rows fits well
# within a single session.

# --- 1. Install deps (Linux + NVIDIA only; bitsandbytes has no Windows wheel) ---
# !pip install -q -r requirements-base.txt -r requirements-train.txt

# --- 2. Get the repo on the path ---
# import sys
# !git clone https://github.com/<you>/fraud-triage-llm.git
# sys.path.insert(0, "/kaggle/working/fraud-triage-llm")
# %cd /kaggle/working/fraud-triage-llm

# --- 3. Bring in the processed splits ---
# data/processed/*.jsonl are gitignored, so either regenerate them or upload as
# a Kaggle Dataset and copy into place:
#   (a) regenerate:  !python -m src.data.load --dataset calls
#   (b) or copy an uploaded dataset:
#       !mkdir -p data/processed
#       !cp /kaggle/input/<your-dataset>/*.jsonl data/processed/

# --- 4. Optional experiment tracking (no HF token needed for Qwen) ---
# import os
# os.environ["WANDB_API_KEY"] = "<from Kaggle Secrets>"   # omit to skip W&B
# # If you switch base_model to gated Mistral, also:
# # from huggingface_hub import login; login(token="<hf token>")

# --- 5. Train: QLoRA SFT -> saves adapter to config.train.output_dir ---
# from src.train.qlora_train import train, load_config
# cfg = load_config()
# adapter_dir = train(cfg)

# --- 6. Predict on the held-out test split -> reports/predictions.jsonl ---
# !python -m src.train.predict --split data/processed/test.jsonl --out reports/predictions.jsonl

# --- 7. Score it through the Phase 2 gate (f1>=0.85, pr_auc>=0.85, json>=0.98) ---
# !python -m src.eval.evaluate --data data/processed --predictions reports/predictions.jsonl

# --- 8. Persist outputs for download (adapter is small — LoRA weights only) ---
# !cp -r models/qwen7b-fraud-qlora /kaggle/working/
# !cp reports/predictions.jsonl reports/metrics.json /kaggle/working/

print("Phase 1 Kaggle driver — run the commented cells in order on a GPU notebook.")
