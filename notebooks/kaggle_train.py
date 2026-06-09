# Kaggle/Colab training driver (Phase 1).
# Paste into a Kaggle Notebook with GPU (T4 x2 or P100) enabled.
# Keep heavy compute here; reusable logic lives in src/train/qlora_train.py.
#
# Steps:
#   1. !pip install -q transformers peft bitsandbytes accelerate datasets trl wandb
#   2. Upload/clone this repo, or copy src/ into the notebook.
#   3. Load processed data (run src/data/load.py locally first, upload as a Kaggle Dataset).
#   4. from src.train.qlora_train import train, load_config; train(load_config())
#   5. Download the adapter from models/ (it's small — just the LoRA weights).
#
# Kaggle free-GPU budget: ~30 hrs/week. A 7B QLoRA run on this data fits well
# within a single session.

print("See header comments — this is the Kaggle Phase 1 driver stub.")
