"""Phase 1 — inference: run the fine-tuned QLoRA model over a test split and
write a predictions file the Phase 2 eval harness can grade.

This closes the loop train -> predict -> evaluate. The output is jsonl with a
"prediction" key, aligned 1:1 (and in order) with the input split's rows, which
is exactly what src.eval.evaluate._load_predictions consumes via
`--predictions`.

Style note: torch / transformers / peft imports live INSIDE function bodies so
this module imports cleanly on Windows-without-GPU and the pure I/O helpers are
unit-testable offline (see tests/test_train.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.train.qlora_train import _build_bnb_config, load_config, read_jsonl


def read_prompts(path: str | Path) -> list[str]:
    """Read the `prompt` field from a {prompt, completion} jsonl, in file order."""
    return [row["prompt"] for row in read_jsonl(path)]


def format_prediction_record(prediction: str) -> dict:
    """One predictions-file record. The "prediction" key is the exact key
    src.eval.evaluate._load_predictions looks for first."""
    return {"prediction": prediction}


def write_predictions(path: str | Path, predictions: list[str]) -> None:
    """Write one JSON object per line, aligned to the input split order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(format_prediction_record(pred), ensure_ascii=False) + "\n")


def load_adapter_model(base_model: str, adapter_dir: str, qcfg: dict):
    """Load the 4-bit base model + LoRA adapter for inference.

    padding_side="left" is required for correct batched generation with a
    decoder-only model. use_cache=True (re-enabled vs training) speeds decoding.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Left truncation is critical: the prompt ends with the "\n\nVerdict:"
    # instruction, so a long transcript truncated on the RIGHT (the default)
    # loses that marker and the model just continues the dialogue instead of
    # emitting a verdict. Truncate the START of the transcript instead.
    tokenizer.truncation_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_build_bnb_config(qcfg),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


# Training fed "{prompt} {completion}", i.e. the prompt's "Verdict:" suffix was
# always followed by " {\"risk\"...}". We replay that by priming each prompt with
# this exact opening, so generation starts INSIDE the JSON object and the model
# cannot drift into prose ("This appears to be a scam...") instead of emitting a
# verdict — the dominant json_validity failure mode. The forced "{" is stripped
# from the model's continuation, so we prepend it back to rebuild valid JSON.
_JSON_PRIMER = " {"


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 96,
    batch_size: int = 8,
    max_length: int = 2048,
) -> list[str]:
    """Greedy-decode a verdict JSON per prompt. Returns only the newly generated
    text (the completion, with the primed "{" restored), aligned 1:1 with
    `prompts`.

    Greedy (do_sample=False) keeps the gated metrics deterministic. Each prompt
    is primed with _JSON_PRIMER so the model resumes a JSON object instead of
    narrating; we decode only tokens past the (primed) prompt and prepend "{".
    """
    import torch

    restored_brace = _JSON_PRIMER.lstrip()  # "{"
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = [p + _JSON_PRIMER for p in prompts[start : start + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Strip the prompt: with left padding, the (primed) prompt occupies the
        # first input_len columns for every row in the batch.
        input_len = enc["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(restored_brace + d for d in decoded)
    return outputs


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 1 inference -> predictions for the eval harness")
    ap.add_argument("--split", type=Path, default=Path("data/processed/test.jsonl"),
                    help="{prompt, completion} jsonl to run the model over")
    ap.add_argument("--adapter", type=str, default=cfg["train"]["output_dir"],
                    help="dir with the trained LoRA adapter + tokenizer")
    ap.add_argument("--out", type=Path, default=Path("reports/predictions.jsonl"),
                    help="predictions file (reports/ is committable; data/ and models/ are gitignored)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    prompts = read_prompts(args.split)
    model, tokenizer = load_adapter_model(cfg["model"]["base_model"], args.adapter, cfg["qlora"])
    preds = generate(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_length=cfg["model"].get("max_seq_len", 2048),
    )
    if len(preds) != len(prompts):
        raise RuntimeError(f"predictions ({len(preds)}) != prompts ({len(prompts)}) — alignment broken")
    write_predictions(args.out, preds)
    print(f"Wrote {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
