"""Phase 1 train/predict tests — pure helpers only, no GPU/network/torch.
The actual QLoRA run (train) and generation (generate) are exercised manually
on Kaggle via notebooks/kaggle_train.py and documented in the devlog."""

import json

from src.data.load import format_example
from src.eval.evaluate import _load_predictions
from src.train.predict import (
    format_prediction_record,
    read_prompts,
    write_predictions,
)
from src.train.qlora_train import RESPONSE_TEMPLATE, build_sft_text

_FRAUD = '{"risk":"high","fraud_type":"reward_scam","reason":"x","flagged_spans":[]}'
_LEGIT = '{"risk":"low","fraud_type":"none","reason":"x","flagged_spans":[]}'


def test_build_sft_text_contains_prompt_completion_and_eos():
    row = {"prompt": "system\n\nTranscript:\nhi\n\nVerdict:", "completion": _FRAUD}
    text = build_sft_text(row, eos="</s>")
    assert row["prompt"] in text
    assert row["completion"] in text
    assert text.endswith("</s>")  # eos lands at the very end -> teaches stopping


def test_response_template_appears_once_at_boundary():
    row = {"prompt": "system\n\nTranscript:\nhi\n\nVerdict:", "completion": _LEGIT}
    text = build_sft_text(row)
    # The marker is unique: it only appears at the prompt/completion boundary so
    # DataCollatorForCompletionOnlyLM masks exactly the prompt.
    assert text.count(RESPONSE_TEMPLATE) == 1


def test_response_template_matches_phase0_format():
    # Guards against the marker drifting out of sync with load.format_example.
    row = format_example("caller: hi there", json.loads(_FRAUD))
    assert row["prompt"].endswith(RESPONSE_TEMPLATE)


def test_format_prediction_record_uses_prediction_key():
    rec = format_prediction_record(_FRAUD)
    assert rec == {"prediction": _FRAUD}


def test_write_predictions_round_trips_through_eval_loader(tmp_path):
    # Producer (predict) <-> consumer (evaluate) contract: what we write must be
    # exactly what the eval harness reads back, in order.
    preds = [_FRAUD, _LEGIT, "wrapped ```{\"risk\":\"medium\"}```"]
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, preds)
    assert _load_predictions(path) == preds


def test_read_prompts_round_trips_format_example(tmp_path):
    rows = [
        format_example("scam transcript", json.loads(_FRAUD)),
        format_example("legit transcript", json.loads(_LEGIT)),
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    prompts = read_prompts(path)
    assert len(prompts) == 2
    assert all(p.endswith("Verdict:") for p in prompts)
