"""Phase 2 eval tests — metric logic only, no network, no GPU, no model.
The XGBoost baseline + full harness are exercised manually via
`python -m src.eval.evaluate`."""

import json

import pytest

from src.data.load import SYSTEM_PROMPT, format_example
from src.eval.evaluate import (
    _extract_transcript,
    evaluate_llm,
    load_split,
)

_FRAUD = '{"risk":"high","fraud_type":"reward_scam","reason":"x","flagged_spans":[]}'
_LEGIT = '{"risk":"low","fraud_type":"none","reason":"x","flagged_spans":[]}'


def test_extract_transcript_round_trips_format_example():
    ex = format_example("caller: hi there", json.loads(_FRAUD))
    assert _extract_transcript(ex["prompt"]) == "caller: hi there"


def test_extract_transcript_without_markers_returns_input():
    assert _extract_transcript("no markers here") == "no markers here"


def test_evaluate_llm_perfect_predictions():
    gold = [True, False, True]
    preds = [_FRAUD, _LEGIT, _FRAUD]
    m = evaluate_llm(preds, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["json_validity"] == 1.0
    assert m["n"] == 3


def test_evaluate_llm_unparseable_counts_as_miss():
    gold = [True, True]
    preds = [_FRAUD, "totally not json"]  # second is a fraud the model missed
    m = evaluate_llm(preds, gold)
    assert m["json_validity"] == 0.5
    assert m["recall"] == 0.5  # caught 1 of 2 frauds
    assert m["precision"] == 1.0  # no false positives


def test_evaluate_llm_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_llm([_FRAUD], [True, False])


def test_load_split_reads_gold_label_from_completion(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        format_example("scam transcript", json.loads(_FRAUD)),
        format_example("legit transcript", json.loads(_LEGIT)),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    texts, gold, completions = load_split(path)
    assert texts == ["scam transcript", "legit transcript"]
    assert gold == [True, False]
    assert SYSTEM_PROMPT not in texts[0]  # transcript only, not the full prompt
    assert len(completions) == 2
