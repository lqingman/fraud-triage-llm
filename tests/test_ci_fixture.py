"""Guards the committed tests/fixtures/ci_eval/* files against silently
rotting out of sync with the {prompt, completion} contract the eval harness
depends on. These rows are synthetic (invented, not scraped from any real
dataset) and exist only to prove the CI eval-gate mechanism runs end to end —
see docs/devlog/phase-5-ci-gate.md. They are not evidence of model accuracy."""

import json
from pathlib import Path

from src.data.schema import parse_verdict
from src.eval.evaluate import _load_predictions, load_split

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ci_eval"


def test_fixture_files_exist():
    for name in ("train.jsonl", "test.jsonl", "predictions.jsonl"):
        assert (FIXTURE_DIR / name).exists(), f"missing CI eval fixture: {name}"


def test_train_and_test_rows_parse_and_have_both_classes():
    for split in ("train.jsonl", "test.jsonl"):
        texts, gold, completions = load_split(FIXTURE_DIR / split)
        assert len(texts) == len(gold) == len(completions) > 0
        assert any(gold), f"{split} has no fraud rows"
        assert not all(gold), f"{split} has no legit rows"
        for c in completions:
            assert parse_verdict(c) is not None


def test_predictions_align_with_test_split_and_are_valid_json():
    _, gold, completions = load_split(FIXTURE_DIR / "test.jsonl")
    preds = _load_predictions(FIXTURE_DIR / "predictions.jsonl")
    assert len(preds) == len(gold)
    # predictions.jsonl is the gold completions verbatim (deterministic,
    # always-passing gate) — assert that contract explicitly so a future edit
    # that breaks it fails loudly instead of just quietly changing the gate's
    # pass/fail behavior.
    assert preds == completions


def test_predictions_file_has_prediction_key():
    lines = [ln for ln in (FIXTURE_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        obj = json.loads(ln)
        assert "prediction" in obj
