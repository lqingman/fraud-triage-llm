import json
import sqlite3

from src.data.load import format_example
from src.features.store import extract_features, materialize


def _completion(risk: str, fraud_type: str) -> dict:
    return {
        "risk": risk,
        "fraud_type": fraud_type,
        "reason": "fixture",
        "flagged_spans": [],
    }


def test_extract_features_is_deterministic():
    first = extract_features(
        "Agent: pay $500\nCustomer: no\nhttps://bad.example",
        dataset_version="v1",
        dataset_name="calls",
        split="train",
        is_fraud=True,
    )
    second = extract_features(
        "Agent:  pay $500 Customer: no https://bad.example",
        dataset_version="v1",
        dataset_name="calls",
        split="train",
        is_fraud=True,
    )
    assert first.entity_id == second.entity_id
    assert first.turn_count == 3
    assert first.has_url == 1
    assert first.digit_ratio > 0


def test_materialize_is_idempotent_and_queryable(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    rows = [
        format_example("call me at 555-0100", _completion("high", "other")),
        format_example("team meeting tomorrow", _completion("low", "none")),
    ]
    split_path = data_dir / "test.jsonl"
    split_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "dataset": "fixture",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "splits": {"test": {"path": str(split_path), "n": 2}},
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = tmp_path / "features.db"
    first = materialize(data_dir, store)
    second = materialize(data_dir, store)

    assert first["materialized_rows"] == 2
    assert second["stored_rows"] == 2
    with sqlite3.connect(store) as connection:
        fraud, total = connection.execute(
            "SELECT SUM(is_fraud), COUNT(*) FROM transcript_features"
        ).fetchone()
    assert (fraud, total) == (1, 2)
