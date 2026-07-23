import json
import sqlite3

from src.data.stream import process_jsonl


def _event(event_id: str, transcript: str = "Please pay 500 dollars") -> str:
    return json.dumps(
        {
            "event_id": event_id,
            "occurred_at": "2026-07-23T10:00:00Z",
            "source": "contact-centre",
            "transcript": transcript,
        }
    )


def test_stream_ingestion_checkpoints_deduplicates_and_dead_letters(tmp_path):
    input_path = tmp_path / "events.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _event("event-1"),
                "{not json",
                _event("event-1", "duplicate event id"),
                json.dumps(
                    {
                        "event_id": "event-2",
                        "occurred_at": "2026-07-23T10:00:00Z",
                        "source": "contact-centre",
                        "transcript": "",
                    }
                ),
                _event("event-3", "Visit https://example.test"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = tmp_path / "state.db"

    first = process_jsonl(input_path, state, partition="calls-0", batch_size=2)
    second = process_jsonl(input_path, state, partition="calls-0", batch_size=2)

    assert first == {
        "input": str(input_path),
        "state": str(state),
        "partition": "calls-0",
        "processed": 5,
        "accepted": 2,
        "dead_lettered": 2,
        "duplicates": 1,
        "skipped_by_checkpoint": 0,
        "next_offset": 5,
    }
    assert second["processed"] == 0
    assert second["next_offset"] == 5

    with sqlite3.connect(state) as connection:
        curated = connection.execute(
            "SELECT event_id, has_url FROM curated_call_events ORDER BY event_id"
        ).fetchall()
        dead_letters = connection.execute(
            "SELECT source_offset, error_code, payload_hash FROM stream_dead_letter "
            "ORDER BY source_offset"
        ).fetchall()
    assert curated == [("event-1", 0), ("event-3", 1)]
    assert [row[0] for row in dead_letters] == [1, 3]
    assert dead_letters[0][1] == "invalid_json"
    assert len(dead_letters[0][2]) == 64
    assert "{not json" not in str(dead_letters)


def test_stream_processes_only_new_appended_records(tmp_path):
    input_path = tmp_path / "events.jsonl"
    input_path.write_text(_event("event-1") + "\n", encoding="utf-8")
    state = tmp_path / "state.db"
    process_jsonl(input_path, state, partition="calls-0")

    with input_path.open("a", encoding="utf-8") as stream:
        stream.write(_event("event-2") + "\n")

    result = process_jsonl(input_path, state, partition="calls-0")
    assert result["processed"] == 1
    assert result["accepted"] == 1
    assert result["next_offset"] == 2
