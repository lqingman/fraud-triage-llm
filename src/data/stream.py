"""Durable micro-batch ingestion for call-event telemetry.

The local adapter reads an append-only JSONL stream, but the processing
boundary is deliberately broker-shaped: partition + monotonic offset + event
payload. Each micro-batch commits curated features, dead-letter metadata, and
the next checkpoint in one SQLite transaction.

Raw transcripts are used only in-memory for feature extraction. The durable
tables retain a SHA-256 entity key and aggregate features, not call content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.features.store import extract_features

SCHEMA = """
CREATE TABLE IF NOT EXISTS curated_call_events (
    event_id TEXT PRIMARY KEY,
    stream_partition TEXT NOT NULL,
    source_offset INTEGER NOT NULL CHECK (source_offset >= 0),
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL,
    transcript_hash TEXT NOT NULL,
    char_count INTEGER NOT NULL CHECK (char_count >= 0),
    word_count INTEGER NOT NULL CHECK (word_count >= 0),
    turn_count INTEGER NOT NULL CHECK (turn_count >= 1),
    digit_ratio REAL NOT NULL CHECK (digit_ratio BETWEEN 0 AND 1),
    has_url INTEGER NOT NULL CHECK (has_url IN (0, 1)),
    ingested_at TEXT NOT NULL,
    UNIQUE (stream_partition, source_offset)
);
CREATE INDEX IF NOT EXISTS idx_curated_events_occurred_at
    ON curated_call_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_curated_events_source
    ON curated_call_events(source);

CREATE TABLE IF NOT EXISTS stream_dead_letter (
    stream_partition TEXT NOT NULL,
    source_offset INTEGER NOT NULL CHECK (source_offset >= 0),
    payload_hash TEXT NOT NULL,
    error_code TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (stream_partition, source_offset)
);

CREATE TABLE IF NOT EXISTS stream_checkpoints (
    stream_partition TEXT PRIMARY KEY,
    next_offset INTEGER NOT NULL CHECK (next_offset >= 0),
    updated_at TEXT NOT NULL
);
"""


class CallEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    source: str = Field(min_length=1, max_length=64)
    transcript: str = Field(min_length=1, max_length=20_000)

    @field_validator("event_id", "source", "transcript")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone is required")
        return value


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def _payload_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error_code(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        field = ".".join(str(item) for item in first["loc"])
        return f"invalid_event:{field}:{first['type']}"
    return "invalid_event"


def checkpoint(connection: sqlite3.Connection, partition: str) -> int:
    row = connection.execute(
        "SELECT next_offset FROM stream_checkpoints WHERE stream_partition = ?",
        (partition,),
    ).fetchone()
    return int(row[0]) if row else 0


def _chunks(
    records: Iterable[tuple[int, str]], batch_size: int
) -> Iterator[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_records(
    connection: sqlite3.Connection,
    records: Iterable[tuple[int, str]],
    *,
    partition: str,
    batch_size: int = 100,
) -> dict:
    """Process offset/payload records with atomic checkpoint advancement."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    current_offset = checkpoint(connection, partition)
    stats = {
        "processed": 0,
        "accepted": 0,
        "dead_lettered": 0,
        "duplicates": 0,
        "skipped_by_checkpoint": 0,
    }

    def pending_records() -> Iterator[tuple[int, str]]:
        last_offset = current_offset - 1
        for offset, payload in records:
            if offset < current_offset:
                stats["skipped_by_checkpoint"] += 1
                continue
            if offset <= last_offset:
                raise ValueError("stream offsets must be strictly increasing")
            last_offset = offset
            yield offset, payload

    pending = pending_records()
    for batch in _chunks(pending, batch_size):
        now = datetime.now(timezone.utc).isoformat()
        with connection:
            for offset, payload in batch:
                stats["processed"] += 1
                try:
                    event = CallEvent.model_validate(json.loads(payload))
                    features = extract_features(
                        event.transcript,
                        dataset_version=f"stream:{partition}",
                        dataset_name="call-event-stream",
                        split="test",
                        is_fraud=False,
                    )
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO curated_call_events (
                            event_id, stream_partition, source_offset, occurred_at,
                            source, transcript_hash, char_count, word_count,
                            turn_count, digit_ratio, has_url, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            partition,
                            offset,
                            event.occurred_at.isoformat(),
                            event.source,
                            features.entity_id,
                            features.char_count,
                            features.word_count,
                            features.turn_count,
                            features.digit_ratio,
                            features.has_url,
                            now,
                        ),
                    )
                    if cursor.rowcount == 1:
                        stats["accepted"] += 1
                    else:
                        stats["duplicates"] += 1
                except (json.JSONDecodeError, ValidationError) as error:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO stream_dead_letter (
                            stream_partition, source_offset, payload_hash,
                            error_code, recorded_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (partition, offset, _payload_hash(payload), _error_code(error), now),
                    )
                    stats["dead_lettered"] += 1

            next_offset = batch[-1][0] + 1
            connection.execute(
                """
                INSERT INTO stream_checkpoints (
                    stream_partition, next_offset, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(stream_partition) DO UPDATE SET
                    next_offset=excluded.next_offset,
                    updated_at=excluded.updated_at
                """,
                (partition, next_offset, now),
            )

    stats["next_offset"] = checkpoint(connection, partition)
    return stats


def process_jsonl(
    input_path: Path,
    state_path: Path,
    *,
    partition: str | None = None,
    batch_size: int = 100,
) -> dict:
    stream_partition = partition or input_path.name
    with connect(state_path) as connection:
        start = checkpoint(connection, stream_partition)
        with input_path.open(encoding="utf-8") as stream:
            records = (
                (offset, payload.rstrip("\n"))
                for offset, payload in enumerate(stream)
                if offset >= start
            )
            result = process_records(
                connection,
                records,
                partition=stream_partition,
                batch_size=batch_size,
            )
    return {
        "input": str(input_path),
        "state": str(state_path),
        "partition": stream_partition,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process an append-only call-event JSONL stream")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("data/stream_state.db"))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    report = process_jsonl(
        args.input,
        args.state,
        partition=args.partition,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
