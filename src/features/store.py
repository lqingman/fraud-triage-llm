"""A small, production-shaped offline feature store backed by SQLite.

It materializes deterministic transcript features from processed JSONL splits.
The schema separates the stable entity key from the dataset version so rebuilds
are idempotent while historical dataset versions remain queryable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.schema import parse_verdict

_TRANSCRIPT_OPEN = "Transcript:\n"
_TRANSCRIPT_CLOSE = "\n\nVerdict:"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript_features (
    entity_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    char_count INTEGER NOT NULL CHECK (char_count >= 0),
    word_count INTEGER NOT NULL CHECK (word_count >= 0),
    turn_count INTEGER NOT NULL CHECK (turn_count >= 1),
    digit_ratio REAL NOT NULL CHECK (digit_ratio BETWEEN 0 AND 1),
    has_url INTEGER NOT NULL CHECK (has_url IN (0, 1)),
    is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
    generated_at TEXT,
    PRIMARY KEY (entity_id, dataset_version)
);
CREATE INDEX IF NOT EXISTS idx_features_version_split
    ON transcript_features(dataset_version, split);
CREATE INDEX IF NOT EXISTS idx_features_version_label
    ON transcript_features(dataset_version, is_fraud);
"""


@dataclass(frozen=True)
class TranscriptFeatures:
    entity_id: str
    dataset_version: str
    dataset_name: str
    split: str
    char_count: int
    word_count: int
    turn_count: int
    digit_ratio: float
    has_url: int
    is_fraud: int
    generated_at: str | None


def _extract_transcript(prompt: str) -> str:
    start = prompt.find(_TRANSCRIPT_OPEN)
    if start == -1:
        return prompt
    start += len(_TRANSCRIPT_OPEN)
    end = prompt.find(_TRANSCRIPT_CLOSE, start)
    return prompt[start:end] if end != -1 else prompt[start:]


def _entity_id(transcript: str) -> str:
    normalized = " ".join(transcript.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_features(
    transcript: str,
    *,
    dataset_version: str,
    dataset_name: str,
    split: str,
    is_fraud: bool,
    generated_at: str | None = None,
) -> TranscriptFeatures:
    chars = len(transcript)
    digit_count = sum(char.isdigit() for char in transcript)
    lowered = transcript.lower()
    nonempty_lines = sum(bool(line.strip()) for line in transcript.splitlines())
    return TranscriptFeatures(
        entity_id=_entity_id(transcript),
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        split=split,
        char_count=chars,
        word_count=len(transcript.split()),
        turn_count=max(nonempty_lines, 1),
        digit_ratio=digit_count / max(chars, 1),
        has_url=int("http://" in lowered or "https://" in lowered or "www." in lowered),
        is_fraud=int(is_fraud),
        generated_at=generated_at,
    )


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def _read_split(
    path: Path,
    *,
    dataset_version: str,
    dataset_name: str,
    split: str,
    generated_at: str | None,
) -> list[TranscriptFeatures]:
    features: list[TranscriptFeatures] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            verdict = parse_verdict(row["completion"])
            if verdict is None:
                raise ValueError(f"invalid completion in {path}:{line_number}")
            features.append(
                extract_features(
                    _extract_transcript(row["prompt"]),
                    dataset_version=dataset_version,
                    dataset_name=dataset_name,
                    split=split,
                    is_fraud=verdict.is_fraud,
                    generated_at=generated_at,
                )
            )
    return features


def upsert_features(connection: sqlite3.Connection, rows: list[TranscriptFeatures]) -> int:
    columns = list(TranscriptFeatures.__dataclass_fields__)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in columns
        if column not in {"entity_id", "dataset_version"}
    )
    sql = (
        f"INSERT INTO transcript_features ({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(entity_id, dataset_version) DO UPDATE SET {updates}"
    )
    connection.executemany(sql, [tuple(asdict(row)[column] for column in columns) for row in rows])
    connection.commit()
    return len(rows)


def materialize(data_dir: Path, store_path: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_metadata = manifest["splits"]
    dataset_version = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    rows: list[TranscriptFeatures] = []
    per_split: dict[str, int] = {}
    for split in ("train", "val", "test"):
        if split not in split_metadata:
            continue
        split_path = data_dir / f"{split}.jsonl"
        split_rows = _read_split(
            split_path,
            dataset_version=dataset_version,
            dataset_name=manifest["dataset"],
            split=split,
            generated_at=manifest.get("generated_at"),
        )
        rows.extend(split_rows)
        per_split[split] = len(split_rows)

    with connect(store_path) as connection:
        upsert_features(connection, rows)
        stored = connection.execute(
            "SELECT COUNT(*) FROM transcript_features WHERE dataset_version = ?",
            (dataset_version,),
        ).fetchone()[0]

    return {
        "dataset": manifest["dataset"],
        "dataset_version": dataset_version,
        "store": str(store_path),
        "materialized_rows": len(rows),
        "stored_rows": stored,
        "splits": per_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize transcript features into SQLite")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--store", type=Path, default=Path("data/features.db"))
    args = parser.parse_args()
    report = materialize(args.data, args.store)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
