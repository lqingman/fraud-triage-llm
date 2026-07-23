# Phase 0g — Durable micro-batch ingestion

## Goal

The batch loader prepares reproducible training datasets, but the Data/AI
Engineer role also calls for streaming ingestion. This phase adds the delivery
semantics and storage model without pretending that a local portfolio project
has a deployed Kafka or Spark cluster.

## Event contract

Each event contains:

```json
{
  "event_id": "01J...",
  "occurred_at": "2026-07-23T10:00:00Z",
  "source": "contact-centre",
  "transcript": "..."
}
```

Pydantic validates identifiers, timestamps, sources, and transcript bounds.
The processor consumes `(partition, offset, payload)` records, matching the
boundary exposed by common message brokers. `process_jsonl` is the local
append-only adapter.

## Delivery and failure semantics

For every micro-batch, one SQLite transaction commits:

1. curated event metadata and deterministic transcript features;
2. invalid-event dead-letter metadata;
3. the partition's next offset.

If the process stops before commit, the whole batch is replayed. Event IDs and
partition offsets are unique, so replay is idempotent. Once committed, a
restart resumes from the stored offset. Appending new JSONL lines processes
only the new range.

The durable store never retains raw transcripts. Curated rows contain the
transcript hash, length/word/turn features, digit ratio, and URL indicator.
Dead letters contain the offset, error code, and payload hash rather than the
invalid payload.

## Run locally

```bash
python -m src.data.stream \
  --input data/incoming/call-events.jsonl \
  --state data/stream_state.db \
  --partition calls-0 \
  --batch-size 100
```

For a production migration, the JSONL adapter can be replaced by a Kafka
consumer while preserving the event contract, partition/offset boundary, and
idempotency keys. SQLite would normally become a lakehouse/warehouse sink plus
a managed checkpoint store.
