# Phase 0f — Data quality gate and offline feature store

## Why this exists

The original loader normalized public datasets and wrote reproducible splits,
but it trusted every upstream row after mapping. That left two production data
engineering gaps:

1. schema-valid output did not prove that transcripts, labels, and evidence
   were mutually consistent;
2. there was no queryable, versioned feature layer for monitoring or classical
   ML consumers.

## Data-quality contract

`src/data/quality.py` runs before train/validation/test splitting and checks:

- transcript type and configurable length boundaries;
- `FraudVerdict` schema validity;
- consistency between risk and fraud type;
- the guarantee that every flagged span is verbatim evidence;
- duplicate transcripts using normalized SHA-256 fingerprints.

Contract violations and duplicates are removed from downstream splits. Rejected
rows are written to `quarantine.jsonl` with only their fingerprint, reason, and
character count, so the report does not create another copy of potentially
sensitive call text. Aggregate counts, rejection reasons, policy values, and
pass/fail state are always written to `quality_report.json` and embedded in
`manifest.json` for successful builds.

Duplicates are reported but do not fail a build because overlapping public
sources are expected. Other contract failures are compared with
`data.quality.max_contract_violation_ratio`; crossing that threshold stops the
pipeline before it can silently train on degraded data.

## Offline feature-store model

`src/features/store.py` materializes deterministic transcript features into
SQLite:

- `entity_id`: normalized transcript SHA-256;
- `dataset_version`: SHA-256 of the immutable manifest;
- dataset name and split;
- character, word, and turn counts;
- digit ratio and URL presence;
- binary fraud label;
- source generation timestamp.

The `(entity_id, dataset_version)` primary key makes rebuilds idempotent and
preserves features from changed dataset versions. Check constraints enforce
valid ranges, and secondary indexes support common version/split and
version/label queries.

```bash
python -m src.data.load --dataset calls --out data/processed
python -m src.features.store \
  --data data/processed \
  --store data/features.db
```

SQLite is deliberate here: it keeps the portfolio project reproducible on a
laptop while exercising relational modeling, constraints, indexes, and upsert
semantics. A production deployment could retain the same logical keys and move
offline storage to a warehouse/lakehouse and online serving to a managed
feature platform.

## Verification

The test suite covers contract failures, duplicate handling, privacy-safe
quarantine metadata, threshold enforcement, deterministic feature extraction,
SQL querying, and idempotent materialization.
