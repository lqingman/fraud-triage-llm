# Phase 0e — dataset manifest / lineage

**Date:** 2026-07-09
**Status:** Done.

## Goal

No dataset versioning existed: `python -m src.data.load` wrote
`train/val/test.jsonl` and nothing else — no record of which git commit
produced them, what config (`test_size`/`val_size`/`seed`) generated the
split, or a way to tell later whether a regenerated split matches the
original byte-for-byte. Full DVC is real infrastructure (a remote
artifact store, `.dvc` files, a separate CLI) that's out of proportion for
this project's scale; a small immutable manifest written alongside the
splits gets most of the actual value — reproducibility proof and drift
detection — for a few lines of code and zero new dependencies.

## What I did

`src/data/load.py`:
- `_sha256_file(path)`: hashes a file already on disk in fixed chunks.
- `_git_commit()`: best-effort `git rev-parse HEAD`, `None` outside a repo
  (mirrors the existing helper of the same name/shape in
  `src/eval/evaluate.py` — kept as a small local duplicate rather than a
  shared utils module, consistent with this codebase's per-module-helper
  style).
- `build_manifest(dataset, split_files, split_rows, cfg) -> dict`: per-split
  row count, fraud ratio, and SHA-256 (read back from the file that was just
  written, not computed from the in-memory rows — so the manifest reflects
  what's actually on disk), plus `git_commit`, `generated_at` (UTC), and the
  `test_size`/`val_size`/`seed` config snapshot that produced the split.
- `write_manifest(out_dir, manifest)`: writes `manifest.json` next to the
  splits.
- Wired into `main()` for both the train/val/test branch and the eval-only
  (CLAIR/DIFRAUD) branch.

## Tests (`tests/test_load.py`, extended)

Pure, no network: hash-matches-known-content, manifest records correct
counts/ratio/hash for a synthetic 2-row split, `write_manifest` round-trips
through JSON, and an empty split doesn't divide-by-zero on fraud ratio.

## Verification — a real dataset, not just synthetic rows

```
python -m src.data.load --dataset bothbosu --out /tmp/manifest_test
```
Real HF download (`BothBosu/scam-dialogue`, 1,600 rows) → real
`train.jsonl`/`val.jsonl`/`test.jsonl` → real `manifest.json`:
```json
{
  "dataset": "bothbosu",
  "generated_at": "2026-07-09T23:16:19.632665+00:00",
  "git_commit": "c9998433880a99fca7a2268ba2da97d507e7546a",
  "config_snapshot": {"test_size": 0.15, "val_size": 0.1, "seed": 42},
  "splits": {
    "train": {"n": 1199, "fraud_ratio": 0.500, "sha256": "2a53eb73..."},
    "val":   {"n": 161,  "fraud_ratio": 0.497, "sha256": "df21d40e..."},
    "test":  {"n": 240,  "fraud_ratio": 0.500, "sha256": "75b5bbe8..."}
  }
}
```
Independently recomputed SHA-256 of each written file and confirmed it
matches the manifest's recorded hash for all three splits.

`pytest -q tests/test_load.py` → 22 passed. Full suite → 84 passed.

## Scope / honesty note

This is dataset **lineage** (what produced this exact data, and a way to
verify it wasn't silently corrupted or regenerated differently) — it is
**not** DVC (no remote storage, no data versioning across multiple retained
versions, no `dvc pull`/`dvc push`). If asked, describe it precisely as "a
lightweight reproducibility manifest," not "we use DVC."

## Follow-ups

- [ ] If the project ever needs to retain multiple historical dataset
      versions (not just verify the current one), a real DVC remote becomes
      worth the operational cost — not needed at this scale today.
