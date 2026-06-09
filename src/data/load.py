"""Phase 0 — load fraud datasets and format them into instruction examples.

Each example becomes:  PROMPT (system + transcript)  ->  COMPLETION (verdict JSON)

TODO(Phase 0):
  - implement _load_teleantifraud / _load_bothbosu / _load_difraud
  - map each dataset's native labels onto FraudVerdict fields
  - write deterministic train/val/test split (config.data.test_size/val_size)
  - the test split must be held out and never seen during training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a fraud-triage analyst. Read the phone-call transcript and respond "
    "with ONLY a JSON object: {risk, fraud_type, reason, flagged_spans}. "
    "flagged_spans must be verbatim phrases from the transcript."
)


def format_example(transcript: str, verdict_json: dict) -> dict:
    """Render one (prompt, completion) instruction pair."""
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nVerdict:",
        "completion": json.dumps(verdict_json, ensure_ascii=False),
    }


def _load_bothbosu():
    """TODO: load BothBosu/scam-dialogue from HF datasets and yield raw rows."""
    raise NotImplementedError("Phase 0: implement BothBosu loader")


def _load_teleantifraud():
    """TODO: load TeleAntiFraud-28k (text side) from HF datasets."""
    raise NotImplementedError("Phase 0: implement TeleAntiFraud loader")


def _load_difraud():
    """TODO: load redasers/difraud — used only as cross-domain eval set."""
    raise NotImplementedError("Phase 0: implement DIFRAUD loader")


LOADERS = {
    "bothbosu": _load_bothbosu,
    "teleantifraud": _load_teleantifraud,
    "difraud": _load_difraud,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=LOADERS, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(LOADERS[args.dataset]())  # noqa: F841  (Phase 0 stub)
    print(f"TODO: format {len(rows)} rows and write splits to {args.out}")


if __name__ == "__main__":
    main()
