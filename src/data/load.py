"""Phase 0 — load fraud datasets and format them into instruction examples.

Each example becomes:  PROMPT (system + transcript)  ->  COMPLETION (verdict JSON)

Currently implemented: BothBosu/scam-dialogue (the text-only prototype set).
TeleAntiFraud-28k / DIFRAUD loaders are still stubs (see _load_* below).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.data.schema import FraudType, FraudVerdict, Risk

SYSTEM_PROMPT = (
    "You are a fraud-triage analyst. Read the phone-call transcript and respond "
    "with ONLY a JSON object: {risk, fraud_type, reason, flagged_spans}. "
    "flagged_spans must be verbatim phrases from the transcript."
)

CONFIG_PATH = Path("config/config.yaml")

# BothBosu `type` -> our FraudType. label-0 types (delivery/insurance/
# telemarketing/wrong) carry no fraud and map to `none`.
BOTHBOSU_TYPE_MAP = {
    "ssn": FraudType.ssn_scam,
    "refund": FraudType.refund_scam,
    "support": FraudType.tech_support_scam,
    "reward": FraudType.reward_scam,
}


def format_example(transcript: str, verdict_json: dict) -> dict:
    """Render one (prompt, completion) instruction pair."""
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nVerdict:",
        "completion": json.dumps(verdict_json, ensure_ascii=False),
    }


def bothbosu_row_to_verdict(row: dict) -> FraudVerdict:
    """Map one BothBosu row {dialogue, type, label} onto a FraudVerdict.

    BothBosu has no gold reason/spans, so we synthesize a short templated
    reason and leave flagged_spans empty. Richer rationales come later from
    TeleAntiFraud. Validating through FraudVerdict makes bad rows fail loudly.
    """
    call_type = str(row["type"]).strip().lower()
    is_scam = int(row["label"]) == 1

    if is_scam:
        fraud_type = BOTHBOSU_TYPE_MAP.get(call_type, FraudType.other)
        risk = Risk.high
        reason = f"Caller exhibits a {call_type} scam pattern."
    else:
        fraud_type = FraudType.none
        risk = Risk.low
        reason = f"No fraud indicators; appears to be a legitimate {call_type} call."

    return FraudVerdict(
        risk=risk,
        fraud_type=fraud_type,
        reason=reason,
        flagged_spans=[],
    )


def _load_bothbosu():
    """Load BothBosu/scam-dialogue and yield (transcript, verdict_dict) pairs.

    We concatenate the dataset's own train/test because we re-split
    deterministically below (config.data.test_size / val_size).
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset("BothBosu/scam-dialogue")
    rows = concatenate_datasets([ds["train"], ds["test"]])
    for row in rows:
        verdict = bothbosu_row_to_verdict(row)
        yield row["dialogue"], verdict.model_dump(mode="json")


def _load_teleantifraud():
    """TODO: load TeleAntiFraud-28k (text side) from HF datasets."""
    raise NotImplementedError("Phase 0 follow-up: implement TeleAntiFraud loader")


def _load_difraud():
    """TODO: load redasers/difraud — used only as cross-domain eval set."""
    raise NotImplementedError("Phase 0 follow-up: implement DIFRAUD loader")


LOADERS = {
    "bothbosu": _load_bothbosu,
    "teleantifraud": _load_teleantifraud,
    "difraud": _load_difraud,
}


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_jsonl(path: Path, pairs: list[tuple[str, dict]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for transcript, verdict in pairs:
            f.write(json.dumps(format_example(transcript, verdict), ensure_ascii=False) + "\n")


def _fraud_ratio(pairs: list[tuple[str, dict]]) -> float:
    if not pairs:
        return 0.0
    fraud = sum(1 for _, v in pairs if v["risk"] in (Risk.medium.value, Risk.high.value))
    return fraud / len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=LOADERS, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    seed = cfg["train"]["seed"]

    pairs = list(LOADERS[args.dataset]())
    if not pairs:
        raise RuntimeError(f"loader '{args.dataset}' produced no rows")

    # Stratify on the binary fraud label so class balance survives the split.
    labels = [int(v["risk"] in (Risk.medium.value, Risk.high.value)) for _, v in pairs]

    # First carve out the held-out test split, then carve val from the rest.
    trainval, test, trainval_labels, _ = train_test_split(
        pairs, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=val_fraction, random_state=seed, stratify=trainval_labels
    )

    splits = {"train": train, "val": val, "test": test}
    for name, rows in splits.items():
        _write_jsonl(args.out / f"{name}.jsonl", rows)

    print(f"Wrote splits to {args.out} (dataset={args.dataset}, total={len(pairs)}):")
    for name, rows in splits.items():
        print(f"  {name:5s}: {len(rows):5d} rows  fraud_ratio={_fraud_ratio(rows):.3f}")


if __name__ == "__main__":
    main()
