"""Phase 0 — load fraud datasets and format them into instruction examples.

Each example becomes:  PROMPT (system + transcript)  ->  COMPLETION (verdict JSON)

Implemented loaders:
  - calls      : combined English phone-call corpus (primary train/val set)
  - bothbosu   : BothBosu/scam-dialogue (original text-only prototype set)
  - difraud    : redasers/difraud (cross-domain, eval-only exam)
TeleAntiFraud-28k was dropped (Chinese + gated); see _load_teleantifraud.
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

# Scam-call `type` string -> our FraudType, shared across the English call
# datasets (BothBosu family, menaattia, shakeleoatmeal). Only consulted for
# scam rows; label-0 types (delivery/insurance/appointment/...) map to `none`
# via the is_scam branch below. Unknown scam types fall back to `other`.
SCAM_TYPE_MAP = {
    "ssn": FraudType.ssn_scam,
    "refund": FraudType.refund_scam,
    "support": FraudType.tech_support_scam,
    "tech_support": FraudType.tech_support_scam,
    "tech": FraudType.tech_support_scam,
    "reward": FraudType.reward_scam,
    "prize": FraudType.reward_scam,
    "lottery": FraudType.reward_scam,
    "gift": FraudType.reward_scam,
    "impersonation": FraudType.impersonation,
    "irs": FraudType.impersonation,
    "bank": FraudType.impersonation,
    "government": FraudType.impersonation,
}


def _scam_type_to_fraudtype(call_type: str | None) -> FraudType:
    """Map a free-text scam `type` onto our taxonomy (exact, then substring)."""
    t = (call_type or "").strip().lower()
    if t in SCAM_TYPE_MAP:
        return SCAM_TYPE_MAP[t]
    for key, ft in SCAM_TYPE_MAP.items():
        if key in t:
            return ft
    return FraudType.other

# DIFRAUD (redasers/difraud): the cross-domain exam. 7 domains, each row a
# {text, label} pair (label 1 = deceptive/fraud, 0 = non-deceptive). Used ONLY
# for out-of-distribution eval, never for training (see EVAL_ONLY below).
DIFRAUD_REPO = "redasers/difraud"
DIFRAUD_DOMAINS = (
    "fake_news",
    "job_scams",
    "phishing",
    "political_statements",
    "product_reviews",
    "sms",
    "twitter_rumours",
)

# The English phone-call corpus = the primary train/val set (TeleAntiFraud-28k
# was dropped: it's Chinese). All four share a {dialogue, label, type?} schema
# with label 1 = scam; we union + dedup them. (repo, label_field, type_field).
CALL_SOURCES = (
    ("menaattia/phone-scam-dataset", "label", None),
    ("shakeleoatmeal/phone-scam-detection-synthetic", "label", "type"),
    ("BothBosu/multi-agent-scam-conversation", "labels", "type"),
    ("BothBosu/single-agent-scam-conversations", "labels", "type"),
)


def format_example(transcript: str, verdict_json: dict) -> dict:
    """Render one (prompt, completion) instruction pair."""
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nVerdict:",
        "completion": json.dumps(verdict_json, ensure_ascii=False),
    }


def call_row_to_verdict(is_scam: bool, call_type: str | None = None) -> FraudVerdict:
    """Map a scam-call row (binary label + optional `type`) onto a FraudVerdict.

    These datasets carry no gold reason/spans, so we synthesize a short templated
    reason and leave flagged_spans empty. Validating through FraudVerdict makes
    bad rows fail loudly.
    """
    if is_scam:
        kind = (call_type or "").strip().lower()
        return FraudVerdict(
            risk=Risk.high,
            fraud_type=_scam_type_to_fraudtype(kind),
            reason=f"Caller exhibits a {kind} scam pattern." if kind else "Caller exhibits a scam pattern.",
            flagged_spans=[],
        )
    kind = (call_type or "").strip().lower()
    return FraudVerdict(
        risk=Risk.low,
        fraud_type=FraudType.none,
        reason=f"No fraud indicators; appears to be a legitimate {kind} call." if kind else "No fraud indicators; appears to be a legitimate call.",
        flagged_spans=[],
    )


def bothbosu_row_to_verdict(row: dict) -> FraudVerdict:
    """Map one BothBosu row {dialogue, type, label} onto a FraudVerdict."""
    return call_row_to_verdict(int(row["label"]) == 1, str(row["type"]))


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


def _load_calls():
    """Load the combined English phone-call corpus (the primary train/val set).

    Unions CALL_SOURCES and de-duplicates on the normalized dialogue text (the
    BothBosu-family sets can share dialogues). Each source has a {dialogue,
    label[, type]} schema with label 1 = scam; main() re-splits the result into
    train/val/test. DIFRAUD is the separate cross-domain test and is never mixed
    in here.
    """
    from datasets import concatenate_datasets, load_dataset

    seen: set[str] = set()
    for repo, label_field, type_field in CALL_SOURCES:
        dd = load_dataset(repo)
        rows = concatenate_datasets(list(dd.values()))
        for row in rows:
            dialogue = row["dialogue"]
            key = " ".join(dialogue.split())
            if key in seen:
                continue
            seen.add(key)
            is_scam = int(row[label_field]) == 1
            call_type = row[type_field] if type_field else None
            verdict = call_row_to_verdict(is_scam, call_type)
            yield dialogue, verdict.model_dump(mode="json")


def difraud_row_to_verdict(row: dict, domain: str) -> FraudVerdict:
    """Map one DIFRAUD {text, label} row + its domain onto a FraudVerdict.

    DIFRAUD's domains (phishing, fake_news, ...) don't map onto our phone-scam
    taxonomy, and DIFRAUD ships no gold rationale/spans — but it's eval-only, so
    only the binary `is_fraud` projection is actually scored. Deceptive rows ->
    high/`other`, non-deceptive -> low/`none`; the domain is kept in the reason
    for traceability. Validating through FraudVerdict makes bad rows fail loudly.
    """
    is_fraud = int(row["label"]) == 1
    pretty = domain.replace("_", " ")
    if is_fraud:
        return FraudVerdict(
            risk=Risk.high,
            fraud_type=FraudType.other,
            reason=f"Cross-domain {pretty} text labeled deceptive.",
            flagged_spans=[],
        )
    return FraudVerdict(
        risk=Risk.low,
        fraud_type=FraudType.none,
        reason=f"Cross-domain {pretty} text labeled non-deceptive.",
        flagged_spans=[],
    )


def _load_teleantifraud():
    """Dropped: TeleAntiFraud-28k is Chinese, and gated on HF. The primary
    train set is now the English call corpus (see _load_calls / CALL_SOURCES)."""
    raise NotImplementedError(
        "TeleAntiFraud-28k dropped (Chinese + gated). Use --dataset calls instead."
    )


def _load_difraud():
    """Load redasers/difraud and yield (text, verdict_dict) pairs.

    Used ONLY as the cross-domain eval set, so we pull each domain's canonical
    `test` split (the benchmark's intended evaluation protocol) and never its
    train/val — DIFRAUD is never trained on. We load per-domain so each row
    carries its domain into the verdict reason. The repo ships a legacy
    `difraud.py` loader script (unsupported by modern `datasets`), so we read
    the raw jsonl directly instead.
    """
    from datasets import load_dataset

    for domain in DIFRAUD_DOMAINS:
        url = f"https://huggingface.co/datasets/{DIFRAUD_REPO}/resolve/main/{domain}/test.jsonl"
        ds = load_dataset("json", data_files=url, split="train")
        for row in ds:
            verdict = difraud_row_to_verdict(row, domain)
            yield row["text"], verdict.model_dump(mode="json")


LOADERS = {
    "bothbosu": _load_bothbosu,
    "calls": _load_calls,
    "teleantifraud": _load_teleantifraud,
    "difraud": _load_difraud,
}

# Datasets used only as out-of-distribution exams: never trained on, so we skip
# the train/val/test re-split and emit a single held-out test.jsonl.
EVAL_ONLY = {"difraud"}


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

    # Eval-only datasets (DIFRAUD) are never trained on: no re-split, just write
    # the held-out cross-domain test set the eval harness reads via --crossdomain.
    if args.dataset in EVAL_ONLY:
        _write_jsonl(args.out / "test.jsonl", pairs)
        print(
            f"Wrote eval-only test split to {args.out} "
            f"(dataset={args.dataset}, n={len(pairs)}, fraud_ratio={_fraud_ratio(pairs):.3f})"
        )
        return

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
