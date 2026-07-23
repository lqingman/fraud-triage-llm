"""Phase 0 — load fraud datasets and format them into instruction examples.

Each example becomes:  PROMPT (system + transcript)  ->  COMPLETION (verdict JSON)

Implemented loaders:
  - calls      : combined English phone-call corpus (primary train/val set)
  - clair      : tasksource/CLAIR_email_fraud (cross-domain, eval-only exam)
  - bothbosu   : BothBosu/scam-dialogue (original text-only prototype set)
  - difraud    : redasers/difraud (RETIRED cross-domain set; kept but unused)
TeleAntiFraud-28k was dropped (Chinese + gated); see _load_teleantifraud.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.quality import QualityPolicy, RejectedRow, enforce_quality, validate_pairs
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

# CLAIR (tasksource/CLAIR_email_fraud): the cross-domain exam. The classic CLAIR
# corpus of advance-fee ("419") scam emails vs. legitimate email, each row a
# {text, label} pair where label is "FRAUD" / "NOT_FRAUD". Used ONLY for
# out-of-distribution eval, never for training (see EVAL_ONLY below). Its label
# is *fraud* (not generic "deception"), and it's a different channel (email) and
# source from the phone-call training set — a clean "same task, different
# channel" generalization test.
CLAIR_REPO = "tasksource/CLAIR_email_fraud"

# DIFRAUD (redasers/difraud): RETIRED as the cross-domain set. Its label is
# "deceptive", not "fraud" — 4 of its 7 domains (fake_news, political_statements,
# product_reviews, twitter_rumours) are deception but NOT fraud, so scoring a
# fraud model against them is unfair. Replaced by CLAIR. Loader kept but unused;
# if ever revived, restrict to the genuinely-fraud domains below.
DIFRAUD_REPO = "redasers/difraud"
DIFRAUD_DOMAINS = (
    "phishing",
    "job_scams",
    "sms",
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


def format_prompt(transcript: str) -> str:
    """The exact prompt template used at both training and inference time.

    Shared by format_example (training/eval data) and src.serve.app (live
    serving) so the two can never drift apart — a serving prompt that differs
    from the training prompt would silently change model behavior.
    """
    return f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nVerdict:"


def format_example(transcript: str, verdict_json: dict) -> dict:
    """Render one (prompt, completion) instruction pair."""
    return {
        "prompt": format_prompt(transcript),
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


def clair_row_to_verdict(row: dict) -> FraudVerdict:
    """Map one CLAIR {text, label: FRAUD|NOT_FRAUD} row onto a FraudVerdict.

    CLAIR has an explicit fraud label but no fine-grained scam type or gold
    rationale, so fraud rows -> high/`other` with a templated reason, non-fraud
    -> low/`none`. Validating through FraudVerdict makes bad rows fail loudly.
    """
    is_fraud = str(row["label"]).strip().upper() == "FRAUD"
    if is_fraud:
        return FraudVerdict(
            risk=Risk.high,
            fraud_type=FraudType.other,
            reason="Cross-domain email labeled fraudulent (advance-fee / scam pattern).",
            flagged_spans=[],
        )
    return FraudVerdict(
        risk=Risk.low,
        fraud_type=FraudType.none,
        reason="Cross-domain email labeled non-fraudulent.",
        flagged_spans=[],
    )


def _load_clair():
    """Load tasksource/CLAIR_email_fraud and yield (text, verdict_dict) pairs.

    Cross-domain eval only, so we pull just the canonical `test` split and never
    train/val — CLAIR is never trained on.
    """
    from datasets import load_dataset

    ds = load_dataset(CLAIR_REPO, split="test")
    for row in ds:
        verdict = clair_row_to_verdict(row)
        yield row["text"], verdict.model_dump(mode="json")


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
    "clair": _load_clair,
    "teleantifraud": _load_teleantifraud,
    "difraud": _load_difraud,
}

# Datasets used only as out-of-distribution exams: never trained on, so we skip
# the train/val/test re-split and emit a single held-out test.jsonl.
EVAL_ONLY = {"clair", "difraud"}


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


def _sha256_file(path: Path) -> str:
    """Hash a file already written to disk, in fixed-size chunks (no need to
    hold the whole split in memory to fingerprint it)."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def build_manifest(
    dataset: str,
    split_files: dict[str, Path],
    split_rows: dict[str, list[tuple[str, dict]]],
    cfg: dict,
) -> dict:
    """A lightweight, dependency-free stand-in for a full DVC pipeline: an
    immutable record of exactly what was generated and from what config, so a
    later run (or a different machine) can tell whether it reproduced the
    same data. Reads back each already-written split file to hash it, so the
    manifest reflects what's actually on disk, not just what main() thinks it
    wrote.
    """
    return {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_snapshot": {
            "test_size": cfg["data"]["test_size"],
            "val_size": cfg["data"]["val_size"],
            "seed": cfg["train"]["seed"],
        },
        "splits": {
            name: {
                "path": str(split_files[name]),
                "n": len(rows),
                "fraud_ratio": _fraud_ratio(rows),
                "sha256": _sha256_file(split_files[name]),
            }
            for name, rows in split_rows.items()
        },
    }


def write_manifest(out_dir: Path, manifest: dict) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_quarantine(out_dir: Path, rejected: list[RejectedRow]) -> Path:
    """Persist rejected row metadata without leaking raw transcript content."""
    path = out_dir / "quarantine.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
    return path


def write_quality_report(out_dir: Path, report: dict) -> Path:
    """Write diagnostics even when the quality gate subsequently stops the run."""
    path = out_dir / "quality_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    # Deferred: only main()'s train/val/test re-split needs scikit-learn, and
    # deferring it keeps format_prompt/format_example importable (e.g. by
    # src.serve.app, which never re-splits data) without scikit-learn
    # installed — matches the heavy-import-deferral convention used elsewhere
    # in this codebase (src/train/qlora_train.py, src/eval/evaluate.py).
    from sklearn.model_selection import train_test_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=LOADERS, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    seed = cfg["train"]["seed"]

    raw_pairs = list(LOADERS[args.dataset]())
    policy = QualityPolicy.from_config(cfg.get("data", {}).get("quality"))
    pairs, rejected, quality_report = validate_pairs(raw_pairs, policy)
    write_quarantine(args.out, rejected)
    write_quality_report(args.out, quality_report)
    enforce_quality(quality_report)
    if not pairs:
        raise RuntimeError(f"loader '{args.dataset}' produced no rows")

    # Eval-only datasets (DIFRAUD) are never trained on: no re-split, just write
    # the held-out cross-domain test set the eval harness reads via --crossdomain.
    if args.dataset in EVAL_ONLY:
        test_path = args.out / "test.jsonl"
        _write_jsonl(test_path, pairs)
        manifest = build_manifest(args.dataset, {"test": test_path}, {"test": pairs}, cfg)
        manifest["data_quality"] = quality_report
        write_manifest(args.out, manifest)
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
    split_files = {name: args.out / f"{name}.jsonl" for name in splits}
    for name, rows in splits.items():
        _write_jsonl(split_files[name], rows)

    manifest = build_manifest(args.dataset, split_files, splits, cfg)
    manifest["data_quality"] = quality_report
    write_manifest(args.out, manifest)

    print(f"Wrote splits to {args.out} (dataset={args.dataset}, total={len(pairs)}):")
    for name, rows in splits.items():
        print(f"  {name:5s}: {len(rows):5d} rows  fraud_ratio={_fraud_ratio(rows):.3f}")


if __name__ == "__main__":
    main()
