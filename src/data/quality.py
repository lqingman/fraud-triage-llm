"""Data-quality contracts for the fraud training pipeline.

The public datasets are external inputs, so they are treated as untrusted.
Rows are validated before splitting, bad rows are quarantined by fingerprint
(not raw text), and aggregate quality metrics are persisted in the dataset
manifest for CI/lineage checks.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from src.data.schema import FraudType, FraudVerdict, Risk

Pair = tuple[str, dict]


@dataclass(frozen=True)
class QualityPolicy:
    min_transcript_chars: int = 1
    max_transcript_chars: int = 20_000
    max_contract_violation_ratio: float = 0.01
    deduplicate: bool = True

    @classmethod
    def from_config(cls, config: dict | None) -> "QualityPolicy":
        values = config or {}
        return cls(
            min_transcript_chars=int(values.get("min_transcript_chars", 1)),
            max_transcript_chars=int(values.get("max_transcript_chars", 20_000)),
            max_contract_violation_ratio=float(
                values.get("max_contract_violation_ratio", 0.01)
            ),
            deduplicate=bool(values.get("deduplicate", True)),
        )


@dataclass(frozen=True)
class RejectedRow:
    fingerprint: str
    reason: str
    transcript_chars: int

    def as_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "reason": self.reason,
            "transcript_chars": self.transcript_chars,
        }


def fingerprint(transcript: object) -> str:
    """Stable, privacy-safe identifier used in quarantine reports."""
    raw = transcript if isinstance(transcript, str) else repr(transcript)
    normalized = " ".join(raw.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _contract_error(transcript: object, verdict_dict: object, policy: QualityPolicy) -> str | None:
    if not isinstance(transcript, str):
        return "transcript_not_string"
    length = len(transcript.strip())
    if length < policy.min_transcript_chars:
        return "transcript_too_short"
    if length > policy.max_transcript_chars:
        return "transcript_too_long"
    if not isinstance(verdict_dict, dict):
        return "verdict_not_object"

    try:
        verdict = FraudVerdict.model_validate(verdict_dict)
    except Exception:
        return "invalid_verdict_schema"

    if verdict.risk == Risk.low and verdict.fraud_type != FraudType.none:
        return "inconsistent_low_risk_fraud_type"
    if verdict.risk in (Risk.medium, Risk.high) and verdict.fraud_type == FraudType.none:
        return "inconsistent_fraud_risk_type"
    if any(span not in transcript for span in verdict.flagged_spans):
        return "flagged_span_not_verbatim"
    return None


def validate_pairs(
    pairs: Iterable[Pair], policy: QualityPolicy
) -> tuple[list[Pair], list[RejectedRow], dict]:
    """Validate and deduplicate rows, returning accepted rows and a quality report.

    Contract violations count against the configured failure threshold.
    Duplicates are tracked separately because repeated public-source records are
    expected and should be removed without making an otherwise healthy run fail.
    """
    accepted: list[Pair] = []
    rejected: list[RejectedRow] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    contract_violations = 0

    for transcript, verdict in pairs:
        total += 1
        row_fingerprint = fingerprint(transcript)
        error = _contract_error(transcript, verdict, policy)
        if error is None and policy.deduplicate and row_fingerprint in seen:
            error = "duplicate_transcript"

        if error is not None:
            rejected.append(
                RejectedRow(
                    fingerprint=row_fingerprint,
                    reason=error,
                    transcript_chars=len(transcript) if isinstance(transcript, str) else 0,
                )
            )
            reasons[error] += 1
            if error != "duplicate_transcript":
                contract_violations += 1
            continue

        seen.add(row_fingerprint)
        accepted.append((transcript, verdict))

    violation_ratio = contract_violations / max(total, 1)
    report = {
        "input_rows": total,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "duplicate_rows": reasons["duplicate_transcript"],
        "contract_violations": contract_violations,
        "contract_violation_ratio": violation_ratio,
        "rejection_reasons": dict(sorted(reasons.items())),
        "policy": {
            "min_transcript_chars": policy.min_transcript_chars,
            "max_transcript_chars": policy.max_transcript_chars,
            "max_contract_violation_ratio": policy.max_contract_violation_ratio,
            "deduplicate": policy.deduplicate,
        },
        "passed": violation_ratio <= policy.max_contract_violation_ratio,
    }
    return accepted, rejected, report


def enforce_quality(report: dict) -> None:
    """Fail the pipeline when upstream data breaks the configured contract."""
    if not report["passed"]:
        actual = report["contract_violation_ratio"]
        allowed = report["policy"]["max_contract_violation_ratio"]
        raise ValueError(
            f"data quality gate failed: contract_violation_ratio={actual:.4f} "
            f"exceeds {allowed:.4f}"
        )
