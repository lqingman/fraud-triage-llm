import pytest

from src.data.quality import QualityPolicy, enforce_quality, fingerprint, validate_pairs
from src.data.schema import FraudType, FraudVerdict, Risk


def _verdict(risk=Risk.high, fraud_type=FraudType.other, spans=None):
    return FraudVerdict(
        risk=risk,
        fraud_type=fraud_type,
        reason="test",
        flagged_spans=spans or [],
    ).model_dump(mode="json")


def test_validation_deduplicates_and_quarantines_without_raw_text():
    pairs = [
        ("send gift cards now", _verdict(spans=["gift cards"])),
        ("send   gift cards now", _verdict()),
        ("", _verdict()),
    ]
    accepted, rejected, report = validate_pairs(pairs, QualityPolicy())

    assert len(accepted) == 1
    assert [row.reason for row in rejected] == [
        "duplicate_transcript",
        "transcript_too_short",
    ]
    assert report["duplicate_rows"] == 1
    assert report["contract_violations"] == 1
    assert "send gift cards now" not in str([row.as_dict() for row in rejected])
    assert rejected[0].fingerprint == fingerprint(pairs[0][0])


def test_validation_rejects_inconsistent_label_and_nonverbatim_span():
    pairs = [
        ("ordinary appointment call", _verdict(Risk.low, FraudType.reward_scam)),
        ("please pay today", _verdict(spans=["gift card"])),
    ]
    _, rejected, report = validate_pairs(
        pairs, QualityPolicy(max_contract_violation_ratio=1.0)
    )
    assert [row.reason for row in rejected] == [
        "inconsistent_low_risk_fraud_type",
        "flagged_span_not_verbatim",
    ]
    assert report["passed"] is True


def test_quality_gate_fails_above_configured_ratio():
    _, _, report = validate_pairs(
        [("", _verdict()), ("valid row", _verdict())],
        QualityPolicy(max_contract_violation_ratio=0.1),
    )
    with pytest.raises(ValueError, match="data quality gate failed"):
        enforce_quality(report)
