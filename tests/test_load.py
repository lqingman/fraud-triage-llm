"""Phase 0 data tests — pure mapping/formatting logic, no network, no GPU.
The actual HF download is exercised manually via `python -m src.data.load`."""

import json

import pytest

from src.data.load import bothbosu_row_to_verdict, format_example
from src.data.schema import FraudType, Risk, parse_verdict


@pytest.mark.parametrize(
    "call_type,expected",
    [
        ("ssn", FraudType.ssn_scam),
        ("refund", FraudType.refund_scam),
        ("support", FraudType.tech_support_scam),
        ("reward", FraudType.reward_scam),
        ("something_new", FraudType.other),  # unknown scam type -> other
    ],
)
def test_scam_rows_map_to_fraud_type(call_type, expected):
    v = bothbosu_row_to_verdict({"dialogue": "...", "type": call_type, "label": 1})
    assert v.fraud_type == expected
    assert v.risk == Risk.high
    assert v.is_fraud is True
    assert v.reason  # non-empty (schema requires min_length=1)


def test_non_scam_row_maps_to_none_and_low():
    v = bothbosu_row_to_verdict({"dialogue": "...", "type": "delivery", "label": 0})
    assert v.fraud_type == FraudType.none
    assert v.risk == Risk.low
    assert v.is_fraud is False


def test_format_example_completion_round_trips_through_schema():
    v = bothbosu_row_to_verdict({"dialogue": "hi", "type": "refund", "label": 1})
    ex = format_example("hi there", v.model_dump(mode="json"))
    assert ex["prompt"].endswith("Verdict:")
    assert "hi there" in ex["prompt"]
    # completion must be valid JSON the schema parser accepts (ties Phase 0 to the contract)
    parsed = parse_verdict(ex["completion"])
    assert parsed is not None
    assert parsed.fraud_type == FraudType.refund_scam
    json.loads(ex["completion"])  # completion is strict JSON, not prose-wrapped
