"""Schema tests run in CI — they don't need a GPU, so they guard the contract
on every push even before a model is trained."""

from src.data.schema import FraudVerdict, Risk, parse_verdict


def test_valid_verdict_parses():
    raw = '{"risk":"high","fraud_type":"tech_support_scam","reason":"asked for gift cards","flagged_spans":["gift cards"]}'
    v = parse_verdict(raw)
    assert v is not None
    assert v.risk == Risk.high
    assert v.is_fraud is True


def test_verdict_wrapped_in_prose_is_recovered():
    raw = 'Here is my answer: {"risk":"low","fraud_type":"none","reason":"legit delivery call","flagged_spans":[]} done.'
    v = parse_verdict(raw)
    assert v is not None
    assert v.is_fraud is False


def test_malformed_returns_none():
    assert parse_verdict("not json at all") is None
    assert parse_verdict('{"risk":"banana"}') is None


def test_low_risk_is_not_fraud():
    v = FraudVerdict(risk="low", fraud_type="none", reason="ok", flagged_spans=[])
    assert v.is_fraud is False
