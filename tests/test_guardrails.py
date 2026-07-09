"""Phase 4 guardrails tests — pure, no network/GPU/model. `generate` is always
a stub Callable[[str], str] (or one that raises), exactly the contract
src.serve.app wires to src.serve.llm_client.complete in production."""

from src.data.schema import FraudVerdict, Risk
from src.serve.guardrails import MAX_RETRIES, safe_generate

_FRAUD = '{"risk":"high","fraud_type":"reward_scam","reason":"x","flagged_spans":[]}'


def test_parses_on_first_try():
    verdict = safe_generate(lambda p: _FRAUD, "prompt")
    assert isinstance(verdict, FraudVerdict)
    assert verdict.risk == Risk.high


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else _FRAUD

    verdict = safe_generate(flaky, "prompt")
    assert verdict.risk == Risk.high
    assert calls["n"] == 2


def test_all_attempts_fail_returns_safe_fallback():
    verdict = safe_generate(lambda p: "not json at all", "prompt")
    assert verdict.risk == Risk.medium
    assert verdict.fraud_type.value == "other"
    assert "manual review" in verdict.reason


def test_generate_raising_degrades_to_fallback_not_a_crash():
    def broken(prompt: str) -> str:
        raise RuntimeError("backend unreachable")

    verdict = safe_generate(broken, "prompt")
    assert verdict.risk == Risk.medium


def test_on_invalid_hook_fires_once_per_failed_attempt():
    counts = {"n": 0}
    safe_generate(lambda p: "garbage", "prompt", on_invalid=lambda: counts.__setitem__("n", counts["n"] + 1))
    assert counts["n"] == MAX_RETRIES + 1


def test_on_invalid_hook_does_not_fire_on_success():
    counts = {"n": 0}
    safe_generate(lambda p: _FRAUD, "prompt", on_invalid=lambda: counts.__setitem__("n", counts["n"] + 1))
    assert counts["n"] == 0
