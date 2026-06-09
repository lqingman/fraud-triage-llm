"""Phase 4 — reliability layer. Production fraud triage cannot crash on a
malformed model response, so we validate and repair.

TODO(Phase 4):
  - retry generation with a stricter 'JSON only' instruction on parse failure
  - after N retries, fall back to a safe default (risk=medium, fraud_type=other)
    so the caller always gets a valid FraudVerdict
"""

from __future__ import annotations

from collections.abc import Callable

from src.data.schema import FraudType, FraudVerdict, Risk, parse_verdict

MAX_RETRIES = 2


def safe_generate(generate: Callable[[str], str], prompt: str) -> FraudVerdict:
    """Call `generate`, parse, retry on failure, then fall back safely."""
    for attempt in range(MAX_RETRIES + 1):
        raw = generate(prompt if attempt == 0 else prompt + "\nRespond with ONLY valid JSON.")
        verdict = parse_verdict(raw)
        if verdict is not None:
            return verdict
    # Fail safe: never silently pass a call through as 'low'.
    return FraudVerdict(
        risk=Risk.medium,
        fraud_type=FraudType.other,
        reason="Model output could not be parsed; flagged for manual review.",
        flagged_spans=[],
    )
