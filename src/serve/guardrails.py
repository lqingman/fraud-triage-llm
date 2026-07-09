"""Phase 4 — reliability layer. Production fraud triage cannot crash on a
malformed model response or a downed inference backend, so we validate,
retry, and fall back to a safe verdict rather than ever raising out to the
API layer.
"""

from __future__ import annotations

from collections.abc import Callable

from src.data.schema import FraudType, FraudVerdict, Risk, parse_verdict

MAX_RETRIES = 2

_FALLBACK_VERDICT = FraudVerdict(
    risk=Risk.medium,
    fraud_type=FraudType.other,
    reason="Model output could not be parsed; flagged for manual review.",
    flagged_spans=[],
)


def safe_generate(
    generate: Callable[[str], str],
    prompt: str,
    on_invalid: Callable[[], None] | None = None,
) -> FraudVerdict:
    """Call `generate`, parse, retry on failure, then fall back safely.

    Any exception from `generate` (network error, timeout, backend down —
    see src.serve.llm_client.LLMBackendError) is treated the same as an
    unparseable response: retried, then degraded to the safe fallback verdict.
    A caller (e.g. the FastAPI layer) can pass `on_invalid` to observe each
    failed attempt — e.g. to increment a metrics counter — without this
    module taking a dependency on any metrics library.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = generate(prompt if attempt == 0 else prompt + "\nRespond with ONLY valid JSON.")
        except Exception:
            raw = ""
        verdict = parse_verdict(raw)
        if verdict is not None:
            return verdict
        if on_invalid is not None:
            on_invalid()
    # Fail safe: never silently pass a call through as 'low'.
    return _FALLBACK_VERDICT
