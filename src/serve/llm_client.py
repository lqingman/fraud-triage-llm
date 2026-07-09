"""Phase 4 — HTTP client to an OpenAI-compatible completions endpoint.

Talks to whatever is listening at VLLM_BASE_URL: `vllm serve <model>` starts
exactly this API, so the same client works unmodified against a real vLLM
deployment or any other OpenAI-compatible server. Config is env-var driven so
the same code runs in tests (stubbed), local dev (pointed at a small local
server), and production (pointed at the real vLLM host) with no code changes.

Reuses the JSON-priming trick from src/train/predict.py: the training
completions are `json.dumps({"risk": ...})`, so priming the prompt with the
exact literal prefix `{"risk": "` biases generation to resume inside the JSON
value instead of narrating prose or echoing the schema template — the same
fix that raised offline json_validity in Phase 1.
"""

from __future__ import annotations

import os

_JSON_PRIMER = ' {"risk": "'


class LLMBackendError(Exception):
    """Raised on any network/timeout/protocol failure talking to the backend.

    Kept narrow and separate from a parse failure (see src/serve/guardrails.py)
    so callers can tell "the model said something unparseable" apart from
    "the backend is unreachable" — both currently degrade to the same safe
    fallback verdict, but they're different failure modes worth distinguishing
    in logs/metrics.
    """


def _base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")


def _model_name() -> str:
    return os.environ.get("VLLM_MODEL", "qwen2.5-7b-fraud-qlora")


def _timeout_s() -> float:
    return float(os.environ.get("VLLM_TIMEOUT_S", "30"))


def complete(prompt: str, max_new_tokens: int = 96) -> str:
    """Send a completion request and return the (primer-restored) text.

    Import of httpx lives inside the function body, matching the repo-wide
    convention (src/train/qlora_train.py, src/train/predict.py) of deferring
    heavier imports so modules stay importable without every optional
    dependency installed.
    """
    import httpx

    primed_prompt = prompt + _JSON_PRIMER
    try:
        resp = httpx.post(
            f"{_base_url()}/completions",
            json={
                "model": _model_name(),
                "prompt": primed_prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            timeout=_timeout_s(),
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        raise LLMBackendError(f"LLM backend request failed: {e}") from e

    restored_prefix = _JSON_PRIMER.lstrip()  # '{"risk": "'
    return restored_prefix + text


def is_healthy() -> bool:
    """Best-effort backend reachability check for the /ready probe.

    Requires exactly 200, not merely "not a 5xx" — a 404 (e.g. hitting a
    server that doesn't expose /v1/models, like this app's own default
    VLLM_BASE_URL pointed at itself when no real backend is configured) would
    otherwise slip through as a false-positive "ready". Caught during real
    container testing (see docs/devlog/phase-4d-docker.md).
    """
    import httpx

    try:
        resp = httpx.get(f"{_base_url()}/models", timeout=5.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
