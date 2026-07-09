# Phase 4 — Serving: wire FastAPI to a real (pluggable) backend + guardrails

**Date:** 2026-07-09
**Status:** Text path done and tested. Audio path (Whisper) explicitly still
`NotImplementedError` — out of scope for this batch; see `src/asr/transcribe.py`.

## Goal

`src/serve/app.py`'s `_generate` was a `NotImplementedError` stub and
`src/serve/guardrails.safe_generate` — fully implemented — was wired to
nothing and had zero tests. Close that gap for the text path specifically
(the audio path additionally needs Whisper, which is separately unimplemented
and out of scope here).

There is no GPU, no running vLLM server, and no trained adapter available in
this environment (`models/` is gitignored and absent locally), so "wire to
vLLM" can't mean "prove it against a live 7B model here." It means: build the
integration against vLLM's real wire protocol (OpenAI-compatible
`/v1/completions`), make it swappable via env var, and prove the guardrail/
error-handling logic with a stubbed backend — the same dependency-injection
pattern the repo already uses for GPU-only code (`src/train/qlora_train.py`,
`src/train/predict.py` defer heavy imports so the pure logic is unit-testable
offline).

## What I did

### 1. `src/serve/llm_client.py` (new)
Thin client to an OpenAI-compatible `/v1/completions` endpoint —
`vllm serve <model>` speaks exactly this API, so the same client works
unmodified against a real vLLM deployment. Config via `VLLM_BASE_URL`
(default `http://localhost:8000/v1`), `VLLM_MODEL`, `VLLM_TIMEOUT_S`. Reuses
the `_JSON_PRIMER` trick from `src/train/predict.py` — priming the prompt with
the literal `{"risk": "` prefix biases generation to resume inside the JSON
value, the same fix that raised offline json_validity in Phase 1. Network/
protocol failures raise a narrow `LLMBackendError`, kept separate from a parse
failure so they're distinguishable in logs even though both currently degrade
to the same safe fallback. `is_healthy()` backs the `/ready` probe.

### 2. `src/serve/guardrails.py` (extended)
`safe_generate` now catches *any* exception from `generate` (not just parse
failures) per attempt and retries/falls back exactly as it already did for
malformed JSON — a downed backend degrades to the safe verdict instead of
crashing the request. Added an optional `on_invalid: Callable[[], None]`
hook, fired once per failed attempt, so a caller can observe failed attempts
(e.g. for metrics) without `guardrails.py` taking a dependency on anything
beyond the standard library.

### 3. `src/data/load.py` — extracted `format_prompt`
Pulled the prompt-template string out of `format_example` into a standalone
`format_prompt(transcript) -> str`, now imported by both `format_example`
(training/eval data) and `src/serve/app.py` (live serving). This closes a real
correctness risk: a hand-copied prompt string in `app.py` could silently drift
from the training prompt and change model behavior at serving time without
any test catching it. Existing `format_example` output is unchanged (verified
by the existing test suite still passing).

### 4. `src/serve/app.py` (wired)
- `_generate` → `llm_client.complete`.
- Request-size limit: `config.serve.max_transcript_chars` (added to
  `config/config.yaml`, default 20,000) — oversized transcripts get a 422
  instead of an unbounded-cost request to the backend.
- `/ready`: 200 when `llm_client.is_healthy()`, 503 otherwise (distinct from
  `/health`, which only says the process itself is up).
- Structured request logging via a middleware: a per-request `uuid4`
  correlation id (also returned as `X-Correlation-ID`), method, path, status,
  latency — logged as `extra={...}` fields, **never the transcript or file
  content**. Verified with a `caplog` test that asserts a planted secret
  transcript string never appears in any emitted log record.

### 5. Tests (new)
- `tests/test_guardrails.py`: parses first try; retries then succeeds; all
  attempts fail → fallback; `generate` raising → fallback (not a crash);
  `on_invalid` fires once per failed attempt and not at all on success.
- `tests/test_serve_app.py`: FastAPI `TestClient` + `monkeypatch` of
  `llm_client.complete`/`is_healthy` (no real network). Covers: `/health`,
  happy-path triage, malformed-output fallback (200, not 500), backend-error
  fallback (200, not 500), oversized-transcript 422, `/ready` 200/503, and
  transcript-never-in-logs.

## Verification

- `pytest -q tests/test_guardrails.py tests/test_serve_app.py` → 14 passed.
- Manual smoke test: ran `uvicorn src.serve.app:app` locally with **no vLLM
  server running** (there is none in this environment) and hit it with curl:
  - `GET /health` → `{"status":"ok"}`
  - `GET /ready` → `{"status":"backend unreachable"}`, HTTP 503 (correct — no
    backend is actually listening)
  - `POST /triage/text {"transcript":"hello there"}` → 200,
    `{"risk":"medium","fraud_type":"other","reason":"Model output could not
    be parsed; flagged for manual review.",...}` — the guardrail correctly
    degrades a connection-refused backend to the safe fallback verdict instead
    of the request failing.

This end-to-end behavior (graceful degradation with no backend at all) is
exactly what the guardrails layer is for, and it's the honest, fully-verified
claim for an interview: *"I built and tested the vLLM integration and
guardrails against a stubbed/absent backend — running it against a live vLLM
server needs a GPU host I don't have in this environment, which I haven't
claimed to have exercised."*

## Follow-ups

- [ ] Whisper audio path (`src/asr/transcribe.py`) — still `NotImplementedError`,
      out of scope for this batch.
- [ ] Once a real vLLM server is available (GPU host), point `VLLM_BASE_URL`
      at it and re-run the manual smoke test end-to-end with a real model.
- [ ] Observability (Prometheus/Grafana), load testing, drift monitoring —
      not attempted yet in this phase.
