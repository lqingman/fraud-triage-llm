# Phase 3 — Whisper ASR (CPU)

**Date:** 2026-07-09
**Status:** Done. Real end-to-end test with a real audio file, no GPU.

## Goal

`src/asr/transcribe.py` was a `NotImplementedError` stub and `/triage/audio`
raised unconditionally. The resume claims "audio inference using Whisper" —
close that gap for real, and specifically address the two items the README
roadmap already called out: temp-file cleanup and wiring into the audio
endpoint.

## What I did

### 1. `src/asr/transcribe.py`
`faster-whisper` on CPU with int8 quantization — no GPU needed, unlike QLoRA
training. Models are cached per size in a module-level dict so a long-running
server process doesn't reload weights on every request. Import of
`faster_whisper` is deferred inside `_load_model`, matching the repo's
existing convention (`qlora_train.py`, `predict.py`) so the module imports
cleanly without the dependency installed and the segment-joining logic is
unit-testable via a stub.

### 2. Dependency reclassification
`faster-whisper` was filed under `requirements-serve.txt` ("Linux + NVIDIA
GPU only") alongside `vllm`. That's wrong — ctranslate2 (which faster-whisper
wraps) runs fine on CPU across platforms; nothing about it needs a GPU. Moved
it to `requirements-base.txt` so ASR unit tests run in CI without the
GPU-only image.

### 3. `src/serve/app.py` — wired `/triage/audio`
- Reads the upload into memory, rejects empty files and files over
  `config.serve.max_audio_bytes` (default 25 MiB) with 422 — same pattern as
  the text path's `max_transcript_chars` limit.
- Writes to a `tempfile.NamedTemporaryFile` (suffix taken from the uploaded
  filename, defaulting to `.wav`), transcribes, and **always** deletes the
  temp file in a `finally` block — including when transcription itself
  raises. A transcription failure (corrupt/unreadable audio) becomes a 422
  with a message, not a bare 500.
- Extracted a shared `_triage(transcript) -> FraudVerdict` helper so both
  `/triage/text` and `/triage/audio` go through the exact same
  `format_prompt` → `safe_generate` → guardrails/metrics path — no duplicated
  logic between the two endpoints.

### 4. Tests
- `tests/test_asr.py`: `transcribe()`'s segment-joining logic via a stub
  model (no network/model download); a caching test confirming `_load_model`
  only instantiates `WhisperModel` once per size (monkeypatches
  `faster_whisper.WhisperModel` directly to assert this, since reloading a
  multi-hundred-MB model per request would be a real production problem).
- `tests/test_serve_app.py`: `/triage/audio` happy path, empty-file 422,
  oversized-file 422, transcription-failure 422 (not 500), temp-file cleanup
  verified even when transcription raises, and a privacy test asserting the
  uploaded filename never appears in logs (mirroring the existing
  transcript-never-in-logs test for the text path).

## Verification — a real audio file, not just mocks

Generated a real WAV file via macOS `say` (text-to-speech), reading a scam
script: *"Hello, this is Microsoft support calling. We detected a virus on
your computer. Please provide remote access and your credit card number to
fix this issue immediately."*

Ran real `faster-whisper` (`tiny` model, CPU, int8) directly against it:
```
LANG: en
TEXT: Hello, this is Microsoft Support Calling. We detected a virus on your
computer. Please provide remote access and your credit card number to fix
this issue immediately.
```
Transcription is accurate (whisper capitalized two words differently;
content is exact).

Then started the real server (`uvicorn src.serve.app:app`, no LLM backend
configured — none exists in this environment) and POSTed the same file to
`/triage/audio`:
```
curl -X POST http://127.0.0.1:8125/triage/audio -F "file=@scam_test.wav;type=audio/wav"
→ 200 {"risk":"medium","fraud_type":"other","reason":"Model output could not
  be parsed; flagged for manual review.",...}
```
The 200 (not a 422 transcription error) confirms Whisper genuinely
transcribed the real audio and the request reached the guardrails layer,
which then degraded to the safe fallback because — exactly as in the text
path — there's no LLM backend attached in this sandbox. Server logs showed no
errors, and no leftover temp file existed after the request.

`pytest -q tests/test_asr.py tests/test_serve_app.py` → 21 passed.

## Scope / honesty note

This closes "Whisper" as a real, working, CPU-only ASR stage with genuine
temp-file cleanup — verified against actual synthesized speech, not just
mocked segments. What it does **not** claim: no GPU-accelerated Whisper (CPU
int8 is what's configured and tested), and end-to-end audio→verdict accuracy
under real ASR noise vs. gold-transcript accuracy hasn't been measured (still
open — see README Priority 5).

## Follow-ups

- [ ] Measure end-to-end (audio→verdict) accuracy vs. gold-transcript
      accuracy once a real LLM backend is available to compare against.
- [ ] Consider a larger Whisper model size (`small`/`medium`) if CPU latency
      budget allows — `tiny`/`base` trade accuracy for speed.
