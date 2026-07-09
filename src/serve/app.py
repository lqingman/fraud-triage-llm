"""Phase 4 — FastAPI serving layer.

Two endpoints:
  POST /triage/text   {transcript}      -> FraudVerdict
  POST /triage/audio  (file upload)     -> FraudVerdict   (faster-whisper -> LLM)

Both paths share one triage core: src.serve.llm_client talks to an
OpenAI-compatible completions endpoint (a real vLLM server, or anything else
that speaks the same protocol), src.serve.guardrails.safe_generate retries and
falls back to a safe verdict on a parse failure or a downed backend, and
src.serve.metrics tracks request volume/latency/invalid-output rate for the
/metrics endpoint. The audio path additionally runs faster-whisper (CPU) on
an uploaded file written to a temp path that is always cleaned up, even if
transcription fails.

Privacy note: request logs carry a correlation id, method, path, status, and
latency — never the transcript, audio content, or uploaded filename.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from src.asr.transcribe import transcribe as asr_transcribe
from src.data.load import format_prompt
from src.data.schema import FraudVerdict
from src.serve import llm_client, metrics
from src.serve.guardrails import safe_generate

logger = logging.getLogger("fraud_triage")

CONFIG_PATH = Path("config/config.yaml")


def _load_config() -> dict:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


_CFG = _load_config()
_SERVE_CFG = _CFG.get("serve", {})
_ASR_CFG = _CFG.get("asr", {})
MAX_TRANSCRIPT_CHARS = _SERVE_CFG.get("max_transcript_chars", 20000)
MAX_AUDIO_BYTES = _SERVE_CFG.get("max_audio_bytes", 25 * 1024 * 1024)
WHISPER_MODEL_SIZE = _ASR_CFG.get("whisper_model", "base")

app = FastAPI(title="Fraud-Triage-LLM")


class TextRequest(BaseModel):
    transcript: str


@app.middleware("http")
async def _request_logging_and_metrics(request: Request, call_next):
    correlation_id = uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    duration_s = time.perf_counter() - start

    endpoint = request.url.path
    metrics.observe_request(endpoint=endpoint, status=response.status_code, duration_s=duration_s)

    # Structured, privacy-aware: correlation id + metadata only, never body content.
    logger.info(
        "request",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": endpoint,
            "status_code": response.status_code,
            "duration_ms": round(duration_s * 1000, 2),
        },
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


def _generate(prompt: str) -> str:
    return llm_client.complete(prompt)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Response:
    if llm_client.is_healthy():
        return Response(content='{"status":"ready"}', media_type="application/json", status_code=200)
    return Response(
        content='{"status":"backend unreachable"}', media_type="application/json", status_code=503
    )


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


def _triage(transcript: str) -> FraudVerdict:
    prompt = format_prompt(transcript)
    return safe_generate(_generate, prompt, on_invalid=metrics.record_invalid_output)


@app.post("/triage/text", response_model=FraudVerdict)
def triage_text(req: TextRequest) -> FraudVerdict:
    if len(req.transcript) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"transcript exceeds max_transcript_chars ({MAX_TRANSCRIPT_CHARS})",
        )
    return _triage(req.transcript)


@app.post("/triage/audio", response_model=FraudVerdict)
async def triage_audio(file: UploadFile) -> FraudVerdict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty audio file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=422, detail=f"audio file exceeds max_audio_bytes ({MAX_AUDIO_BYTES})"
        )

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            transcript = asr_transcribe(tmp_path, model_size=WHISPER_MODEL_SIZE)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"could not transcribe audio: {e}") from e
    finally:
        # Always clean up the temp file, even if transcription raised.
        if tmp_path is not None:
            os.unlink(tmp_path)

    return _triage(transcript)
