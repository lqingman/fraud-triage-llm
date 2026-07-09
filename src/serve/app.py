"""Phase 4 — FastAPI serving layer.

Two endpoints:
  POST /triage/text   {transcript}      -> FraudVerdict
  POST /triage/audio  (file upload)     -> FraudVerdict   (Whisper -> LLM, still
                                                            Phase 3 scaffolding —
                                                            see src/asr/transcribe.py)

The text path is fully wired: src.serve.llm_client talks to an
OpenAI-compatible completions endpoint (a real vLLM server, or anything else
that speaks the same protocol), src.serve.guardrails.safe_generate retries and
falls back to a safe verdict on a parse failure or a downed backend, and
src.serve.metrics tracks request volume/latency/invalid-output rate for the
/metrics endpoint.

Privacy note: request logs carry a correlation id, method, path, status, and
latency — never the transcript or uploaded file content.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from src.data.load import format_prompt
from src.data.schema import FraudVerdict
from src.serve import llm_client, metrics
from src.serve.guardrails import safe_generate

logger = logging.getLogger("fraud_triage")

CONFIG_PATH = Path("config/config.yaml")


def _load_serve_config() -> dict:
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return cfg.get("serve", {})
    except FileNotFoundError:
        return {}


_SERVE_CFG = _load_serve_config()
MAX_TRANSCRIPT_CHARS = _SERVE_CFG.get("max_transcript_chars", 20000)

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


@app.post("/triage/text", response_model=FraudVerdict)
def triage_text(req: TextRequest) -> FraudVerdict:
    if len(req.transcript) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"transcript exceeds max_transcript_chars ({MAX_TRANSCRIPT_CHARS})",
        )
    prompt = format_prompt(req.transcript)
    return safe_generate(_generate, prompt, on_invalid=metrics.record_invalid_output)


@app.post("/triage/audio", response_model=FraudVerdict)
async def triage_audio(file: UploadFile) -> FraudVerdict:
    # from src.asr.transcribe import transcribe
    raise NotImplementedError("Phase 3+4: Whisper -> LLM path (see src/asr/transcribe.py)")
