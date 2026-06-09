"""Phase 4 — FastAPI serving layer.

Two endpoints:
  POST /triage/text   {transcript}      -> FraudVerdict
  POST /triage/audio  (file upload)     -> FraudVerdict   (Whisper -> LLM)

Backed by vLLM for the LLM (continuous batching). Includes structured request
logging + a /metrics endpoint for observability.

TODO(Phase 4):
  - wire vLLM engine (or an OpenAI-compatible vLLM server) as `generate`
  - wrap generation with guardrails.safe_generate
  - add latency logging (p50/p95) and a /metrics endpoint
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from src.data.schema import FraudVerdict
from src.serve.guardrails import safe_generate

app = FastAPI(title="Fraud-Triage-LLM")


class TextRequest(BaseModel):
    transcript: str


def _generate(prompt: str) -> str:  # TODO(Phase 4): replace with vLLM call
    raise NotImplementedError("Phase 4: connect vLLM engine")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage/text", response_model=FraudVerdict)
def triage_text(req: TextRequest) -> FraudVerdict:
    return safe_generate(_generate, req.transcript)


@app.post("/triage/audio", response_model=FraudVerdict)
async def triage_audio(file: UploadFile) -> FraudVerdict:
    # from src.asr.transcribe import transcribe
    # transcript = transcribe(saved_path)
    raise NotImplementedError("Phase 3+4: Whisper -> LLM path")
