"""Phase 4 serving tests — FastAPI TestClient + a stubbed llm_client, no
network/GPU/real vLLM server. Exercises the full request path: guardrails,
privacy-safe logging, size limits, and /ready."""

import logging

from fastapi.testclient import TestClient

from src.serve import llm_client
from src.serve.app import MAX_TRANSCRIPT_CHARS, app

client = TestClient(app)

_FRAUD_JSON = '{"risk":"high","fraud_type":"reward_scam","reason":"caller offered a prize","flagged_spans":["free prize"]}'


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_triage_text_happy_path(monkeypatch):
    monkeypatch.setattr(llm_client, "complete", lambda prompt, **kw: _FRAUD_JSON)
    resp = client.post("/triage/text", json={"transcript": "You've won a free prize!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk"] == "high"
    assert body["fraud_type"] == "reward_scam"


def test_triage_text_malformed_output_falls_back_not_500(monkeypatch):
    monkeypatch.setattr(llm_client, "complete", lambda prompt, **kw: "not valid json at all")
    resp = client.post("/triage/text", json={"transcript": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk"] == "medium"
    assert body["fraud_type"] == "other"


def test_triage_text_backend_error_falls_back_not_500(monkeypatch):
    def raising(prompt, **kw):
        raise llm_client.LLMBackendError("connection refused")

    monkeypatch.setattr(llm_client, "complete", raising)
    resp = client.post("/triage/text", json={"transcript": "hello"})
    assert resp.status_code == 200
    assert resp.json()["risk"] == "medium"


def test_triage_text_oversized_transcript_rejected():
    oversized = "a" * (MAX_TRANSCRIPT_CHARS + 1)
    resp = client.post("/triage/text", json={"transcript": oversized})
    assert resp.status_code == 422


def test_ready_ok_when_backend_healthy(monkeypatch):
    monkeypatch.setattr(llm_client, "is_healthy", lambda: True)
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_503_when_backend_unhealthy(monkeypatch):
    monkeypatch.setattr(llm_client, "is_healthy", lambda: False)
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_transcript_never_appears_in_logs(monkeypatch, caplog):
    monkeypatch.setattr(llm_client, "complete", lambda prompt, **kw: _FRAUD_JSON)
    secret_marker = "SUPER-SECRET-CALLER-TRANSCRIPT-CONTENT-12345"
    with caplog.at_level(logging.INFO):
        client.post("/triage/text", json={"transcript": secret_marker})
    for record in caplog.records:
        assert secret_marker not in record.getMessage()
        assert secret_marker not in str(record.__dict__)
