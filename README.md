# Fraud-Triage-LLM

Production-oriented telecom-fraud triage: **call transcript → fine-tuned LLM → structured, explainable fraud verdict**. The completed pipeline covers data preparation, QLoRA training, deterministic inference, and evaluation; audio ingestion and production serving remain roadmap work.

Rather than returning only a binary label, the model emits a schema-validated analyst-style verdict with risk level, fraud type, rationale, and supporting transcript spans. The evaluation suite compares it with a TF-IDF/XGBoost control so the accuracy, reliability, and explainability trade-offs are visible rather than assumed.

## Resume-ready description

**Fraud-Triage-LLM | Python, PyTorch, Hugging Face, QLoRA/PEFT, XGBoost, Pydantic, pytest, GitHub Actions**

- Built a modular Python ML pipeline that transforms English call transcripts into schema-validated fraud risk, fraud type, rationale, and supporting evidence; prepared an approximately 9,000-transcript corpus with reproducible train/validation/test splits.
- Fine-tuned Qwen2.5-7B-Instruct with 4-bit QLoRA on a single 16 GB T4 GPU using completion-only loss, gradient checkpointing, and deterministic inference, achieving **0.941 F1**, **0.937 PR-AUC**, and **94.7% valid JSON** on 1,350 held-out calls.
- Developed an evaluation and reliability harness with a TF-IDF/XGBoost baseline, out-of-domain fraud-email testing, Pydantic output contracts, regression thresholds, and **36 unit tests**; automated CPU-safe tests with GitHub Actions and documented model, data, and infrastructure trade-offs.

The repository also contains FastAPI, Docker, retry/fallback guardrail, Whisper, and vLLM scaffolding. Those components are intentionally described as work in progress until the end-to-end service and load/observability tests are complete.

## Pipeline

```
Scam call audio
   │  Whisper (ASR, planned)             src/asr/
   ▼
Transcript
   │  QLoRA-fine-tuned 7B LLM            src/train/  src/serve/
   ▼
Structured verdict (JSON):
   { risk, fraud_type, reason, flagged_spans }
   │  FastAPI + vLLM in Docker (planned) src/serve/
   ▼
API + metrics/logging (planned)
```

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| **English call corpus** (`--dataset calls`) | Primary train/val/test | ~9k phone-call transcripts unioned from [menaattia](https://huggingface.co/datasets/menaattia/phone-scam-dataset), [shakeleoatmeal](https://huggingface.co/datasets/shakeleoatmeal/phone-scam-detection-synthetic), [BothBosu multi-agent](https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation) + [single-agent](https://huggingface.co/datasets/BothBosu/single-agent-scam-conversations); ~50% fraud |
| [BothBosu/scam-dialogue](https://huggingface.co/datasets/BothBosu/scam-dialogue) | Prototype | Original pipeline scaffold, more scam types |
| [tasksource/CLAIR_email_fraud](https://huggingface.co/datasets/tasksource/CLAIR_email_fraud) | Cross-domain eval | ~12k advance-fee ("419") fraud emails, explicit FRAUD label — out-of-distribution check (never trained on) |

> *Dataset history: TeleAntiFraud-28k (original primary) was dropped — Chinese + gated ([phase-0c](docs/devlog/phase-0c-call-corpus.md)); redasers/difraud (original cross-domain set) was retired — its label is "deceptive", not "fraud" ([phase-0d](docs/devlog/phase-0d-clair-crossdomain.md)).*

## Model output schema

Every prediction is validated against `src/data/schema.py`:

```json
{
  "risk": "high",
  "fraud_type": "tech_support_scam",
  "reason": "Caller claimed to be Microsoft support and requested remote access and gift-card payment.",
  "flagged_spans": ["remote access to your computer", "pay with gift cards"]
}
```

## Implementation roadmap

### Completed: model development and evaluation

- [x] Build a configuration-driven data pipeline for loading, normalizing, instruction formatting, and reproducible train/validation/test splitting.
- [x] Fine-tune Qwen2.5-7B with 4-bit QLoRA on a single T4 GPU.
- [x] Implement deterministic batch inference and schema-validated structured predictions.
- [x] Evaluate precision, recall, F1, PR-AUC, and JSON validity on a held-out test set.
- [x] Train and persist a TF-IDF/XGBoost baseline for controlled comparison.
- [x] Add a CLAIR fraud-email split for out-of-distribution evaluation.
- [x] Add unit tests for data mappings, schema contracts, training boundaries, prediction artifacts, and evaluation logic.

### Priority 1: complete the end-to-end inference service

- [ ] Implement `faster-whisper` audio transcription and temporary-file cleanup.
- [ ] Connect the FastAPI text endpoint to a vLLM OpenAI-compatible inference server.
- [ ] Complete the audio endpoint: upload validation → transcription → model inference → validated verdict.
- [ ] Integrate retry/fallback guardrails into both inference paths and test malformed model outputs.
- [ ] Add request-size limits, timeouts, structured errors, and `/health` and `/ready` probes.
- [ ] Pin the GPU/CUDA runtime and produce a reproducible multi-stage Docker image.

### Priority 2: experiment tracking and model lineage

- [ ] Integrate MLflow tracking for training parameters, dataset version, Git commit, metrics, and artifacts.
- [ ] Log the LoRA adapter, tokenizer, evaluation report, and model card for every candidate run.
- [ ] Register promoted model versions in the MLflow Model Registry with staging/production aliases.
- [ ] Add DVC or immutable dataset manifests to reproduce the exact training and evaluation splits.
- [ ] Define a promotion policy based on F1, PR-AUC, JSON validity, latency, and responsible-AI checks.

### Priority 3: CI/CD and quality gates

- [ ] Extend GitHub Actions with formatting, linting, type checking, security scanning, and the full CPU unit-test suite.
- [ ] Commit a small, license-safe evaluation fixture so model-regression logic runs in CI instead of being skipped when prediction artifacts are absent.
- [ ] Build and scan the Docker image in CI and publish versioned images only after quality gates pass.
- [ ] Add a deployment workflow with environment approval and an automated post-deployment smoke test.
- [ ] Protect the main branch with required reviews and passing CI checks.

### Priority 4: observability, performance, and drift

- [ ] Expose Prometheus metrics for request volume, p50/p95 latency, throughput, failures, retries, and invalid-output rate.
- [ ] Add structured logs with correlation IDs while redacting transcripts and other sensitive data.
- [ ] Build a Grafana dashboard and alerts for latency, error rate, service health, and model-output quality.
- [ ] Run Locust or k6 load tests; document concurrency, throughput, GPU utilization, and p95 latency.
- [ ] Monitor input/output drift and fraud-type distribution changes without storing raw sensitive calls.
- [ ] Define rollback and human-review procedures for degraded or uncertain predictions.

### Priority 5: model validation and portfolio demo

- [ ] Generate real Qwen predictions on CLAIR and report LLM versus XGBoost cross-domain results.
- [ ] Report confusion matrices, per-fraud-type metrics, calibration, and performance under simulated ASR errors.
- [ ] Add responsible-AI tests for demographic cues, false positives, prompt injection, and unsupported explanations.
- [ ] Build a Gradio demo for transcript and audio inputs backed by the deployed API.
- [ ] Publish an architecture diagram, API examples, model card, limitations, and a short demonstration video.

### Optional extension

- [ ] Add retrieval over a versioned scam-pattern knowledge base and evaluate whether it improves explanation quality without reducing detection performance.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-base.txt              # light, cross-platform (Phase 0 + 2)
pytest                                            # schema + data tests
python -m src.data.load --dataset bothbosu --out data/processed   # Phase 0
```

Heavy training (Phase 1) runs on Kaggle/Colab — see `notebooks/kaggle_train.py`
(`requirements-train.txt`). Serving (Phase 4) runs in Docker (`requirements-serve.txt`).
Both pull `vllm`/`bitsandbytes`, which have no Windows wheels — keep them off local installs.
