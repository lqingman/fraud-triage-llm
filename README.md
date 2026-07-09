# Fraud-Triage-LLM

Production-oriented telecom-fraud triage: **call transcript → fine-tuned LLM → structured, explainable fraud verdict**. The completed pipeline covers data preparation, QLoRA training, deterministic inference, evaluation with a CI regression gate, a FastAPI text-triage service with retry/fallback guardrails and Prometheus metrics, and MLflow experiment tracking on the eval side. Audio ingestion (Whisper), a live GPU/vLLM deployment, and full observability (Grafana, load testing, drift monitoring) remain roadmap work.

Rather than returning only a binary label, the model emits a schema-validated analyst-style verdict with risk level, fraud type, rationale, and supporting transcript spans. The evaluation suite compares it with a TF-IDF/XGBoost control so the accuracy, reliability, and explainability trade-offs are visible rather than assumed.

## Resume-ready description

**Fraud-Triage-LLM | Python, PyTorch, Hugging Face, QLoRA/PEFT, XGBoost, FastAPI, MLflow, Prometheus, Pydantic, pytest, GitHub Actions**

- Built a modular Python ML pipeline that transforms English call transcripts into schema-validated fraud risk, fraud type, rationale, and supporting evidence; prepared an approximately 9,000-transcript corpus with reproducible train/validation/test splits.
- Fine-tuned Qwen2.5-7B-Instruct with 4-bit QLoRA on a single 16 GB T4 GPU using completion-only loss, gradient checkpointing, and deterministic inference, achieving **0.941 F1**, **0.937 PR-AUC**, and **94.7% valid JSON** on 1,350 held-out calls; the same fine-tuned model generalizes to an out-of-domain fraud-email set (CLAIR) at **0.803 F1** versus a call-trained TF-IDF/XGBoost baseline's **0.578 F1**.
- Developed an evaluation and reliability harness with a TF-IDF/XGBoost baseline, out-of-domain fraud-email testing, Pydantic output contracts, regression thresholds, and **70 unit tests**; a committed synthetic fixture makes the CI regression gate run unconditionally on every push (it previously pointed at a path that never existed and silently skipped — see [phase-5-ci-gate](docs/devlog/phase-5-ci-gate.md)).
- Built a FastAPI text-triage service backed by a pluggable OpenAI-compatible (vLLM-style) LLM client, with retry/fallback guardrails that degrade a malformed response or a downed backend to a safe verdict instead of erroring, privacy-aware structured request logging, and Prometheus request/latency/invalid-output metrics — unit-tested end-to-end against a stubbed backend (no GPU/vLLM server in this dev environment to run it against a live model; see [phase-4-serving](docs/devlog/phase-4-serving.md), [phase-4c-prometheus](docs/devlog/phase-4c-prometheus.md)).
- Added MLflow experiment tracking and alias-based Model Registry promotion on the evaluation side (params, metrics, and artifacts logged on every run, no GPU required); the training-side integration is code-complete but only exercised on Kaggle (see [phase-4b-mlflow](docs/devlog/phase-4b-mlflow.md)).

The repository also contains Docker and Whisper scaffolding, plus a vLLM-compatible client that hasn't been run against a live GPU deployment. Those components — along with Grafana dashboards, load testing, and drift monitoring — are intentionally described as work in progress until they've actually been exercised end-to-end.

## Pipeline

```
Scam call audio
   │  Whisper (ASR, planned)                        src/asr/
   ▼
Transcript
   │  QLoRA-fine-tuned 7B LLM                        src/train/  src/serve/
   ▼
Structured verdict (JSON):
   { risk, fraud_type, reason, flagged_spans }
   │  FastAPI + guardrails (wired, tested against    src/serve/
   │  a stubbed backend; vLLM client code-complete,
   │  not yet run against a live GPU deployment)
   ▼
API + Prometheus /metrics + privacy-safe logs        src/serve/metrics.py
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
- [x] Connect the FastAPI text endpoint to an OpenAI-compatible (vLLM-style) inference client — wired and unit-tested against a stubbed backend; not yet run against a live vLLM/GPU server ([phase-4-serving](docs/devlog/phase-4-serving.md)).
- [ ] Complete the audio endpoint: upload validation → transcription → model inference → validated verdict (blocked on Whisper above).
- [x] Integrate retry/fallback guardrails into the text inference path and test malformed model outputs and backend failures (audio path still blocked on Whisper).
- [x] Add request-size limits, structured errors, and `/health` and `/ready` probes.
- [ ] Pin the GPU/CUDA runtime and produce a reproducible multi-stage Docker image.

### Priority 2: experiment tracking and model lineage

- [x] Integrate MLflow tracking for evaluation parameters, dataset paths, Git commit, and metrics — runs locally on every `evaluate.py` call, no GPU required; the training-side hook is code-complete but only exercised on Kaggle ([phase-4b-mlflow](docs/devlog/phase-4b-mlflow.md)).
- [ ] Log the LoRA adapter, tokenizer, evaluation report, and model card for every candidate run (adapter/tokenizer logging code is written but unexercised — no GPU training run since it was added).
- [x] Register promoted model versions in the MLflow Model Registry with alias-based promotion (`--register-model`) — currently scoped to the classical baseline; the LLM adapter isn't registered yet.
- [ ] Add DVC or immutable dataset manifests to reproduce the exact training and evaluation splits.
- [ ] Define a full promotion policy based on F1, PR-AUC, JSON validity, latency, and responsible-AI checks (today's policy compares F1 only).

### Priority 3: CI/CD and quality gates

- [x] Commit a small, license-safe evaluation fixture so model-regression logic runs in CI instead of being skipped when prediction artifacts are absent — the gate previously pointed at a path that never existed and silently skipped on every push ([phase-5-ci-gate](docs/devlog/phase-5-ci-gate.md)).
- [ ] Extend GitHub Actions with formatting, linting, type checking, and security scanning (the full CPU unit-test suite already runs unconditionally on every push).
- [ ] Build and scan the Docker image in CI and publish versioned images only after quality gates pass.
- [ ] Add a deployment workflow with environment approval and an automated post-deployment smoke test.
- [ ] Protect the main branch with required reviews and passing CI checks.

### Priority 4: observability, performance, and drift

- [x] Expose Prometheus metrics for request volume, latency, and invalid-output rate at `/metrics` ([phase-4c-prometheus](docs/devlog/phase-4c-prometheus.md)).
- [x] Add structured logs with correlation IDs while redacting transcripts and other sensitive data.
- [ ] Build a Grafana dashboard and alerts for latency, error rate, service health, and model-output quality.
- [ ] Run Locust or k6 load tests; document concurrency, throughput, GPU utilization, and p95 latency.
- [ ] Monitor input/output drift and fraud-type distribution changes without storing raw sensitive calls.
- [ ] Define rollback and human-review procedures for degraded or uncertain predictions (the guardrails' safe fallback verdict already flags uncertain output as "manual review"; no formal rollback procedure is documented).

### Priority 5: model validation and portfolio demo

- [x] Generate real Qwen predictions on CLAIR and report LLM versus XGBoost cross-domain results — **0.803 vs 0.578 F1** ([phase-2b](docs/devlog/phase-2b-llm-crossdomain.md)).
- [ ] Report per-fraud-type metrics, calibration, and performance under simulated ASR errors (a confusion matrix + failure-mode breakdown already exists in `reports/error_analysis_*.md`).
- [ ] Add responsible-AI tests for demographic cues, false positives, prompt injection, and unsupported explanations.
- [ ] Build a Gradio demo for transcript and audio inputs backed by the deployed API.
- [ ] Publish an architecture diagram, API examples, model card, limitations, and a short demonstration video.

### Optional extension

- [ ] Add retrieval over a versioned scam-pattern knowledge base and evaluate whether it improves explanation quality without reducing detection performance.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-base.txt              # light, cross-platform (Phase 0, 2, 4 unit tests, MLflow)
pytest                                            # 70 tests: schema, data, eval, serving, guardrails, MLflow
python -m src.data.load --dataset bothbosu --out data/processed   # Phase 0
uvicorn src.serve.app:app --reload                # Phase 4: serve locally (falls back to a safe verdict with no backend configured)
```

`requirements-base.txt` now also covers the FastAPI serving layer, guardrails,
Prometheus metrics, and MLflow tracking — all CPU-only and runnable without a
GPU. Heavy training (Phase 1) runs on Kaggle/Colab — see
`notebooks/kaggle_train.py` (`requirements-train.txt`). Running against a real
vLLM inference server (Phase 4, GPU) uses Docker (`requirements-serve.txt`).
Both pull `vllm`/`bitsandbytes`, which have no Windows wheels — keep them off
local installs.
