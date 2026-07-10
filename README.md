# Fraud-Triage-LLM

Production-oriented telecom-fraud triage: **call transcript → fine-tuned LLM → structured, explainable fraud verdict**. The completed pipeline covers data preparation with lineage tracking, QLoRA training, deterministic inference, evaluation with a CI regression gate, a FastAPI text-and-audio triage service (real CPU Whisper transcription) with retry/fallback guardrails, a scanned/smoke-tested Docker image, Prometheus metrics with PSI-based drift detection visualized in Grafana, real load-test numbers, and MLflow experiment tracking on the eval side. A live GPU/vLLM deployment and any actual running/cloud deployment remain roadmap work — everything above has been built and verified locally or in CI, not deployed anywhere.

Rather than returning only a binary label, the model emits a schema-validated analyst-style verdict with risk level, fraud type, rationale, and supporting transcript spans. The evaluation suite compares it with a TF-IDF/XGBoost control so the accuracy, reliability, and explainability trade-offs are visible rather than assumed.

## Resume-ready description

**Fraud-Triage-LLM | Python, PyTorch, Hugging Face, QLoRA/PEFT, XGBoost, FastAPI, Docker, MLflow, Prometheus, Grafana, Locust, Pydantic, pytest, GitHub Actions**

- Built a modular Python ML pipeline that transforms English call transcripts into schema-validated fraud risk, fraud type, rationale, and supporting evidence; prepared an approximately 9,000-transcript corpus with reproducible train/validation/test splits and a generated lineage manifest (SHA-256 hashes, git commit, config snapshot) for every run (see [phase-0e-dataset-manifest](docs/devlog/phase-0e-dataset-manifest.md)).
- Fine-tuned Qwen2.5-7B-Instruct with 4-bit QLoRA on a single 16 GB T4 GPU using completion-only loss, gradient checkpointing, and deterministic inference, achieving **0.941 F1**, **0.937 PR-AUC**, and **94.7% valid JSON** on 1,350 held-out calls; the same fine-tuned model generalizes to an out-of-domain fraud-email set (CLAIR) at **0.803 F1** versus a call-trained TF-IDF/XGBoost baseline's **0.578 F1**.
- Developed an evaluation and reliability harness with a TF-IDF/XGBoost baseline, out-of-domain fraud-email testing, Pydantic output contracts, regression thresholds, and **93 unit tests**; a committed synthetic fixture makes the CI regression gate run unconditionally on every push (it previously pointed at a path that never existed and silently skipped — see [phase-5-ci-gate](docs/devlog/phase-5-ci-gate.md)).
- Built a FastAPI text-**and-audio** triage service — real CPU `faster-whisper` transcription verified against synthesized scam-call audio, a pluggable OpenAI-compatible (vLLM-style) LLM client, retry/fallback guardrails that degrade malformed output or a downed backend to a safe verdict, and privacy-aware structured logging — packaged into a Docker image that's built, Trivy-scanned (fixed real CVEs found on the first scan), and smoke-tested via GitHub Actions (no GPU/vLLM server in this dev environment to run it against a live model; see [phase-3-whisper](docs/devlog/phase-3-whisper.md), [phase-4-serving](docs/devlog/phase-4-serving.md), [phase-4d-docker](docs/devlog/phase-4d-docker.md)).
- Instrumented the service with Prometheus request/latency/invalid-output metrics and PSI-based fraud-rate drift detection, visualized through a provisioned Grafana dashboard, and load-tested with Locust (**38 req/s, p95 17ms** under 20 concurrent users against the guardrails path with no backend attached) (see [phase-4c-prometheus](docs/devlog/phase-4c-prometheus.md), [phase-4f-drift](docs/devlog/phase-4f-drift.md), [phase-4g-grafana](docs/devlog/phase-4g-grafana.md), [phase-4e-load-test](docs/devlog/phase-4e-load-test.md)).
- Added MLflow experiment tracking and alias-based Model Registry promotion on the evaluation side (params, metrics, and artifacts logged on every run, no GPU required); the training-side integration is code-complete but only exercised on Kaggle (see [phase-4b-mlflow](docs/devlog/phase-4b-mlflow.md)).

The vLLM client, Docker image, drift monitor, and Grafana dashboard have all been built and verified locally or in CI — but none of it runs anywhere in production: there is no live GPU host serving the fine-tuned model and no cloud deployment. The Grafana dashboard was confirmed correct via its own API (real provisioning + real query results) rather than a visual screenshot, since browser tooling wasn't available in this environment. Say so precisely if asked, rather than implying a running deployment exists.

## Pipeline

```
Scam call audio
   │  faster-whisper, CPU (verified on real audio)   src/asr/
   ▼
Transcript
   │  QLoRA-fine-tuned 7B LLM                        src/train/  src/serve/
   ▼
Structured verdict (JSON):
   { risk, fraud_type, reason, flagged_spans }
   │  FastAPI + guardrails, in a scanned/smoke-       src/serve/
   │  tested Docker image (vLLM client code-
   │  complete, not yet run against a live GPU)
   ▼
API + Prometheus /metrics + drift alerts +           src/serve/metrics.py
privacy-safe logs, visualized in Grafana             src/serve/drift.py
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

- [x] Implement `faster-whisper` audio transcription and temporary-file cleanup — CPU + int8, verified against a real synthesized scam-call audio clip, temp file always deleted (including on transcription failure) ([phase-3-whisper](docs/devlog/phase-3-whisper.md)).
- [x] Connect the FastAPI text endpoint to an OpenAI-compatible (vLLM-style) inference client — wired and unit-tested against a stubbed backend; not yet run against a live vLLM/GPU server ([phase-4-serving](docs/devlog/phase-4-serving.md)).
- [x] Complete the audio endpoint: upload validation → transcription → model inference → validated verdict ([phase-3-whisper](docs/devlog/phase-3-whisper.md)).
- [x] Integrate retry/fallback guardrails into both the text and audio inference paths and test malformed model outputs and backend failures.
- [x] Add request-size limits, structured errors, and `/health` and `/ready` probes (audio path also gets a `max_audio_bytes` limit).
- [x] Produce a reproducible, scanned Docker image for the API/guardrails/ASR layer — pinned base image, `apt`/`pip` upgraded (closed real CVEs found by Trivy), non-root user, healthcheck, built+scanned+smoke-tested in CI. Deliberately CPU-only and GPU/CUDA-free: the vLLM inference server is its own deployable unit (real production would run vLLM's own official image), not something built or pinned here ([phase-4d-docker](docs/devlog/phase-4d-docker.md)).

### Priority 2: experiment tracking and model lineage

- [x] Integrate MLflow tracking for evaluation parameters, dataset paths, Git commit, and metrics — runs locally on every `evaluate.py` call, no GPU required; the training-side hook is code-complete but only exercised on Kaggle ([phase-4b-mlflow](docs/devlog/phase-4b-mlflow.md)).
- [ ] Log the LoRA adapter, tokenizer, evaluation report, and model card for every candidate run (adapter/tokenizer logging code is written but unexercised — no GPU training run since it was added).
- [x] Register promoted model versions in the MLflow Model Registry with alias-based promotion (`--register-model`) — currently scoped to the classical baseline; the LLM adapter isn't registered yet.
- [x] Add immutable dataset manifests to reproduce the exact training and evaluation splits — a lightweight, dependency-free stand-in for full DVC: SHA-256 per split, git commit, config snapshot, generated on every `python -m src.data.load` run and verified against a real HF download ([phase-0e-dataset-manifest](docs/devlog/phase-0e-dataset-manifest.md)). Not full DVC — no remote storage, no retained historical versions.
- [ ] Define a full promotion policy based on F1, PR-AUC, JSON validity, latency, and responsible-AI checks (today's policy compares F1 only).

### Priority 3: CI/CD and quality gates

- [x] Commit a small, license-safe evaluation fixture so model-regression logic runs in CI instead of being skipped when prediction artifacts are absent — the gate previously pointed at a path that never existed and silently skipped on every push ([phase-5-ci-gate](docs/devlog/phase-5-ci-gate.md)).
- [ ] Extend GitHub Actions with formatting, linting, and type checking for the Python code itself.
- [x] Build and scan the Docker image in CI (Trivy, fails on fixable CRITICAL/HIGH) — publishing versioned images to a registry is still open, since there's nowhere to deploy them yet ([phase-4d-docker](docs/devlog/phase-4d-docker.md)).
- [x] Add an automated post-build smoke test (run the container, poll `/health`, assert `/ready` behaves correctly with no backend, teardown) — this is the CI-appropriate stand-in for "deployment validation" in a repo with no actual cloud target; a real deployment workflow with environment approval still needs somewhere to deploy to ([phase-4d-docker](docs/devlog/phase-4d-docker.md)).
- [ ] Protect the main branch with required reviews and passing CI checks.

### Priority 4: observability, performance, and drift

- [x] Expose Prometheus metrics for request volume, latency, and invalid-output rate at `/metrics` ([phase-4c-prometheus](docs/devlog/phase-4c-prometheus.md)).
- [x] Add structured logs with correlation IDs while redacting transcripts and other sensitive data.
- [x] Build a Grafana dashboard for latency, error rate, invalid-output rate, fraud-rate, and drift status, provisioned from files against a real Prometheus scraping the real service — verified via Grafana's own API (real datasource, real dashboard, real query results); no alerting rules yet, and no visual screenshot since browser tooling wasn't available in this sandbox ([phase-4g-grafana](docs/devlog/phase-4g-grafana.md)).
- [x] Run Locust load tests; document concurrency, throughput, and p95 latency — 20 concurrent users, 38.4 req/s, p95 17ms on `/triage/text` with the backend down, measuring guardrails/HTTP overhead rather than real LLM inference latency (no GPU to run a real backend against) ([phase-4e-load-test](docs/devlog/phase-4e-load-test.md)).
- [x] Monitor output drift (fraud-rate distribution vs. training baseline, via PSI) without storing raw sensitive calls — demonstrated catching a real backend-outage scenario live. Per-fraud-type distribution and input-side drift are still open ([phase-4f-drift](docs/devlog/phase-4f-drift.md)).
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

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-base.txt    # light, cross-platform (Phase 0, 2, 4 unit tests, MLflow, ASR)
python -m pytest                                  # 93 tests: schema, data, eval, serving, guardrails, ASR, drift, MLflow
python -m src.data.load --dataset bothbosu --out data/processed   # Phase 0 (writes a manifest.json too)
python -m uvicorn src.serve.app:app --reload      # Phase 4: serve locally
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-base.txt
python -m pytest
python -m src.data.load --dataset bothbosu --out data/processed
python -m uvicorn src.serve.app:app --reload
```

`requirements-base.txt` now also covers the FastAPI serving layer, guardrails,
`faster-whisper` ASR, Prometheus metrics, and MLflow tracking — all CPU-only
and runnable without a GPU. Heavy training (Phase 1) runs on Kaggle/Colab —
see `notebooks/kaggle_train.py` (`requirements-train.txt`). Running against a
real vLLM inference server (Phase 4, GPU) uses Docker
(`requirements-serve.txt`). Both pull `vllm`/`bitsandbytes`, which have no
Windows wheels — keep them off local installs.

The QLoRA adapter produced by the Kaggle training run is the model artifact
this service is intended to use. The adapter weights are gitignored and are
not automatically available to the local API: download the Kaggle output
directory (`models/qwen7b-fraud-qlora`) and serve it together with the
`Qwen/Qwen2.5-7B-Instruct` base model from a GPU inference host.

Until an LLM backend is running and configured, `/triage/text` and
`/triage/audio` return the guardrails' safe fallback verdict (`risk=medium`)
instead of erroring — that's the intended behavior, not evidence that the
model was not trained. Point
`VLLM_BASE_URL` (default `http://localhost:8000/v1`) at a real
OpenAI-compatible vLLM server to get real verdicts:

```bash
export VLLM_BASE_URL=http://your-gpu-host:8000/v1
export VLLM_MODEL=qwen2.5-7b-fraud-qlora
export VLLM_TIMEOUT_S=30
python -m uvicorn src.serve.app:app --reload
```

The FastAPI service and vLLM backend are separate processes. If both run on
the same machine, give them different ports (for example FastAPI on `8000`
and vLLM on `8001`) and set `VLLM_BASE_URL=http://127.0.0.1:8001/v1`.

On a Linux/NVIDIA GPU host, serve the downloaded Kaggle adapter under the
same model name expected by `src/serve/llm_client.py`:

```bash
python -m pip install -r requirements-serve.txt
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules qwen2.5-7b-fraud-qlora=/absolute/path/to/models/qwen7b-fraud-qlora

curl http://127.0.0.1:8001/v1/models
```

The adapter directory should contain at least `adapter_config.json`, the
adapter weights (normally `adapter_model.safetensors`), and the tokenizer
files saved by the training script. vLLM exposes the LoRA module name as a
model ID, so `VLLM_MODEL=qwen2.5-7b-fraud-qlora` selects the fine-tuned
adapter rather than the untouched base model. See the
[vLLM LoRA serving guide](https://docs.vllm.ai/en/stable/features/lora/).

```bash
curl -X POST http://127.0.0.1:8000/triage/text \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Congratulations, you won a prize! Just pay a shipping fee with a gift card."}'

curl -X POST http://127.0.0.1:8000/triage/audio -F "file=@call.wav;type=audio/wav"

curl http://127.0.0.1:8000/health     # process liveness
curl http://127.0.0.1:8000/ready      # 200 if VLLM_BASE_URL is reachable, else 503
curl http://127.0.0.1:8000/metrics    # Prometheus exposition format
```

### Docker (Phase 4d)

```bash
docker build -f src/serve/Dockerfile -t fraud-triage-llm-api .
docker run -p 8000:8000 fraud-triage-llm-api
# optional: -e VLLM_BASE_URL=http://your-vllm-host:8000/v1 to point at a real backend
```

### Observability stack — Prometheus + Grafana (Phase 4g)

```bash
docker compose -f monitoring/docker-compose.yml up --build
```
Then open **http://localhost:3000** (Grafana — anonymous viewer access is
enabled for this dev stack, no login needed; the "Fraud-Triage-LLM" dashboard
is pre-provisioned) and **http://localhost:9090** (Prometheus). Generate some
traffic first (the curl commands above, or the load test below) so the
dashboard has something to show. Tear down with
`docker compose -f monitoring/docker-compose.yml down`.

### Load testing (Phase 4e)

```bash
pip install -r requirements-loadtest.txt
uvicorn src.serve.app:app &
locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 \
  --headless -u 20 -r 5 -t 30s --csv reports/load_test
```
Results land in `reports/load_test_stats.csv`; see `reports/load_test.md` for
the last documented run (38 req/s, p95 17ms on `/triage/text` with no backend
attached — see the caveat there about what that number does and doesn't mean).
