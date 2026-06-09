# Fraud-Triage-LLM

Explainable telecom-fraud triage: **speech → transcript → fine-tuned LLM → structured fraud verdict**, served in production.

This is not a binary classifier. The model is QLoRA-fine-tuned to emit a *structured, explainable verdict* — a fraud-analyst-style judgment with the specific phrases that triggered it — which is something a classical classifier (XGBoost) cannot do. The eval suite explicitly compares against that baseline to justify the LLM.

## Pipeline

```
Scam call audio
   │  Whisper (ASR)                      src/asr/
   ▼
Transcript
   │  QLoRA-fine-tuned 7B LLM            src/train/  src/serve/
   ▼
Structured verdict (JSON):
   { risk, fraud_type, reason, flagged_spans }
   │  FastAPI + vLLM in Docker           src/serve/
   ▼
Gradio demo  +  metrics/logging
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

## Build phases

- [ ] **Phase 0** — Data: load, format to instruction, held-out split, baseline number
- [ ] **Phase 1** — QLoRA fine-tune (Kaggle free GPU) + W&B tracking
- [ ] **Phase 2** — Eval harness: F1 / PR-AUC / JSON-validity + XGBoost baseline + DIFRAUD cross-domain
- [ ] **Phase 3** — Whisper ASR frontend (off-the-shelf, not trained)
- [ ] **Phase 4** — Serving: vLLM + FastAPI + AWQ + Docker + load test + guardrails + observability
- [ ] **Phase 5** — CI/CD: GitHub Actions eval-regression gate, README polish, demo link
- [ ] **Phase 6 (optional)** — RAG over known scam-script knowledge base for richer explanations

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
