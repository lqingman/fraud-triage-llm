# Phase 4 — lightweight serving image: FastAPI + guardrails + ASR
# (faster-whisper, CPU) + Prometheus metrics. Talks to a real vLLM inference
# server over HTTP (VLLM_BASE_URL) rather than bundling vllm/torch/CUDA in
# this image — the GPU inference server is its own deployable unit (in
# production, run from vLLM's official image), so this image stays small,
# CPU-only, and fast to build + scan in CI. See requirements-docker.txt.
#
# Build from the repo root (not this directory) so the COPY paths resolve:
#   docker build -f src/serve/Dockerfile -t fraud-triage-llm-api .

FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Patch base-OS packages (openssl/perl/sqlite etc. in the upstream image tag
# lag behind their own Debian security fixes) — found via a real Trivy scan
# during development; this is what actually closes those CVEs, not a scanner
# config tweak. See docs/devlog/phase-4d-docker.md.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Non-root, but WITH a real home directory: faster-whisper (via
# huggingface_hub) caches downloaded model weights under $HOME/.cache and
# hard-fails with EACCES if that path doesn't exist and isn't writable —
# caught by actually running a real transcription through the built
# container, not just hitting /health. See docs/devlog/phase-4d-docker.md.
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app
ENV HOME=/home/app

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY config/ ./config/

USER app

EXPOSE 8000

# No curl in the slim base — use Python's stdlib so we don't add an extra
# package (and its CVEs) just for the healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
