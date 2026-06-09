# Phase 4 — multi-stage build for the serving image.
# TODO: pin CUDA base image matching your vLLM/host GPU.

FROM python:3.11-slim AS base
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
