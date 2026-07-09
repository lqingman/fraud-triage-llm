# Phase 4d — reproducible Docker image, container scanning, deployment validation

**Date:** 2026-07-09
**Status:** Done. Built, scanned, and smoke-tested locally with real Docker;
wired into CI. No actual cloud deployment — see scope note.

## Goal

Close three related resume gaps at once: "reproducible Docker environments,"
"container scanning," and "deployment validation ... through GitHub Actions."
The existing `Dockerfile` had a literal `# TODO: pin CUDA base image` comment,
had never been built once, `COPY`'d the full `requirements.txt` (pulling in
`vllm`/`torch`/`bitsandbytes` — GPU-only deps a CPU-only image build would
choke on or bloat enormously), and CI had no Docker step at all.

## Key design decision: split the image from the GPU inference server

`vllm` doesn't belong in this image. In a real deployment, the vLLM inference
server is its own unit — typically vLLM's own official image, running on a
GPU host — and this repo's `src/serve/app.py` only ever talks to it over
HTTP (`VLLM_BASE_URL`). So the serving image here is just FastAPI +
guardrails + ASR + Prometheus: CPU-only, no GPU dependency, genuinely
buildable and scannable in ordinary GitHub Actions runners.

New `requirements-docker.txt`, deliberately **not** `-r requirements-base.txt`:
that file also pulls in `xgboost`/`pandas`/`datasets`/`mlflow` for Phase 0/2/
MLflow, none of which `src/serve/app.py`'s import chain needs — and
`xgboost`'s wheel drags in ~300MB of `nvidia-nccl-cu12` it never uses on CPU.
Along the way, found `src/data/load.py` importing `sklearn.model_selection`
at module top level even though it's only used inside `main()`'s re-split —
moved it inside the function (matching the deferred-import convention
already used in `qlora_train.py`/`evaluate.py`), so `format_prompt` (imported
by `app.py`) no longer transitively requires scikit-learn.

## What I did

### 1. `src/serve/Dockerfile` — rewritten
- `python:3.11.9-slim-bookworm` (pinned patch version, not floating `slim`).
- `apt-get upgrade` before anything else, and `pip install --upgrade pip
  setuptools wheel` before installing requirements — both found necessary by
  actually running Trivy (see below), not assumed upfront.
- Non-root `app` user **with a real home directory** — `faster-whisper` (via
  `huggingface_hub`) caches downloaded model weights under `$HOME/.cache`
  and hard-fails with `EACCES` if that path doesn't exist/isn't writable.
  First pass used `--no-create-home` (seemed more locked-down) and broke
  audio transcription inside the container; caught only by actually running
  a real transcription through the built container, not just `/health`.
- `HEALTHCHECK` via Python's stdlib `urllib` (no `curl` in the slim image —
  not worth adding a package, and its CVEs, just for a healthcheck).
- `.dockerignore` added (`.venv`, `data/`, `models/`, `mlruns/`, `.git`, etc.)
  so the build context doesn't ship gigabytes of local dev state.

### 2. `.github/workflows/docker.yml` (new)
Build → Trivy scan (`aquasecurity/trivy-action`, `CRITICAL,HIGH`,
`--ignore-unfixed`, fails the job on a hit) → smoke test (run the container,
poll `/health` until it's up, assert `/ready` is exactly `503` — correct,
since no backend is configured in CI — then tear down). This is the
CI-appropriate stand-in for "deployment validation" in a repo with no actual
cloud target: it proves the built artifact starts and serves correctly, not
that it was pushed anywhere.

## Verification — real Docker, not just Dockerfile review

Docker Desktop wasn't running in this sandbox; started it and confirmed the
whole pipeline for real:

- **Build**: succeeded. First pass (before splitting from
  `requirements-base.txt`) produced an **2.81 GB** image in ~69s; after the
  `requirements-docker.txt` split it's **860 MB** in ~8.5s.
- **Trivy scan, first real run**: found real, fixable CVEs — `CRITICAL`
  `libssl3`/`openssl` (multiple), `HIGH` `setuptools`/`wheel` in the base
  image and its bundled pip. These weren't hypothetical: `apt-get upgrade`
  and `pip install --upgrade pip setuptools wheel` closed every one of them
  — rescanned afterward with **0 CRITICAL/HIGH findings**, `exit-code 1`
  passing (0).
- **Container run + real bug #1**: `/health` → 200, but `/ready` incorrectly
  returned **200** with no backend configured — traced to
  `src/serve/llm_client.is_healthy()` accepting any `status_code < 500`; the
  container's default `VLLM_BASE_URL` (`http://localhost:8000/v1`) happens
  to alias back to the app's own port, so `GET /v1/models` hit FastAPI's own
  404-not-found handler, and 404 slipped through the `< 500` check. Fixed to
  require exactly `200`. Rebuilt, reran: `/ready` → 503, correctly.
- **Container run + real bug #2**: uploading the real synthesized scam-call
  WAV (from the Phase 3 Whisper work) to `/triage/audio` returned
  `{"detail":"could not transcribe audio: [Errno 13] Permission denied:
  '/home/app'"}` — the non-root user had no real home directory
  (`--no-create-home`), so `faster-whisper`'s model-weight cache write
  failed. Switched to `--create-home` + explicit `ENV HOME=/home/app`.
  Rebuilt, re-uploaded the same file: 200, real transcription, guardrail
  fallback verdict (no LLM backend in this sandbox) — exactly the same
  behavior already verified outside Docker in Phase 3/Phase 4.
- **Smoke-test script**: ran the exact bash logic that's now in
  `docker.yml` locally against the final image — passed.
- Confirmed the process runs as `app`, not `root` (`docker exec ... whoami`).

Both bugs were only found by actually running the container end-to-end —
reviewing the Dockerfile or running `pytest` outside Docker would not have
caught either one.

## Scope / honesty note

This closes "reproducible Docker environments" and "container scanning" for
real — built, scanned, fixed, rescanned clean. "Deployment validation" is
scoped to CI-side validation of the built artifact (it starts, serves,
degrades correctly); there is no actual cloud deployment, no registry push,
and no live host running this image anywhere. That would need real
infrastructure/credentials this project doesn't have — say so plainly if
asked, rather than implying a running production deployment exists.

## Follow-ups

- [ ] Push the scanned image to a registry (e.g. GHCR) once there's an
      actual deployment target to pull it from.
- [ ] A real deployment workflow (environment approval, post-deploy smoke
      test against a live host) once there's somewhere to deploy to.
