# Phase 4b — MLflow: tracking + registry (eval-side real, training-side code-only)

**Date:** 2026-07-09
**Status:** Eval-side done and tested, runs locally with no GPU. Training-side
is a few lines added to `src/train/qlora_train.py` and is code-complete but,
like the rest of the QLoRA trainer, only ever exercised on Kaggle/Colab — no
GPU exists in this environment to run it end to end.

## Goal

There was zero MLflow code anywhere in the repo before this. Add real
experiment tracking + model registry, scoped honestly: the classical baseline
(TF-IDF/XGBoost) is the only model actually *fit* inside `src/eval/evaluate.py`
today, so that's what gets registered and promoted. The QLoRA adapter is a
Kaggle-only artifact — instrument the trainer to log it, but don't pretend to
have run that path here.

## What I did

### Eval-side (`src/eval/evaluate.py`) — runs now, no GPU needed
- Every `main()` run now logs to MLflow unconditionally: `mlflow.start_run()`
  under experiment `"fraud-triage-llm"` (overridable via `--mlflow-experiment`,
  used by tests to isolate runs), params (data paths, `config.eval`
  thresholds, git commit via a best-effort `subprocess` call), the flattened
  metrics report, the `gate_failed` tag, and `reports/metrics.json` as an
  artifact.
- Extracted `_flatten_metrics(report) -> dict[str, float]`: a pure function
  (no MLflow import needed to call it) that turns the nested
  `{in_distribution: {baseline, llm}, crossdomain: {baseline, llm}}` report
  into flat keys like `id_baseline_f1`, `crossdomain_llm_pr_auc` — skipping
  any `None` block (e.g. no `--predictions` supplied, or no LLM crossdomain
  predictions). Unit-tested directly, no MLflow run required.
- **Registry write + promotion is opt-in** (`--register-model`, off by
  default). Reasoning: `main()` runs unconditionally in CI against the tiny
  synthetic fixture (Phase 5), and registering a model version for every CI
  push — including throwaway 8-row-test-set runs scoring F1 0 — would make
  the registry's version history meaningless for anyone looking at it later.
  `--register-model` is for a deliberate "I have a real candidate, evaluate
  and register it" step, not routine CI noise.
- `_maybe_promote_champion(client, name, version, candidate_f1)`: fetches the
  current `"champion"`-aliased version's `id_baseline_f1` (via
  `MlflowClient.get_model_version_by_alias` + `get_run`), and calls
  `set_registered_model_alias(name, "champion", version)` if the candidate
  matches or beats it (or there is no champion yet). This is MLflow's modern
  alias API (stage transitions like "Staging"/"Production" are deprecated in
  favor of aliases as of MLflow 2.9+).

### A real, unplanned wrinkle: MLflow 3.x's stricter defaults
Installed `mlflow>=2.14` resolved to 3.14.0 (latest). Two of its newer
defaults broke the naive implementation, both worth knowing about if this
comes up:
1. **The plain filesystem tracking store (`./mlruns`) is now in "maintenance
   mode"** — `mlflow.set_experiment()` against a bare `file://` URI raises
   unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Fixed by using a
   database-backed URI (`sqlite:///...`) instead, which is also required for
   the Model Registry API to work reliably — not just a workaround, but the
   more correct choice for anything using the registry.
2. **`mlflow.sklearn.log_model`'s default serializer (`skops`) refuses to
   deserialize types it doesn't recognize** — logging the TF-IDF/XGBoost
   pipeline raised `UntrustedTypesFoundException` on
   `xgboost.core.Booster`/`xgboost.sklearn.XGBClassifier`. This is a real
   safety feature (skops replaces unsafe pickle deserialization), not a bug —
   fixed by explicitly passing
   `skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]`
   rather than reverting to raw pickle serialization.

### Training-side (`src/train/qlora_train.py`) — code-complete, Kaggle-only
- Added `_mlflow_available()`, mirroring the existing `_wandb_available()`
  guard (no API key needed for MLflow — it defaults to a local store).
  `report_to` now includes `"mlflow"` when available, so TRL's `SFTTrainer`
  logs per-epoch training/eval metrics via its built-in MLflow integration —
  zero additional code for that part.
- After `trainer.save_model`, a follow-up run (experiment
  `"fraud-triage-llm-training"`) logs the training config params, git commit,
  and the adapter directory as an artifact (`mlflow.log_artifacts`).
- This only runs where GPU + `transformers`/`peft`/`trl` are installed
  (Kaggle/Colab, per the rest of this file's existing import-deferral
  pattern) — it has **not** been exercised end-to-end in this environment.
  Say this plainly if asked: the code is written and reviewed, not run.

## Tests (`tests/test_evaluate_mlflow.py`, new)
Points `MLFLOW_TRACKING_URI` at a `tmp_path`-scoped sqlite file per test
(fully isolated, no shared state) and runs the real `evaluate.main()` against
the Phase 5 CI fixture:
- `_flatten_metrics` unit tests (two cases: full report, and no-llm/
  no-crossdomain).
- A full run logs params/metrics/tags to the expected experiment and run.
- `--register-model` registers a model version and promotes it to `"champion"`
  (first version, no prior champion to compare against).

## Verification

- `pytest -q tests/test_evaluate_mlflow.py` → 4 passed.
- Full suite: `pytest -q` → 70 passed.
- Manually ran `python -m src.eval.evaluate --data tests/fixtures/ci_eval
  --predictions tests/fixtures/ci_eval/predictions.jsonl --register-model`
  locally and confirmed a run + a registered `fraud-triage-baseline` v1
  aliased `"champion"` — this created a local `mlruns/` dir, which is
  deliberately **not** committed (added to `.gitignore`) since it's local
  tracking-store state, not project source.

## Scope / honesty note

"MLflow experiment tracking, dataset lineage, artifact management, and
registry-based model promotion" is now literally true **for the baseline
model, on the eval side, runnable today without a GPU**. It is not yet true
end-to-end for the fine-tuned LLM/QLoRA adapter — that requires a real Kaggle
training run with the training-side code above actually exercised, which
hasn't happened. If asked in an interview: "the eval-side MLflow integration
is real and I can demo it live; the training-side hook is written and
consistent with how the rest of the trainer defers GPU-only work, but I
haven't had a GPU session to run it since adding it."

## Follow-ups

- [ ] Next real Kaggle training run: confirm the `report_to=["mlflow"]` path
      actually logs per-epoch metrics as expected, and that the adapter
      artifact upload doesn't blow the Kaggle session's time/disk budget.
- [ ] Consider registering/promoting the LLM itself (not just the baseline)
      once there's a real registered "model" for it — likely the adapter
      directory logged as a `pyfunc` custom model, not `mlflow.sklearn`.
