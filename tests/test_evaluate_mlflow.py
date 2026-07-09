"""Phase 4b MLflow tests. Runs the real eval harness (CPU-only, no GPU/model)
against the synthetic CI fixture with MLFLOW_TRACKING_URI pointed at a
tmp_path file store, then asserts MLflow actually recorded the run — no
network, no shared/global mlruns/ state."""

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.eval.evaluate import _flatten_metrics, main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ci_eval"


def _run_main(monkeypatch, tmp_path, extra_args=None):
    # A DB-backed tracking store (sqlite) rather than the plain filesystem
    # store: newer MLflow (3.x) puts the raw './mlruns' filestore in
    # maintenance mode and requires opting in via MLFLOW_ALLOW_FILE_STORE, and
    # the Model Registry API (used below) needs a database-backed store
    # anyway — sqlite is the standard local choice for both.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    argv = [
        "evaluate",
        "--data", str(FIXTURE_DIR),
        "--predictions", str(FIXTURE_DIR / "predictions.jsonl"),
        "--metrics-out", str(tmp_path / "metrics.json"),
        "--baseline-out", str(tmp_path / "baseline.joblib"),
        "--mlflow-experiment", "test-fraud-triage-llm",
    ]
    if extra_args:
        argv += extra_args
    monkeypatch.setattr(sys, "argv", argv)
    return main()


def test_flatten_metrics_pure_helper():
    report = {
        "in_distribution": {
            "baseline": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "pr_auc": 0.95, "n": 10},
            "llm": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "pr_auc": 1.0, "json_validity": 1.0, "n": 8},
        },
        "crossdomain": {
            "baseline": {"precision": 0.4, "recall": 0.8, "f1": 0.5, "pr_auc": 0.4, "n": 100},
            "llm": None,
        },
    }
    flat = _flatten_metrics(report)
    assert flat["id_baseline_f1"] == 0.85
    assert flat["id_llm_json_validity"] == 1.0
    assert flat["crossdomain_baseline_f1"] == 0.5
    assert "crossdomain_llm_f1" not in flat  # llm=None -> skipped


def test_flatten_metrics_handles_no_llm_no_crossdomain():
    report = {"in_distribution": {"baseline": {"f1": 0.9, "n": 5}, "llm": None}, "crossdomain": None}
    flat = _flatten_metrics(report)
    assert flat == {"id_baseline_f1": 0.9, "id_baseline_n": 5.0}


def test_eval_run_logs_to_mlflow(monkeypatch, tmp_path):
    exit_code = _run_main(monkeypatch, tmp_path)
    assert exit_code == 0  # gate passes: predictions == gold completions

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("test-fraud-triage-llm")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.metrics["id_llm_f1"] == 1.0
    assert run.data.metrics["id_llm_json_validity"] == 1.0
    assert run.data.params["min_f1"] == "0.85" or float(run.data.params["min_f1"]) == 0.85
    assert run.data.tags["gate_failed"] == "False"


def test_register_model_flag_registers_and_promotes(monkeypatch, tmp_path):
    exit_code = _run_main(monkeypatch, tmp_path, extra_args=["--register-model"])
    assert exit_code == 0

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    versions = client.search_model_versions("name='fraud-triage-baseline'")
    assert len(versions) == 1

    champion = client.get_model_version_by_alias("fraud-triage-baseline", "champion")
    assert champion.version == versions[0].version
