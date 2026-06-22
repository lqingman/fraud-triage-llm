"""Phase 2 — the evaluation harness. This is the project's biggest credibility
signal, so it is intentionally explicit.

Reports, on the held-out test set:
  - fraud detection: precision / recall / F1 / PR-AUC  (PR-AUC matters most for
    imbalanced fraud; accuracy is misleading)
  - JSON-validity rate of model output (via schema.parse_verdict)
  - comparison vs. an XGBoost/TF-IDF baseline -> answers "why not classical ML?"
  - cross-domain generalization on a held-out OOD set, e.g. CLAIR (--crossdomain)

The XGBoost baseline is a *control group*, not a quality bar: it is always
reported but never gates the build. The config.eval thresholds gate only the
LLM, and only when its predictions are supplied (--predictions) — that is the
CI regression target (.github/workflows/eval.yml). Until Phase 1 produces a
trained model there are no LLM predictions, so the gate is skipped and this
still runs end to end on the baseline alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml

from src.data.schema import Risk, parse_verdict

# Markers used by src.data.load.format_example, so we can recover the raw
# transcript from a formatted prompt without re-touching the dataset.
_TRANSCRIPT_OPEN = "Transcript:\n"
_TRANSCRIPT_CLOSE = "\n\nVerdict:"

# Ordinal risk -> graded fraud score, so PR-AUC sees the model's confidence
# (a "medium" hedge ranks between "low" and "high") instead of a flat 0/1.
_RISK_SCORE = {Risk.low: 0.0, Risk.medium: 0.5, Risk.high: 1.0}


def _extract_transcript(prompt: str) -> str:
    """Recover the transcript text from a formatted instruction prompt."""
    start = prompt.find(_TRANSCRIPT_OPEN)
    if start == -1:
        return prompt
    start += len(_TRANSCRIPT_OPEN)
    end = prompt.find(_TRANSCRIPT_CLOSE, start)
    return prompt[start:end] if end != -1 else prompt[start:]


def load_split(path: Path) -> tuple[list[str], list[bool], list[str]]:
    """Read a processed {prompt, completion} jsonl split.

    Returns (transcripts, gold_is_fraud, gold_completions). The gold label is
    derived from the completion verdict's risk, so it stays consistent with the
    same is_fraud projection the model is graded against.
    """
    transcripts: list[str] = []
    gold_is_fraud: list[bool] = []
    completions: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            verdict = parse_verdict(ex["completion"])
            if verdict is None:
                raise ValueError(f"gold completion failed schema parse in {path}: {ex['completion']!r}")
            transcripts.append(_extract_transcript(ex["prompt"]))
            gold_is_fraud.append(verdict.is_fraud)
            completions.append(ex["completion"])
    return transcripts, gold_is_fraud, completions


def _binary_metrics(y_true: list[bool], y_pred: list[bool], y_score: list[float]) -> dict:
    """Precision / recall / F1 / PR-AUC for the binary fraud label."""
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # average_precision_score needs at least one positive in y_true.
    pr_auc = float(average_precision_score(y_true, y_score)) if any(y_true) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
    }


def evaluate_llm(predictions: list[str], gold_is_fraud: list[bool]) -> dict:
    """Parse raw model outputs, project to binary, compute metrics.

    Unparseable outputs are counted (json_validity) and treated as a non-fraud
    prediction with score 0 — i.e. a miss whenever the gold label is fraud, so
    a model that emits garbage is penalised rather than silently dropped.
    """
    if len(predictions) != len(gold_is_fraud):
        raise ValueError(f"predictions ({len(predictions)}) != gold ({len(gold_is_fraud)})")

    parsed = [parse_verdict(p) for p in predictions]
    json_validity = sum(v is not None for v in parsed) / max(len(parsed), 1)
    pred_is_fraud = [bool(v and v.is_fraud) for v in parsed]
    pred_score = [_RISK_SCORE[v.risk] if v else 0.0 for v in parsed]

    metrics = _binary_metrics(gold_is_fraud, pred_is_fraud, pred_score)
    metrics["json_validity"] = json_validity
    metrics["n"] = len(predictions)
    return metrics


def train_baseline(train_texts: list[str], train_labels: list[bool]):
    """TF-IDF + XGBoost classical baseline to compare against.

    The control group for "why not classical ML?": it predicts the binary fraud
    label but, unlike the LLM, can produce no rationale or flagged spans. Returns
    a fitted pipeline exposing .predict / .predict_proba.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train_texts, [int(y) for y in train_labels])
    return pipeline


def evaluate_baseline(pipeline, texts: list[str], gold_is_fraud: list[bool]) -> dict:
    """Run the fitted baseline on a split and compute the same binary metrics."""
    pred = [bool(p) for p in pipeline.predict(texts)]
    score = [float(p) for p in pipeline.predict_proba(texts)[:, 1]]
    metrics = _binary_metrics(gold_is_fraud, pred, score)
    metrics["n"] = len(texts)
    return metrics


def evaluate_crossdomain(
    pipeline,
    xd_texts: list[str],
    xd_gold: list[bool],
    xd_preds: list[str] | None = None,
) -> dict:
    """Score the cross-domain (OOD) split: always the XGBoost baseline, and the
    fine-tuned LLM too when its predictions are supplied.

    The baseline is the call-trained control reused on the new channel (e.g.
    CLAIR emails); the LLM block answers "does the fine-tuned model generalize
    better than classical ML off-domain?". Returns {"baseline", "llm"} where
    "llm" is None if no predictions were given. This is reported, never gated:
    OOD F1 is expected below the in-distribution floor, so gating it would turn
    an honest generalization check into a false regression.
    """
    base = evaluate_baseline(pipeline, xd_texts, xd_gold)
    llm = evaluate_llm(xd_preds, xd_gold) if xd_preds is not None else None
    return {"baseline": base, "llm": llm}


def _load_predictions(path: Path) -> list[str]:
    """Read raw LLM outputs aligned 1:1 with the test split.

    Accepts a .jsonl (each line an object with a prediction/completion/output
    key) or a plain-text file with one raw output per line.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if path.suffix == ".jsonl":
        out = []
        for ln in lines:
            obj = json.loads(ln)
            for key in ("prediction", "completion", "output"):
                if key in obj:
                    out.append(obj[key])
                    break
            else:
                raise ValueError(f"no prediction/completion/output key in line: {ln!r}")
        return out
    return lines


def _fmt(metrics: dict) -> str:
    keys = ["precision", "recall", "f1", "pr_auc", "json_validity"]
    return "  ".join(f"{k}={metrics[k]:.3f}" for k in keys if k in metrics) + f"  (n={metrics['n']})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 fraud-triage eval harness")
    ap.add_argument("--data", type=Path, default=Path("data/processed"),
                    help="dir with train/val/test.jsonl (in-distribution split)")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="raw LLM outputs aligned to the test split; enables LLM eval + the CI gate")
    ap.add_argument("--crossdomain", type=Path, default=None,
                    help="dir with a cross-domain test.jsonl (e.g. CLAIR) for out-of-distribution eval")
    ap.add_argument("--crossdomain-predictions", type=Path, default=None,
                    help="raw LLM outputs aligned to the cross-domain test split; adds an LLM "
                         "row to the cross-domain report (reported, never gated)")
    ap.add_argument("--baseline-out", type=Path, default=Path("models/baseline_xgb.joblib"),
                    help="where to persist the fitted TF-IDF+XGBoost baseline")
    ap.add_argument("--metrics-out", type=Path, default=Path("reports/metrics.json"),
                    help="where to write the metrics report (JSON)")
    args = ap.parse_args()

    if args.crossdomain_predictions is not None and args.crossdomain is None:
        ap.error("--crossdomain-predictions requires --crossdomain (the split they align to)")

    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))["eval"]

    train_texts, train_labels, _ = load_split(args.data / "train.jsonl")
    test_texts, test_gold, _ = load_split(args.data / "test.jsonl")

    report: dict = {"data": str(args.data), "in_distribution": {}, "crossdomain": None}

    print(f"== In-distribution test ({args.data}) ==")

    # --- Baseline (control group; reported, never gates). Fit once, persist,
    # and reuse for the cross-domain split below. ---
    pipeline = train_baseline(train_texts, train_labels)
    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.baseline_out)
    base = evaluate_baseline(pipeline, test_texts, test_gold)
    report["in_distribution"]["baseline"] = base
    report["baseline_model"] = str(args.baseline_out)
    print(f"  XGBoost baseline : {_fmt(base)}")

    # --- LLM (the thing we're actually building; gated when predictions exist) ---
    failed = False
    if args.predictions is not None:
        preds = _load_predictions(args.predictions)
        llm = evaluate_llm(preds, test_gold)
        report["in_distribution"]["llm"] = llm
        print(f"  Fine-tuned LLM   : {_fmt(llm)}")
        failed = (
            llm["f1"] < cfg["min_f1"]
            or llm["pr_auc"] < cfg["min_pr_auc"]
            or llm["json_validity"] < cfg["min_json_validity"]
        )
        if failed:
            print(
                f"  GATE FAILED: f1>={cfg['min_f1']} pr_auc>={cfg['min_pr_auc']} "
                f"json_validity>={cfg['min_json_validity']}"
            )
    else:
        report["in_distribution"]["llm"] = None
        print("  Fine-tuned LLM   : (no --predictions; skipping LLM eval + gate)")

    # --- Cross-domain (CLAIR): honest out-of-distribution check. Reported, never
    # gated — OOD F1 is expected below the in-distribution floor. ---
    if args.crossdomain is not None:
        xd_texts, xd_gold, _ = load_split(args.crossdomain / "test.jsonl")
        xd_preds = _load_predictions(args.crossdomain_predictions) if args.crossdomain_predictions else None
        xd = evaluate_crossdomain(pipeline, xd_texts, xd_gold, xd_preds)
        report["crossdomain"] = {"data": str(args.crossdomain), **xd}
        print(f"\n== Cross-domain test ({args.crossdomain}) ==")
        print(f"  XGBoost baseline : {_fmt(xd['baseline'])}")
        if xd["llm"] is not None:
            print(f"  Fine-tuned LLM   : {_fmt(xd['llm'])}")
        else:
            print("  Fine-tuned LLM   : (no --crossdomain-predictions; baseline only)")

    report["gate_failed"] = failed
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved baseline -> {args.baseline_out}\nSaved metrics  -> {args.metrics_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
