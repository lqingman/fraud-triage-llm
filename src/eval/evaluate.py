"""Phase 2 — the evaluation harness. This is the project's biggest credibility
signal, so it is intentionally explicit.

Reports, on the held-out test set:
  - fraud detection: precision / recall / F1 / PR-AUC  (PR-AUC matters most for
    imbalanced fraud; accuracy is misleading)
  - JSON-validity rate of model output (via schema.parse_verdict)
  - comparison vs. an XGBoost/TF-IDF baseline -> answers "why not classical ML?"
  - cross-domain generalization on DIFRAUD

Exit code is non-zero if metrics fall below config.eval thresholds, so this
doubles as the CI regression gate (.github/workflows/eval.yml).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src.data.schema import parse_verdict


def evaluate_llm(predictions: list[str], gold_is_fraud: list[bool]) -> dict:
    """Parse raw model outputs, project to binary, compute metrics.

    TODO(Phase 2):
      - from sklearn.metrics import precision_recall_fscore_support, average_precision_score
      - handle unparseable outputs (count them; treat as wrong prediction)
    """
    parsed = [parse_verdict(p) for p in predictions]
    json_validity = sum(v is not None for v in parsed) / max(len(parsed), 1)
    pred_is_fraud = [bool(v and v.is_fraud) for v in parsed]
    # TODO: compute precision/recall/f1/pr_auc from pred_is_fraud vs gold_is_fraud
    return {"json_validity": json_validity, "n": len(predictions)}


def train_baseline(train_texts, train_labels):
    """TODO(Phase 2): TF-IDF + XGBoost classical baseline to compare against."""
    raise NotImplementedError("Phase 2: classical baseline")


def main() -> int:
    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))["eval"]
    # TODO: load predictions + gold, call evaluate_llm, print a results table
    metrics = {"f1": 0.0, "pr_auc": 0.0, "json_validity": 0.0}  # placeholder

    failed = (
        metrics["f1"] < cfg["min_f1"]
        or metrics["pr_auc"] < cfg["min_pr_auc"]
        or metrics["json_validity"] < cfg["min_json_validity"]
    )
    print(metrics)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
