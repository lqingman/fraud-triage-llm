"""Deterministic per-example error analysis for saved LLM predictions.

This module deliberately consumes the same split and prediction formats as the
main evaluator. It turns aggregate metrics into a confusion matrix, structured
output failure categories, and review samples without running the LLM again.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from src.data.schema import FraudVerdict
from src.eval.evaluate import _load_predictions, evaluate_llm, load_split, train_baseline


def parse_with_failure(raw: str) -> tuple[FraudVerdict | None, str | None]:
    """Parse exactly one verdict and explain a schema failure when possible."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "missing_json_object"

    fragment = raw[start : end + 1]
    try:
        obj = json.loads(fragment)
    except json.JSONDecodeError:
        return None, "malformed_json"

    try:
        return FraudVerdict.model_validate(obj), None
    except ValidationError as exc:
        errors = exc.errors()
        if any(err["type"] == "missing" for err in errors):
            return None, "missing_required_field"
        if any(err.get("loc", (None,))[0] in {"risk", "fraud_type"} for err in errors):
            return None, "invalid_enum"
        if any(err.get("loc", (None,))[0] == "reason" for err in errors):
            return None, "invalid_reason"
        if any(err.get("loc", (None,))[0] == "flagged_spans" for err in errors):
            return None, "invalid_flagged_spans"
        return None, "schema_validation"


def _confusion(gold: list[bool], pred: list[bool]) -> dict[str, int]:
    return {
        "tn": sum(not y and not p for y, p in zip(gold, pred)),
        "fp": sum(not y and p for y, p in zip(gold, pred)),
        "fn": sum(y and not p for y, p in zip(gold, pred)),
        "tp": sum(y and p for y, p in zip(gold, pred)),
    }


def _binary_from_confusion(cm: dict[str, int]) -> dict[str, float]:
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if cm["tp"] + cm["fp"] else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if cm["tp"] + cm["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _risk_hint(raw: str) -> str | None:
    """Recover a valid binary risk value from otherwise schema-invalid JSON."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        risk = json.loads(raw[start : end + 1]).get("risk")
    except (json.JSONDecodeError, AttributeError):
        return None
    return risk if risk in {"low", "medium", "high"} else None


def analyse_predictions(
    predictions: list[str],
    gold: list[bool],
    baseline_predictions: list[bool] | None = None,
) -> tuple[dict, list[dict]]:
    """Return an aggregate summary and one compact record per evaluated row."""
    if len(predictions) != len(gold):
        raise ValueError(f"predictions ({len(predictions)}) != gold ({len(gold)})")
    if baseline_predictions is not None and len(baseline_predictions) != len(gold):
        raise ValueError(
            f"baseline predictions ({len(baseline_predictions)}) != gold ({len(gold)})"
        )

    rows: list[dict] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_by_gold: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    false_positive_outputs: Counter[str] = Counter()
    invalid_risk_values: Counter[str] = Counter()
    llm_pred: list[bool] = []
    risk_fallback_pred: list[bool] = []

    for index, (raw, gold_label) in enumerate(zip(predictions, gold)):
        verdict, failure = parse_with_failure(raw)
        pred_label = bool(verdict and verdict.is_fraud)
        llm_pred.append(pred_label)
        fallback_label = pred_label

        outcome = (
            "true_positive" if gold_label and pred_label
            else "false_negative" if gold_label
            else "false_positive" if pred_label
            else "true_negative"
        )
        categories = [outcome]
        case_counts[outcome] += 1

        if failure is not None:
            categories.append("invalid_json")
            case_counts["invalid_json"] += 1
            invalid_reasons[failure] += 1
            invalid_by_gold["fraud" if gold_label else "non_fraud"] += 1
            risk_hint = _risk_hint(raw)
            if risk_hint is not None:
                fallback_label = risk_hint in {"medium", "high"}
                invalid_risk_values[risk_hint] += 1
        risk_fallback_pred.append(fallback_label)

        if not gold_label and pred_label:
            false_positive_outputs[raw] += 1

        baseline_label = baseline_predictions[index] if baseline_predictions is not None else None
        if baseline_label is not None and baseline_label != gold_label and pred_label == gold_label:
            categories.append("baseline_wrong_llm_correct")
            case_counts["baseline_wrong_llm_correct"] += 1

        rows.append(
            {
                "index": index,
                "gold_is_fraud": gold_label,
                "predicted_is_fraud": pred_label,
                "json_valid": failure is None,
                "invalid_reason": failure,
                "risk": verdict.risk.value if verdict else None,
                "fraud_type": verdict.fraud_type.value if verdict else None,
                "baseline_predicted_is_fraud": baseline_label,
                "categories": categories,
            }
        )

    fallback_cm = _confusion(gold, risk_fallback_pred)
    dominant_fp, dominant_fp_count = (
        false_positive_outputs.most_common(1)[0] if false_positive_outputs else (None, 0)
    )
    summary = {
        "n": len(gold),
        "gold_counts": {"fraud": sum(gold), "non_fraud": len(gold) - sum(gold)},
        "llm_metrics": evaluate_llm(predictions, gold),
        "llm_confusion_matrix": _confusion(gold, llm_pred),
        "invalid_outputs": {
            "total": sum(invalid_reasons.values()),
            "by_reason": dict(sorted(invalid_reasons.items())),
            "by_gold_label": dict(sorted(invalid_by_gold.items())),
        },
        "case_counts": dict(sorted(case_counts.items())),
        "diagnostics": {
            "false_negatives": {
                "valid_output": sum(
                    "false_negative" in row["categories"] and row["json_valid"] for row in rows
                ),
                "invalid_output": sum(
                    "false_negative" in row["categories"] and not row["json_valid"] for row in rows
                ),
            },
            "invalid_outputs_with_valid_risk": {
                "count": sum(invalid_risk_values.values()),
                "risk_values": dict(sorted(invalid_risk_values.items())),
                "counterfactual_confusion_matrix": fallback_cm,
                "counterfactual_metrics": _binary_from_confusion(fallback_cm),
                "note": "Diagnostic only: applies a risk-only fallback to schema-invalid outputs.",
            },
            "dominant_false_positive_output": {
                "count": dominant_fp_count,
                "share_of_false_positives": (
                    dominant_fp_count / case_counts["false_positive"]
                    if case_counts["false_positive"] else 0.0
                ),
                "output": dominant_fp,
            },
        },
    }
    if baseline_predictions is not None:
        summary["local_baseline_confusion_matrix"] = _confusion(gold, baseline_predictions)
    return summary, rows


def _short(value: str, limit: int = 180) -> str:
    value = " ".join(value.split()).replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_markdown(
    summary: dict,
    rows: list[dict],
    transcripts: list[str],
    predictions: list[str],
    sample_size: int = 20,
    seed: int = 42,
    dataset_name: str = "Prediction",
    input_channel: str | None = None,
) -> str:
    """Render a deterministic review report with bounded example excerpts."""
    cm = summary["llm_confusion_matrix"]
    invalid = summary["invalid_outputs"]
    diagnostics = summary["diagnostics"]
    fn = diagnostics["false_negatives"]
    fallback = diagnostics["invalid_outputs_with_valid_risk"]
    dominant_fp = diagnostics["dominant_false_positive_output"]
    findings = [
        f"- Of {fn['valid_output'] + fn['invalid_output']} false negatives, **{fn['invalid_output']}** come from schema-invalid outputs; only {fn['valid_output']} are valid low-risk verdicts.",
        f"- Missing required fields account for **{invalid['by_reason'].get('missing_required_field', 0)} / {invalid['total']}** invalid outputs.",
        f"- Invalid outputs are concentrated on fraud examples: **{invalid['by_gold_label'].get('fraud', 0)} / {invalid['total']}** have a fraud gold label.",
        f"- {fallback['count']} invalid outputs still contain a valid `risk` value. A diagnostic risk-only fallback would yield F1 **{fallback['counterfactual_metrics']['f1']:.3f}**; this is not the reported production metric.",
        f"- One generic output accounts for **{dominant_fp['count']} / {summary['case_counts']['false_positive']}** false positives ({dominant_fp['share_of_false_positives']:.1%}), indicating a repeated default-high response.",
    ]
    if (
        input_channel
        and input_channel != "phone calls"
        and "Caller" in (dominant_fp.get("output") or "")
    ):
        findings.append(
            f"- The dominant false-positive rationale says `Caller` even though the inputs are {input_channel}, exposing cross-channel training-template leakage."
        )

    lines = [
        f"# {dataset_name} error analysis",
        "",
        f"Evaluated {summary['n']} examples. Invalid structured outputs are treated as non-fraud, matching the main evaluator.",
        "",
        "## Key findings",
        "",
        *findings,
        "",
        "## LLM confusion matrix",
        "",
        "| | predicted non-fraud | predicted fraud |",
        "|---|---:|---:|",
        f"| gold non-fraud | {cm['tn']} | {cm['fp']} |",
        f"| gold fraud | {cm['fn']} | {cm['tp']} |",
        "",
        "## Invalid structured outputs",
        "",
        f"Total: **{invalid['total']}** ({invalid['total'] / max(summary['n'], 1):.1%})",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in invalid["by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "By gold label: " + ", ".join(
        f"{label}={count}" for label, count in invalid["by_gold_label"].items()
    ) + "."])

    rng = random.Random(seed)
    for category in ("false_positive", "false_negative", "invalid_json", "baseline_wrong_llm_correct"):
        candidates = [row for row in rows if category in row["categories"]]
        chosen = rng.sample(candidates, min(sample_size, len(candidates))) if candidates else []
        lines.extend([
            "",
            f"## Sample: {category} ({len(candidates)} total)",
            "",
        ])
        if not chosen:
            lines.append("No examples available.")
            continue
        lines.extend([
            "| index | gold | pred | valid | failure | transcript excerpt | raw output |",
            "|---:|---|---|---|---|---|---|",
        ])
        for row in sorted(chosen, key=lambda item: item["index"]):
            index = row["index"]
            lines.append(
                f"| {index} | {row['gold_is_fraud']} | {row['predicted_is_fraud']} | "
                f"{row['json_valid']} | {row['invalid_reason'] or '—'} | "
                f"{_short(transcripts[index])} | {_short(predictions[index])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse saved fraud predictions example by example")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--baseline-data", type=Path, default=None,
                    help="optional train/test directory for local XGBoost comparison")
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--rows-out", type=Path, required=True)
    ap.add_argument("--markdown-out", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-name", default="Prediction")
    ap.add_argument("--input-channel", default=None)
    args = ap.parse_args()

    transcripts, gold, _ = load_split(args.split)
    predictions = _load_predictions(args.predictions)
    baseline_predictions = None
    baseline_environment = None
    if args.baseline_data is not None:
        train_texts, train_gold, _ = load_split(args.baseline_data / "train.jsonl")
        pipeline = train_baseline(train_texts, train_gold)
        baseline_predictions = [bool(value) for value in pipeline.predict(transcripts)]
        import sklearn
        import xgboost
        baseline_environment = {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
        }

    summary, rows = analyse_predictions(predictions, gold, baseline_predictions)
    if baseline_environment is not None:
        summary["local_baseline_environment"] = baseline_environment

    for path in (args.summary_out, args.rows_out, args.markdown_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.rows_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.markdown_out.write_text(
        render_markdown(
            summary, rows, transcripts, predictions, args.sample_size, args.seed,
            args.dataset_name, args.input_channel,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary -> {args.summary_out}")
    print(f"Wrote rows    -> {args.rows_out}")
    print(f"Wrote report  -> {args.markdown_out}")


if __name__ == "__main__":
    main()
