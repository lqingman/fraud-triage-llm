"""Multi-signal promotion gate for an LLM release candidate.

Offline quality alone is insufficient for production promotion. This gate
combines model metrics, serving SLOs, and a responsible-AI test result into an
auditable decision artifact and a CI-friendly exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromotionPolicy:
    min_f1: float
    min_pr_auc: float
    min_json_validity: float
    max_p95_latency_ms: float
    max_error_rate: float
    require_responsible_ai: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "PromotionPolicy":
        return cls(
            min_f1=float(config["min_f1"]),
            min_pr_auc=float(config["min_pr_auc"]),
            min_json_validity=float(config["min_json_validity"]),
            max_p95_latency_ms=float(config["max_p95_latency_ms"]),
            max_error_rate=float(config["max_error_rate"]),
            require_responsible_ai=bool(config.get("require_responsible_ai", True)),
        )


def evaluate_promotion(
    llm_metrics: dict | None,
    operational_metrics: dict | None,
    responsible_ai: dict | None,
    policy: PromotionPolicy,
) -> dict:
    checks: list[dict] = []

    def minimum(name: str, source: dict | None, threshold: float) -> None:
        value = source.get(name) if source else None
        passed = value is not None and float(value) >= threshold
        checks.append(
            {
                "name": name,
                "operator": ">=",
                "threshold": threshold,
                "value": value,
                "passed": passed,
            }
        )

    def maximum(name: str, source: dict | None, threshold: float) -> None:
        value = source.get(name) if source else None
        passed = value is not None and float(value) <= threshold
        checks.append(
            {
                "name": name,
                "operator": "<=",
                "threshold": threshold,
                "value": value,
                "passed": passed,
            }
        )

    minimum("f1", llm_metrics, policy.min_f1)
    minimum("pr_auc", llm_metrics, policy.min_pr_auc)
    minimum("json_validity", llm_metrics, policy.min_json_validity)
    maximum("p95_latency_ms", operational_metrics, policy.max_p95_latency_ms)
    maximum("error_rate", operational_metrics, policy.max_error_rate)

    if policy.require_responsible_ai:
        value = responsible_ai.get("passed") if responsible_ai else None
        checks.append(
            {
                "name": "responsible_ai",
                "operator": "==",
                "threshold": True,
                "value": value,
                "passed": value is True,
            }
        )

    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "decision": "promote" if not failed else "reject",
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the model promotion policy")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--operational-metrics", type=Path, required=True)
    parser.add_argument("--responsible-ai", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/promotion_decision.json"))
    args = parser.parse_args()

    metrics_report = _read_json(args.metrics)
    operational = _read_json(args.operational_metrics)
    responsible_ai = _read_json(args.responsible_ai)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    policy = PromotionPolicy.from_config(config["promotion"])

    llm_metrics = (metrics_report.get("in_distribution") or {}).get("llm")
    decision = evaluate_promotion(llm_metrics, operational, responsible_ai, policy)
    decision["inputs"] = {
        "metrics": str(args.metrics),
        "operational_metrics": str(args.operational_metrics),
        "responsible_ai": str(args.responsible_ai),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
