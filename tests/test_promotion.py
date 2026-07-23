from src.eval.promotion import PromotionPolicy, evaluate_promotion


POLICY = PromotionPolicy(
    min_f1=0.85,
    min_pr_auc=0.85,
    min_json_validity=0.94,
    max_p95_latency_ms=2_000,
    max_error_rate=0.02,
)


def test_promotion_passes_only_when_all_signals_pass():
    decision = evaluate_promotion(
        {"f1": 0.94, "pr_auc": 0.93, "json_validity": 0.95},
        {"p95_latency_ms": 850, "error_rate": 0.005},
        {"passed": True},
        POLICY,
    )
    assert decision["decision"] == "promote"
    assert decision["failed_checks"] == []


def test_promotion_reports_every_failed_signal():
    decision = evaluate_promotion(
        {"f1": 0.80, "pr_auc": 0.90, "json_validity": 0.90},
        {"p95_latency_ms": 2_500, "error_rate": 0.01},
        {"passed": False},
        POLICY,
    )
    assert decision["decision"] == "reject"
    assert decision["failed_checks"] == [
        "f1",
        "json_validity",
        "p95_latency_ms",
        "responsible_ai",
    ]


def test_promotion_fails_closed_when_evidence_is_missing():
    decision = evaluate_promotion(None, None, None, POLICY)
    assert decision["passed"] is False
    assert decision["failed_checks"] == [
        "f1",
        "pr_auc",
        "json_validity",
        "p95_latency_ms",
        "error_rate",
        "responsible_ai",
    ]
