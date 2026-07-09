"""Phase 4f drift-monitoring tests — pure, no network/GPU/model."""

import math

from src.serve.drift import DriftTracker


def test_empty_tracker_reports_zero_rate_and_no_drift():
    t = DriftTracker(window=10, baseline_fraud_rate=0.5)
    assert t.fraud_rate() == 0.0
    assert t.psi() == 0.0
    assert t.is_drifting() is False
    assert len(t) == 0


def test_matching_baseline_does_not_drift():
    t = DriftTracker(window=10, baseline_fraud_rate=0.5, alert_threshold=0.25)
    for is_fraud in [True, False] * 5:  # exactly 50% fraud, matches baseline
        t.record(is_fraud)
    assert t.fraud_rate() == 0.5
    assert math.isclose(t.psi(), 0.0, abs_tol=1e-9)
    assert t.is_drifting() is False


def test_sustained_all_fraud_triggers_drift_alert():
    t = DriftTracker(window=20, baseline_fraud_rate=0.5, alert_threshold=0.25)
    for _ in range(20):
        t.record(True)
    assert t.fraud_rate() == 1.0
    assert t.psi() > 0.25
    assert t.is_drifting() is True


def test_sustained_all_legit_triggers_drift_alert():
    t = DriftTracker(window=20, baseline_fraud_rate=0.5, alert_threshold=0.25)
    for _ in range(20):
        t.record(False)
    assert t.fraud_rate() == 0.0
    assert t.is_drifting() is True


def test_mild_shift_below_threshold_does_not_alert():
    t = DriftTracker(window=20, baseline_fraud_rate=0.5, alert_threshold=0.25)
    # 55% fraud vs 50% baseline: a small, plausible sampling wobble, not a real shift.
    for is_fraud in ([True] * 11 + [False] * 9):
        t.record(is_fraud)
    assert t.fraud_rate() == 0.55
    assert t.is_drifting() is False


def test_window_is_bounded_and_drops_oldest():
    t = DriftTracker(window=5, baseline_fraud_rate=0.5)
    for is_fraud in [True, True, True, True, True]:
        t.record(is_fraud)
    assert t.fraud_rate() == 1.0
    t.record(False)  # pushes out the oldest True
    assert len(t) == 5
    assert t.fraud_rate() == 0.8


def test_reset_clears_window():
    t = DriftTracker(window=10, baseline_fraud_rate=0.5)
    t.record(True)
    t.record(True)
    assert len(t) == 2
    t.reset()
    assert len(t) == 0
    assert t.fraud_rate() == 0.0


def test_guardrail_outage_scenario_backend_down_flips_to_all_medium_risk():
    # Documents the exact scenario the module-level docstring calls out: a
    # sustained backend outage makes every verdict the guardrails' fallback
    # (risk=medium -> is_fraud=True), which this should catch as drift.
    t = DriftTracker(window=50, baseline_fraud_rate=0.5, alert_threshold=0.25)
    for _ in range(50):
        t.record(True)  # every request returns the "medium" fallback verdict
    assert t.is_drifting() is True
