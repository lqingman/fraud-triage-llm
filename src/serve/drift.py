"""Phase 4f — drift monitoring: track the rolling distribution of predicted
risk and flag when it deviates from the training-time baseline fraud rate.

Uses the Population Stability Index (PSI) collapsed to two buckets (fraud vs.
not) — a standard industry heuristic: PSI < 0.1 is no significant shift,
0.1-0.25 is moderate, > 0.25 is significant. This project's training corpus
is ~50% fraud (see docs/devlog/data-strategy-and-baseline.md), so that's the
default baseline; a sustained shift away from it — e.g. because the model or
input distribution changed, or because the guardrails' fallback verdict
(always "medium" == is_fraud) is firing on every request during a backend
outage — should page someone, not go unnoticed.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

DEFAULT_WINDOW = 200
DEFAULT_BASELINE_FRAUD_RATE = 0.5
DEFAULT_PSI_ALERT_THRESHOLD = 0.25
_EPS = 1e-6

CONFIG_PATH = Path("config/config.yaml")


def _clip(p: float) -> float:
    """Keep proportions strictly inside (0, 1) so PSI's log() never blows up
    on a window that's briefly all-fraud or all-legitimate."""
    return min(max(p, _EPS), 1 - _EPS)


def _psi_component(observed: float, baseline: float) -> float:
    return (observed - baseline) * math.log(observed / baseline)


class DriftTracker:
    """A fixed-size rolling window of recent is_fraud predictions, compared
    against a baseline fraud rate via PSI."""

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        baseline_fraud_rate: float = DEFAULT_BASELINE_FRAUD_RATE,
        alert_threshold: float = DEFAULT_PSI_ALERT_THRESHOLD,
    ):
        self._window: deque[bool] = deque(maxlen=window)
        self.baseline_fraud_rate = baseline_fraud_rate
        self.alert_threshold = alert_threshold

    def record(self, is_fraud: bool) -> None:
        self._window.append(bool(is_fraud))

    def fraud_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def psi(self) -> float:
        if not self._window:
            return 0.0
        observed = _clip(self.fraud_rate())
        baseline = _clip(self.baseline_fraud_rate)
        return _psi_component(observed, baseline) + _psi_component(1 - observed, 1 - baseline)

    def is_drifting(self) -> bool:
        return bool(self._window) and self.psi() > self.alert_threshold

    def reset(self) -> None:
        self._window.clear()

    def __len__(self) -> int:
        return len(self._window)


def _load_monitoring_config() -> dict:
    try:
        import yaml

        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return cfg.get("monitoring", {})
    except FileNotFoundError:
        return {}


_MONITORING_CFG = _load_monitoring_config()

# Module-level singleton, mirroring src.serve.metrics's pattern: one shared
# tracker per process, wired into src.serve.app's request path.
TRACKER = DriftTracker(
    window=_MONITORING_CFG.get("drift_window", DEFAULT_WINDOW),
    baseline_fraud_rate=_MONITORING_CFG.get("baseline_fraud_rate", DEFAULT_BASELINE_FRAUD_RATE),
    alert_threshold=_MONITORING_CFG.get("psi_alert_threshold", DEFAULT_PSI_ALERT_THRESHOLD),
)


def record(is_fraud: bool) -> None:
    TRACKER.record(is_fraud)


def fraud_rate() -> float:
    return TRACKER.fraud_rate()


def is_drifting() -> bool:
    return TRACKER.is_drifting()
