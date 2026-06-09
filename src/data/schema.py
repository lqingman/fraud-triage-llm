"""The structured-verdict contract. Everything (training labels, model output,
serving, eval) validates against this. This is what makes the project
'explainable triage' rather than a black-box classifier."""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field, ValidationError


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class FraudType(str, Enum):
    none = "none"
    ssn_scam = "ssn_scam"
    refund_scam = "refund_scam"
    tech_support_scam = "tech_support_scam"
    reward_scam = "reward_scam"
    impersonation = "impersonation"
    other = "other"


class FraudVerdict(BaseModel):
    """The single source of truth for model output shape."""

    risk: Risk
    fraud_type: FraudType
    reason: str = Field(..., min_length=1, description="Short analyst-style justification.")
    flagged_spans: list[str] = Field(
        default_factory=list,
        description="Verbatim phrases from the transcript that triggered the verdict.",
    )

    @property
    def is_fraud(self) -> bool:
        """Binary projection used for F1 / PR-AUC against gold labels."""
        return self.risk in (Risk.medium, Risk.high)


def parse_verdict(raw: str) -> FraudVerdict | None:
    """Best-effort parse of a model's raw text into a verdict.

    Returns None on failure so callers (eval, serving guardrails) can count
    JSON-validity rate and trigger a repair/retry. See src/serve/guardrails.py.
    """
    try:
        # Tolerate models that wrap JSON in prose / code fences.
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return None
        return FraudVerdict.model_validate_json(raw[start : end + 1])
    except (ValidationError, json.JSONDecodeError):
        return None
