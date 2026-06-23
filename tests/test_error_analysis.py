import pytest

from src.eval.error_analysis import analyse_predictions, parse_with_failure, render_markdown

_FRAUD = '{"risk":"high","fraud_type":"other","reason":"x","flagged_spans":[]}'
_LEGIT = '{"risk":"low","fraud_type":"none","reason":"x","flagged_spans":[]}'


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("no object", "missing_json_object"),
        ('{"risk":', "missing_json_object"),
        ('{"risk": nope}', "malformed_json"),
        ('{"risk":"high"}', "missing_required_field"),
        ('{"risk":"extreme","fraud_type":"other","reason":"x","flagged_spans":[]}', "invalid_enum"),
        ('{"risk":"high","fraud_type":"other","reason":"","flagged_spans":[]}', "invalid_reason"),
    ],
)
def test_parse_with_failure_categories(raw, reason):
    verdict, failure = parse_with_failure(raw)
    assert verdict is None
    assert failure == reason


def test_analyse_predictions_confusion_invalid_and_baseline_pairing():
    predictions = [_FRAUD, _FRAUD, _LEGIT, "not json"]
    gold = [True, False, False, True]
    baseline = [False, False, False, True]

    summary, rows = analyse_predictions(predictions, gold, baseline)

    assert summary["llm_confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert summary["invalid_outputs"]["total"] == 1
    assert summary["invalid_outputs"]["by_gold_label"] == {"fraud": 1}
    assert summary["case_counts"]["baseline_wrong_llm_correct"] == 1
    assert summary["diagnostics"]["false_negatives"] == {
        "valid_output": 0,
        "invalid_output": 1,
    }
    assert summary["diagnostics"]["invalid_outputs_with_valid_risk"]["count"] == 0
    assert summary["diagnostics"]["dominant_false_positive_output"]["count"] == 1
    assert rows[3]["categories"] == ["false_negative", "invalid_json"]


def test_analyse_predictions_rejects_misalignment():
    with pytest.raises(ValueError, match="predictions"):
        analyse_predictions([_FRAUD], [True, False])


def test_report_only_calls_out_channel_leakage_for_non_call_inputs():
    caller_fraud = _FRAUD.replace('"reason":"x"', '"reason":"Caller exhibits a scam pattern."')
    summary, rows = analyse_predictions([caller_fraud], [False])
    email_report = render_markdown(
        summary, rows, ["hello"], [caller_fraud], dataset_name="CLAIR", input_channel="emails"
    )
    call_report = render_markdown(
        summary, rows, ["hello"], [caller_fraud], dataset_name="Calls", input_channel="phone calls"
    )
    assert email_report.startswith("# CLAIR error analysis")
    assert "cross-channel training-template leakage" in email_report
    assert "cross-channel training-template leakage" not in call_report
