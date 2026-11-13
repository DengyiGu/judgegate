import pytest

pytest.importorskip("retrieval_regression")

from examples.joint_evaluation import compare_with_labels, synthetic_inputs


def test_shared_precision_reference_and_partial_label_decision():
    qrels, baseline, candidate = synthetic_inputs()
    result = compare_with_labels(qrels, baseline, candidate)
    assert result["reference"]["delta"] == pytest.approx(-0.2)
    assert result["sequential"]["decision"] == result["reference"]["decision"] == "fail"
    assert result["sequential"]["labels"] < result["sequential"]["nonzero_weight_items"]
    reverse = compare_with_labels(qrels, candidate, baseline)
    assert reverse["sequential"]["decision"] == reverse["reference"]["decision"] == "pass"
    assert reverse["reference"]["delta"] == pytest.approx(0.2)


def test_shared_queries_ties_and_budget_with_optional_judge():
    qrels = {"a": {"d1": 2}, "missing": {"d1": 1}}
    baseline = {"a": {"d2": 0.5, "d1": 0.5}}
    candidate = {"a": {"d2": 1.0}}
    cache = {("a", d): {"probs": [0.1, 0.1, 0.8]} for d in ("d1", "d2")}
    result = compare_with_labels(qrels, baseline, candidate, k=1, max_labels=1, judge_cache=cache)
    assert result["query_count"] == 2
    assert result["reference"]["delta"] == -0.5
    assert result["sequential"]["predictor"] == "recalibrated-judge"
    budget = compare_with_labels(*synthetic_inputs(), max_labels=0)
    assert budget["sequential"]["decision"] == "undecided"
    assert budget["sequential"]["agrees_with_reference"] is None
    identical = compare_with_labels(qrels, baseline, baseline, k=1)
    assert identical["sequential"]["decision"] == "pass"
    assert identical["sequential"]["labels"] == 0
