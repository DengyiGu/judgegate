import math

import numpy as np
import pytest

from judgegate import delta_items, grade_values, metric_value
from judgegate.metrics import discounts

QRELS = {"q1": {"a": 2, "b": 1, "c": 0}, "q2": {"x": 1}, "q3": {"m": 2}}
BASE = {"q1": {"a": 3.0, "c": 2.0, "b": 1.0}, "q2": {"x": 1.0, "y": 0.5}}
CAND = {"q1": {"a": 3.0, "b": 2.0, "z": 1.0}, "q2": {"y": 1.0, "x": 0.5}, "q3": {"m": 1.0}}


def gains(metric):
    values = grade_values(metric, 2)
    return {q: {d: float(values[g]) for d, g in docs.items()} for q, docs in QRELS.items()}


def test_discounts_are_normalized_and_validated():
    assert math.isclose(discounts("sdcg", 10).sum(), 1.0)
    assert math.isclose(discounts("precision", 5).sum(), 1.0)
    assert discounts("rbp", 3, rbp_p=0.5).tolist() == [0.5, 0.25, 0.125]
    with pytest.raises(ValueError):
        discounts("ndcg", 10)
    with pytest.raises(ValueError):
        discounts("sdcg", 0)


@pytest.mark.parametrize("metric", ["sdcg", "precision", "rbp"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_weighted_sum_equals_metric_difference(metric, k):
    g = gains(metric)
    items = delta_items(QRELS, BASE, CAND, metric=metric, k=k)
    expected = metric_value(QRELS, CAND, g, metric=metric, k=k) - metric_value(
        QRELS, BASE, g, metric=metric, k=k
    )
    assert math.isclose(items.delta(g), expected, abs_tol=1e-12)


def test_documents_at_the_same_rank_need_no_label():
    items = delta_items(QRELS, BASE, CAND, metric="sdcg", k=3)
    assert ("q1", "a") not in items.keys()
    # q3 is missing from the baseline, so its candidate document carries full weight.
    assert ("q3", "m") in items.keys()
    assert items.query_count == 3


def test_precision_by_hand():
    # P@2: baseline q1 = {a, c} -> 1/2, q2 = {x, y} -> 1/2, q3 missing -> 0; mean 1/3.
    # candidate q1 = {a, b} -> 1, q2 = {y, x} -> 1/2, q3 = {m} -> 1/2; mean 2/3.
    g = gains("precision")
    assert math.isclose(metric_value(QRELS, BASE, g, metric="precision", k=2), 1 / 3)
    assert math.isclose(metric_value(QRELS, CAND, g, metric="precision", k=2), 2 / 3)
    items = delta_items(QRELS, BASE, CAND, metric="precision", k=2)
    assert math.isclose(items.delta(g), 1 / 3)
    # Swapping x and y inside the top 2 changes nothing for precision.
    assert not any(q == "q2" for q, _ in items.keys())


def test_precision_matches_independent_evaluator():
    rr = pytest.importorskip("retrieval_regression")
    g = gains("precision")
    for run in (BASE, CAND):
        reference = rr.evaluate(QRELS, run, k=[2]).summary["precision@2"]
        assert math.isclose(metric_value(QRELS, run, g, metric="precision", k=2), reference)


def test_ties_break_by_document_id():
    items = delta_items(["q"], {"q": {"b": 1.0, "a": 1.0}}, {"q": {"a": 1.0, "b": 1.0}}, k=1)
    assert len(items) == 0
    assert np.isclose(items.delta({}), 0.0)


def test_grade_values():
    assert grade_values("sdcg", 2).tolist() == [0.0, 0.5, 1.0]
    assert grade_values("precision", 2).tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        grade_values("sdcg", 0)
