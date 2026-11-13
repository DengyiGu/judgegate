import json
import math

import numpy as np
import pytest

from judgegate.agreement import cohen_kappa, confusion_matrix, kendall_tau
from judgegate.files import write_trec_qrels, write_trec_run
from judgegate.judge import (
    build_messages,
    clip_words,
    grade_probabilities,
    load_cache,
    predicted_gain,
    prompt_hash,
)


def test_kappa_by_hand():
    # Observed agreement 0.7, chance agreement 0.5 -> kappa 0.4.
    matrix = confusion_matrix([0] * 50 + [1] * 50, [0] * 35 + [1] * 15 + [0] * 15 + [1] * 35, 2)
    assert matrix.tolist() == [[35, 15], [15, 35]]
    assert math.isclose(cohen_kappa(matrix), 0.4)
    assert math.isclose(cohen_kappa(np.diag([5, 5, 5])), 1.0)
    # Adjacent-grade confusion is penalized less under linear weights.
    near = np.array([[10, 5, 0], [5, 10, 5], [0, 5, 10]])
    assert cohen_kappa(near, weighted=True) > cohen_kappa(near)


def test_kendall_tau():
    assert kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert math.isclose(kendall_tau([1, 2, 3], [10, 30, 20]), 1 / 3)
    with pytest.raises(ValueError):
        kendall_tau([1], [1])


def test_trec_files_load_in_an_independent_reader(tmp_path):
    rr = pytest.importorskip("retrieval_regression")
    write_trec_qrels({"q2": {"d1": 1}, "q1": {"d9": 0, "d2": 2}}, tmp_path / "qrels.trec")
    write_trec_run({"q1": {"d2": 0.5, "d9": 0.5, "d3": 1.25}}, tmp_path / "run.trec", "demo")
    assert rr.load_qrels(tmp_path / "qrels.trec", format="trec") == {
        "q1": {"d2": 2.0, "d9": 0.0},
        "q2": {"d1": 1.0},
    }
    assert rr.load_run(tmp_path / "run.trec", format="trec") == {
        "q1": {"d3": 1.25, "d2": 0.5, "d9": 0.5}
    }
    assert (tmp_path / "run.trec").read_text().splitlines()[1] == "q1 Q0 d2 2 0.5 demo"
    with pytest.raises(ValueError):
        write_trec_qrels({"q 1": {"d": 1}}, tmp_path / "bad.trec")
    with pytest.raises(ValueError):
        write_trec_qrels({"q": {"d": -1}}, tmp_path / "bad.trec")


def test_prompt_and_cache(tmp_path):
    assert clip_words("a  b\nc d", 3) == "a b c"
    messages = build_messages("what   is x", "one two three", max_words=2)
    assert "Query: what is x" in messages[1]["content"]
    assert "Passage: one two\n" in messages[1]["content"]
    assert len(prompt_hash()) == 64
    path = tmp_path / "cache.jsonl"
    row = {"qid": "q", "docid": "d", "probs": [0.2, 0.3, 0.5], "digit_mass": 0.99}
    path.write_text(json.dumps(row) + "\n" + '{"qid": "q", "doc', encoding="utf-8")
    cache = load_cache(path)
    assert list(cache) == [("q", "d")]
    assert load_cache(tmp_path / "missing.jsonl") == {}
    probs = grade_probabilities(cache, [("q", "d")])
    mean, spread = predicted_gain(probs, np.array([0.0, 0.5, 1.0]))
    assert math.isclose(mean[0], 0.65)
    assert math.isclose(spread[0], math.sqrt(0.3 * 0.25 + 0.5 - 0.65**2))
