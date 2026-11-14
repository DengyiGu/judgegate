import math

import numpy as np
import pytest

from judgegate import (
    Predictor,
    bounds,
    draw_values,
    gate_path,
    labeling_order,
    run_gate,
    sampling_probabilities,
    wealth_paths,
)


def scenario(shift, seed=0, n=400, overlap=0.0, accuracy=0.85):
    """Half the items have positive weight; ``shift`` moves relevance toward them.

    Returns weights, gains and the grade probabilities of a two-grade judge.
    """
    rng = np.random.default_rng(seed)
    weights = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * (1.0 - overlap) / n
    relevant = rng.uniform(size=n) < np.where(weights > 0, 0.4 + shift, 0.4 - shift)
    gains = relevant.astype(float)
    said = np.where(rng.uniform(size=n) < accuracy, gains, 1.0 - gains)
    confidence = rng.uniform(0.7, 0.99, size=n)
    judge = np.column_stack((np.where(said == 0, confidence, 1 - confidence), np.zeros(n)))
    judge[:, 1] = 1.0 - judge[:, 0]
    return weights, gains, judge


def decisions(weights, gains, probs, predictor, max_drop, alpha, runs, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        order = labeling_order(probs, rng)
        out.append(
            gate_path(
                weights, probs, order, gains[order], predictor, max_drop=max_drop, alpha=alpha
            )
        )
    return out


def test_wealth_rarely_grows_against_the_truth():
    # Thresholds on the wrong side of the truth: their wealth must stay below 1/alpha in all
    # but a fraction alpha of the labeling orders, at every label simultaneously.
    weights, gains, judge = scenario(0.03, seed=3, n=600)
    truth = float(np.sum(weights * gains))
    probs = sampling_probabilities(weights)
    predictor = Predictor.from_judge(judge, [0.0, 1.0], weights)
    rng = np.random.default_rng(7)
    alpha, need = 0.1, math.log(10.0)
    wrong_pass = wrong_fail = 0
    for _ in range(400):
        order = labeling_order(probs, rng)
        draws = draw_values(weights, probs, order, gains[order], predictor)
        log_pass, log_fail = wealth_paths(draws, np.array([truth + 0.004, truth - 0.004]))
        wrong_pass += bool(log_pass[:, 0].max() >= need)  # claims delta >= truth + 0.004
        wrong_fail += bool(log_fail[:, 1].max() >= need)  # claims delta < truth - 0.004
    assert wrong_pass / 400 <= alpha
    assert wrong_fail / 400 <= alpha


def test_gate_decides_clear_cases_from_a_fraction_of_the_labels():
    for shift, expected in ((0.25, "pass"), (-0.25, "fail")):
        weights, gains, judge = scenario(shift)
        truth = float(np.sum(weights * gains))
        assert (truth >= -0.02) == (expected == "pass")
        predictor = Predictor.from_judge(judge, [0.0, 1.0], weights)
        results = decisions(
            weights, gains, sampling_probabilities(weights), predictor, 0.02, 0.05, 50, seed=5
        )
        assert {r.decision for r in results} == {expected}
        assert np.mean([r.labels for r in results]) < len(weights) / 4
        assert all(r.low <= truth <= r.high for r in results)


def test_gate_error_rate_is_controlled_near_the_threshold():
    for seed in (2, 11):
        weights, gains, judge = scenario(0.0, seed=seed)
        truth = float(np.sum(weights * gains))
        probs = sampling_probabilities(weights)
        for predictor in (
            Predictor.without_judge(weights),
            Predictor.from_judge(judge, [0.0, 1.0], weights),
        ):
            # Put the threshold just beside the truth, once on each side.
            for max_drop, correct in ((-truth + 0.003, "pass"), (-truth - 0.003, "fail")):
                if max_drop < 0:
                    continue
                results = decisions(weights, gains, probs, predictor, max_drop, 0.1, 300, seed=9)
                # Labeling everything settles any case, so a full order is never undecided.
                assert all(r.decision != "undecided" for r in results)
                assert np.mean([r.decision != correct for r in results]) <= 0.1


def test_a_good_judge_saves_labels_and_a_useless_one_costs_little():
    weights, gains, judge = scenario(0.12, seed=4, n=2000, overlap=0.5)
    _, _, coin = scenario(0.12, seed=4, n=2000, overlap=0.5, accuracy=0.5)
    probs = sampling_probabilities(weights)
    mean_labels = {}
    for name, predictor in (
        ("plain", Predictor.zero(len(weights))),
        ("learned mean", Predictor.without_judge(weights)),
        ("judge", Predictor.from_judge(judge, [0.0, 1.0], weights)),
        ("coin trusted", Predictor.fixed(coin[:, 1])),
        ("coin refitted", Predictor.from_judge(coin, [0.0, 1.0], weights)),
    ):
        results = decisions(weights, gains, probs, predictor, 0.02, 0.05, 60, seed=0)
        assert {r.decision for r in results} == {"pass"}
        mean_labels[name] = np.mean([r.labels for r in results])
    assert mean_labels["learned mean"] < mean_labels["plain"]
    assert mean_labels["judge"] < 0.7 * mean_labels["learned mean"]
    assert mean_labels["coin refitted"] < mean_labels["coin trusted"]
    assert mean_labels["coin refitted"] < 1.3 * mean_labels["learned mean"]


def test_bounds_contain_the_truth_and_tighten():
    weights, gains, judge = scenario(0.1, seed=6, n=800)
    truth = float(np.sum(weights * gains))
    probs = sampling_probabilities(weights)
    predictor = Predictor.from_judge(judge, [0.0, 1.0], weights)
    order = labeling_order(probs, np.random.default_rng(3))
    widths = []
    for count in (50, 200, 800):
        low, high = bounds(weights, probs, order[:count], gains[order[:count]], predictor)
        assert low <= truth <= high
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2]
    assert widths[2] == pytest.approx(0.0, abs=1e-12)


def test_interactive_gate():
    same = run_gate(np.array([]), lambda i: pytest.fail("no label needed"), max_drop=0.02)
    assert same.decision == "pass" and same.labels == 0 and same.low == same.high == 0.0
    weights, gains, judge = scenario(0.25)
    truth = float(np.sum(weights * gains))
    predictor = Predictor.from_judge(judge, [0.0, 1.0], weights)
    asked = []

    def label(index):
        asked.append(index)
        return gains[index]

    result = run_gate(weights, label, max_drop=0.02, predictor=predictor, seed=3)
    assert result.decision == "pass"
    assert result.labels == len(asked) == len(set(asked)) < len(weights) / 2
    assert -0.02 <= result.low <= truth <= result.high

    # No judge and default sampling still decide, with more labels.
    plain = run_gate(weights, lambda i: gains[i], max_drop=0.02, seed=3)
    assert plain.decision == "pass" and plain.labels >= result.labels

    capped = run_gate(weights, lambda i: gains[i], max_drop=0.02, max_labels=5)
    assert capped.decision == "undecided" and capped.labels == 5

    # A difference exactly on the threshold counts as a pass and is always resolved.
    w = np.array([0.5, -0.5, 0.25, -0.25])
    g = np.array([1.0, 1.0, 0.0, 0.0])
    edge = run_gate(w, lambda i: g[i], max_drop=0.0, batch=1)
    assert edge.decision == "pass" and edge.labels <= 4
    assert edge.low <= 0.0 <= edge.high
    with pytest.raises(ValueError):
        run_gate(weights, lambda i: 2.0, max_drop=0.02)
    with pytest.raises(ValueError):
        gate_path(
            weights,
            sampling_probabilities(weights),
            np.arange(3),
            gains[:3],
            predictor,
            max_drop=-1,
        )


def test_reported_bounds_agree_with_the_decision():
    rng = np.random.default_rng(1)
    low_miss = high_miss = 0
    trials = 150
    for trial in range(trials):
        n = int(rng.integers(200, 1500))
        weights = rng.choice([-1.0, 1.0], size=n) * rng.uniform(0.2, 1.0, size=n) / n
        shift = rng.uniform(-0.15, 0.15)
        chance = np.where(weights > 0, 0.4 + shift, 0.4 - shift)
        gains = (rng.uniform(size=n) < chance).astype(float)
        truth = float(np.sum(weights * gains))
        max_drop = float(rng.choice([0.0, 0.01, 0.02, 0.05]))
        result = run_gate(weights, lambda i, g=gains: g[i], max_drop=max_drop, seed=trial)
        if result.decision == "pass":
            assert result.low >= -max_drop
        else:
            assert result.decision == "fail" and result.high <= -max_drop
        assert result.low <= result.high + 1e-12
        low_miss += result.low > truth + 1e-12
        high_miss += result.high < truth - 1e-12
    assert low_miss / trials <= 0.1 and high_miss / trials <= 0.1


def test_bounds_anchored_at_a_threshold_cover_at_the_stated_rate():
    weights, gains, judge = scenario(0.04, seed=8, n=900)
    truth = float(np.sum(weights * gains))
    probs = sampling_probabilities(weights)
    predictor = Predictor.from_judge(judge, [0.0, 1.0], weights)
    rng = np.random.default_rng(21)
    runs = 300
    for threshold in (truth + 0.02, truth - 0.02):
        low_miss = high_miss = 0
        for _ in range(runs):
            order = labeling_order(probs, rng)[:200]
            low, high = bounds(
                weights, probs, order, gains[order], predictor, alpha=0.1, threshold=threshold
            )
            low_miss += low > truth + 1e-12
            high_miss += high < truth - 1e-12
        assert low_miss / runs <= 0.1 and high_miss / runs <= 0.1
