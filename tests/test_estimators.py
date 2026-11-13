import itertools
import math

import numpy as np
import pytest

from judgegate import Predictor, draw_values, labeling_order, sampling_probabilities


def problem(seed=0, n=200):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=n) / n
    weights[weights == 0] = 1e-3
    gains = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.6, 0.25, 0.15])
    pred = np.clip(gains + rng.normal(scale=0.25, size=n), 0, 1)
    spread = rng.uniform(0.05, 0.5, size=n)
    return weights, gains, pred, spread


def judge_probs(gains, accuracy, rng):
    """Three-grade judge that names the true grade with the given probability."""
    truth = np.rint(gains * 2).astype(int)
    said = np.where(rng.uniform(size=len(gains)) < accuracy, truth, rng.integers(0, 3, len(gains)))
    probs = np.full((len(gains), 3), 0.05)
    probs[np.arange(len(gains)), said] = 0.9
    probs = probs * rng.uniform(0.9, 1.1, size=probs.shape)
    return probs / probs.sum(axis=1, keepdims=True)


def test_probabilities_are_distributions_with_bounded_ratio():
    weights, _, _, spread = problem()
    for probs in (
        sampling_probabilities(weights, uniform=True),
        sampling_probabilities(weights),
        sampling_probabilities(weights, spread, mix=0.2),
    ):
        assert math.isclose(probs.sum(), 1.0)
        assert np.all(probs > 0)
    active = sampling_probabilities(weights, spread, mix=0.2)
    assert np.max(np.abs(weights) / active) <= np.abs(weights).sum() / 0.2 + 1e-12
    # No uncertainty anywhere falls back to |w| sampling.
    assert np.allclose(
        sampling_probabilities(weights, np.zeros_like(weights)), sampling_probabilities(weights)
    )


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sampling_probabilities(np.array([0.1, 0.0]))
    with pytest.raises(ValueError):
        sampling_probabilities(np.array([0.1, 0.2]), np.array([0.1, 0.2]), mix=0.0)
    with pytest.raises(ValueError):
        sampling_probabilities(np.array([0.1, 0.2]), np.array([0.1]))
    weights, gains, pred, _ = problem(n=5)
    probs = sampling_probabilities(weights)
    predictor = Predictor.fixed(pred)
    with pytest.raises(ValueError):
        draw_values(weights, probs, np.array([0, 0]), gains[[0, 0]], predictor)
    with pytest.raises(ValueError):
        draw_values(weights, probs, np.array([0, 1]), np.array([0.5, 1.5]), predictor)
    with pytest.raises(ValueError):
        draw_values(weights, probs, np.array([0, 1]), gains[:2], Predictor.zero(4))
    with pytest.raises(ValueError):
        Predictor(np.array([1.5]), np.array([0]))
    with pytest.raises(ValueError):
        Predictor(np.array([0.5]), np.array([0]), strength=0.0)


def order_probability(order, probs):
    """Probability of a full labeling order under successive sampling."""
    left, result = 1.0, 1.0
    for index in order:
        result *= probs[index] / left
        left -= probs[index]
    return result


def predictors(weights, gains, pred, rng):
    return {
        "zero": Predictor.zero(len(weights)),
        "fixed": Predictor.fixed(pred),
        "learned mean": Predictor.without_judge(weights, strength=1.0),
        "judge": Predictor.from_judge(judge_probs(gains, 0.7, rng), [0, 0.5, 1], weights),
    }


def test_every_label_is_exactly_unbiased_given_the_past():
    # Enumerate all 720 labeling orders of six items and take exact expectations.
    rng = np.random.default_rng(4)
    weights = np.array([0.30, -0.25, 0.20, -0.10, 0.08, -0.07])
    gains = np.array([1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
    pred = np.clip(gains + rng.normal(scale=0.3, size=6), 0, 1)
    probs = sampling_probabilities(weights, np.array([0.5, 0.1, 0.4, 0.2, 0.3, 0.3]))
    truth = float(np.sum(weights * gains))
    for name, predictor in predictors(weights, gains, pred, rng).items():
        expected = np.zeros(6)
        total = 0.0
        for order in itertools.permutations(range(6)):
            order = np.array(order)
            chance = order_probability(order, probs)
            draws = draw_values(weights, probs, order, gains[order], predictor)
            assert np.all(np.abs(draws.values - draws.center) <= draws.radius + 1e-12), name
            assert np.all(draws.sure_low <= truth + 1e-12), name
            assert np.all(draws.sure_high >= truth - 1e-12), name
            assert math.isclose(draws.values[-1], truth, abs_tol=1e-12), name
            expected += chance * draws.values
            total += chance
        assert math.isclose(total, 1.0)
        assert np.allclose(expected, truth, atol=1e-12), name


def test_a_prefix_gives_the_same_values_as_the_full_order():
    weights, gains, pred, spread = problem(n=50)
    probs = sampling_probabilities(weights, spread)
    order = labeling_order(probs, np.random.default_rng(2))
    predictor = Predictor.without_judge(weights)
    full = draw_values(weights, probs, order, gains[order], predictor)
    part = draw_values(weights, probs, order[:20], gains[order[:20]], predictor)
    for field in ("values", "center", "radius", "remaining", "sure_low", "sure_high"):
        assert np.allclose(getattr(part, field), getattr(full, field)[:20])


def test_labeling_order_follows_the_propensities():
    probs = np.array([0.6, 0.3, 0.1])
    rng = np.random.default_rng(0)
    first = np.bincount([labeling_order(probs, rng)[0] for _ in range(6000)], minlength=3)
    assert np.allclose(first / 6000, probs, atol=0.02)
    assert sorted(labeling_order(probs, rng).tolist()) == [0, 1, 2]


def test_predictions_are_learned_from_labels():
    rng = np.random.default_rng(5)
    weights, gains, pred, _ = problem(n=600)
    probs = sampling_probabilities(weights)
    truth = float(np.sum(weights * gains))
    # A judge whose grades are informative but mislabeled: it calls every grade one too high.
    shifted = np.roll(judge_probs(gains, 0.85, rng), 1, axis=1)
    options = {
        "zero": Predictor.zero(len(weights)),
        "learned mean": Predictor.without_judge(weights),
        "shifted judge, trusted": Predictor.fixed(shifted @ np.array([0, 0.5, 1])),
        "shifted judge, refitted": Predictor.from_judge(shifted, [0, 0.5, 1], weights),
    }
    error = {}
    for name, predictor in options.items():
        values = np.array(
            [
                draw_values(weights, probs, order, gains[order], predictor).values[100:300]
                for order in (labeling_order(probs, rng) for _ in range(150))
            ]
        )
        error[name] = float(np.sqrt(np.mean((values - truth) ** 2)))
    assert error["learned mean"] < error["zero"]
    assert error["shifted judge, refitted"] < 0.7 * error["learned mean"]
    assert error["shifted judge, refitted"] < 0.7 * error["shifted judge, trusted"]


def test_judge_groups_follow_grade_and_confidence():
    probs = np.array(
        [[0.9, 0.05, 0.05], [0.5, 0.4, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    )
    weights = np.array([0.2, 0.2, -0.2, 0.2, -0.2])
    predictor = Predictor.from_judge(probs, [0.0, 0.5, 1.0], weights)
    groups = predictor.groups
    assert groups[0] != groups[1]  # same grade, different confidence
    assert groups[0] == groups[2] and groups[3] == groups[4]  # sides are split later
    assert len(set(groups.tolist())) == 3
    assert predictor.prior[3] == pytest.approx(0.85)
    with pytest.raises(ValueError):
        Predictor.from_judge(probs[:2], [0.0, 0.5, 1.0], weights)
