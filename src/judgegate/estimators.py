"""Unbiased estimation of a weighted label sum while labeling items one at a time.

Target: ``delta = sum_i w_i * g_i`` over N items with known weights ``w`` and unknown gains
``g`` in [0, 1].

Items are labeled in a random order drawn by successive sampling: each next item is picked
from the still unlabeled ones with probability proportional to ``p``. Before the t-th label,
the labeled items contribute their exact sum ``A``, the unlabeled set ``R`` has total
propensity ``P(R)``, every unlabeled item has a current prediction ``h_t(j)`` in [0, 1], and
the item drawn, ``i``, gives

    y_t = A + sum_{j in R} w_j h_t(j) + w_i * (g_i - h_t(i)) * P(R) / p_i.

Given everything labeled so far, ``E[y_t] = delta`` for any predictions that were fixed
before the draw: the first term is exact, the second is the predicted remainder, and the
third is an importance-weighted correction of that prediction. Good predictions make the
correction small; bad ones only add variance. After N labels ``y`` equals ``delta``.

The predictions are refitted as labels arrive. Items are partitioned into groups, and an
item's prediction is its prior shrunk toward the mean of the labels seen so far in its
group. With groups built from an LLM judge's grades this recalibrates the judge on the fly:
the gate learns what each judge grade is worth in human terms on this collection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sampling_probabilities(
    weights: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    mix: float = 0.2,
    uniform: bool = False,
) -> np.ndarray:
    """Propensities over items.

    ``uniform=True`` gives equal propensities. Otherwise the base is proportional to
    ``|w|``. With ``uncertainty`` (for example the judge's predictive standard deviation of
    each gain), the result is ``(1 - mix) * p_active + mix * p_base`` where ``p_active`` is
    proportional to ``|w| * uncertainty``. The ``mix`` floor keeps every importance ratio
    ``|w| / p`` at most ``sum(|w|) / mix``.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a non-empty finite vector")
    if np.any(weights == 0.0):
        raise ValueError("items with zero weight do not need labels; remove them")
    if uniform:
        return np.full(len(weights), 1.0 / len(weights))
    base = np.abs(weights) / np.abs(weights).sum()
    if uncertainty is None:
        return base
    if not 0.0 < mix <= 1.0:
        raise ValueError("mix must be in (0, 1]")
    uncertainty = np.asarray(uncertainty, dtype=float)
    if uncertainty.shape != weights.shape or np.any(uncertainty < 0):
        raise ValueError("uncertainty must be a non-negative vector matching weights")
    active = np.abs(weights) * uncertainty
    if active.sum() <= 0.0:
        return base
    return (1.0 - mix) * active / active.sum() + mix * base


def labeling_order(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A full labeling order from successive sampling with propensities ``probs``.

    Sorting independent exponentials divided by the propensities is equivalent to drawing
    one item at a time, without replacement, proportionally to ``probs`` among those left.
    """
    return np.argsort(rng.exponential(size=len(probs)) / probs, kind="stable")


@dataclass(frozen=True)
class Predictor:
    """Per-item predictions that are refitted from the labels seen so far.

    Items fall into cells: their group, split by the side they are on (positive weight =
    ranked better by the candidate, negative = by the baseline). ``prior[i]`` in [0, 1] is
    the prediction for item ``i`` before any label. After ``n`` labels with sum ``S`` in the
    item's cell, its prediction is

        (strength/2 * prior[i] + strength/2 * side_mean + S) / (strength + n),

    where ``side_mean`` is the running mean of all labels on the item's side. Cells with
    few labels therefore lean on the prior and on what the whole side looks like; cells
    with many labels speak for themselves. An infinite ``strength`` never updates.
    """

    prior: np.ndarray
    groups: np.ndarray
    strength: float = 4.0

    def __post_init__(self) -> None:
        if self.prior.shape != self.groups.shape or self.prior.ndim != 1:
            raise ValueError("prior and groups must be vectors of equal length")
        if np.any(self.prior < 0) or np.any(self.prior > 1):
            raise ValueError("prior predictions must be gains in [0, 1]")
        if np.any(self.groups < 0) or not self.strength > 0:
            raise ValueError("groups must be non-negative and strength positive")

    @staticmethod
    def without_judge(weights: np.ndarray, *, strength: float = 4.0) -> Predictor:
        """No judge: learn one mean gain for candidate-side and one for baseline-side items."""
        return Predictor(np.full(len(weights), 0.5), np.zeros(len(weights), dtype=int), strength)

    @staticmethod
    def zero(size: int) -> Predictor:
        """Predict zero gain forever: plain importance sampling of human labels."""
        return Predictor(np.zeros(size), np.zeros(size, dtype=int), np.inf)

    @staticmethod
    def fixed(pred: np.ndarray) -> Predictor:
        """Trust the given predictions as they are and never update them."""
        pred = np.asarray(pred, dtype=float)
        return Predictor(pred, np.zeros(len(pred), dtype=int), np.inf)

    @staticmethod
    def from_judge(
        grade_probs: np.ndarray,
        grade_gains: np.ndarray,
        weights: np.ndarray,
        *,
        strength: float = 4.0,
    ) -> Predictor:
        """Group items by what the judge said and learn what that is worth.

        ``grade_probs`` has one row per item and one column per grade; ``grade_gains`` maps
        each grade to its gain. A group is the judge's most likely grade together with
        whether the judge was less sure than usual for that grade. The prior of an item is
        the judge's own mean prediction over its group.
        """
        grade_probs = np.asarray(grade_probs, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if grade_probs.ndim != 2 or len(grade_probs) != len(weights):
            raise ValueError("grade_probs needs one row per item")
        grade = np.argmax(grade_probs, axis=1)
        ranked = np.sort(grade_probs, axis=1)
        margin = np.log(ranked[:, -1] + 1e-300) - np.log(ranked[:, -2] + 1e-300)
        unsure = np.zeros(len(weights), dtype=int)
        for value in np.unique(grade):
            member = grade == value
            unsure[member] = margin[member] < np.median(margin[member])
        _, groups = np.unique(grade * 2 + unsure, return_inverse=True)
        expected = grade_probs @ np.asarray(grade_gains, dtype=float)
        sums = np.bincount(groups, weights=expected)
        prior = (sums / np.bincount(groups))[groups]
        return Predictor(np.clip(prior, 0.0, 1.0), groups, strength)


@dataclass(frozen=True)
class Draws:
    """Per-label quantities for the first T items of a labeling order."""

    values: np.ndarray  # y_t, each with conditional mean delta
    center: np.ndarray  # part of y_t known before label t
    radius: np.ndarray  # y_t is certain to lie within center +- radius
    remaining: np.ndarray  # P(R) just before label t
    sure_low: np.ndarray  # lowest delta consistent with the first t labels
    sure_high: np.ndarray  # highest delta consistent with the first t labels


def _before(matrix: np.ndarray) -> np.ndarray:
    """Column-wise running totals over the rows strictly before each row."""
    return np.concatenate((np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)[:-1]))


def draw_values(
    weights: np.ndarray,
    probs: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    predictor: Predictor,
) -> Draws:
    """Compute ``y_1..y_T`` for the first T items of ``order`` with their ``labels``.

    ``order`` may be a prefix of a labeling order; ``labels[t]`` is the gain of item
    ``order[t]``.
    """
    order = np.asarray(order, dtype=int)
    labels = np.asarray(labels, dtype=float)
    if len(order) != len(labels) or len(order) == 0:
        raise ValueError("order and labels must be equally long and non-empty")
    if len(set(order.tolist())) != len(order):
        raise ValueError("an item cannot be labeled twice")
    if np.any(labels < 0) or np.any(labels > 1):
        raise ValueError("labels must be gains in [0, 1]")
    if len(predictor.prior) != len(weights):
        raise ValueError("predictor does not match the items")
    steps = np.arange(len(order))
    w, p, prior = weights[order], probs[order], predictor.prior[order]
    side_of_item = (weights > 0).astype(int)
    cell_of_item = predictor.groups * 2 + side_of_item
    cells = int(cell_of_item.max()) + 1
    cell, side = cell_of_item[order], side_of_item[order]
    in_cell = np.zeros((len(order), cells))
    in_cell[steps, cell] = 1.0
    # Unlabeled weight, and unlabeled weight times prior, per cell before each label.
    weight_left = np.bincount(cell_of_item, weights, cells) - _before(in_cell * w[:, None])
    prior_left = np.bincount(cell_of_item, weights * predictor.prior, cells) - _before(
        in_cell * (w * prior)[:, None]
    )
    if np.isinf(predictor.strength):
        predicted_rest = prior_left.sum(axis=1)
        prediction = prior
    else:
        half = predictor.strength / 2.0
        in_side = np.zeros((len(order), 2))
        in_side[steps, side] = 1.0
        side_mean = (0.5 + _before(in_side * labels[:, None])) / (1.0 + _before(in_side))
        side_mean = side_mean[:, np.arange(cells) % 2]  # one column per cell
        seen = _before(in_cell)
        label_sum = _before(in_cell * labels[:, None])
        total = predictor.strength + seen
        predicted_rest = (
            (half * prior_left + (half * side_mean + label_sum) * weight_left) / total
        ).sum(axis=1)
        prediction = (
            half * prior + half * side_mean[steps, cell] + label_sum[steps, cell]
        ) / total[steps, cell]
    remaining = np.maximum(1.0 - np.concatenate(([0.0], np.cumsum(p)[:-1])), p)
    exact = np.cumsum(w * labels)
    center = np.concatenate(([0.0], exact[:-1])) + predicted_rest
    values = center + w * (labels - prediction) * remaining / p
    # Largest importance ratio among the items still unlabeled before each label.
    ratio = np.abs(weights) / probs
    unseen = np.ones(len(weights), dtype=bool)
    unseen[order] = False
    outside = float(ratio[unseen].max()) if unseen.any() else 0.0
    largest_left = np.maximum(np.maximum.accumulate(ratio[order][::-1])[::-1], outside)
    positive_left = float(np.sum(np.clip(weights, 0, None))) - np.cumsum(np.clip(w, 0, None))
    negative_left = float(np.sum(np.clip(-weights, 0, None))) - np.cumsum(np.clip(-w, 0, None))
    return Draws(
        values,
        center,
        remaining * largest_left,
        remaining,
        exact - np.clip(negative_left, 0, None),
        exact + np.clip(positive_left, 0, None),
    )
