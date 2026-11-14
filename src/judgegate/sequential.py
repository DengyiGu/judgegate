"""Sequential pass/fail gate and one-sided bounds by testing with bets.

For a threshold ``m`` two wealth processes are tracked over the labels:

    pass side:  K+_t = prod_s (1 + b_s * (y_s - m)),   b_s >= 0
    fail side:  K-_t = prod_s (1 - b_s * (y_s - m)),   b_s >= 0

Every ``y_s`` has conditional mean ``delta``. If ``delta < m`` the pass-side wealth is a
non-negative supermartingale, so by Ville's inequality it reaches ``1 / alpha`` with
probability at most ``alpha``; likewise the fail side when ``delta >= m``. The bets only
have to be fixed before the label they apply to. They follow the approximate Kelly rule of
Waudby-Smith and Ramdas (2023) and use two facts known in advance of each label: the
variance of ``y_s`` scales with the square of the unlabeled propensity mass, and ``y_s`` is
certain to fall within a known, shrinking range.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .estimators import Draws, Predictor, draw_values, labeling_order, sampling_probabilities

CAP = 0.75  # largest share of the current wealth a single bet may lose


def _bets(draws: Draws, thresholds: np.ndarray | float, cap: float):
    """Bets and payoffs per label and threshold for the pass and fail sides."""
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must be between 0 and 1")
    m = np.atleast_1d(np.asarray(thresholds, dtype=float))[None, :]
    y, center, radius, remaining = draws.values, draws.center, draws.radius, draws.remaining
    count = np.arange(1, len(y) + 1, dtype=float)
    # Running estimate of delta: precision-weighted, started at the first known center.
    precision = 1.0 / remaining**2
    mean = (center[0] + np.cumsum(precision * y)) / (1.0 + np.cumsum(precision))
    mean_before = np.concatenate(([center[0]], mean[:-1]))
    # Variance of y at full remaining mass, started from a wide prior.
    prior = (radius[0] / 2.0) ** 2
    unit = (prior + np.cumsum(((y - mean_before) / remaining) ** 2)) / (count + 1.0)
    variance = remaining**2 * np.concatenate(([prior], unit[:-1]))
    gap = mean_before[:, None] - m
    kelly = gap / (variance[:, None] + gap**2 + 1e-300)
    deviation = y[:, None] - m
    below = m - (center - radius)[:, None]  # how far under m the next value could fall
    above = (center + radius)[:, None] - m
    with np.errstate(divide="ignore"):
        bet_pass = np.clip(kelly, 0.0, np.where(below > 0, cap / below, np.inf))
        bet_fail = np.clip(-kelly, 0.0, np.where(above > 0, cap / above, np.inf))
    # Where the whole range is on one side, rounding must not produce a losing bet.
    gain_pass = np.where(below > 0, deviation, np.maximum(deviation, 0.0))
    gain_fail = np.where(above > 0, -deviation, np.maximum(-deviation, 0.0))
    return bet_pass, gain_pass, bet_fail, gain_fail


def wealth_paths(
    draws: Draws, thresholds: np.ndarray | float, *, cap: float = CAP
) -> tuple[np.ndarray, np.ndarray]:
    """Log-wealth after each label for each threshold, shape (labels, thresholds).

    Returns the pass-side and fail-side processes. ``cap`` is the largest share of the
    current wealth a single bet may lose.
    """
    bet_pass, gain_pass, bet_fail, gain_fail = _bets(draws, thresholds, cap)
    return (
        np.cumsum(np.log1p(bet_pass * gain_pass), axis=0),
        np.cumsum(np.log1p(bet_fail * gain_fail), axis=0),
    )


@dataclass(frozen=True)
class GateDecision:
    """Outcome of a sequential gate.

    ``decision`` is "pass", "fail" or "undecided". From ``gate_path``, ``low`` and ``high``
    are the bounds on the difference that are certain given the labels seen. ``run_gate``
    replaces them with the tighter statistical bounds of ``bounds``.
    """

    decision: str
    labels: int
    low: float
    high: float


def gate_path(
    weights: np.ndarray,
    probs: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    predictor: Predictor,
    *,
    max_drop: float,
    alpha: float = 0.05,
) -> GateDecision:
    """Follow a labeling order and stop at the first conclusive label.

    The gate passes when the candidate is shown to lose at most ``max_drop`` and fails when
    the loss is shown to exceed it. A wrong pass has probability at most ``alpha``, and so
    has a wrong fail.
    """
    if max_drop < 0:
        raise ValueError("max_drop must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if len(weights) == 0:
        return GateDecision("pass", 0, 0.0, 0.0)
    draws = draw_values(weights, probs, order, labels, predictor)
    threshold = -max_drop
    log_pass, log_fail = wealth_paths(draws, threshold)
    need = math.log(1.0 / alpha)
    passed = np.flatnonzero((log_pass[:, 0] >= need) | (draws.sure_low >= threshold))
    failed = np.flatnonzero((log_fail[:, 0] >= need) | (draws.sure_high < threshold))
    first_pass = int(passed[0]) if len(passed) else len(order)
    first_fail = int(failed[0]) if len(failed) else len(order)
    stop = min(first_pass, first_fail)
    if stop == len(order):
        return GateDecision(
            "undecided", len(order), float(draws.sure_low[-1]), float(draws.sure_high[-1])
        )
    return GateDecision(
        "pass" if first_pass <= first_fail else "fail",
        stop + 1,
        float(draws.sure_low[stop]),
        float(draws.sure_high[stop]),
    )


def _grid_bound(crossed: np.ndarray, candidates: np.ndarray, start: float) -> float:
    """Farthest candidate reached from ``start`` through an unbroken run of crossings."""
    run = len(crossed) if crossed.all() else int(np.argmin(crossed))
    return float(candidates[run - 1]) if run > 0 else start


def _anchored_bound(
    values: np.ndarray, bets: np.ndarray, sign: float, start: float, end: float, need: float
) -> float:
    """Farthest point from ``end`` toward ``start`` established with one fixed bet sequence.

    The bets are those the gate placed at its own threshold ``end``. With them the wealth
    against a candidate is monotone in the candidate, so the established region is an
    interval and bisection finds its edge. Returns ``start`` when nothing is established.
    """

    def established(candidate: float) -> bool:
        payoff = np.maximum(sign * bets * (values - candidate), -CAP)
        return float(np.cumsum(np.log1p(payoff)).max()) >= need

    if not established(start):
        return start
    near, far = start, end  # established at near, not established at far
    for _ in range(60):
        middle = 0.5 * (near + far)
        if established(middle):
            near = middle
        else:
            far = middle
    return near


def bounds(
    weights: np.ndarray,
    probs: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    predictor: Predictor,
    *,
    alpha: float = 0.05,
    grid: int = 201,
    threshold: float | None = None,
) -> tuple[float, float]:
    """One-sided bounds on the difference after the given labels.

    ``delta >= low`` holds with confidence ``1 - alpha`` and so does ``delta <= high``;
    both hold together with confidence ``1 - 2 * alpha``. The bounds are valid at any
    stopping point.

    Without ``threshold`` each of ``grid`` candidate values is tested with its own bets and
    the bound is the edge of the unbroken run of established candidates. With the gate's
    ``threshold`` (``-max_drop``) the bounds agree with the gate: candidates on the far side
    of the threshold are tested with the bets the gate itself placed, which makes the
    established region an interval that includes the threshold exactly when the gate has
    crossed it. Only then are candidates beyond the threshold examined on the grid.
    """
    draws = draw_values(weights, probs, order, labels, predictor)
    floor = -float(np.sum(np.clip(-weights, 0, None)))
    ceiling = float(np.sum(np.clip(weights, 0, None)))
    need = math.log(1.0 / alpha)
    if threshold is None or not floor < threshold < ceiling:
        candidates = np.linspace(floor, ceiling, grid)
        log_pass, log_fail = wealth_paths(draws, candidates)
        low = _grid_bound(log_pass.max(axis=0) >= need, candidates, floor)
        high = _grid_bound((log_fail.max(axis=0) >= need)[::-1], candidates[::-1], ceiling)
    else:
        bet_pass, _, bet_fail, _ = _bets(draws, threshold, CAP)
        log_pass, log_fail = wealth_paths(draws, threshold)
        if log_pass.max() >= need:
            # The gate showed delta >= threshold; look for more above it.
            above = np.linspace(threshold, ceiling, grid)[1:]
            crossed = wealth_paths(draws, above)[0].max(axis=0) >= need
            low = _grid_bound(crossed, above, threshold)
        else:
            low = _anchored_bound(draws.values, bet_pass[:, 0], 1.0, floor, threshold, need)
        if log_fail.max() >= need:
            below = np.linspace(threshold, floor, grid)[1:]
            crossed = wealth_paths(draws, below)[1].max(axis=0) >= need
            high = _grid_bound(crossed, below, threshold)
        else:
            high = _anchored_bound(draws.values, bet_fail[:, 0], -1.0, ceiling, threshold, need)
    return max(low, float(draws.sure_low[-1])), min(high, float(draws.sure_high[-1]))


def run_gate(
    weights: np.ndarray,
    label: Callable[[int], float],
    *,
    max_drop: float,
    predictor: Predictor | None = None,
    probs: np.ndarray | None = None,
    alpha: float = 0.05,
    max_labels: int | None = None,
    batch: int = 25,
    seed: int = 0,
) -> GateDecision:
    """Interactive gate: ``label(i)`` returns the human gain of item ``i``.

    Items are requested in batches until the gate is conclusive or the label budget is
    spent. With no budget the gate always ends in pass or fail, because labeling every item
    gives the exact difference. The result counts every label requested and carries the
    statistical bounds from ``bounds``, which agree with the decision: a pass has
    ``low >= -max_drop`` and a fail has ``high <= -max_drop``. Without a ``predictor`` no
    judge is used; without ``probs`` items are drawn proportionally to their absolute weight.
    """
    if batch <= 0:
        raise ValueError("batch must be positive")
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return GateDecision("pass", 0, 0.0, 0.0)
    if probs is None:
        probs = sampling_probabilities(weights)
    if predictor is None:
        predictor = Predictor.without_judge(weights)
    order = labeling_order(probs, np.random.default_rng(seed))
    budget = len(weights) if max_labels is None else min(max_labels, len(weights))
    gains: list[float] = []
    decision = "undecided"
    while len(gains) < budget and decision == "undecided":
        for index in order[len(gains) : min(len(gains) + batch, budget)].tolist():
            gains.append(float(label(index)))
        decision = gate_path(
            weights,
            probs,
            order[: len(gains)],
            np.array(gains),
            predictor,
            max_drop=max_drop,
            alpha=alpha,
        ).decision
    if not gains:
        return GateDecision("undecided", 0, -math.inf, math.inf)
    low, high = bounds(
        weights,
        probs,
        order[: len(gains)],
        np.array(gains),
        predictor,
        alpha=alpha,
        threshold=-max_drop,
    )
    return GateDecision(decision, len(gains), low, high)
