"""Label-linear ranking metrics and the item weights of a paired difference.

A metric is *label-linear* when its value is a fixed weighted sum of per-document gains:
``M(run) = mean_q sum_r c(r) * gain(q, doc at rank r)``. For such metrics the difference
between two runs is itself a weighted sum over (query, document) items, and every document
that both runs place at the same rank drops out. Only the remaining items need labels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

Run = Mapping[str, Mapping[str, float]]
Gains = Mapping[str, Mapping[str, float]]

METRICS = ("sdcg", "precision", "rbp")


def discounts(metric: str, k: int, *, rbp_p: float = 0.8) -> np.ndarray:
    """Per-rank coefficients for ranks 1..k. Every metric lies in [0, 1] for gains in [0, 1].

    ``sdcg`` is DCG@k divided by the DCG of k maximally relevant documents, so it needs no
    per-query ideal ranking. ``precision`` is P@k. ``rbp`` is rank-biased precision truncated
    at k.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    ranks = np.arange(1, k + 1, dtype=float)
    if metric == "sdcg":
        raw = 1.0 / np.log2(ranks + 1.0)
        return raw / raw.sum()
    if metric == "precision":
        return np.full(k, 1.0 / k)
    if metric == "rbp":
        if not 0.0 < rbp_p < 1.0:
            raise ValueError("rbp_p must be between 0 and 1")
        return (1.0 - rbp_p) * rbp_p ** (ranks - 1.0)
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")


def top_k(scores: Mapping[str, float], k: int) -> list[str]:
    """Rank by descending score, then ascending document ID."""
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:k]


@dataclass(frozen=True)
class DeltaItems:
    """Items whose labels determine ``metric(candidate) - metric(baseline)``."""

    qids: tuple[str, ...]
    docids: tuple[str, ...]
    weights: np.ndarray
    query_count: int

    def __len__(self) -> int:
        return len(self.qids)

    def keys(self) -> list[tuple[str, str]]:
        return list(zip(self.qids, self.docids, strict=True))

    def gather(self, gains: Gains, *, default: float = 0.0) -> np.ndarray:
        """Look up one gain per item; unlabeled documents take ``default``."""
        return np.array(
            [gains.get(q, {}).get(d, default) for q, d in zip(self.qids, self.docids, strict=True)],
            dtype=float,
        )

    def delta(self, gains: Gains | np.ndarray, *, default: float = 0.0) -> float:
        values = gains if isinstance(gains, np.ndarray) else self.gather(gains, default=default)
        return float(math.fsum(self.weights * values))


def delta_items(
    qids: Iterable[str],
    baseline: Run,
    candidate: Run,
    *,
    metric: str = "sdcg",
    k: int = 10,
    rbp_p: float = 0.8,
) -> DeltaItems:
    """Weights ``w`` such that ``sum(w * gain) = metric(candidate) - metric(baseline)``.

    ``qids`` fixes the query universe: a query missing from a run contributes an empty
    ranking, and the mean is always taken over every listed query.
    """
    universe = sorted(set(qids))
    if not universe:
        raise ValueError("at least one query is required")
    coefficient = discounts(metric, k, rbp_p=rbp_p)
    out_q: list[str] = []
    out_d: list[str] = []
    out_w: list[float] = []
    for qid in universe:
        weight: dict[str, float] = {}
        for sign, run in ((1.0, candidate), (-1.0, baseline)):
            for rank, doc in enumerate(top_k(run.get(qid, {}), k)):
                weight[doc] = weight.get(doc, 0.0) + sign * float(coefficient[rank])
        for doc in sorted(weight):
            if weight[doc] != 0.0:
                out_q.append(qid)
                out_d.append(doc)
                out_w.append(weight[doc] / len(universe))
    return DeltaItems(tuple(out_q), tuple(out_d), np.array(out_w, dtype=float), len(universe))


def metric_value(
    qids: Iterable[str],
    run: Run,
    gains: Gains,
    *,
    metric: str = "sdcg",
    k: int = 10,
    rbp_p: float = 0.8,
) -> float:
    """Mean metric over the full query universe; unlabeled documents have zero gain."""
    universe = sorted(set(qids))
    if not universe:
        raise ValueError("at least one query is required")
    coefficient = discounts(metric, k, rbp_p=rbp_p)
    total = math.fsum(
        float(coefficient[rank]) * gains.get(qid, {}).get(doc, 0.0)
        for qid in universe
        for rank, doc in enumerate(top_k(run.get(qid, {}), k))
    )
    return total / len(universe)


def grade_values(metric: str, max_grade: int) -> np.ndarray:
    """Gain of each integer grade 0..max_grade: graded for sdcg/rbp, binary for precision."""
    if isinstance(max_grade, bool) or not isinstance(max_grade, int) or max_grade <= 0:
        raise ValueError("max_grade must be a positive integer")
    grades = np.arange(max_grade + 1, dtype=float)
    if metric == "precision":
        return (grades > 0).astype(float)
    if metric in ("sdcg", "rbp"):
        return grades / max_grade
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
