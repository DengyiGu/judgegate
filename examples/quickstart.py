"""Offline walk-through on synthetic data: no model, no downloads, numpy only.

A baseline and a candidate ranker each return ten documents for 200 queries. A simulated
annotator knows the true relevance, and a simulated judge is right most of the time and
reports how sure it is. The gate asks the annotator for labels one batch at a time and stops
as soon as it can tell whether the candidate lost more than the allowed budget.
"""

from __future__ import annotations

import numpy as np

from judgegate import Predictor, delta_items, run_gate

rng = np.random.default_rng(0)
QUERIES, DOCS, K = 200, 60, 10

relevance, baseline, candidate = {}, {}, {}
for q in range(QUERIES):
    qid = f"q{q}"
    relevant = rng.uniform(size=DOCS) < 0.15
    relevance[qid] = {f"d{d}": float(relevant[d]) for d in range(DOCS)}
    # Both rankers see a noisy relevance signal; the candidate's is slightly noisier.
    signal = relevant.astype(float)
    baseline[qid] = {f"d{d}": signal[d] + rng.normal(scale=0.60) for d in range(DOCS)}
    candidate[qid] = {f"d{d}": signal[d] + rng.normal(scale=0.66) for d in range(DOCS)}

items = delta_items(relevance, baseline, candidate, metric="precision", k=K)
truth = items.gather(relevance)
print(f"{2 * K * QUERIES} ranked slots, {len(items)} items whose label can change the result")
print(f"true precision@{K} difference (unknown to the gate): {items.delta(truth):+.4f}")

# A judge that is right 85% of the time and less confident when it is wrong.
correct = rng.uniform(size=len(items)) < 0.85
confidence = np.where(
    correct, rng.uniform(0.80, 0.99, len(items)), rng.uniform(0.55, 0.90, len(items))
)
says_relevant = np.where(correct, truth, 1.0 - truth)
predicted = np.where(says_relevant == 1.0, confidence, 1.0 - confidence)
predictor = Predictor.from_judge(
    np.column_stack((1.0 - predicted, predicted)), np.array([0.0, 1.0]), items.weights
)
print(f"judge-only estimate (no guarantee):               {items.delta(predicted):+.4f}")

asked = []


def ask_annotator(index: int) -> float:
    """Stand-in for a labeling tool: return the human gain of one item."""
    asked.append(index)
    return float(truth[index])


for max_drop in (0.05, 0.01):
    asked.clear()
    decision = run_gate(
        items.weights,
        ask_annotator,
        predictor=predictor,
        max_drop=max_drop,
        alpha=0.05,
    )
    print(
        f"max_drop={max_drop}: {decision.decision.upper()} after {decision.labels} labels "
        f"({decision.labels / len(items):.0%} of the items); "
        f"one-sided 95% bounds [{decision.low:+.4f}, {decision.high:+.4f}]"
    )
