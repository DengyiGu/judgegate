"""Compare a full-qrels reference gate with a simulated sequential labeling gate.

Run without arguments for synthetic data, or supply exported qrels and two ranking files.
The optional retrieval-regression package must be installed from its repository.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from retrieval_regression import Threshold, compare, load_qrels, load_run

from judgegate import Predictor, delta_items, run_gate
from judgegate.judge import grade_probabilities, load_cache


def synthetic_inputs():
    """Small deterministic fixture with an intentionally worse candidate; no real model."""
    qrels, baseline, candidate = {}, {}, {}
    for number in range(120):
        qid = f"q{number:03d}"
        # Four shared documents cancel in precision. The candidate loses two positives.
        qrels[qid] = {f"d{i}": int(i < 8) for i in range(16)}
        baseline[qid] = {f"d{i}": float(10 - i) for i in range(10)}
        ids = list(range(4)) + list(range(6, 12))
        candidate[qid] = {f"d{i}": float(10 - rank) for rank, i in enumerate(ids)}
    return qrels, baseline, candidate


def compare_with_labels(
    qrels,
    baseline,
    candidate,
    *,
    k=10,
    max_drop=0.02,
    alpha=0.05,
    max_labels=None,
    seed=0,
    judge_cache=None,
):
    """Reference lookup and simulated human requests over precisely the same inputs.

    Positive grades become binary gains; unjudged documents count as zero in both tools.
    The complete qrels are available only because this is a replay experiment.
    """
    metric = f"precision@{k}"
    reference = compare(
        qrels,
        baseline,
        candidate,
        k=[k],
        thresholds=[Threshold(metric, max_drop=max_drop)],
        gate_mode="point",
        bootstrap_samples=0,
    )
    items = delta_items(qrels, baseline, candidate, metric="precision", k=k)
    gains = {q: {d: float(g > 0) for d, g in docs.items()} for q, docs in qrels.items()}
    labels = items.gather(gains)
    # This parity check is essential: do not compare decisions about different estimands.
    if not math.isclose(items.delta(labels), reference.delta[metric], abs_tol=1e-12):
        raise AssertionError("the two tools disagree on the full-label metric difference")
    predictor = None
    if judge_cache is not None and len(items):
        probs = grade_probabilities(judge_cache, items.keys())
        predictor = Predictor.from_judge(probs, np.array([0.0, 1.0, 1.0]), items.weights)
    result = run_gate(
        items.weights,
        lambda i: float(labels[i]),
        predictor=predictor,
        max_drop=max_drop,
        alpha=alpha,
        max_labels=max_labels,
        seed=seed,
    )
    expected = "pass" if reference.passed else "fail"
    return {
        "metric": metric,
        "query_count": reference.baseline.query_count,
        "max_drop": max_drop,
        "reference": {
            "baseline": reference.baseline.summary[metric],
            "candidate": reference.candidate.summary[metric],
            "delta": reference.delta[metric],
            "decision": expected,
        },
        "sequential": {
            "decision": result.decision,
            "alpha": alpha,
            "labels": result.labels,
            "nonzero_weight_items": len(items),
            "label_share": result.labels / len(items) if len(items) else 0.0,
            "predictor": "recalibrated-judge" if judge_cache is not None else "learned-human-mean",
            "agrees_with_reference": result.decision == expected
            if result.decision != "undecided"
            else None,
        },
        "label_source": "qrels replay; unjudged documents score zero",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels")
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--qrels-format", choices=["json", "beir", "trec"], default="json")
    parser.add_argument("--run-format", choices=["json", "trec"], default="json")
    parser.add_argument(
        "--judge-cache", help="JudgeGate JSONL cache with grade 0/1/2 probabilities"
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument("--max-labels", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    files = [args.qrels, args.baseline, args.candidate]
    if any(files) and not all(files):
        parser.error("supply --qrels, --baseline and --candidate together")
    inputs = (
        (
            load_qrels(args.qrels, format=args.qrels_format),
            load_run(args.baseline, format=args.run_format),
            load_run(args.candidate, format=args.run_format),
        )
        if all(files)
        else synthetic_inputs()
    )
    record = compare_with_labels(
        *inputs,
        k=args.k,
        max_drop=args.max_drop,
        max_labels=args.max_labels,
        seed=args.seed,
        judge_cache=load_cache(args.judge_cache) if args.judge_cache else None,
    )
    record["data_kind"] = "file inputs" if all(files) else "synthetic"
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
