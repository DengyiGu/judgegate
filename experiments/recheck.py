"""Stage 5: re-run the comparisons with the highest observed wrong-decision rates.

Fifty labeling orders cannot tell a 3% error rate from a 7% one. This stage takes every
(comparison, method) cell of the main study whose observed rate exceeded alpha and repeats
it with many more orders and a different seed. The cells are picked for looking bad, so
their main-study rates are biased upward; the fresh rates here are not.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from judgegate import delta_items, gate_path, grade_values, labeling_order
from judgegate.judge import grade_probabilities, load_cache

from .data import OUTPUT, ROOT, fetch_dataset, load_config, read_dataset, read_run
from .simulate import method_setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args(argv)
    config = load_config()
    settings = config["evaluation"]
    alpha, max_drop = settings["alpha"], settings["max_drop"]
    cells = []
    start = time.perf_counter()
    for index, (name, info) in enumerate(config["datasets"].items()):
        study = json.loads((OUTPUT / "simulation" / f"{name}.json").read_text("utf-8"))
        flagged = [
            (pair, method)
            for pair in study["pairs"]
            for method, cell in pair["methods"].items()
            if cell["wrong"] > alpha
        ]
        if not flagged:
            continue
        _, _, qrels, _ = read_dataset(fetch_dataset(name, config), name, info["split"])
        cache = load_cache(OUTPUT / "judgments" / f"{name}.jsonl")
        values_of = {m: grade_values(m, info["max_grade"]) for m in settings["metrics"]}
        for number, (pair, method) in enumerate(flagged):
            runs = {
                s: read_run(OUTPUT / "runs" / name / f"{s}.trec")
                for s in (pair["baseline"], pair["candidate"])
            }
            values = values_of[pair["metric"]]
            items = delta_items(
                qrels,
                runs[pair["baseline"]],
                runs[pair["candidate"]],
                metric=pair["metric"],
                k=settings["k"],
            )
            keys = items.keys()
            truth = values[np.array([qrels[q].get(d, 0) for q, d in keys])]
            setup, _ = method_setup(
                items.weights, grade_probabilities(cache, keys), values, settings
            )
            probs, predictor = setup[method]
            correct = "pass" if float(np.sum(items.weights * truth)) >= -max_drop else "fail"
            rng = np.random.default_rng([settings["seed"], 1, index, number])
            wrong = 0
            for _ in range(args.repetitions):
                order = labeling_order(probs, rng)
                decision = gate_path(
                    items.weights,
                    probs,
                    order,
                    truth[order],
                    predictor,
                    max_drop=max_drop,
                    alpha=alpha,
                ).decision
                wrong += decision != correct
            rate = wrong / args.repetitions
            cells.append(
                {
                    "dataset": name,
                    "metric": pair["metric"],
                    "baseline": pair["baseline"],
                    "candidate": pair["candidate"],
                    "method": method,
                    "true_delta": pair["true_delta"],
                    "main_study_wrong": pair["methods"][method]["wrong"],
                    "recheck_wrong": rate,
                    "recheck_standard_error": float(np.sqrt(rate * (1 - rate) / args.repetitions)),
                }
            )
            print(
                name, pair["metric"], pair["baseline"], pair["candidate"], method, rate, flush=True
            )
    record = {
        "alpha": alpha,
        "max_drop": max_drop,
        "main_study_repetitions": settings["repetitions"],
        "recheck_repetitions": args.repetitions,
        "selection": "main-study wrong-decision rate above alpha",
        "cells": cells,
        "highest_recheck_wrong": max((c["recheck_wrong"] for c in cells), default=None),
        "wall_seconds": time.perf_counter() - start,
    }
    target = ROOT / "docs/reports/error-recheck.json"
    target.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print("rechecked", len(cells), "cells; highest rate", record["highest_recheck_wrong"])


if __name__ == "__main__":
    main()
