"""Stage 3: judge diagnostics and the label-efficiency study.

Existing human judgments play the role of the annotator: a simulated label request looks up
the human grade, and a document without a human judgment counts as not relevant, the same
convention the full-judgment evaluation uses. Every ordered pair of systems is one
regression test with the first system as baseline and the second as candidate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from judgegate import (
    Predictor,
    delta_items,
    gate_path,
    grade_values,
    labeling_order,
    sampling_probabilities,
)
from judgegate.agreement import cohen_kappa, confusion_matrix, kendall_tau
from judgegate.judge import grade_probabilities, load_cache, predicted_gain
from judgegate.metrics import metric_value, top_k

from .data import OUTPUT, fetch_dataset, load_config, read_dataset, read_run
from .systems import SYSTEMS

# name: (sampling scheme, predictor)
METHODS = {
    "human-uniform-plain": ("uniform", "zero"),
    "human-weighted-plain": ("weighted", "zero"),
    "human-weighted": ("weighted", "none"),
    "judge-weighted-trusted": ("weighted", "trusted"),
    "judge-weighted": ("weighted", "judge"),
    "judge-active": ("active", "judge"),
}


def pooled_pairs(runs, k):
    return sorted({(q, d) for run in runs.values() for q in run for d in top_k(run[q], k)})


def diagnostics(qrels, runs, cache, pooled, settings, max_grade):
    """Label agreement, system orderings and plug-in gate decisions for one dataset."""
    k = settings["k"]
    judged = [(q, d) for q, d in pooled if d in qrels[q]]
    judged_set = set(judged)
    llm_grade = {key: int(np.argmax(cache[key]["probs"])) for key in pooled}
    matrix = confusion_matrix(
        [qrels[q][d] for q, d in judged], [llm_grade[key] for key in judged], max_grade + 1
    )
    binary = np.array(
        [[matrix[0, 0], matrix[0, 1:].sum()], [matrix[1:, 0].sum(), matrix[1:, 1:].sum()]]
    )
    result = {
        "pooled_pairs": len(pooled),
        "human_judged_pairs": len(judged),
        "confusion_human_rows_llm_columns": matrix.tolist(),
        "kappa": cohen_kappa(matrix),
        "kappa_linear_weighted": cohen_kappa(matrix, weighted=True),
        "binary_kappa": cohen_kappa(binary),
        "llm_grades_on_unjudged_pairs": np.bincount(
            [llm_grade[key] for key in pooled if key not in judged_set], minlength=max_grade + 1
        ).tolist(),
        "metrics": {},
    }
    for metric in settings["metrics"]:
        values = grade_values(metric, max_grade)
        human = {q: {d: float(values[g]) for d, g in docs.items()} for q, docs in qrels.items()}
        llm = {}
        for q, d in pooled:
            llm.setdefault(q, {})[d] = float(np.dot(cache[(q, d)]["probs"], values))
        scores = {
            name: {
                "human": metric_value(qrels, run, human, metric=metric, k=k),
                "llm": metric_value(qrels, run, llm, metric=metric, k=k),
                "judged_at_k": float(
                    np.mean([d in qrels[q] for q in qrels for d in top_k(run.get(q, {}), k)])
                ),
            }
            for name, run in runs.items()
        }
        agree = false_pass = false_fail = 0
        for base, cand in itertools.permutations(runs, 2):
            truth = scores[cand]["human"] - scores[base]["human"] >= -settings["max_drop"]
            guess = scores[cand]["llm"] - scores[base]["llm"] >= -settings["max_drop"]
            agree += truth == guess
            false_pass += guess and not truth
            false_fail += truth and not guess
        result["metrics"][metric] = {
            "systems": scores,
            "kendall_tau": kendall_tau(
                [scores[s]["human"] for s in runs], [scores[s]["llm"] for s in runs]
            ),
            "llm_only_gate": {
                "pairs": agree + false_pass + false_fail,
                "agree": agree,
                "false_pass": false_pass,
                "false_fail": false_fail,
            },
        }
    return result


def method_setup(weights, grade_probs, values, settings):
    """Propensities and predictor of every method, plus the judge's raw mean predictions."""
    mean, spread = predicted_gain(grade_probs, values)
    predictors = {
        "zero": Predictor.zero(len(weights)),
        "none": Predictor.without_judge(weights, strength=settings["strength"]),
        "trusted": Predictor.fixed(mean),
        "judge": Predictor.from_judge(grade_probs, values, weights, strength=settings["strength"]),
    }
    setup = {}
    for name, (scheme, kind) in METHODS.items():
        probs = sampling_probabilities(
            weights,
            spread if scheme == "active" else None,
            mix=settings["mix"],
            uniform=scheme == "uniform",
        )
        setup[name] = (probs, predictors[kind])
    return setup, mean


def study_pair(weights, truth, grade_probs, values, settings, rng):
    """Run every method on one regression test; return per-method summaries.

    Every repetition draws a full labeling order, so each one ends in pass or fail.
    """
    exact = float(np.sum(weights * truth))
    correct = "pass" if exact >= -settings["max_drop"] else "fail"
    reps = settings["repetitions"]
    setup, mean = method_setup(weights, grade_probs, values, settings)
    out = {}
    for name, (probs, predictor) in setup.items():
        labels, wrong = [], 0
        for _ in range(reps):
            order = labeling_order(probs, rng)
            gate = gate_path(
                weights,
                probs,
                order,
                truth[order],
                predictor,
                max_drop=settings["max_drop"],
                alpha=settings["alpha"],
            )
            labels.append(gate.labels)
            wrong += gate.decision != correct
        out[name] = {
            "wrong": wrong / reps,
            "mean_labels": float(np.mean(labels)),
            "mean_labels_mcse": float(np.std(labels, ddof=1) / np.sqrt(reps)),
            "median_labels": float(np.median(labels)),
            "p90_labels": float(np.quantile(labels, 0.9)),
        }
    return exact, correct, out, mean


def synthetic_judge(grades, levels, accuracy, rng):
    """Grade probabilities of a made-up judge that names the human grade with a given chance.

    Otherwise it names one of the other grades at random. It is always 90% confident.
    """
    wrong = (grades + rng.integers(1, levels, size=len(grades))) % levels
    said = np.where(rng.uniform(size=len(grades)) < accuracy, grades, wrong)
    probs = np.full((len(grades), levels), 0.1 / (levels - 1))
    probs[np.arange(len(grades)), said] = 0.9
    return probs


def study_synthetic(weights, truth, grades, values, settings, rng):
    """Mean labels of the recalibrated-judge gate for made-up judges of varying accuracy."""
    probs = sampling_probabilities(weights)
    out = {}
    for accuracy in settings["synthetic_judge_accuracy"]:
        labels = []
        for _ in range(settings["repetitions"]):
            judge = synthetic_judge(grades, len(values), accuracy, rng)
            predictor = Predictor.from_judge(judge, values, weights, strength=settings["strength"])
            order = labeling_order(probs, rng)
            labels.append(
                gate_path(
                    weights,
                    probs,
                    order,
                    truth[order],
                    predictor,
                    max_drop=settings["max_drop"],
                    alpha=settings["alpha"],
                ).labels
            )
        out[str(accuracy)] = float(np.mean(labels))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    config = load_config()
    settings = dict(config["evaluation"])
    if args.repetitions:
        settings["repetitions"] = args.repetitions
    target = OUTPUT / "simulation"
    target.mkdir(parents=True, exist_ok=True)

    for index, (name, info) in enumerate(config["datasets"].items()):
        start = time.perf_counter()
        _, _, qrels, _ = read_dataset(fetch_dataset(name, config), name, info["split"])
        runs = {s: read_run(OUTPUT / "runs" / name / f"{s}.trec") for s in SYSTEMS}
        cache = load_cache(OUTPUT / "judgments" / f"{name}.jsonl")
        pooled = pooled_pairs(runs, settings["k"])
        max_grade = info["max_grade"]
        record = {
            "dataset": name,
            "queries": len(qrels),
            "settings": settings,
            "diagnostics": diagnostics(qrels, runs, cache, pooled, settings, max_grade),
            "pairs": [],
        }
        for m, metric in enumerate(settings["metrics"]):
            values = grade_values(metric, max_grade)
            for p, (base, cand) in enumerate(itertools.permutations(SYSTEMS, 2)):
                items = delta_items(qrels, runs[base], runs[cand], metric=metric, k=settings["k"])
                keys = items.keys()
                grades = np.array([qrels[q].get(d, 0) for q, d in keys])
                truth = values[grades]
                rng = np.random.default_rng([settings["seed"], index, m, p])
                exact, correct, methods, raw = study_pair(
                    items.weights, truth, grade_probabilities(cache, keys), values, settings, rng
                )
                record["pairs"].append(
                    {
                        "metric": metric,
                        "baseline": base,
                        "candidate": cand,
                        "items": len(items),
                        "ranked_slots": 2 * settings["k"] * len(qrels),
                        "weight_mass": float(np.abs(items.weights).sum()),
                        "true_delta": exact,
                        "llm_only_delta": float(np.sum(items.weights * raw)),
                        "judge_rmse": float(np.sqrt(np.mean((truth - raw) ** 2))),
                        "std_of_human_gain": float(np.std(truth)),
                        "correct_decision": correct,
                        "methods": methods,
                        "synthetic_judge_mean_labels": study_synthetic(
                            items.weights, truth, grades, values, settings, rng
                        ),
                    }
                )
            print(name, metric, f"{time.perf_counter() - start:.0f}s", flush=True)
        record["wall_seconds"] = time.perf_counter() - start
        (target / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
