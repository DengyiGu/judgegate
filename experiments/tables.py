"""Stage 4: turn raw simulation output into a compact summary and Markdown tables."""

from __future__ import annotations

import json
from statistics import mean, median

from .data import OUTPUT, ROOT, load_config
from .simulate import METHODS

PUBLIC = ROOT / "docs" / "reports"
BINS = (("under 0.02", 0.0, 0.02), ("0.02 to 0.05", 0.02, 0.05), ("over 0.05", 0.05, 9.0))


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def summarize(record):
    settings = record["settings"]
    diag = record["diagnostics"]
    summary = {
        "queries": record["queries"],
        "judge": {
            key: diag[key]
            for key in (
                "pooled_pairs",
                "human_judged_pairs",
                "confusion_human_rows_llm_columns",
                "kappa",
                "kappa_linear_weighted",
                "binary_kappa",
                "llm_grades_on_unjudged_pairs",
            )
        },
        "metrics": {},
    }
    for metric in settings["metrics"]:
        pairs = [p for p in record["pairs"] if p["metric"] == metric]
        block = {
            "systems": diag["metrics"][metric]["systems"],
            "kendall_tau": diag["metrics"][metric]["kendall_tau"],
            "llm_only_gate": diag["metrics"][metric]["llm_only_gate"],
            "tests": len(pairs),
            "mean_items": mean(p["items"] for p in pairs),
            "mean_item_share_of_ranked_slots": mean(p["items"] / p["ranked_slots"] for p in pairs),
            "judge_rmse": mean(p["judge_rmse"] for p in pairs),
            "std_of_human_gain": mean(p["std_of_human_gain"] for p in pairs),
            "methods": {},
            "by_margin": {},
            "synthetic": {},
        }
        for name in METHODS:
            cells = [p["methods"][name] for p in pairs]
            block["methods"][name] = {
                "wrong": mean(c["wrong"] for c in cells),
                "worst_wrong": max(c["wrong"] for c in cells),
                "mean_labels": mean(c["mean_labels"] for c in cells),
                "label_share_mcse": sum(
                    (c["mean_labels_mcse"] / p["items"]) ** 2
                    for c, p in zip(cells, pairs, strict=True)
                )
                ** 0.5
                / len(pairs),
                "label_share": mean(
                    c["mean_labels"] / p["items"] for c, p in zip(cells, pairs, strict=True)
                ),
                "median_test_label_share": median(
                    c["median_labels"] / p["items"] for c, p in zip(cells, pairs, strict=True)
                ),
            }
        for label, low, high in BINS:
            chosen = [p for p in pairs if low <= abs(p["true_delta"] + settings["max_drop"]) < high]
            block["by_margin"][label] = {
                "tests": len(chosen),
                "methods": {
                    name: {
                        "mean_labels": mean(p["methods"][name]["mean_labels"] for p in chosen),
                        "label_share": mean(
                            p["methods"][name]["mean_labels"] / p["items"] for p in chosen
                        ),
                    }
                    for name in METHODS
                }
                if chosen
                else {},
            }
        for accuracy in map(str, settings["synthetic_judge_accuracy"]):
            block["synthetic"][accuracy] = {
                "all": mean(p["synthetic_judge_mean_labels"][accuracy] / p["items"] for p in pairs),
                **{
                    label: mean(
                        p["synthetic_judge_mean_labels"][accuracy] / p["items"]
                        for p in pairs
                        if low <= abs(p["true_delta"] + settings["max_drop"]) < high
                    )
                    for label, low, high in BINS
                    if block["by_margin"][label]["tests"]
                },
            }
        summary["metrics"][metric] = block
    return summary


def render(name, summary, settings):
    out = [f"## {name}\n"]
    judge = summary["judge"]
    out.append(
        f"{summary['queries']} queries; {judge['pooled_pairs']} pooled query-document pairs, "
        f"{judge['human_judged_pairs']} of them with a human judgment.\n"
    )
    out.append(
        f"Judge agreement on human-judged pairs: kappa {judge['kappa']:.3f}, linear-weighted "
        f"kappa {judge['kappa_linear_weighted']:.3f}, binary kappa {judge['binary_kappa']:.3f}. "
        f"LLM grades on pairs without a human judgment (0/1/2): "
        f"{judge['llm_grades_on_unjudged_pairs']}.\n"
    )
    matrix = judge["confusion_human_rows_llm_columns"]
    out.append(
        table(
            ["human \\ LLM"] + [str(g) for g in range(len(matrix))],
            [[g] + row for g, row in enumerate(matrix)],
        )
    )
    for metric, block in summary["metrics"].items():
        out.append(f"### {name}, {metric}@{settings['k']}\n")
        out.append(
            table(
                ["system", "human labels", "LLM labels only", "judged@k"],
                [
                    [s, f"{v['human']:.4f}", f"{v['llm']:.4f}", f"{v['judged_at_k']:.3f}"]
                    for s, v in block["systems"].items()
                ],
            )
        )
        gate = block["llm_only_gate"]
        out.append(
            f"Kendall tau between the two system orderings: {block['kendall_tau']:.3f}. "
            f"Gate decided from LLM labels alone (max drop {settings['max_drop']}): "
            f"{gate['agree']}/{gate['pairs']} tests agree with human labels, "
            f"{gate['false_pass']} false passes, {gate['false_fail']} false fails.\n"
        )
        out.append(
            f"Sequential gate over {block['tests']} tests, {settings['repetitions']} labeling "
            f"orders each, alpha {settings['alpha']}. A test has {block['mean_items']:.0f} items "
            f"that need labels on average, {block['mean_item_share_of_ranked_slots']:.1%} of the "
            f"ranked slots in the two top-{settings['k']} lists. Label share is labels used "
            f"divided by those items. The judge's raw predictions have RMSE "
            f"{block['judge_rmse']:.3f} against the human gains, whose standard deviation is "
            f"{block['std_of_human_gain']:.3f}.\n"
        )
        out.append(
            table(
                ["method", "mean labels", "mean label share", "median test", "wrong", "worst test"],
                [
                    [
                        m,
                        f"{v['mean_labels']:.0f}",
                        f"{v['label_share']:.1%} ± {100 * v['label_share_mcse']:.2f} pp MCSE",
                        f"{v['median_test_label_share']:.1%}",
                        f"{v['wrong']:.2%}",
                        f"{v['worst_wrong']:.1%}",
                    ]
                    for m, v in block["methods"].items()
                ],
            )
        )
        out.append(
            "Mean labels (and label share) by distance of the true difference from the "
            "threshold; the number of tests is in parentheses:\n"
        )
        heads = [f"{label} ({cell['tests']})" for label, cell in block["by_margin"].items()]
        out.append(
            table(
                ["method"] + heads,
                [
                    [m]
                    + [
                        f"{cell['methods'][m]['mean_labels']:.0f} "
                        f"({cell['methods'][m]['label_share']:.0%})"
                        if cell["methods"]
                        else "-"
                        for cell in block["by_margin"].values()
                    ]
                    for m in METHODS
                ],
            )
        )
        out.append(
            "Label share with a made-up judge that names the human grade with the given "
            "probability, recalibrated the same way (not a real model):\n"
        )
        columns = list(next(iter(block["synthetic"].values())))
        out.append(
            table(
                ["judge accuracy"] + columns,
                [
                    [accuracy] + [f"{cell[column]:.1%}" for column in columns]
                    for accuracy, cell in block["synthetic"].items()
                ],
            )
        )
    return "\n".join(out)


def main(argv=None):
    config = load_config()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    summary, sections, settings = {}, [], {}
    for name in config["datasets"]:
        record = json.loads((OUTPUT / "simulation" / f"{name}.json").read_text("utf-8"))
        settings = record["settings"]
        summary[name] = summarize(record)
        summary[name]["judge_run"] = json.loads(
            (OUTPUT / "judgments" / f"{name}.json").read_text("utf-8")
        )
        summary[name]["data"] = json.loads(
            (OUTPUT / "runs" / name / "dataset.json").read_text("utf-8")
        )
        summary[name]["tests"] = [pair for pair in record["pairs"]]
        sections.append(render(name, summary[name], settings))
    payload = {"config": config, "settings_used": settings, "datasets": summary}
    (PUBLIC / "summary.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    header = (
        "# Result tables\n\nGenerated by `python -m experiments.tables` from the raw simulation "
        "output. Sampling is `uniform`, `weighted` (proportional to an item's weight) or "
        "`active` (weight times judge uncertainty). `plain` predicts zero for unlabeled "
        "items, `human-weighted` learns a mean gain per side, `trusted` uses the judge's raw "
        "predictions unchanged, and `judge-weighted` / `judge-active` recalibrate the judge "
        "from the labels as they arrive.\n\n"
    )
    (PUBLIC / "tables.md").write_text(header + "\n".join(sections), encoding="utf-8", newline="\n")
    print("wrote", PUBLIC / "tables.md")


if __name__ == "__main__":
    main()
