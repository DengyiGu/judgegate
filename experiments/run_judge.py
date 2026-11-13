"""Stage 2: judge the top-k documents of every system with the local LLM."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version

import numpy as np

from judgegate.files import write_trec_qrels
from judgegate.judge import LocalJudge, judge_pairs, prompt_hash
from judgegate.metrics import top_k

from .data import OUTPUT, fetch_dataset, load_config, read_dataset, read_run
from .systems import SYSTEMS


def pooled_items(runs, k):
    """Every (qid, docid) that some system places in its top k."""
    return sorted(
        {(qid, doc) for run in runs.values() for qid in run for doc in top_k(run[qid], k)}
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = load_config()
    spec, settings = config["models"]["judge"], config["judge"]
    identity = {
        "model": spec["id"],
        "revision": spec["revision"],
        "prompt_sha256": prompt_hash(),
        "max_words": settings["max_words"],
    }
    target = OUTPUT / "judgments"
    target.mkdir(parents=True, exist_ok=True)
    identity_path = target / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text("utf-8")) != identity:
        raise SystemExit("judge model or prompt changed; move reports/judgments aside first")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    work = {}
    for name, info in config["datasets"].items():
        if (target / f"{name}.json").exists():
            print(name, "already judged", flush=True)
            continue
        corpus, queries, _, _ = read_dataset(fetch_dataset(name, config), name, info["split"])
        runs = {s: read_run(OUTPUT / "runs" / name / f"{s}.trec") for s in SYSTEMS}
        work[name] = (corpus, queries, pooled_items(runs, config["evaluation"]["k"]))
    if not work:
        return

    import torch
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        spec["id"],
        revision=spec["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        local_files_only=not args.download,
    )
    judge = LocalJudge(model_path, device=args.device, max_words=settings["max_words"])
    for name, (corpus, queries, items) in work.items():
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        cache = judge_pairs(
            judge,
            items,
            queries,
            corpus,
            target / f"{name}.jsonl",
            batch_size=settings["batch_size"],
        )
        elapsed = time.perf_counter() - start
        grades = {}
        for qid, docid in items:
            grades.setdefault(qid, {})[docid] = int(np.argmax(cache[(qid, docid)]["probs"]))
        write_trec_qrels(grades, target / f"{name}-llm.qrels")
        record = {
            "dataset": name,
            "pairs": len(items),
            "wall_seconds_this_run": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None,
            "mean_digit_mass": float(np.mean([cache[i]["digit_mass"] for i in items])),
            "min_digit_mass": float(np.min([cache[i]["digit_mass"] for i in items])),
            "environment": {
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            },
            **identity,
        }
        (target / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(record, flush=True)


if __name__ == "__main__":
    main()
