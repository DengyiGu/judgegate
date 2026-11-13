"""Stage 1: verify datasets and write each system's run in TREC format."""

from __future__ import annotations

import argparse
import json
import time

from judgegate.files import write_trec_run

from .data import OUTPUT, fetch_dataset, file_hash, load_config, read_dataset
from .systems import build_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = load_config()
    for name, info in config["datasets"].items():
        target = OUTPUT / "runs" / name
        if (target / "dataset.json").exists():
            print(name, "runs already built", flush=True)
            continue
        archive = fetch_dataset(name, config, download=args.download)
        corpus, queries, qrels, clamped = read_dataset(archive, name, info["split"])
        start = time.perf_counter()
        runs = build_runs(name, corpus, queries, config, args.device)
        for system, run in runs.items():
            write_trec_run(run, target / f"{system}.trec", system)
        record = {
            "dataset": name,
            "archive_sha256": file_hash(archive),
            "documents": len(corpus),
            "queries": len(queries),
            "judgments": sum(map(len, qrels.values())),
            "negative_grades_clamped": int(clamped),
            "build_seconds": time.perf_counter() - start,
        }
        (target / "dataset.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(record, flush=True)


if __name__ == "__main__":
    main()
