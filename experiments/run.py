"""Run every stage: retrieval runs, LLM judgments, the simulation study, tables and recheck.

Finished stages are skipped, so the command can be repeated after an interruption.
"""

from __future__ import annotations

import argparse

from . import build_runs, evidence, recheck, run_judge, simulate, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="allow dataset/model downloads")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    shared = ["--device", args.device] + (["--download"] if args.download else [])
    build_runs.main(shared)
    run_judge.main(shared)
    simulate.main([])
    tables.main([])
    recheck.main([])
    evidence.main()


if __name__ == "__main__":
    main()
