"""TREC-format output for judgments and runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def _check(identifier: str) -> str:
    if not identifier or any(character.isspace() for character in identifier):
        raise ValueError(f"TREC identifiers must be non-empty and whitespace-free: {identifier!r}")
    return identifier


def write_trec_qrels(grades: Mapping[str, Mapping[str, int]], path: str | Path) -> None:
    """Write ``qid 0 docid grade`` rows sorted by query then document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(grades):
            for docid in sorted(grades[qid]):
                grade = grades[qid][docid]
                if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
                    raise ValueError("grades must be non-negative integers")
                handle.write(f"{_check(qid)} 0 {_check(docid)} {grade}\n")


def write_trec_run(run: Mapping[str, Mapping[str, float]], path: str | Path, tag: str) -> None:
    """Write ``qid Q0 docid rank score tag`` rows; ties order by ascending document ID."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(run):
            ranked = sorted(run[qid], key=lambda doc: (-run[qid][doc], doc))
            for rank, docid in enumerate(ranked, 1):
                handle.write(
                    f"{_check(qid)} Q0 {_check(docid)} {rank} {run[qid][docid]!r} {_check(tag)}\n"
                )
