"""Download, verify and read BEIR archives. Raw data stays in the ignored data/ directory."""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "reports"


def load_config():
    return json.loads((ROOT / "experiments/config.json").read_text(encoding="utf-8"))


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dataset(name, config, *, download=False):
    """Return the verified archive path, downloading it once when allowed."""
    path = DATA / f"{name}.zip"
    if not path.exists():
        if not download:
            raise SystemExit(f"{name} is not cached; rerun with --download")
        import truststore

        DATA.mkdir(parents=True, exist_ok=True)
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        request = urllib.request.Request(
            config["dataset_url"].format(name=name), headers={"User-Agent": "judgegate-experiment"}
        )
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(request, context=context, timeout=120) as response:
            with temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        temporary.replace(path)
    if file_hash(path, "md5") != config["datasets"][name]["md5"]:
        raise SystemExit(f"{name}: archive checksum does not match the configured MD5")
    return path


def read_dataset(path, name, split):
    """Return corpus texts, query texts, integer qrels and a count of clamped grades.

    Negative grades are stored as 0; the count says how many rows that affected.
    """
    with zipfile.ZipFile(path) as archive:
        corpus_lines = archive.read(f"{name}/corpus.jsonl").decode("utf-8").splitlines()
        query_lines = archive.read(f"{name}/queries.jsonl").decode("utf-8").splitlines()
        qrel_lines = archive.read(f"{name}/qrels/{split}.tsv").decode("utf-8").splitlines()
    corpus = {}
    for line in corpus_lines:
        row = json.loads(line)
        if row["_id"] in corpus:
            raise ValueError("duplicate corpus ID")
        corpus[row["_id"]] = (row.get("title", "") + " " + row.get("text", "")).strip()
    all_queries = {}
    for line in query_lines:
        row = json.loads(line)
        all_queries[row["_id"]] = row["text"]
    if qrel_lines[0].split("\t") != ["query-id", "corpus-id", "score"]:
        raise ValueError("unexpected qrels header")
    qrels, clamped = {}, 0
    for line in qrel_lines[1:]:
        if not line.strip():
            continue
        qid, docid, grade = line.split("\t")
        if docid in qrels.setdefault(qid, {}):
            raise ValueError("duplicate judgment")
        if docid not in corpus:
            raise ValueError("judgment references an unknown document")
        value = int(grade)
        clamped += value < 0
        qrels[qid][docid] = max(value, 0)
    queries = {qid: all_queries[qid] for qid in qrels}
    return corpus, queries, qrels, clamped


def read_run(path):
    """Read a run written by judgegate.files.write_trec_run."""
    run = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            qid, _, docid, _, score, _ = line.split()
            run.setdefault(qid, {})[docid] = float(score)
    return run
