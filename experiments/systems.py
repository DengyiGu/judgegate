"""Build the retrieval runs that are compared pairwise.

Seven systems share each corpus: a lexical baseline, two small dense encoders, two
deliberately degraded variants of one encoder (binary-quantized vectors and passages cut to
32 tokens), a reciprocal-rank fusion and a cross-encoder reranker. The degraded variants
give realistic small and large regressions next to the natural differences between systems.
"""

from __future__ import annotations

import numpy as np

from .data import DATA

SYSTEMS = ("bm25", "bge", "bge-binary", "bge-trunc", "e5", "rrf", "bm25-ce")


def top_run(scores, docids, depth):
    """Top ``depth`` documents; ``docids`` must be sorted so ties break by ascending ID."""
    depth = min(depth, len(docids))
    order = np.argsort(-scores, kind="stable")[:depth]
    return {docids[i]: float(scores[i]) for i in order}


def bm25_runs(corpus, queries, docids, settings):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", dtype=np.float32)
    counts = vectorizer.fit_transform(corpus[d] for d in docids)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    norm = settings["bm25_k1"] * (
        1 - settings["bm25_b"] + settings["bm25_b"] * lengths / lengths.mean()
    )
    frequency = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((len(docids) - frequency + 0.5) / (frequency + 0.5))
    columns = counts.tocsc()
    analyze = vectorizer.build_analyzer()
    runs = {}
    for qid in sorted(queries):
        scores = np.zeros(len(docids))
        for term in sorted(set(analyze(queries[qid]))):
            j = vectorizer.vocabulary_.get(term)
            if j is None:
                continue
            column = columns[:, j]
            tf = column.data
            scores[column.indices] += (
                idf[j] * tf * (settings["bm25_k1"] + 1) / (tf + norm[column.indices])
            )
        runs[qid] = top_run(scores, docids, settings["depth"])
    return runs


def embed(name, spec, texts, kind, max_tokens, dataset, device):
    """Normalized embeddings, cached on disk per dataset, model, text kind and length."""
    path = DATA / "embeddings" / f"{dataset}-{name}-{kind}-{max_tokens}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device)
    model.max_seq_length = max_tokens
    prefix = spec["query_prefix"] if kind == "query" else spec["passage_prefix"]
    vectors = model.encode(
        [prefix + text for text in texts],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def dense_runs(query_vectors, passage_vectors, qids, docids, depth):
    scores = query_vectors @ passage_vectors.T
    return {qid: top_run(scores[i], docids, depth) for i, qid in enumerate(qids)}


def fuse(first, second, rrf_k, depth):
    runs = {}
    for qid in first:
        fused = {}
        for run in (first[qid], second[qid]):
            ranked = sorted(run, key=lambda doc: (-run[doc], doc))
            for rank, doc in enumerate(ranked, 1):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank)
        keep = sorted(fused, key=lambda doc: (-fused[doc], doc))[:depth]
        runs[qid] = {doc: fused[doc] for doc in keep}
    return runs


def rerank(candidates, corpus, queries, spec, settings, device):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        spec["id"], revision=spec["revision"], max_length=settings["max_tokens"], device=device
    )
    runs = {}
    for qid in sorted(candidates):
        run = candidates[qid]
        pool = sorted(run, key=lambda doc: (-run[doc], doc))[: settings["rerank_depth"]]
        scores = model.predict(
            [(queries[qid], corpus[doc]) for doc in pool], batch_size=128, show_progress_bar=False
        )
        runs[qid] = {doc: float(score) for doc, score in zip(pool, scores, strict=True)}
    return runs


def build_runs(dataset, corpus, queries, config, device="cuda"):
    settings, models = config["retrieval"], config["models"]
    docids, qids = sorted(corpus), sorted(queries)
    passages, questions = [corpus[d] for d in docids], [queries[q] for q in qids]
    depth, full = settings["depth"], settings["max_tokens"]
    runs = {"bm25": bm25_runs(corpus, queries, docids, settings)}
    print(dataset, "bm25 done", flush=True)
    bge_q = embed("bge", models["bge"], questions, "query", full, dataset, device)
    bge_p = embed("bge", models["bge"], passages, "passage", full, dataset, device)
    runs["bge"] = dense_runs(bge_q, bge_p, qids, docids, depth)
    runs["bge-binary"] = dense_runs(
        np.where(bge_q > 0, 1.0, -1.0).astype(np.float32),
        np.where(bge_p > 0, 1.0, -1.0).astype(np.float32),
        qids,
        docids,
        depth,
    )
    short = embed(
        "bge", models["bge"], passages, "passage", settings["truncated_tokens"], dataset, device
    )
    runs["bge-trunc"] = dense_runs(bge_q, short, qids, docids, depth)
    print(dataset, "bge variants done", flush=True)
    e5_q = embed("e5", models["e5"], questions, "query", full, dataset, device)
    e5_p = embed("e5", models["e5"], passages, "passage", full, dataset, device)
    runs["e5"] = dense_runs(e5_q, e5_p, qids, docids, depth)
    runs["rrf"] = fuse(runs["bm25"], runs["bge"], settings["rrf_k"], depth)
    runs["bm25-ce"] = rerank(
        runs["bm25"], corpus, queries, models["cross_encoder"], settings, device
    )
    print(dataset, "all systems done", flush=True)
    return {name: runs[name] for name in SYSTEMS}
