"""A local LLM relevance judge that returns a probability for every grade.

One forward pass per (query, passage) pair reads the next-token distribution over the grade
digits, so each judgment comes with an uncertainty for free. Judgments are appended to a
JSON Lines cache keyed by pair, and an interrupted run resumes where it stopped. torch and
transformers are imported only when a model is loaded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

SYSTEM_PROMPT = "You are a careful search quality rater."

USER_PROMPT = """Rate how well the passage answers the query.

2 = relevant: the passage directly addresses the query and gives useful information for it.
1 = partially relevant: the passage is on the topic of the query but only partly addresses it.
0 = not relevant: the passage does not help with the query.

Query: {query}

Passage: {passage}

Answer with a single digit (0, 1 or 2)."""

GRADES = (0, 1, 2)


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode("utf-8")).hexdigest()


def clip_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def build_messages(query: str, passage: str, *, max_words: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                query=" ".join(query.split()), passage=clip_words(passage, max_words)
            ),
        },
    ]


def load_cache(path: str | Path) -> dict[tuple[str, str], dict[str, object]]:
    """Read cached judgments; a truncated final line from an interrupted run is ignored."""
    cache: dict[tuple[str, str], dict[str, object]] = {}
    target = Path(path)
    if not target.exists():
        return cache
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[(row["qid"], row["docid"])] = row
    return cache


class LocalJudge:
    """Causal LM scorer. ``model_path`` is a local directory or cached snapshot."""

    def __init__(self, model_path: str | Path, *, device: str = "cuda", max_words: int = 250):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_words = max_words
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=False)
            .to(device)
            .eval()
        )
        self.digit_ids = []
        for grade in GRADES:
            ids = self.tokenizer.encode(str(grade), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"grade {grade} is not a single token for this tokenizer")
            self.digit_ids.append(ids[0])

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[list[float], list[float], float]]:
        """Return (grade probabilities, grade logits, digit mass) for each pair.

        Grade probabilities are renormalized over the grade digits; the raw logits are kept
        so the distributions can be rescaled later without another model pass. The digit
        mass is the share
        of the full next-token distribution on those digits; a low value means the model
        wanted to say something other than a grade.
        """
        torch = self.torch
        texts = [
            self.tokenizer.apply_chat_template(
                build_messages(query, passage, max_words=self.max_words),
                tokenize=False,
                add_generation_prompt=True,
            )
            for query, passage in pairs
        ]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            # Inputs are left-padded, so the last position is each prompt's final token.
            # Only that position goes through the vocabulary projection.
            hidden = self.model.base_model(**batch, use_cache=False).last_hidden_state[:, -1, :]
            logits = self.model.get_output_embeddings()(hidden).float()
        full = torch.softmax(logits, dim=-1)[:, self.digit_ids]
        mass = full.sum(dim=-1)
        probs = (full / mass.unsqueeze(-1)).cpu().tolist()
        digits = logits[:, self.digit_ids].cpu().tolist()
        return list(zip(probs, digits, mass.cpu().tolist(), strict=True))


def judge_pairs(
    judge: LocalJudge,
    items: Iterable[tuple[str, str]],
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    cache_path: str | Path,
    *,
    batch_size: int = 16,
    progress_every: int = 50,
) -> dict[tuple[str, str], dict[str, object]]:
    """Judge every uncached (qid, docid) item and return the complete cache."""
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(target)
    todo = sorted({item for item in items if item not in cache})
    # Similar lengths per batch keep padding, and therefore memory, small.
    todo.sort(key=lambda item: len(corpus[item[1]].split()[: judge.max_words]))
    with target.open("a", encoding="utf-8") as handle:
        for number, offset in enumerate(range(0, len(todo), batch_size), 1):
            chunk = todo[offset : offset + batch_size]
            scored = judge.score([(queries[q], corpus[d]) for q, d in chunk])
            for (qid, docid), (probs, logits, mass) in zip(chunk, scored, strict=True):
                row = {
                    "qid": qid,
                    "docid": docid,
                    "probs": probs,
                    "logits": logits,
                    "digit_mass": mass,
                }
                cache[(qid, docid)] = row
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            if number % progress_every == 0:
                print(f"judged {offset + len(chunk)}/{len(todo)} new pairs", flush=True)
    return cache


def grade_probabilities(
    cache: Mapping[tuple[str, str], Mapping[str, object]], keys: Sequence[tuple[str, str]]
) -> np.ndarray:
    """Matrix of grade probabilities, one row per requested item."""
    return np.array([cache[key]["probs"] for key in keys], dtype=float)


def predicted_gain(probs: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the gain under each row's grade distribution."""
    mean = np.clip(probs @ values, values.min(), values.max())
    variance = np.clip(probs @ (values**2) - mean**2, 0.0, None)
    return mean, np.sqrt(variance)
