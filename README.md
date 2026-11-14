# JudgeGate

[![tests](https://github.com/DengyiGu/judgegate/actions/workflows/ci.yml/badge.svg)](https://github.com/DengyiGu/judgegate/actions/workflows/ci.yml)

**Decide whether a ranking change exceeds a quality budget before labeling everything.**

JudgeGate compares a baseline and candidate ranking on a fixed query set. It requests human
relevance labels only for documents that can change the comparison, uses a local LLM judge
to predict the remaining labels, recalibrates those predictions as human labels arrive, and
stops when a sequential test establishes pass or fail. Core dependency: NumPy.

Status: research alpha, 0.1.0. Install from source; no PyPI release is claimed.

## First measured result

Seven retrievers, two BEIR datasets, 13,418 locally judged query-document pairs. Each cell
below averages 42 ordered system comparisons and 50 random labeling orders per comparison.
Numbers are the fraction of nonzero-weight items whose human label was requested, for
scaled DCG@10 with an allowed absolute drop of 0.02.

| Method | NFCorpus | TREC-COVID |
| --- | ---: | ---: |
| Uniform sampling, no prediction | 39.3% | 65.4% |
| Weighted sampling, learned human mean | 31.0% | 42.6% |
| Weighted sampling, recalibrated LLM judge | 29.2% | 39.1% |
| Weighted sampling, raw LLM judge trusted unchanged | 57.6% | 44.0% |

The judge adds a modest gain beyond the learned-mean baseline. Trusting its raw predictions
can cost more labels. Sampling by judge confidence did not help. These are simulated label
requests against existing qrels, with unjudged documents scored zero; they do not measure
fresh human annotation quality or future-query performance. The study is exploratory.

[Report](docs/reports/2026-11-13-first-results.md) ·
[All tables](docs/reports/tables.md) · [Machine-readable evidence](docs/reports/summary.json)

## Install and run

Python 3.10+:

```console
git clone https://github.com/DengyiGu/judgegate.git
cd judgegate
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe examples/quickstart.py
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python examples/quickstart.py
.venv/bin/python -m pytest -q
```

The quickstart is fully synthetic, offline and needs no model. Runtime-only installation is
`python -m pip install .`. Tests optionally cross-check file formats and precision against
[retrieval-regression](https://github.com/DengyiGu/retrieval-regression); those two tests skip
when that separate package is not installed. The joint-example tests also skip without it.

## Companion project: Retrieval Regression

[Retrieval Regression](https://github.com/DengyiGu/retrieval-regression) evaluates complete
qrels, compares retrieval runs, and produces quality gates and offline reports. JudgeGate
uses sequential sampling to reduce the labels needed for a decision on a fixed test set.
Both remain independently installable; neither core requires the other.

The [joint example](docs/INTEGRATION.md) uses both tools on the same Precision@K target:
complete-qrels reference first, then simulated label requests with optional cached LLM
predictions. It includes an offline demo and a measured NFCorpus replay. This complements
Retrieval Regression's query-bootstrap gates, whose uncertainty concerns query sampling;
JudgeGate's uncertainty concerns which document labels were requested on fixed queries.

## Use your own annotator

```python
import numpy as np
from judgegate import Predictor, delta_items, run_gate

# Fix the query universe independently of which queries the runs returned.
items = delta_items(query_ids, baseline_run, candidate_run, metric="sdcg", k=10)
# judge_probs: one row per items.keys(), columns are grades 0, 1, 2.
predictor = Predictor.from_judge(judge_probs, np.array([0.0, 0.5, 1.0]), items.weights)


def label(index):
    query_id, document_id = items.keys()[index]
    return ask_human(query_id, document_id) / 2.0  # human grade 0, 1 or 2


result = run_gate(
    items.weights, label, predictor=predictor, max_drop=0.02, alpha=0.05, max_labels=500
)
print(result.decision, result.labels)  # pass / fail / undecided
```

This integration sketch assumes your query IDs, runs, judge probabilities and labeling
function already exist. `examples/quickstart.py` supplies a complete runnable version.
Omit `predictor` to use a learned human-only mean. Labels must be fixed gains in [0, 1].
Each item is requested at most once; the default batch size is 25. A budget can leave the
gate undecided. Identical rankings pass with zero labels.

Supported metrics are precision@k, scaled DCG@k and truncated rank-biased precision.
NDCG, MAP and reciprocal rank are not supported because their usual forms are not a fixed
linear sum of document gains. All queries stay in the denominator; missing rankings score
zero. Score ties break by ascending document ID.

For one fixed gate with correctly sampled labels, a wrong decision has probability at most
alpha under the method's assumptions. The guarantee is over the sampling order on this
fixed test set. Repeated releases, metrics and slices need a multiple-testing policy.
Printed bounds are one-sided 1-alpha bounds (jointly at least 1-2*alpha) and always agree
with the decision: a pass has `low >= -max_drop`, a fail has `high <= -max_drop`.
See the [estimator, stopping argument and assumptions](docs/METHOD.md).

## Model compatibility

The built-in local judge has been tested with **Qwen3-4B-Instruct-2507**. To use another
model, it needs:

- Transformers `AutoModelForCausalLM` loading with `trust_remote_code=False` and a tokenizer
  with a chat template.
- Each grade digit `0`, `1` and `2` encoded as a single token.
- A base model that returns `last_hidden_state` and an output embedding layer that maps it
  to vocabulary logits, as used by [the scorer](src/judgegate/judge.py).

Other models need a compatibility check; remote chat APIs are not wired into this scorer.
Change `models.judge.id` and `revision` in `experiments/config.json` and start a fresh
judgments cache when replacing the model. Weights are downloaded locally, not bundled with
the repository. The core statistical methods and human-only gate need no language model.

## Reproduce the public-data experiment

Install `.[experiments]` in an environment with an appropriate PyTorch build. The recorded
run used an RTX 4090 Laptop GPU with 16 GB VRAM; judge peak tensor allocation was about
8.5 GiB after memory optimization. The pinned judge snapshot is about 8 GB on disk.

```console
python -m experiments.run --download
```

The command downloads and verifies BEIR archives, builds the seven runs, judges their top-10
union with Qwen3-4B-Instruct-2507, runs the CPU simulation, and writes result tables. Existing
retrieval and judging stages resume from local artifacts. Simulation always reruns.

```console
python -m experiments.simulate --repetitions 50
python -m experiments.tables
python -m experiments.recheck
python -m experiments.evidence
```

Config and model revisions: `experiments/config.json`. To change model, prompt or retrieval
settings, use a new output/cache directory or move the existing ignored artifacts aside;
the caches are not a general experiment manager. Data, weights, runs and logs remain local.
For a quick pipeline check use `--repetitions 4`; it overwrites the local simulation output,
so regenerate publication tables only after the intended run.

## Project layout

- `src/judgegate/`: metrics, unbiased correction, sequential tests, local judge and TREC writers.
- `experiments/`: retrieval systems, judging, simulation and evidence generation.
- `examples/quickstart.py`: complete offline example.
- `examples/joint_evaluation.py`: the same decision checked against Retrieval Regression.
- `tests/`: hand calculations, exact conditional-expectation checks and sampling simulations.
- `docs/`: method and measured evidence.

Code is MIT licensed. Dataset and model terms remain with their original owners. Raw data
and model weights are not redistributed. [Data and model attribution](docs/DATA.md).
