# Using JudgeGate with Retrieval Regression

The tools share a useful input contract, while answering different questions:

| Tool | Inputs | Decision |
| --- | --- | --- |
| Retrieval Regression | Full qrels and two rankings | Quality budget on observed metrics; optional query-bootstrap evidence |
| JudgeGate | Rankings, predictions and requested human labels | Sequential quality decision about the fixed query set |

This integration is a source example, not a runtime dependency or a combined framework.
It needs no changes to either library's metric or gate implementation. Both remain usable
alone, and Retrieval Regression keeps its dependency-free runtime.

## Offline demo

Clone the repositories as siblings. In the JudgeGate environment, install the optional
companion from its source (no PyPI publication is assumed):

```console
python -m pip install -e . -e ../retrieval-regression
python examples/joint_evaluation.py
```

Use the actual sibling directory name if it differs. The rest of the commands run from
the JudgeGate repository root.
Python can be replaced by `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on POSIX.
Use Retrieval Regression 0.4.0; the integration CI pins commit
`8707649dfa41f377fb6b09e2566be40035b2e5e2` for reproducibility.

The default fixture is explicitly synthetic: baseline Precision@10 is 0.8 and candidate
Precision@10 is 0.6 over 120 queries. With a 0.02 budget, the complete-qrels gate fails.
At seed 0 the sequential gate also fails after 25 of 480 nonzero-weight items. This is
an intentionally easy regression, not evidence of performance on real traffic.

## Exported rankings and judgments

```console
python examples/joint_evaluation.py --qrels qrels.tsv --qrels-format beir --baseline baseline.trec --candidate candidate.trec --run-format trec --k 10 --max-drop 0.02
```

The example uses Retrieval Regression's JSON/BEIR/TREC readers. Rankings may be JSON or
TREC. The resulting query universe is fixed by qrels. Strict input defaults apply, including
nonnegative grades and at least one positive judgment per query. A query missing from a
ranking contributes zero. Both tools rank scores descending with ascending document IDs
for ties, divide precision by K even for short lists, and give unjudged documents zero.
Grades above zero are converted to binary gain, so graded qrels work for Precision@K.

An optional `--judge-cache judgments.jsonl` reads JudgeGate's cached grade probabilities
for the relevant query-document pairs. All nonzero-weight pairs must be present, with
three columns corresponding to grades 0, 1 and 2. The expected precision gain is
`P(grade=1) + P(grade=2)`. The online predictor is then recalibrated from requested labels.
Without a cache, the gate learns human-only mean gains. No model is downloaded or run by
this example. A `--max-labels` budget can return `undecided`, which is not counted as an
agreement or a failure. Output is JSON; process success means the replay ran successfully,
not that the candidate passed its quality gate.

## NFCorpus replay

The original experiment's cached bge and sign-quantized bge runs were reused, with no
retrieval changes or model retraining. On 323 test queries:

| Reference / replay | Result |
| --- | --- |
| Baseline bge Precision@10 | 0.2563467492 |
| Candidate bge-binary Precision@10 | 0.2037151703 |
| Full-qrels difference | -0.0526315789 |
| Retrieval Regression point gate, max_drop=0.02 | fail |
| JudgeGate, human-only mean, seed 0 | fail; 325 / 3698 labels (8.8%) |
| JudgeGate, cached Qwen predictions, seed 0 | fail; 325 / 3698 labels (8.8%) |

The two libraries agree numerically on the full-label difference before the sequential
gate runs. This single labeling order demonstrates interoperability; it is not an estimate
of average label savings and shows no extra saving from the LLM. See the broader
[50-order study](reports/2026-11-13-first-results.md) for label-cost comparisons.
Existing qrels simulate human responses, and unjudged documents score zero. This is not a
fresh human-annotation study. One item's label can be reused in a real workflow, but this
demo does not assume a shared labeling budget across multiple gates.

The measured JSON is in [joint-evaluation.json](reports/joint-evaluation.json).
After running the original `python -m experiments.run --download` pipeline, create the
qrels file in an ignored directory without redistributing its text:

```python
from pathlib import Path
from zipfile import ZipFile

Path("work").mkdir(exist_ok=True)
with ZipFile("data/nfcorpus.zip") as archive:
    Path("work/nfcorpus-qrels.tsv").write_bytes(archive.read("nfcorpus/qrels/test.tsv"))
```

```console
python examples/joint_evaluation.py --qrels work/nfcorpus-qrels.tsv --qrels-format beir --baseline reports/runs/nfcorpus/bge.trec --candidate reports/runs/nfcorpus/bge-binary.trec --run-format trec
```

Repeat with `--judge-cache reports/judgments/nfcorpus.jsonl` for the judge-assisted replay.
Dataset/model attribution and original snapshot hashes are in [DATA.md](DATA.md) and
[provenance.json](reports/provenance.json). No raw text or weights are included here.

## Interpretation

The complete-label point gate is the reference answer under the supplied qrels convention.
It uses `gate_mode="point"` and no bootstrap. JudgeGate's chance of a wrong decision is
controlled over random labeling orders for a fixed query set under its assumptions.
Retrieval Regression's bootstrap gates instead resample queries; they are not a substitute
for the sequential label-sampling test, and their confidence levels should not be equated.

This integration intentionally uses Precision@K. JudgeGate's scaled DCG and Retrieval
Regression's NDCG have different denominators and cannot be compared as the same metric.
Never export LLM predictions as human ground truth simply to make the point gate pass.
For a live annotation workflow, replace the qrels-lookup callback with your labeling tool;
the complete-qrels reference will then only become available after a full annotation pass.
