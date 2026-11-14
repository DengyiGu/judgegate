# First results - 2026-11-13

This exploratory study asks how many labels a sequential ranking regression gate buys,
and whether a small local LLM judge reduces that cost beyond a human-only predictor.
Existing qrels simulate the annotator: unjudged documents are assigned zero, not relabeled.
No new human annotation was collected. The seven systems are fixed pretrained/lexical runs;
no model was trained in this project.

## Design and repetitions

Seven systems give 42 ordered baseline/candidate comparisons per dataset and metric.
We test precision@10 and scaled DCG@10 with max_drop=0.02 and alpha=0.05. Each method
gets 50 successive-sampling orders per comparison. These reuse the same queries, systems
and labels: they measure random-order variation, not independent retrieval experiments.
Both directions of a comparison share the same rankings. Do not count 42 tests as 42
independent datasets.

The original 200 repetitions were a planning placeholder. Fifty were chosen for this first
release to estimate aggregate annotation cost without spending compute on fine per-test
error rates. Each method table reports Monte Carlo SE of the mean label share, propagated
from independent random-order simulations across comparisons. These SEs are conditional
on the fixed benchmark, not uncertainty about generalization. At a 5% error rate the
single-comparison binomial SE is 3.1 percentage points at 50 repetitions and 1.5 at 200;
neither count certifies a theoretical error bound. Error rates are diagnostics only.

Four- and twenty-repetition pilots informed estimator development, including the online
recalibration rule. Final runs are not held-out confirmation and should not be called
preregistered. The synthetic-judge sweep is a labeled diagnostic, not a real model result.

## Label cost

| dataset / metric | uniform plain | weighted learned mean | recalibrated judge | raw judge trusted |
| --- | ---: | ---: | ---: | ---: |
| nfcorpus / sdcg | 39.3% | 31.0% | 29.2% | 57.6% |
| nfcorpus / precision | 45.9% | 44.0% | 42.1% | 61.7% |
| trec-covid / sdcg | 65.4% | 42.6% | 39.1% | 44.0% |
| trec-covid / precision | 63.1% | 45.4% | 42.8% | 47.6% |

Label share divides labels used by the number of nonzero-weight query-document items for
that comparison, then averages comparisons equally. It is not a fraction of all corpus
judgments. Common documents with cancelling metric weights need no label. Different
comparisons are evaluated separately; costs do not assume a shared label cache across gates.
The simulation checks after each label, while the interactive API checks after batches.

The main gain is already present in the learned-mean human baseline. Adding this judge
reduces average label share by about 1.8-3.4 percentage points across the four blocks.
This modest extra saving does not establish broad superiority or account for GPU judging
cost. Trusting raw judge predictions can increase variance enough to cost more labels than
using no judge, especially on NFCorpus. Judge-uncertainty sampling did not improve on the
recalibrated weighted sampler. Near-threshold comparisons are the most expensive; see the
margin bins in [the complete tables](tables.md).

## Judge quality and incorrect decisions

- nfcorpus: 11,414 pooled pairs; 1,437 explicitly judged. Cohen's kappa on that subset is -0.002; binary kappa is 0.000.
  sdcg: judge-only gates agree in 32/42 comparisons, with 0 false passes and 10 false fails.
  precision: judge-only gates agree in 35/42 comparisons, with 0 false passes and 7 false fails.
- trec-covid: 2,004 pooled pairs; 1,654 explicitly judged. Cohen's kappa on that subset is 0.223; binary kappa is 0.278.
  sdcg: judge-only gates agree in 37/42 comparisons, with 5 false passes and 0 false fails.
  precision: judge-only gates agree in 36/42 comparisons, with 6 false passes and 0 false fails.

NFCorpus has no explicitly non-relevant labels in the pooled judged subset, making its
kappa particularly uninformative about distinguishing relevant from irrelevant documents.
The many unjudged pairs scored as zero explain part of the discrepancy with the LLM.

Across the four blocks, mean observed wrong-decision fractions for the recalibrated
weighted judge are 0.29-1.05%. Fifty orders cannot separate a 3% rate from a 7% one, and
20 of the 1,008 comparison-method cells showed more than 5%, up to 10%. Each of those 20
was repeated with 500 fresh labeling orders (`python -m experiments.recheck`). Their rates
came out between 0.6% and 3.6%, each with a standard error of at most 0.83 percentage
points; [error-recheck.json](error-recheck.json) lists every cell. The cells were chosen
for looking bad, so their 50-order rates were biased upward and the fresh ones are not.
This checks the flagged cells only. It is consistent with the guarantee and does not prove
it. The method's conditional-expectation and supermartingale arguments are given separately
in [METHOD.md](../METHOD.md), with exact-expectation and synthetic error checks in tests.

## Resources and traceability

Qwen3-4B-Instruct-2507 judged 13,418 pairs locally on an RTX 4090 Laptop GPU. Final recorded
peak PyTorch allocation was about 8.5 GiB, excluding driver and other allocations. NFCorpus
judging was interrupted and resumed; its recorded 573-second final segment is not total
judging time. TREC-COVID judging took 131 seconds. Reported retrieval construction times
include first-use model downloads, so they are not fair throughput benchmarks.

Dataset archive hashes, counts, all 168 comparison records, simulation settings and aggregate
methods are in [summary.json](summary.json). Source, TREC run, judgment cache and simulation
fingerprints plus package versions are in [provenance.json](provenance.json). Models and
prompt are pinned. Reproducing fresh model inference may introduce small hardware/backend
floating-point differences; the stored caches define this reported study. Raw data and
weights stay local under their source licenses; see [attribution](../DATA.md).

## Limits and next experiment

Only two collections, one judge and a single rubric were tested. Qrels incompleteness is a
large limitation; gains concern the qrels lookup convention, not hidden true relevance.
Metric linearity excludes usual NDCG and MAP. Labels are treated as error-free and fixed.
The theoretical guarantee concerns one fixed test set and one gate, not future traffic or
all releases simultaneously. Predictions were revised during pilot analysis.

The useful next experiment is a second judge or an independently annotated pooled subset
under a matched rubric. More labeling-order repetitions alone will not resolve these
limitations. The synthetic accuracy sweep suggests that better judge quality could improve
label efficiency, but it is not evidence that a larger real model will do so.

