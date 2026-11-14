# Method

## Target

For a fixed universe of Q queries, compare candidate C with baseline B and pass when
`delta = M(C) - M(B) >= -max_drop`. Gains are deterministic values in [0, 1].
A missing query contributes an empty ranking; ties sort by ascending document ID.

For a label-linear metric, `M = mean_q sum_r c(r) g(q, doc_r)`, the difference is
`delta = sum_i w_i g_i`, where `w_i = (c(rank_C) - c(rank_B)) / Q`.
Keep only nonzero weights. Identical precision top-k sets cancel even if their order differs.
The package supports precision@k, scaled DCG@k and truncated rank-biased precision.
Scaled DCG divides DCG by the discount sum of k maximally relevant documents, and gains by
the maximum grade. It is **not NDCG**. NDCG, MAP and reciprocal rank are not implemented.

## Sampling and unbiased correction

Each item has a positive propensity p_i, summing to one. Draw successively without
replacement, proportional to p among items left. Sorting independent exponential random
variables divided by p produces this order. Do not choose the order after looking at labels.

Before label t let R be the remaining items, P their total propensity, A the exact weighted
sum of labeled gains, and h_t(j) a prediction fixed using earlier labels only. The draw gives

```
y_t = A + sum_{j in R} w_j h_t(j) + w_i (g_i - h_t(i)) P / p_i.
```

The conditional selection probability is p_i/P, so
`E[y_t | earlier labels] = A + sum_R w_j g_j = delta`.
This holds for biased predictions and for predictions learned from earlier labels. The
sequence is not independent, but its conditional mean is constant. Each label is bought once.

The default propensity is proportional to |w_i|. The experimental active alternative mixes
80% weight-times-judge-standard-deviation sampling with 20% weight sampling. That floor
bounds importance ratios. Judge uncertainty is a heuristic and did not improve the first study.
Propensities stay fixed during a gate; selection renormalizes them over the remaining items.

## Recalibrating predictions

The judge reads a query and a passage, returning probabilities for grades 0, 1, 2 from the
next-token distribution. Items are grouped by the most probable grade and whether the log
margin between the two leading probabilities is below that grade's median. Each group is
split by the sign of its metric weight. Groups are fixed before human labels are read.

Let f_i be the judge's mean expected gain within its group. In an item's cell, n earlier
labels sum to S. Let u be the earlier mean gain for its whole sign/side, initialized by
`u = (0.5 + sum_side gains) / (1 + n_side)`. The prediction is

```
h_t(i) = (s/2 * f_i + s/2 * u + S) / (s + n),  s = 4.
```

This bounded shrinkage rule learns a collection-specific mapping from judge outputs to human
gains. Sampled-cell means are prediction heuristics, not unbiased population estimates;
only the importance-corrected y_t must be unbiased. No human label used to predict itself.
The human-only baseline learns one mean per side with prior 0.5. Other ablations predict
zero forever or trust raw judge predictions forever. Recalibration can help a poor judge,
but does not guarantee lower variance than human-only prediction.

## Sequential gate

Given h in [0, 1], y_t is within a predictable interval centered at
`A + sum_R w_j h_t(j)` with radius `P * max_R |w_j|/p_j`.
At threshold m = -max_drop maintain two wealth processes:

```
K_pass(t) = product_s (1 + b_pass(s) * (y_s - m))
K_fail(t) = product_s (1 - b_fail(s) * (y_s - m)).
```

Bets are nonnegative, chosen using earlier y values and the known next-value range. The
implementation uses a running precision-weighted mean, a variance estimate scaled by the
remaining mass, and an approximate Kelly fraction `gap / (variance + gap^2)`. Each bet is
capped so its worst loss is at most 75% of current wealth. These are efficiency choices;
validity requires predictability and nonnegative wealth, not a correct variance model.

If delta < m, K_pass is a nonnegative supermartingale. Ville's inequality bounds its chance
of ever reaching 1/alpha by alpha. The symmetric statement holds for K_fail when delta >= m.
The gate stops on the first crossing. It may also decide from the exact range consistent
with observed labels: `[A - sum_R max(-w,0), A + sum_R max(w,0)]`. A full census is exact.

For one fixed gate, the probability of the wrong decision is at most alpha under the stated
sampling/label assumptions. This is not a guarantee across many gates simultaneously.
`gate_path` checks every label; `run_gate` buys batches and may therefore buy extra labels.
A label budget can return `undecided`. Identical runs pass without labels.

`bounds` reports a one-sided lower and upper bound, each with confidence 1-alpha and jointly
at least 1-2*alpha, intersected with the certain range.

Called on its own it tests each value of a fixed grid with that value's own bets and reports
the edge of the unbroken run of rejected values. Bets tuned per value do not make wealth
monotone in the value, so a bound has to stop at the first gap; taking the farthest rejected
value instead would pool many tests and lose the level.

`run_gate` passes the gate threshold m, and the bounds are then anchored there. For values c
on the far side of m (below it for the lower bound) the test reuses the bets the gate placed
at m: `prod_s (1 + b_pass(s) * (y_s - c))`. Each factor is at least the gate's own factor, so
wealth stays nonnegative, and it is decreasing in c. The established values form an interval
whose edge is found by bisection, and that interval contains m exactly when the gate has
crossed. If the true delta is below m, a lower bound above delta implies the single test at
c = delta crossed, which has probability at most alpha. Only after the gate has crossed are
values beyond m examined on the grid; an error there needs delta >= m and a crossing at the
first grid value past delta, again one fixed test. The upper bound is symmetric. A pass
therefore always reports `low >= m` and a fail `high <= m`. The price is that the bound on
the side the evidence points away from uses bets that were mostly zero and can stay wide.

## Scope and evidence

The target is the fully labeled **fixed test set**, not future queries. Human gains are
assumed fixed; disagreement and labeling errors are not modeled. Predictions and groups
must be built before a label is requested, and the random order must follow the propensities.
Multiple metrics, groups or releases require a separate multiple-testing policy.

The experiment simulates human requests with existing qrels and treats unjudged items as
zero. Thus its target is the complete lookup table under that convention, not a claim that
unjudged documents really are irrelevant. NFCorpus has especially sparse judgments.
The estimator and betting arguments support the guarantee; empirical error counts are
checks, not proofs. The implementation and first study are exploratory research software.

## Related work

Every ingredient here has a published source.

- Judging only the documents that separate two systems: Carterette, Allan and Sitaraman,
  Minimal test collections for retrieval evaluation, SIGIR 2006.
- Importance sampling of judgments tuned to the difference between two rankers: Schnabel,
  Swaminathan, Frazier and Joachims, Unbiased comparative evaluation of ranking functions,
  ICTIR 2016.
- Correcting model predictions with a small labeled sample:
  [Angelopoulos et al., Prediction-powered inference](https://arxiv.org/abs/2301.09633);
  applied to retrieval metrics with LLM labels by
  [Oosterhuis et al., Reliable Confidence Intervals for Information Retrieval Evaluation
  Using Generative A.I, KDD 2024](https://arxiv.org/abs/2407.02464).
- Choosing which items to label from model uncertainty:
  [Zrnic and Candes, Active statistical inference, ICML 2024](https://arxiv.org/abs/2403.03208).
- Anytime-valid tests by betting:
  [Waudby-Smith and Ramdas, Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686).

What this project adds is the assembly around one decision, whether a candidate lost more
than a budget on a fixed test set: exact item weights of the paired difference, successive
sampling without replacement so that labeled items count exactly and the estimate ends at
the true value, predictions refitted from the labels the gate has already bought, and a
test whose bets use the known shrinking range of the next value. A literature search did
not find this combination as an existing method, but the search was not exhaustive and the
claim is no stronger than that. The measured benefit of each part is in the first-results
report, including the parts that did not help.
