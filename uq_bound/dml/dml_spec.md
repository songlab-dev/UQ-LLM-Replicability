# Pre-registration: cross-fitted stage attribution $\theta_m$ (Appendix C)

**Spec:** `dml_spec.json` · **id** `dml_theta_v1` · **frozen** 2026-09-10 (revised same day, pre-commit)
**SHA-256 of `dml_spec.json`:** `39448bec61da415b6165ab56f7d11cb23ffe083cf33491eaf03f0103b8ca7ac0`

This file has two parts. **Definitions** restates the appendix's own math in
full — every symbol carries its indices, nothing is abbreviated to a code
column name yet. **Step 1–14** then walks through exactly how `dml_theta.py`
computes those quantities on the three corpora. `dml_spec.json` is the
machine-readable authority; this file is for checking the plan against the
appendix symbol by symbol.

Re-verify the hash before trusting a result that claims to follow this spec:

```bash
shasum -a 256 uq_bound/dml/dml_spec.json
```

Every run of `dml_theta.py` prints this hash into its log. A mismatch means
the spec was edited after freezing, and the run is not covered by this
pre-registration — bump `spec_id` to `dml_theta_v2` rather than editing `v1`
in place.

> **Revision notes.** (1) An earlier draft of Step 4 below estimated $h_0$
> from a three-covariate regression across studies instead of from each
> study's own runs. (2) A later draft of Step 9/13 added small-$G$
> corrections (a jackknife/CR3 standard error, Webb-weight wild-cluster
> bootstrap) and expanded the coverage simulation into a six-arm design
> (misspecification, a null stage, unequal-cluster designs, an $R$-sweep) —
> none of that is in the request this spec implements, which asks only for
> **cluster-robust and bootstrap bands** on all three corpora and **one**
> coverage simulation at $n=90$, $R=100$. Both drafts were caught and
> reverted before any committed result relied on them, so this remains a
> revision of `v1`, not a `v2`. Once this file is committed, that door
> closes.

## Definitions

**Indices.** $i = 1, \ldots, n$ indexes a study (a claim's cluster); $r = 1,
\ldots, R_i$ indexes one of that study's runs; $m = 0, \ldots, s-1$ indexes a
stage of the chain, where $s$ is the number of stages (in this chain, $s=5$).

**The model's own sub-outputs.** $p_{i,m}^{(r)}$ is the model's own output at
stage $m$, for run $r$ of study $i$ — one number per (study, run, stage). The
prefix through stage $m$ is the vector $p_{i,1:m}^{(r)} =
\big(p_{i,1}^{(r)}, \ldots, p_{i,m}^{(r)}\big)$.

**The score.** $P = p_{i,s}^{(r)}$ is the final-stage output — the quantity
every stage attribution explains the variance of. It is a sub-output like any
other, just the one at position $s$, and it is deliberately never included in
any conditioning set below.

**What has been revealed through stage $m$.**

$$W_m = \big(X_i,\ p_{i,1:m}^{(r)}\big), \qquad m = 0, \ldots, s-1, \qquad W_0 = X_i.$$

$X_i$ — the full context of study $i$ (everything about that paper the
generating model can see) — sits inside every $W_m$, including $W_0$. The
score $P$ itself never appears in any $W_m$, since $p_{i,1:m}^{(r)}$ only
reaches stage $m \le s-1$.

**The nuisance functions.**

$$h_m(W_m) = E\big[P \mid W_m\big], \qquad m = 0, \ldots, s-1.$$

$h_m$ is **one fixed function of its argument**, for each $m$ — not indexed
by $i$ or $r$. The expectation pools over a two-stage population: draw a
study (fixing $X_i$), then draw a run (realizing $p_{i,1:m}^{(r)}$ and $P$
given that study). Because $X_i$ sits inside every $W_m$, evaluating this one
pooled function at observation $(i,r)$'s own realized $W_m$ recovers the
per-study Doob mean:

$$h_m(W_m) = h_{i,m}^{(r)} \ \text{a.s.}, \qquad \text{in particular} \quad h_0(X_i) = \bar p_i,$$

the study's own mean score. Pooling across $(i,r)$ is purely a way of
estimating $h_m$; it is not part of what $h_m$ means.

**The stage attributions.** Let $D_m = h_m(W_m) - h_{m-1}(W_{m-1})$ for
$m = 1, \ldots, s-1$, and $D_s = P - h_{s-1}(W_{s-1})$. Define

$$\theta_m = E\big[D_m^2\big] = E\Big[\big(h_m(W_m) - h_{m-1}(W_{m-1})\big)^2\Big], \qquad m = 1, \ldots, s-1,$$
$$\theta_{\text{res}} = E\big[D_s^2\big] = E\Big[\big(P - h_{s-1}(W_{s-1})\big)^2\Big].$$

Each $\theta_m$ and $\theta_{\text{res}}$ is **a single scalar** — one number
per stage, averaged over both the study draw and the run draw, not one
number per study. They add up to the total variance left after only $X_i$ is
known:

$$E\big[(P - h_0(X))^2\big] = \sum_{m<s} \theta_m + \theta_{\text{res}} = E_i[V_i],$$

where $V_i$ is study $i$'s own run-to-run variance of $P$. So the $\theta$'s
are the study-averaged stage attributions, and their sum is the
study-averaged run-to-run variance.

**Orthogonal scores.** For any fitted nuisances $\hat h_0, \ldots,
\hat h_{s-1}$ — not necessarily equal to the true $h_m$'s — define, for a
single observation $(i,r)$:

$$\phi_m = 2\big(\hat h_m - \hat h_{m-1}\big) P - \hat h_m^2 + \hat h_{m-1}^2, \qquad m = 1, \ldots, s-1,$$
$$\phi_{\text{res}} = \big(P - \hat h_{s-1}\big)^2.$$

At the truth, $E[\phi_m] = \theta_m$, and the score is Neyman-orthogonal:
nuisance error enters only quadratically, not linearly.

**Proposition C.1 (exact empirical telescoping).** For every observation and
*any* fitted $\hat h_0, \ldots, \hat h_{s-1}$,

$$\sum_{m=1}^{s-1} \phi_m + \phi_{\text{res}} = \big(P - \hat h_0\big)^2.$$

This is pure algebra — summing $\phi_m$ telescopes to
$2P(\hat h_{s-1} - \hat h_0) - (\hat h_{s-1}^2 - \hat h_0^2)$, and adding
$\phi_{\text{res}} = P^2 - 2P\hat h_{s-1} + \hat h_{s-1}^2$ cancels every
$\hat h_{s-1}$ term and leaves $(P - \hat h_0)^2$ — so it holds regardless of
how good the fitted nuisances are.

**Cross-fitted estimator and inference.** Partition the $n$ studies (never
the $nR$ runs) into $K$ folds, keeping every run of a study together. Fit
$\hat h_m^{(-k)}$ off-fold, evaluate $\phi_{ir,m}$ on-fold, and average:

$$\hat\theta_m = \frac{1}{n} \sum_i \frac{1}{R_i} \sum_r \hat\phi_{ir,m}.$$

With study-level mean scores $\bar\phi_{i,m}$, the cluster-robust standard
error is

$$\widehat{se}(\hat\theta_m) = \{n(n-1)\}^{-1/2} \Big\{ \sum_i (\bar\phi_{i,m} - \hat\theta_m)^2 \Big\}^{1/2},$$

with simultaneous bands from a multiplier bootstrap over studies. The
root-$n$ normal limit needs independent study clusters, exchangeable runs,
fixed $s$, bounded outputs, and $\|\hat h_m - h_m\|_2 = o_p(n^{-1/4})$.
Negative $\hat\theta_m$ are not truncated before inference.

**Connection to Section 4.1's existing plug-in.** Applying Proposition C.1 to
one study $i$ and averaging over that study's own runs instead of over the
whole corpus gives $\hat v_i = \sum_m \hat\theta_{i,m}$ — exactly Section
4.1's Freedman plug-in already computed by `common.py`'s
`estimate_chain_of_action()`. That existing code is a correct instance of
this same telescoping identity; what it lacks is cross-fitting and $X_i$ in
its own $\hat h_1, \ldots, \hat h_{s-1}$ (see Steps 4–5).

## What gets computed

Everything below is produced once per corpus (RPP, CB, SSRP separately).
The table gives each quantity's role before the steps derive it in full;
"Reported?" marks what ends up in the output files of Step 14 versus what
exists only to validate the pipeline along the way.

| # | Quantity | Symbol | Interpretation | From | Reported? |
| -: | - | - | - | - | - |
| 1 | Fitted nuisances | $\hat h_0, \ldots, \hat h_4$ | The off-fold best guess of the final score $P$, using only what the model had revealed through stage $m$. Not a result on its own — an intermediate model used to build row 2. | Steps 4–5 | No |
| 2 | Per-observation scores | $\phi_1, \ldots, \phi_4, \phi_{\text{res}}$ | One value per (claim, run) — how much stage $m$'s revelation moved the best guess, squared, corrected so nuisance error doesn't bias it. The raw material row 4 averages. | Step 6 | No |
| 3 | Telescoping check | $\sum_m \phi_m + \phi_{\text{res}} \overset{?}{=} (P-\hat h_0)^2$ | A per-observation identity that must hold exactly regardless of nuisance quality. Confirms Steps 4–6 were implemented correctly; not itself a scientific finding. | Step 7 | No (assertion only) |
| 4 | Stage attributions | $\hat\theta_1, \ldots, \hat\theta_4$ | **The headline result.** How much of the score's corpus-wide variance each of the four upstream reasoning stages (Extraction, Stat review, Researcher DoF, Theory/context) explains on its own, beyond the stages before it. | Step 8 | Yes |
| 5 | Residual | $\hat\theta_{\text{res}}$ | What's left unexplained after all four stages — the model's own run-to-run inconsistency in the final score, given everything it already reasoned through. Not a fifth stage. | Step 8 | Yes |
| 6 | Standard errors | $\widehat{se}(\hat\theta_m)$ | How uncertain each of rows 4–5 is, accounting for claims that share a cluster (a paper with several effects, for CB). Small-cluster corrected for CB/SSRP. | Step 9 | Yes |
| 7 | Simultaneous band | 95% band over all 5 scalars jointly | An interval that covers all five estimates at once at 95%, not just each one individually — the right comparison when reading rows 4–5 as a set. | Step 10 | Yes |
| 8 | Adding-up check | $\sum_m \hat\theta_m + \hat\theta_{\text{res}} \overset{?}{\approx} E_i[V_i]$ | Compares rows 4–5 summed against the run-to-run variance computed directly from the raw runs (no fitting at all). Confirms the whole pipeline, not just Step 7's per-observation arithmetic. | Step 11 | Yes (as a diagnostic line) |
| 9 | Per-claim profile | $\hat\theta_{i,m}$ | How much each stage moved *one specific claim's* prediction — a descriptive picture for the panel figures, not a second estimand. Carries no confidence interval. | Step 12 | Yes (labeled descriptive) |
| 10 | Coverage simulation | pointwise & simultaneous coverage, bias, band width | Checks — on data with a known answer — that rows 4–7's procedure actually has the coverage it claims, before any of it is trusted on the real corpora. | Step 13 | Yes (separate file) |

## Step 1 — Map the definitions onto this chain and assemble the observations

This chain has $s = 5$. Stage $m$'s sub-output $p_{i,m}^{(r)}$ and the final
score $P = p_{i,s}^{(r)}$ correspond to the repo's own per-run columns as
follows:

| $m$ | Stage name | Column |
| -: | - | - |
| 1 | Extraction | $p_{i,1}^{(r)} = $ `q_1` |
| 2 | Statistical review | $p_{i,2}^{(r)} = $ `q_3` |
| 3 | Researcher DoF concern | $p_{i,3}^{(r)} = $ `q_4` |
| 4 | Theory and context | $p_{i,4}^{(r)} = $ `q_5` |
| 5 (= $s$) | — (the score itself) | $P = p_{i,5}^{(r)} = $ `q_6` |

Unit 2 (Credibility) has no parsed numeric output and is not part of the
chain; unit 7 (Verdict) is a deterministic function of `q_6` generated one
step later and contributes no information beyond it, so it is excluded by
construction rather than assigned a stage. This gives four stage attributions
$\theta_1, \ldots, \theta_4$ plus $\theta_{\text{res}}$ — never a fifth
stage — which is the paper-notation restatement of why the existing chain
code's fifth increment (`D5 = q6 - h4`) is $\phi_{\text{res}}$, not a stage.

$i$ is a *claim* and its cluster is the unit that gets its own cross-fitting
fold and its own robust-inference weight. For two of the three corpora, a
cluster holds more than one claim:

| Corpus | Claim key ($i$) | Cluster key | Claims ($n$) | Clusters | Runs/claim ($R_i$) |
| - | - | - | -: | -: | -: |
| RPP | `altmejd_id` | `Study Title (O)` | 90 | 88 | 100 |
| CB | `(Paper #, Experiment #, Effect #)` | `Paper #` | 158 | 23 | 100 |
| SSRP | `Study Num` | `Study Num` | 21 | 21 | 100 |

RPP is keyed by `altmejd_id`, not by title as the existing chain code does.
Two title clusters carry two claims each, and they aren't the same case:
`rpp.49`/`rpp.50` have genuinely distinct 100-run prediction vectors (two
different effects, which title-level grouping would pool into one 200-run
claim), while `rpp.148`/`rpp.149` have byte-identical vectors (one generation
set surfaced under two ids). Both pairs are kept as two claims, matching the
$n=90$ convention defined by Altmejd et al.'s `drop == False` filter and used by `predict_embed_lr.py`
already use; the within-paper dependence is absorbed by the cluster in every
later step.

CB's 158 effects sit in only 23 papers, so 23 — not 158 — is the $n$ that
drives every fold split and every standard error from here on. Its primary
scope is further narrowed to complete-chain effects: a claim only enters if
$p_{i,1}^{(r)}, \ldots, p_{i,5}^{(r)}$ (i.e. `q_1..q_6`) are jointly non-null
on at least one of its runs (the current `dropna` behaviour of
`common.py`'s CB section, stated here rather than left incidental). That leaves roughly
104 of the 158 effects; the realized count is printed and logged on every
run. A second, all-158 arm (S4) imputes the missing $p_{i,1}^{(r)}$ at its
within-study mean plus an availability flag.

Each surviving observation $(i,r)$ carries: $P = p_{i,5}^{(r)}$, the prefix
$\big(p_{i,1}^{(r)}, \ldots, p_{i,4}^{(r)}\big)$, its claim id $i$, and its
cluster id.

## Step 2 — Split the clusters into study folds

Assign each *cluster* — never a claim, never a run — to one of $K=5$ folds
with `GroupKFold`, seed 2024. Every claim and every run belonging to a
cluster travels with it, so a fold boundary never puts part of one cluster on
the training side and the rest on the evaluation side. This is the
"partition the $n$ studies, not the $nR$ runs" rule from Definitions, applied
to the object that pools across studies in Step 5. Sensitivity arm S1
repeats this split at $K \in \{2, 10, \text{leave-one-cluster-out}\}$.

## Step 3 — Split each claim's own runs into run folds

Independently of Step 2, and within each claim separately, split that
claim's $R_i \approx 100$ runs into $K = 5$ folds, seed 2024. This second
partition exists only to keep Step 4's per-claim mean honest: without it, a
claim's own $\hat h_0$ would be estimated in part from the very run it later
scores, biasing $\hat\theta_{\text{res}}$ down by roughly $V_i / R_i$.
Sensitivity arm S3 repeats this split at
$K \in \{2, 10, \text{leave-one-run-out}\}$.

## Step 4 — Estimate the baseline $\hat h_0$ from each claim's own runs

For a run sitting in run-fold $k$ of claim $i$, set

$$\hat h_0(X_i) = \text{mean of } P \text{ over claim } i\text{'s runs in the other four run-folds.}$$

Per Definitions, this is exact, not approximate: $h_0(X_i) = \bar p_i$
identically, and on a claim-identifying $X_i$ the only functions of $X_i$
are "one number per claim" — so the claim's own out-of-fold run mean is the
saturated fit of $h_0$.

A different design was tried first and measured before being set aside:
regress $P$ on three covariates drawn from the *original* study ($\log N$,
$|r|$, and whether $p < .001$), pooled across claims, instead of using a
claim's own runs. The measurement is why it was dropped — the cross-fitted
$R^2$ of that regression is *negative* in every corpus (it predicts a
claim's mean score worse than the grand mean does), while the variance
across claims is 2–4 times the variance the stage attributions are supposed
to sum to:

| Corpus | Target: $E_i[V_i]$ | Between-claim variance | Cross-fitted $R^2$ | Shortfall | Would inflate the total by |
| - | -: | -: | -: | -: | -: |
| RPP | 0.00307 | 0.01120 | −0.033 | 0.01144 | +373% |
| CB | 0.00473 | 0.01029 | −0.115 | 0.01140 | +241% |
| SSRP | 0.00361 | 0.01279 | −0.185 | 0.01443 | +399% |

Swapping in the richest study-level text summary available in this repo —
the embedding-LR out-of-fold probability, itself trained on replication
outcomes — does not close the gap: cross-fitted $R^2$ of −0.009 on RPP, still
leaving a shortfall 3.6 times the target. Whatever a pooled predictor misses
about a claim's mean does not cancel; per the population identity in
Definitions it is added on top of $E_i[V_i]$, which is why this option is
kept only as diagnostic arm S7 and never used to produce $\hat h_0$.

This choice also settles the estimation-error budget for free. Double
machine learning needs $\sqrt{n} \times \|\hat h_0 - h_0\|^2$ to be
negligible; with $\hat h_0$ built from a claim's own runs, that squared error
is about $V_i / R_i$, so the bound is driven by $R$, not by $n$:

| Corpus | $\sqrt{n} \times E_i[V_i] / R$ |
| - | -: |
| RPP | 0.00029 |
| CB | 0.00023 |
| SSRP | 0.00017 |

Negligible in all three at $R = 100$ — a bound the rejected pooled-covariate
version would not have met.

## Step 5 — Fit the pooled prefix maps $\hat g_1, \ldots, \hat g_4$

For every run on the training side of the current study fold (Step 2),
subtract that claim's own $\hat h_0$ (Step 4) from $P$, giving a within-claim
score deviation, and subtract each $p_{i,m}^{(r)}$'s own within-claim mean
from itself. Then fit four ordinary-least-squares regressions of the
centered deviation on the centered prefix, each nested in the last:

- $\hat g_1$: deviation on centered $p_{i,1}^{(r)}$
- $\hat g_2$: deviation on centered $\big(p_{i,1}^{(r)}, p_{i,2}^{(r)}\big)$
- $\hat g_3$: deviation on centered $\big(p_{i,1}^{(r)}, p_{i,2}^{(r)}, p_{i,3}^{(r)}\big)$
- $\hat g_4$: deviation on centered $\big(p_{i,1}^{(r)}, p_{i,2}^{(r)}, p_{i,3}^{(r)}, p_{i,4}^{(r)}\big)$

Each $\hat g_m$ is **one fitted object per study fold**, shared by every
claim on that fold's evaluation side — never refit per claim, matching
Definitions' requirement that $h_m$ is a single function, not indexed by $i$
or $r$. Set $\hat h_m = \hat h_0 + \hat g_m(\text{centered prefix})$, clipped
to $[0,1]$.

Centering within claim before fitting is what makes $W_m$ nest correctly:
without it, $\hat g_1$ would be a function of $p_{i,1}^{(r)}$ alone with no
claim term anywhere, so knowing $p_{i,1}^{(r)}$ would not be a strict
refinement of knowing which claim is being scored — $W_1$ would not contain
$W_0$ — and $\hat\theta_1$ would not be measuring stage 1's own share of the
variance, even though Proposition C.1's arithmetic (Step 7) would still
balance regardless. This is exactly the gap in the existing chain code: it
sets $\hat h_0$ to the claim's own mean (correct, per Step 4) but fits
$\hat h_1, \ldots, \hat h_4$ as pooled regressions on the prefix alone, with
no claim term at all. Sensitivity arm S2 refits this step with depth-3
gradient boosting in place of OLS, on the identical folds; arm S6 relaxes
the additive form with claim-covariate-by-prefix interactions.

## Step 6 — Score every observation with nuisances it never trained

For observation $(i,r)$, using $\hat h_0, \ldots, \hat h_4$ fitted without
that claim's cluster on the training side of its study fold (Step 2) and
without that run on the training side of its run fold (Step 3), compute

$$\phi_m^{(i,r)} = 2\big(\hat h_m - \hat h_{m-1}\big) P - \hat h_m^2 + \hat h_{m-1}^2, \qquad m = 1, \ldots, 4,$$
$$\phi_{\text{res}}^{(i,r)} = \big(P - \hat h_4\big)^2,$$

with $P = p_{i,5}^{(r)}$ (`q_6`), exactly as defined for general $m$ and $s$
in Definitions, here with $s = 5$.

## Step 7 — Check that the five scores telescope exactly

For every single observation, verify

$$\phi_1 + \phi_2 + \phi_3 + \phi_4 + \phi_{\text{res}} = \big(P - \hat h_0\big)^2$$

to within $10^{-10}$ — Proposition C.1, which holds for *any* fitted
nuisances. A failure here means an indexing bug in Steps 4–6, never a
statistical problem, and the run stops.

## Step 8 — Average into five scalars and correct the residual for finite $R$

$$\hat\theta_m = \frac{1}{n} \sum_i \frac{1}{R_i} \sum_r \phi_m^{(i,r)}, \qquad m = 1, \ldots, 4, \text{res},$$

exactly the cross-fitted estimator from Definitions — one number per stage
over the whole corpus, not one per claim. Because $\hat h_0$ in Step 4 was
fit from $R_{\text{train}} < R_i$ runs rather than infinitely many,
$E[(P - \hat h_0)^2]$ is inflated by a factor of $1 + 1/R_{\text{train}}$
(about 1.25% at $R_{\text{train}} = 80$). The reported
$\hat\theta_{\text{res}}$ multiplies by
$R_{\text{train}} / (R_{\text{train}} + 1)$ to remove that inflation; the
uncorrected value is kept alongside it. Negative $\hat\theta_m$ are reported
as estimated, with no truncation, per Definitions.

## Step 9 — Attach a cluster-robust standard error to each scalar

Average each claim's own $\phi_m$ values into $\bar\phi_{i,m}$, then use the
formula from Definitions:

$$\widehat{se}(\hat\theta_m) = \{n(n-1)\}^{-1/2} \Big\{ \sum_i (\bar\phi_{i,m} - \hat\theta_m)^2 \Big\}^{1/2}.$$

This is the request's literal formula, applied identically to all three
corpora — no small-$G$ variant (jackknife, CR3, wild-cluster) is part of
what was asked, and none is used.

## Step 10 — Build one simultaneous band across all five scalars

Draw $B = 10{,}000$ Rademacher weights $\xi_i$ per cluster, seed 0, and form

$$\sup_m \; \frac{\big| \sum_i \xi_i (\bar\phi_{i,m} - \hat\theta_m) \big|}{\widehat{se}(\hat\theta_m)}$$

jointly over the five scalars on each draw — the multiplier bootstrap over
studies from Definitions. The 95th percentile of that maximum, times each
$\widehat{se}(\hat\theta_m)$, gives a band that covers all five estimates
simultaneously at 95%, reported alongside the five separate pointwise
intervals from Step 9. The same draws also give a per-scalar percentile
interval — the "study-cluster bootstrap intervals" the appendix says to
report alongside the normal-based ones, for all three corpora.

## Step 11 — Check the estimates add up to something observable

Sum the five corrected scalars from Step 8 and compare to $E_i[V_i]$
computed directly from the raw runs, with no fitting at all — the mean
within-claim sample variance of $P$: **0.00307** (RPP), **0.00473** (CB),
**0.00361** (SSRP). Agreement is required to within the finite-$R$ correction
of about 1.25%. A larger gap means $\hat h_0$ is not behaving like $\bar p_i$
somewhere upstream, and the run fails loudly rather than reporting a number.

This is the pooled-corpus version of the identity in Definitions'
"Connection to Section 4.1's existing plug-in": applying Proposition C.1 per
claim and averaging over $i$ gives $E_i\big[\sum_m \hat\theta_{i,m}\big]$,
which this check compares against the same $E_i[V_i]$ from the other
direction.

## Step 12 — Separately, build a descriptive per-claim profile

For one claim $i$, evaluate that claim's own fitted $\hat h_0, \ldots,
\hat h_4$ (from whichever study fold placed it on the evaluation side) on
its own runs and average within the claim, giving $\hat\theta_{i,m}$ for
$m = 1, \ldots, 4, \text{res}$. Summed over $m$, this is exactly
$\hat v_i = \sum_m \hat\theta_{i,m}$ from Definitions — the same construction
Section 4.1's `estimate_chain_of_action()` already computes, just now built
from the cross-fitted, claim-conditioning $\hat h_m$'s of Steps 4–5 rather
than the existing code's in-sample, prefix-only ones.

This per-claim profile is used only in the panel figures and is labeled
**descriptive**: it carries no confidence interval and is not a second
estimand — Steps 9 and 10's inference attaches only to the five pooled
scalars from Step 8.

## Step 13 — Validate the whole procedure on simulated data first

This is the request's own item, taken literally: "the $n=90$, $R=100$
coverage simulation." One run, not a battery of stress tests. Before Steps
1–12 are trusted on the real corpora, run them once on simulated data where
the truth is known: $n = 90$ studies, $R = 100$ runs, 2000 replicates, seed
7, drawn from the same two-stage law (a claim-level $\bar p_i$ that Step 4's
measurement says is *not* predictable from any observed covariate, then
runs drawn given that $\bar p_i$, through a correctly-specified additive
linear stage map — the same functional form Step 5 assumes). The check is
whether the cluster-robust pointwise CI (Step 9) and the multiplier-bootstrap
simultaneous band (Step 10) — the same two intervals reported on the real
corpora — achieve their nominal 95% coverage at this $n$ and $R$, reporting
per-stage bias, pointwise and simultaneous coverage, mean band width, and
the Step 11 adding-up error.

## Step 14 — Write the outputs

- `results/{RPP,CB,SSRP}/dml_theta.csv` — $\hat\theta_1, \ldots, \hat\theta_4,
  \hat\theta_{\text{res}}$ per corpus, with standard errors and bands
  (Steps 8–10)
- `results/{RPP,CB,SSRP}/dml_theta_per_claim.csv` — the descriptive
  $\hat\theta_{i,m}$ profiles from Step 12
- `results/coverage_sim.csv` — the Step 13 validation results
- `table_dml_theta.tex` and `figures/dml_theta_attribution_cluster_robust.pdf` — the
  corpus-by-stage table and figure built from the first file

Section 4.1's existing plug-in $\hat v_i$ in `freedman_bound.py` and
`validation_plot*.py` is untouched by any of the above; the gap between
$\sum_m \hat\theta_m$ from Step 8 and the mean of the existing (in-sample,
prefix-only) $\hat v_i$ is reported as a diagnostic of how optimistic that
plug-in is.

## Sensitivity arms referenced above

| Arm | Varies |
| - | - |
| S1 | study-fold $K \in \{2, 10, \text{leave-one-cluster-out}\}$ (Step 2) |
| S2 | gradient-boosting prefix map (Step 5) |
| S3 | run-fold $K \in \{2, 10, \text{leave-one-run-out}\}$ (Step 3) |
| S4 | CB at all 158 effects, with $p_{i,1}^{(r)}$ imputed (Step 1) |
| S5 | temperature-0.2 prediction files in place of 0.7 |
| S6 | claim-covariate-by-prefix interactions in $\hat g_m$, relaxing Step 5's additivity |
| S7 | the rejected pooled-covariate $\hat h_0$, kept only as the diagnostic in Step 4 |
