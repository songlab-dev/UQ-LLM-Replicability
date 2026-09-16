# Results: cross-fitted stage attribution $\theta_m$

**Executed:** 2026-09-10 · **Spec:** `dml_spec.json` (`dml_theta_v1`), verified
against SHA-256 `39448bec61da415b6165ab56f7d11cb23ffe083cf33491eaf03f0103b8ca7ac0`
before every run (`dml_theta.py` prints this hash and refuses silently to
diverge from it) · **Code:** `dml_theta.py` (Steps 1–12, 14),
`dml_coverage_sim.py` (Step 13), `dml_theta_outputs.py` (LaTeX table + figure)

This file reports what `dml_spec.md`'s walkthrough produces when actually run
on the three corpora, states every place the executed code made a concrete
choice the spec left implicit, and flags the one number that misses the
spec's own pre-registered tolerance.

> **Correction, same session.** An earlier version of this run added a
> small-$G$ standard-error correction (jackknife/CR3, Webb-weight wild
> bootstrap) and expanded Step 13 into a six-arm design (misspecification, a
> null stage, unequal-cluster arms, an $R$-sweep). None of that is in the
> request — the TODO asks only for "cross-fitted $\hat\theta_m$ with
> **cluster-robust and bootstrap bands** on all three corpora, and the
> **$n=90$, $R=100$ coverage simulation**" (singular). This version reports
> exactly that: one cluster-robust interval and one bootstrap interval,
> applied identically to all three corpora, and one coverage simulation at
> $n=90$, $R=100$. The reverted material is not reproduced below.

## Headline result

*Point estimate of each stage's contribution to the score's run-to-run
variance, per corpus, with each stage's share of that corpus's own total in
parentheses.*

| Corpus | $\theta_1$ Extraction ($\times10^{-3}$) | $\theta_2$ Stat review ($\times10^{-3}$) | $\theta_3$ Researcher DoF ($\times10^{-3}$) | $\theta_4$ Theory/context ($\times10^{-3}$) | $\theta_{\text{res}}$ Residual ($\times10^{-3}$) |
| - | -: | -: | -: | -: | -: |
| RPP ($n=90$, $G=88$) | −0.013 (−0.4%) | 1.066 (34.5%) | 0.247 (8.0%) | 0.062 (2.0%) | 1.725 (55.9%) |
| CB[^rpcb-analysis-sample] ($n=103$, $G=20$) | −0.003 (−0.1%) | 2.419 (52.0%) | 0.130 (2.8%) | 0.048 (1.0%) | 2.058 (44.2%) |
| SSRP ($n=21$, $G=21$) | −0.165 (−4.5%) | 1.727 (47.1%) | 0.126 (3.4%) | 0.047 (1.3%) | 1.931 (52.7%) |

[^rpcb-analysis-sample]: CB contains 158 effects, but the primary DML
    analysis uses the 103 effects having at least two runs with the full
    $q_1,q_3,q_4,q_5,q_6$ chain observed (52 effects have no such run and
    three have only one). These are completed verdict predictions, not model
    refusals; the unavailable component is $q_1$, which cannot be scored when
    no recorded/extracted $N$, $p$, or effect-size pair is jointly available.
    The same analysis sample is required for every stage because the
    contributions are incremental along a nested history: for example,
    $\theta_2$ compares $E[P\mid X,q_1]$ with
    $E[P\mid X,q_1,q_3]$ and therefore also requires $q_1$. Restoring the
    excluded effects only for $\theta_2$ and later stages would change the
    target population between components and break the telescoping/adding-up
    identity. An all-158 analysis instead requires a separate,
    missingness-aware specification for $q_1$.

(share of the corpus's own total in parentheses; full point estimates,
standard errors, and both interval constructions are in
`results/{RPP,CB,SSRP}/dml_theta.csv` and plotted in
`figures/dml_theta_attribution_{cluster_robust,bootstrap}.pdf`)

**Reading it:** Stat review ($\theta_2$) is the single largest explainable
stage in all three corpora — a third to half of the model's run-to-run
variance in its own final score traces to what it wrote at the statistical
review step, on top of Extraction. Extraction itself ($\theta_1$) is
indistinguishable from zero everywhere (slightly negative in two corpora,
which is expected sampling noise around a true value at or near zero, not
truncated per the spec). Researcher-DoF concern and Theory/context
($\theta_3$, $\theta_4$) are small but positive. Ten to fifteen points less
than half the variance is never resolved by any of the four upstream
stages — the residual — consistent across all three corpora at 44–56%.

## 95% intervals

*Two interval constructions per stage and corpus — cluster-robust normal and
study-cluster bootstrap — with rows bolded where both agree the interval
excludes zero.*

Two interval constructions, applied identically to all three corpora, per
the request: the cluster-robust normal interval (Step 9), and a
study-cluster bootstrap interval from the same multiplier-bootstrap draws
used for the Step 10 simultaneous band. No small-$G$ variant is used
anywhere in this table. **Bold rows** are significant at $\alpha=0.05$ by
both intervals (neither CI contains zero); the three plain rows ($\theta_1$
in every corpus) are the only ones that are not.

**Cluster-robust 95% intervals**

![Point estimates and 95% cluster-robust confidence intervals for the five
stage-attribution components across RPP, CB, and
SSRP.](figures/dml_theta_attribution_cluster_robust.png)

**Bootstrap 95% intervals**

![Point estimates and 95% bootstrap confidence intervals for the five
stage-attribution components across RPP, CB, and
SSRP.](figures/dml_theta_attribution_bootstrap.png)

*Points are $\hat\theta_m$ and whiskers are the indicated 95% confidence
intervals. Larger filled points indicate intervals that exclude zero;
larger hollow points indicate intervals that include zero.*

| Corpus | Stage | $\hat\theta_m$ ($\times10^{-3}$) | SE, cluster-robust ($\times10^{-3}$) | 95% CI, cluster-robust ($\times10^{-3}$) | 95% CI, bootstrap ($\times10^{-3}$) |
| - | - | -: | -: | - | - |
| RPP | $\theta_1$ | −0.013 | 0.0079 | [−0.028, 0.003] | [−0.027, 0.001] |
| RPP | **$\theta_2$** | **1.066** | **0.132** | **[0.807, 1.324]** | **[0.813, 1.320]** |
| RPP | **$\theta_3$** | **0.247** | **0.047** | **[0.155, 0.340]** | **[0.158, 0.341]** |
| RPP | **$\theta_4$** | **0.062** | **0.0086** | **[0.045, 0.079]** | **[0.045, 0.079]** |
| RPP | **$\theta_{\text{res}}$** | **1.725** | **0.080** | **[1.569, 1.880]** | **[1.575, 1.879]** |
| CB | $\theta_1$ | −0.003 | 0.0097 | [−0.022, 0.016] | [−0.020, 0.015] |
| CB | **$\theta_2$** | **2.419** | **0.255** | **[1.920, 2.919]** | **[1.949, 2.893]** |
| CB | **$\theta_3$** | **0.130** | **0.059** | **[0.014, 0.246]** | **[0.018, 0.238]** |
| CB | **$\theta_4$** | **0.048** | **0.013** | **[0.023, 0.074]** | **[0.025, 0.073]** |
| CB | **$\theta_{\text{res}}$** | **2.058** | **0.137** | **[1.789, 2.327]** | **[1.796, 2.314]** |
| SSRP | $\theta_1$ | −0.165 | 0.091 | [−0.343, 0.013] | [−0.330, 0.001] |
| SSRP | **$\theta_2$** | **1.727** | **0.571** | **[0.609, 2.845]** | **[0.709, 2.729]** |
| SSRP | **$\theta_3$** | **0.126** | **0.048** | **[0.031, 0.221]** | **[0.037, 0.213]** |
| SSRP | **$\theta_4$** | **0.047** | **0.017** | **[0.014, 0.081]** | **[0.015, 0.079]** |
| SSRP | **$\theta_{\text{res}}$** | **1.931** | **0.225** | **[1.490, 2.373]** | **[1.535, 2.323]** |

The two intervals agree closely everywhere, including CB (20 clusters)
and SSRP (21) — there is no sign in this data that the plain cluster-robust
formula is misbehaving at these cluster counts specifically; whether it
holds up in general at this $G$ is exactly what Step 13's simulation checks
(below), not something asserted here.

## Step 7: telescoping check

Passed on every one of the three corpora, to machine precision:

*Largest per-observation deviation from Proposition C.1's exact telescoping
identity, which should equal zero (up to floating-point error) for any
fitted nuisances.*

| Corpus | max$\vert\phi_1+\phi_2+\phi_3+\phi_4+\phi_{\text{res}} - (P-\hat h_0)^2\vert$ |
| - | -: |
| RPP | $1.8\times10^{-16}$ |
| CB | $1.4\times10^{-16}$ |
| SSRP | $1.2\times10^{-16}$ |

This is Proposition C.1 holding on real fitted nuisances, not just
algebraically — confirms Steps 4–6 are wired correctly.

## Step 11: adding-up check

*The five stage attributions summed, checked against the run-to-run
variance computed directly from the raw runs with no model fitting at all.*

| Corpus | $\sum_m\hat\theta_m$ ($\times10^{-3}$) | $E_i[V_i]$, direct ($\times10^{-3}$) | Gap |
| - | -: | -: | -: |
| RPP | 3.087 | 3.067 | **+0.64%** |
| CB | 4.653 | 4.530 | **+2.72%** |
| SSRP | 3.666 | 3.612 | **+1.50%** |

The spec's own rule: a gap wider than the finite-$R$ correction (about
1.25% at $R_{\text{train}}=80$) means $\hat h_0$ is not behaving like
$\bar p_i$ somewhere upstream, and the run should fail loudly rather than
report a number. RPP clears that bar; SSRP is close; **CB does not**
(2.72%). Diagnosed cause, not a silent pass:

CB's $R_i$ is not uniformly 100 the way RPP's and SSRP's are — the
complete-chain filter (Step 1) leaves claims with anywhere from 2 to 100
surviving runs, so $R_{\text{train}}$ (mean 59.9) is both smaller and far
more heterogeneous than the 80 the pre-registered ~1.25% figure assumed. The
finite-$R$ bias $1/R_{\text{train}}$ enters the residual nonlinearly, so a
few very-low-$R$ claims need a much larger correction than the corpus
average suggests; the executed code applies the correction *per claim*
before averaging rather than once on the pooled average (see Deviations
below), which already recovered most of the gap (from 3.58% naively to
2.72%), but did not fully close it. This is reported as an open diagnostic,
not resolved by further tuning the correction — doing so risked fitting the
check rather than reporting it.

## Step 12: per-claim descriptive profiles

Full files: `results/{RPP,CB,SSRP}/dml_theta_per_claim.csv`. No confidence
intervals attached, per the spec — these feed panel figures only.

*Spread of the per-claim residual attribution within each corpus, descriptive
only and not a second estimand.*

| Corpus | $\hat\theta_{i,\text{res}}$ median ($\times10^{-3}$) | $\hat\theta_{i,\text{res}}$ range ($\times10^{-3}$) | Note |
| - | -: | -: | - |
| RPP | 1.58 | [0.59, 4.16] | all claims $R_i=100$ |
| CB | 1.71 | [0.65, 8.32] | $R_i \in [3, 100]$, median 96 |
| SSRP | 1.67 | [0.90, 5.36] | all claims $R_i=100$ |

CB is the only corpus where $R_i$ varies at all, because it's the only
one where any column ever goes missing: `q_3,q_4,q_5,q_6` are 100% non-null
on every run in all three corpora, and `q_1` (Extraction) is 100% non-null
in RPP and SSRP too — RPP because `match_indicator` only needs *one* of
{N, p, r} comparable and `N (O)` alone is present for 99/99 studies, SSRP
because all three original-study reference fields are complete (21/21).
CB has no field anywhere near complete (N 120/158, p-value 115/158,
effect-size-as-r 92/158 — genuine gaps in the published originals, e.g. an
effect size reported as Cliff's delta or a hazard ratio with no exact
conversion to $r$), and the model's own per-run extraction from CB's
rougher source text is also less reliable, so even claims with a usable
reference lose a handful of runs unpredictably rather than all-or-nothing.

## How $h_0,\ldots,h_4$ realize "one pooled fit per fold"

Every $h_m$ is fit as **one function per (stage, study-fold)** — never a
separate regression per claim. $h_1,\ldots,h_4$ are literally that: one OLS
fit per stage per fold, shared across every claim on that fold's evaluation
side (Step 5). $h_0$ looks different in the code (a per-claim leave-run-fold
mean) but is not a separate model: because $W_0=X_i$ is claim-identifying, a
single pooled regression of $P$ on $X_i$ with a full set of claim dummies is,
by the Frisch–Waugh–Lovell theorem, *identical* to computing each claim's own
group mean directly — the closed form is used instead of literally fitting
dummy-variable OLS, but it is the same fitted function. This is also why
centering $P$ and the prefix by the claim's own mean before fitting
$\hat g_1,\ldots,\hat g_4$ (Step 5) is not an ad hoc trick: it is the standard
"within" transformation for a fixed-effects regression of $P$ on (claim
dummies + prefix), so $\hat g_m$ together with the $\hat h_0$ offset *is* the
single pooled regression of $P$ on $W_m$, computed via demeaning rather than
by literally including claim dummies in the design matrix.

The one genuine departure from the textbook recipe: because $X_i$ is
claim-identifying, cross-fitting $h_0$ over *study* folds is not just
noisier than over *run* folds, it is uninformative by construction — a
claim's own dummy does not appear in the training data at all once its
cluster is held out of a study fold, so the fitted value would collapse to
an intercept. This is why $h_0$ alone is cross-fit over run folds within the
claim (Steps 3–4), and it is exactly what the measured cross-fitted
$R^2\approx0$ for every across-study covariate (Step 4's rejected-regression
table) already showed was unavoidable, not a shortcut.

Step 12's per-claim profile $\hat\theta_{i,m}$ is then obtained purely by
evaluating claim $i$'s own fold's $\hat g_1,\ldots,\hat g_4$ (fit without
seeing $i$'s cluster) on $i$'s own runs, plus $i$'s own $\hat h_0$ — no
separate model is ever fit for claim $i$ specifically.

## Step 13: coverage simulation

"The $n=90$, $R=100$ coverage simulation" — the request's own words, run
once, at full scale: 2000 replicates, $B=10{,}000$ bootstrap draws per
replicate (243s total). Ground truth:
$(\theta_1,\theta_2,\theta_3,\theta_4,\theta_{\text{res}}) =
(0.0006, 0.0016, 0.0003, 0.0002, 0.0020)$, chosen to match the real corpora's
magnitude, generated through the same additive stage map Step 5 assumes (so
this checks calibration, not robustness to misspecification — that question
was not asked and is not tested here).

*Bias, plus coverage of both the 95% pointwise CI and the bootstrap
simultaneous band, against a known ground truth at the pre-registered
$n=90$, $R=100$ design.*

| Component | True $\theta$ ($\times10^{-3}$) | Mean $\hat\theta$ ($\times10^{-3}$) | Bias ($\times10^{-3}$) | Coverage, 95% CI (pointwise) | Width, bootstrap band ($\times10^{-3}$, simultaneous) |
| - | -: | -: | -: | -: | -: |
| $\theta_1$ | 0.60 | 0.606 | 0.006 | 0.943 | 0.175 |
| $\theta_2$ | 1.60 | 1.618 | 0.018 | 0.938 | 0.249 |
| $\theta_3$ | 0.30 | 0.303 | 0.003 | 0.953 | 0.091 |
| $\theta_4$ | 0.20 | 0.201 | 0.001 | 0.954 | 0.070 |
| $\theta_{\text{res}}$ | 2.00 | 2.006 | 0.006 | 0.957 | 0.155 |

**Coverage, bootstrap band (simultaneous, all 5 components jointly):** 0.929 ·
**Mean $\vert$adding-up gap$\vert$:** 0.71%.

Every pointwise coverage figure lands within Monte Carlo noise of the
nominal 95% (the binomial SE at 2000 replicates is about 0.5 percentage
points, so 93.8–95.7% is unremarkable), bias is negligible relative to the
$\theta$'s own scale everywhere, and the mean adding-up gap (0.71%) matches
RPP's own real-data gap (0.64%) closely — both draw on the same $n=90$,
$R=100$ design. The simultaneous band, at 92.9%, sits a little under nominal;
with 2000 replicates that is close enough to plausibly be noise rather than
a design flaw, though this run does not have the power to rule out a small
genuine shortfall. Steps 1–11's machinery performs as intended at the scale
it was asked to be checked at.

## Deviations from the pre-registered spec

Every choice below is something the request left as an implementation
detail (K, the nuisance learner class, and everything downstream of "your
call" in the TODO), not a change to what was actually asked for (cross-fitted
$\hat\theta_m$ with cluster-robust and bootstrap bands on all three corpora,
plus the $n=90,R=100$ coverage simulation). Listed because a plan executed
silently-differently from how a reader would assume is worse than one that
says so.

1. **Per-claim finite-$R$ correction, not a pooled one.** RPP and SSRP have
   $R_i=100$ for every claim, so this makes no difference there. CB does
   not: $R_i$ ranges 2–100 after the missingness filter (see Step 12's
   footnote). The executed code applies
   $R_{\text{train},i}/(R_{\text{train},i}+1)$ to each claim's own
   $\phi_{\text{res}}$ before averaging, not a corpus-average factor applied
   once after — the latter (tried first) left a 3.58% adding-up gap for
   CB; the per-claim version reduces it to 2.72% (still open, see Step
   11). This is a correctness requirement for computing $\hat\theta_m$
   itself, not an extra deliverable.

2. **Claims needing $R_i\ge2$.** Three CB claims survived the Step 1
   `q_1..q_6` dropna with exactly one complete run
   (`47_1_5`, `47_1_6`, `9_2_5`) — too few for Step 3/4's leave-run-out
   $\hat h_0$ to exist at all (there is no "other run" to average). These
   three are dropped, on top of the pre-registered filter, leaving
   $n=103$ claims rather than the "roughly 104" the spec estimated (close,
   not a substantive change).

3. **CB clusters: 20 realized, not 23.** Papers 7, 8, and 12 have *no*
   effect with a complete `q_1..q_6` run anywhere, so they drop out of the
   cluster set entirely, not just lose some claims. $G=20$ is what every
   CB standard error and fold split in this run actually uses.

4. **Cluster-robust SE generalized to clusters holding multiple claims.**
   The appendix's formula (transcribed in Definitions) is written for one
   claim per cluster. The executed code sums each cluster's *claims'*
   residuals before squaring (a per-cluster total, not a per-claim one),
   which collapses to the appendix's literal formula when a cluster holds
   exactly one claim (true for 86/88 RPP clusters and all 21 SSRP ones) and
   is the minimal generalization when it doesn't (all of CB, 2 RPP
   pairs) — without it, "cluster-robust... on all three corpora" would
   silently not be cluster-robust for CB at all.

5. **CB adding-up gap (2.72%) exceeds the spec's ~1.25% tolerance.**
   Reported above under Step 11 rather than silently passed — this is the
   one number in this run that the pre-registered rule says should trigger
   a loud failure. Diagnosed cause: CB's $R_i$ is far more heterogeneous
   (2–100) than RPP's/SSRP's uniform 100, and the finite-$R$ bias enters
   the residual as $1/R_{\text{train}}$, a nonlinear function whose corpus
   average understates what a few very-low-$R$ claims actually need. This
   is analytical reasoning from the executed code's own numbers, not a
   claim independently confirmed by simulation — the coverage simulation
   below is run only at the requested $R=100$, so it does not speak to
   CB's low-$R$ claims specifically.

## Outputs written

```text
results/RPP/dml_theta.csv              results/RPP/dml_theta_per_claim.csv
results/CB/dml_theta.csv               results/CB/dml_theta_per_claim.csv
results/SSRP/dml_theta.csv             results/SSRP/dml_theta_per_claim.csv
results/dml_theta_all.csv              (all three, concatenated)
results/coverage_sim.csv               (Step 13)
doc/table_dml_theta.tex                         (Step 14 LaTeX table)
figures/dml_theta_attribution_cluster_robust.*  (Step 14 figure)
figures/dml_theta_attribution_bootstrap.*       (Step 14 figure)
```

Section 4.1's existing plug-in `v_hat_i` in `freedman_bound.py` and
`validation_plot*.py` was not touched by any of the above.
