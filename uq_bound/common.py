"""Shared bound inputs for RPP, CB, and SSRP.

``delta_i = |qbar_i - pstar_i|`` measures the gap between the mean Scoring
prediction and the embedding-model reference. ``D1``--``D5`` are
within-run chain increments from Extraction through Scoring; their squared
means define the Freedman variance term ``v_i``. The chain uses units 1, 3,
4, 5, and 6 because Credibility has no numeric output and Verdict is derived
from Scoring.

RPP uses held-out ``proba_cv`` as its reference; CB and SSRP use
``proba_cv_avg``. Observed outcomes are used only for empirical exceedance
rates. Corpus-specific loaders retain the source keys needed to keep claims
separate.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402
import metrics as llm_metrics  # noqa: E402
from rpp_outcomes import attach_canonical_rpp_outcomes  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RPP_CSV = os.path.join(REPO, "data", "rpp_data_cleaned.csv")
ALTMEJD_CSV = os.path.join(REPO, "data", "altmejd_osf", "data.csv")
RPP_PRED_CSV = os.path.join(REPO, "prediction", "LLM_Reasoning", "RPP", "text_predictions_high_temp0.7.csv")
RPP_EMBED_LR_CSV = os.path.join(REPO, "prediction", "embed_pred", "RPP", "embed_lr_study_predictions.csv")

CB_CSV = os.path.join(REPO, "data", "cb_data_cleaned.csv")
CB_EFF_CSV = os.path.join(REPO, "data", "RP_CB_eff.csv")
CB_PRED_CSV = os.path.join(REPO, "prediction", "LLM_Reasoning", "CB", "text_predictions_high_temp0.7.csv")
CB_EMBED_LR_CSV = os.path.join(REPO, "prediction", "embed_pred", "CB", "embed_lr_study_predictions.csv")

SSRP_CSV = os.path.join(REPO, "data", "ssrp_data_cleaned.csv")
SSRP_PRED_CSV = os.path.join(REPO, "prediction", "LLM_Reasoning", "SSRP", "text_predictions_high_temp0.7.csv")
SSRP_EMBED_LR_CSV = os.path.join(REPO, "prediction", "embed_pred", "SSRP", "embed_lr_study_predictions.csv")

# ---------------------------------------------------------------------------
# Constants shared identically across all three datasets
# ---------------------------------------------------------------------------
CHAIN = [1, 3, 4, 5, 6]  # chain-of-action steps in generation order:
                         # Extraction, Stat review, Researcher DoF concern,
                         # Theory and context, Scoring (unit 2 is unscored)
ULAB = {1: "Extraction", 3: "Stat review", 4: "Researcher DoF concern",
        5: "Theory and context", 6: "Scoring", 7: "Verdict"}
ALPHA_POWER = 0.05   # two-tailed test size for the unit-3 power-analytic reference

# Original-study columns used to score Extraction.
REC_N_COL = "N (O)"
REC_P_COL = "P-value (O)"
REC_R_COL = "Effect size r (O)"

# CB effect-size types with direct conversions to r.
SMD_TYPES = {"Cohen's d", "Cohen's dz", "Glass' delta"}
R_TYPES = {"r", "Pearson's r", "Spearman's r"}


# ---------------------------------------------------------------------------
# Shared generic helpers
# ---------------------------------------------------------------------------

def normalize_title(s):
    """Normalize titles for matching."""
    s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def correlation_power(r, n, alpha=ALPHA_POWER):
    """Fisher-z approximation to two-sided correlation-test power."""
    if r is None or n is None or not np.isfinite(r) or n <= 3:
        return np.nan
    z = np.arctanh(min(abs(r), 0.999)) * np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z - zc) + norm.cdf(-z - zc))


def chain_bound_constants():
    """Return the unit increment bound and its squared sum for AH."""
    c_hat = {m: 1.0 for m in CHAIN}
    return c_hat, float(len(CHAIN))


def estimate_chain_of_action(unit_scores, key_col):
    """Estimate per-run conditional-expectation increments.

    Pooled OLS estimates the expected Scoring value after each revealed
    prefix. The first and final conditional expectations are the claim mean
    and observed Scoring value. Returns the long increment table, per-claim
    squared-increment sum, and maximum absolute increment.
    """
    cols = ["q_1", "q_3", "q_4", "q_5", "q_6"]
    d = unit_scores[[key_col, "run_id"] + cols].dropna().copy()

    lr1 = LinearRegression().fit(d[["q_1"]], d["q_6"])
    lr2 = LinearRegression().fit(d[["q_1", "q_3"]], d["q_6"])
    lr3 = LinearRegression().fit(d[["q_1", "q_3", "q_4"]], d["q_6"])
    lr4 = LinearRegression().fit(d[["q_1", "q_3", "q_4", "q_5"]], d["q_6"])

    d["h0"] = d.groupby(key_col)["q_6"].transform("mean")
    d["h1"] = np.clip(lr1.predict(d[["q_1"]]), 0.0, 1.0)
    d["h2"] = np.clip(lr2.predict(d[["q_1", "q_3"]]), 0.0, 1.0)
    d["h3"] = np.clip(lr3.predict(d[["q_1", "q_3", "q_4"]]), 0.0, 1.0)
    d["h4"] = np.clip(lr4.predict(d[["q_1", "q_3", "q_4", "q_5"]]), 0.0, 1.0)
    d["h5"] = d["q_6"]

    d["D1"] = d["h1"] - d["h0"]
    d["D2"] = d["h2"] - d["h1"]
    d["D3"] = d["h3"] - d["h2"]
    d["D4"] = d["h4"] - d["h3"]
    d["D5"] = d["h5"] - d["h4"]

    dcols = ["D1", "D2", "D3", "D4", "D5"]
    v_i = (d.groupby(key_col)[dcols]
             .apply(lambda g: (g ** 2).mean().sum()))
    v_i.name = "v_i"
    max_abs_D = float(d[dcols].abs().to_numpy().max())

    return d[[key_col, "run_id"] + dcols], v_i, max_abs_D


def per_run_exceedance_indexed(claims, unit_scores, key_col, taus):
    """Compute per-claim run-level exceedance rates for CB and SSRP."""
    p4 = claims["p4"]
    d = unit_scores[unit_scores[key_col].isin(p4.index)][[key_col, "q_6"]].dropna()
    d = d.merge(p4.rename("p4"), left_on=key_col, right_index=True)
    absdev = (d["q_6"] - d["p4"]).abs()
    out = {t: (absdev > t).groupby(d[key_col]).mean() for t in taus}
    return pd.DataFrame(out)


def broadcast_to_claims_indexed(claims, indexed):
    """Align a claim-indexed result with the claims table."""
    return indexed.loc[claims.index]


def compute_unit_scores_keyed(pred_df, scope, key_col, row_key):
    """Shared CB/SSRP per-run score mapping: identical field-by-field logic,
    differing only in which column of `scope` keys a prediction row's
    original-study reference (key_col) and how a prediction row's own key is
    derived (row_key(row) -> CB's composite claim id or SSRP's plain
    Study Num). RPP does not use this -- it reuses scoring.compute_unit_scores
    directly (see build_claims_and_deviations_rpp)."""
    rec = scope.set_index(key_col)
    rows = []
    for _, r in pred_df.iterrows():
        key = row_key(r)
        rr = rec.loc[key]

        q1 = scoring.match_indicator(
            scoring.to_float(rr.get(REC_N_COL)), scoring.to_float(rr.get(REC_P_COL)),
            scoring.to_float(rr.get(REC_R_COL)),
            scoring.to_float(r.get("extract_N")), scoring.to_float(r.get("extract_p")),
            scoring.to_float(r.get("extract_r")),
        )
        q3 = scoring.to_float(r.get("stat_score"))
        sev = r.get("qrp_severity")
        q4 = (1.0 - float(sev) / 3.0
              if sev is not None and not (isinstance(sev, float) and np.isnan(sev)) else np.nan)
        q5 = scoring.surprise_repl_score(r.get("surprise_rating"))
        q6 = scoring.to_float(r.get("q_hat"))
        vrd = str(r.get("verdict", "")).strip().lower()
        q7 = 1.0 if vrd == "replicable" else (0.0 if vrd == "unreplicable" else np.nan)

        rows.append({
            key_col: key, "run_id": r.get("run_id"),
            "q_1": q1, "q_3": q3, "q_4": q4, "q_5": q5, "q_6": q6, "q_7": q7,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# RPP
# ===========================================================================

def load_rpp_scope():
    """The 90-study RPP subset, keyed by rpp_data_cleaned.csv's "Study Title
    (O)", with each study's Altmejd project row (effect_size.o, n.o, ...)
    attached via normalized-title match and its outcome taken from the
    original RPP OSF export."""
    rpp = pd.read_csv(RPP_CSV, encoding="latin1")
    rpp["norm_title"] = rpp["Study Title (O)"].apply(normalize_title)
    assert not rpp["norm_title"].duplicated().any()

    alt = pd.read_csv(ALTMEJD_CSV)
    alt90 = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
    alt90["norm_title"] = alt90["title"].apply(normalize_title)

    scope = alt90.merge(rpp[["Study Title (O)", "norm_title"]], on="norm_title", how="inner")
    return attach_canonical_rpp_outcomes(scope)


def load_pstar4_rpp(scope):
    """Load one held-out embedding-model probability for each RPP claim."""
    d = pd.read_csv(RPP_EMBED_LR_CSV)
    assert len(d) == len(scope) and (d["title"].values == scope["title"].values).all(), \
        "embed_lr_study_predictions.csv row order no longer matches the Altmejd scope table"
    return d["proba_cv"].values


def per_run_exceedance_rpp(claims, unit_scores, taus):
    """Compute RPP run-level exceedance rates by Altmejd claim ID."""
    p4 = claims["p4"]
    d = unit_scores[unit_scores["altmejd_id"].isin(p4.index)][["altmejd_id", "q_6"]].dropna()
    d = d.merge(p4.rename("p4"), left_on="altmejd_id", right_index=True)
    absdev = (d["q_6"] - d["p4"]).abs()
    out = {t: (absdev > t).groupby(d["altmejd_id"]).mean() for t in taus}
    return pd.DataFrame(out)


def broadcast_to_claims_rpp(claims, title_indexed):
    """Align a claim-id-indexed RPP result with the claims table."""
    return title_indexed.loc[claims.index]


def build_claims_and_deviations_rpp(pred_csv=None):
    """Build RPP claims, chain increments, scores, and bound constants.

    Claims are indexed by Altmejd ID, preserving the 90 study-effect rows.
    The chain is keyed by the stored prediction-run claim ID.
    """
    scope = load_rpp_scope()
    scope["p_star_4"] = load_pstar4_rpp(scope)
    print(f"claims in scope: {len(scope)}/90 (matched to both gpt-5.4-mini predictions "
          f"and the embed_lr nested-CV table; includes both rows of the 2 "
          f"multi-effect papers)")

    rpp = pd.read_csv(RPP_CSV, encoding="latin1")
    pred = pd.read_csv(pred_csv or RPP_PRED_CSV)
    pred = pred[pred["Study Title (O)"].isin(scope["Study Title (O)"])]
    unit_scores = scoring.compute_unit_scores(pred, rpp)

    # Outcomes are shared by effects from the same RPP paper.
    p_star_4_by_title = scope.drop_duplicates("Study Title (O)").set_index("Study Title (O)")["p_star_4"]
    p_star_5_by_title = scope.drop_duplicates("Study Title (O)").set_index("Study Title (O)")["replicated"].astype(float)
    pstar_by_unit = {
        6: p_star_4_by_title.rename("p_star").reset_index(),
        7: p_star_5_by_title.rename("p_star").reset_index(),
    }
    met = llm_metrics.monte_carlo_metrics(unit_scores, pstar_by_unit)

    # Altmejd ID distinguishes effects with a shared title.
    q4_by_claim = met.loc[met.unit == 6].set_index("altmejd_id")["q_bar"]
    assert q4_by_claim.index.is_unique, "Expected one RPP scoring q_bar per altmejd_id"
    chain, v_i_by_claim, max_abs_D = estimate_chain_of_action(unit_scores, "altmejd_id")

    claims = scope.set_index("id")[["Study Title (O)", "p_star_4", "replicated"]].copy()
    claims = claims.rename(columns={"p_star_4": "p4", "replicated": "Y"})
    claims["Y"] = claims["Y"].astype(float)
    claims["q4"] = claims.index.map(q4_by_claim)
    claims["v_i"] = claims.index.map(v_i_by_claim)
    claims = claims.dropna(subset=["q4", "p4", "v_i", "Y"])
    claims["delta"] = (claims["q4"] - claims["p4"]).abs()
    print(f"n = {len(claims)} claims ({claims['Study Title (O)'].nunique()} unique papers)")

    c_hat, C2 = chain_bound_constants()

    return claims, chain, unit_scores, met, c_hat, C2, max_abs_D


# ===========================================================================
# CB
# ===========================================================================
# CB uses one composite claim key per paper, experiment, and effect.

def _claim_key(paper, experiment, effect):
    """Return a string claim ID from paper, experiment, and effect numbers."""
    return f"{int(paper)}_{int(experiment)}_{int(effect)}"


def load_original_stats():
    """Load CB original-study statistics by composite claim ID.

    Internal replications share the original-study statistics, so the first
    row supplies each effect's reference values. Standardized mean differences
    and correlation measures are converted to r; other effect-size measures
    remain missing. The output also retains replication sample size for VEA.
    """
    eff = pd.read_csv(CB_EFF_CSV)
    eff = eff.sort_values(["Paper #", "Experiment #", "Effect #", "Internal replication #"])
    eff = eff.drop_duplicates(subset=["Paper #", "Experiment #", "Effect #"], keep="first")
    eff["claim_id"] = [_claim_key(p, e, f) for p, e, f in
                        zip(eff["Paper #"], eff["Experiment #"], eff["Effect #"])]

    def to_r(row):
        t, v = row["Effect size type"], row["Original effect size"]
        if pd.isna(t) or pd.isna(v):
            return np.nan
        if t in SMD_TYPES:
            return v / np.sqrt(v ** 2 + 4)
        if t in R_TYPES:
            return v
        return np.nan

    eff["orig_r"] = eff.apply(to_r, axis=1)
    out = eff.set_index("claim_id")[
        ["Original sample size", "Original p value", "orig_r", "Replication sample size"]]
    return out.rename(columns={"Original sample size": REC_N_COL,
                                "Original p value": REC_P_COL,
                                "orig_r": REC_R_COL})


def load_scope_cb():
    """Load completed CB effects with their original-study statistics."""
    cb = pd.read_csv(CB_CSV)
    cb["claim_id"] = [_claim_key(p, e, f) for p, e, f in
                       zip(cb["Paper #"], cb["Experiment #"], cb["Effect #"])]
    assert not cb["claim_id"].duplicated().any(), "cb_data_cleaned.csv has a duplicate (Paper #, Experiment #, Effect #)"
    cb = cb.merge(load_original_stats(), left_on="claim_id", right_index=True, how="left")
    return cb


def load_pstar4_cb(scope):
    """Load CB embedding-model reference probabilities by claim ID."""
    d = pd.read_csv(CB_EMBED_LR_CSV)
    d["claim_id"] = [_claim_key(p, e, f) for p, e, f in
                      zip(d["paper_num"], d["experiment_num"], d["effect_num"])]
    d = d.set_index("claim_id")["proba_cv_avg"]
    assert set(scope["claim_id"]) <= set(d.index), \
        "embed_lr_study_predictions.csv is missing some cb_data_cleaned.csv effects"
    return scope["claim_id"].map(d).values


def compute_unit_scores_cb(pred_df, cb_scope):
    """Compute CB run-level unit scores."""
    return compute_unit_scores_keyed(
        pred_df, cb_scope, "claim_id",
        lambda r: _claim_key(r["paper_num"], r["experiment_num"], r["effect_num"]))


def build_claims_and_deviations_cb(pred_csv=None):
    """Build CB claims, chain increments, scores, and bound constants."""
    scope = load_scope_cb().reset_index(drop=True)
    scope["p_star_4"] = load_pstar4_cb(scope)

    pred = pd.read_csv(pred_csv or CB_PRED_CSV)
    unit_scores = compute_unit_scores_cb(pred, scope)

    q4_by_claim = unit_scores.groupby("claim_id")["q_6"].mean()
    chain, v_i_by_claim, max_abs_D = estimate_chain_of_action(unit_scores, "claim_id")

    claims = scope.set_index("claim_id")[
        ["Study Title (O)", "Paper #", "Experiment #", "Effect #", "p_star_4"]].copy()
    claims = claims.rename(columns={"p_star_4": "p4"})
    claims["Y"] = scope.set_index("claim_id")["Replicate (R)"].map({"yes": 1.0, "no": 0.0})
    claims["q4"] = claims.index.map(q4_by_claim)
    claims["v_i"] = claims.index.map(v_i_by_claim)
    n_total = len(claims)
    claims = claims.dropna(subset=["q4", "p4", "v_i", "Y"])
    claims["delta"] = (claims["q4"] - claims["p4"]).abs()
    print(f"n = {len(claims)} claims (CB, {claims['Paper #'].nunique()} papers) "
          f"with a scorable Extraction match, out of {n_total} completed effects")

    c_hat, C2 = chain_bound_constants()

    return claims, chain, unit_scores, c_hat, C2, max_abs_D


# ===========================================================================
# SSRP
# ===========================================================================
# SSRP is indexed by its 21 study numbers.

def load_scope_ssrp():
    return pd.read_csv(SSRP_CSV)


def load_pstar4_ssrp(scope):
    """Load SSRP embedding-model reference probabilities by study number."""
    d = pd.read_csv(SSRP_EMBED_LR_CSV).set_index("study_num")["proba_cv_avg"]
    assert set(scope["Study Num"]) <= set(d.index), \
        "embed_lr_study_predictions.csv is missing some ssrp_data_cleaned.csv studies"
    return scope["Study Num"].map(d).values


def compute_unit_scores_ssrp(pred_df, ssrp):
    """Compute SSRP run-level unit scores."""
    return compute_unit_scores_keyed(pred_df, ssrp, "Study Num", lambda r: r["study_num"])


def build_claims_and_deviations_ssrp(pred_csv=None):
    """Build SSRP claims, chain increments, scores, and bound constants."""
    scope = load_scope_ssrp().reset_index(drop=True)
    scope["p_star_4"] = load_pstar4_ssrp(scope)

    pred = pd.read_csv(pred_csv or SSRP_PRED_CSV)
    unit_scores = compute_unit_scores_ssrp(pred, scope)

    q4_by_study = unit_scores.groupby("Study Num")["q_6"].mean()
    chain, v_i_by_study, max_abs_D = estimate_chain_of_action(unit_scores, "Study Num")

    claims = scope.set_index("Study Num")[["Study Title (O)", "p_star_4"]].copy()
    claims = claims.rename(columns={"p_star_4": "p4"})
    claims["Y"] = scope.set_index("Study Num")["Replicate (R)"].map({"yes": 1.0, "no": 0.0})
    claims["q4"] = claims.index.map(q4_by_study)
    claims["v_i"] = claims.index.map(v_i_by_study)
    claims = claims.dropna(subset=["q4", "p4", "v_i", "Y"])
    claims["delta"] = (claims["q4"] - claims["p4"]).abs()
    print(f"n = {len(claims)} claims (SSRP, 1 per study)")

    c_hat, C2 = chain_bound_constants()

    return claims, chain, unit_scores, c_hat, C2, max_abs_D


# ===========================================================================
# Public, dataset-parameterized dispatchers
# ===========================================================================
# Dataset-dispatch helpers for multi-corpus callers.

_INDEXED_KEY_COL = {"cb": "claim_id", "ssrp": "Study Num"}


def build_claims_and_deviations(dataset, pred_csv=None):
    """Build bounds inputs for the selected dataset."""
    if dataset == "rpp":
        return build_claims_and_deviations_rpp(pred_csv)
    if dataset == "cb":
        return build_claims_and_deviations_cb(pred_csv)
    if dataset == "ssrp":
        return build_claims_and_deviations_ssrp(pred_csv)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def per_run_exceedance(dataset, claims, unit_scores, taus):
    if dataset == "rpp":
        return per_run_exceedance_rpp(claims, unit_scores, taus)
    return per_run_exceedance_indexed(claims, unit_scores, _INDEXED_KEY_COL[dataset], taus)


def broadcast_to_claims(dataset, claims, indexed):
    if dataset == "rpp":
        return broadcast_to_claims_rpp(claims, indexed)
    return broadcast_to_claims_indexed(claims, indexed)
