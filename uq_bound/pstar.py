"""Target p_hat*_{i,m} for the calibration term in metrics.py.

p* is the out-of-sample (5-fold CV) logistic-regression replication probability
on human-rated RPP features — the same construction used in the llm_direct task.
Two predictor sets are supported:

  ``post_replication_similarity``: the legacy specification, which adds the
      replication-rated Effect/Findings similarity fields and is not suitable
      as a deployment proxy because those fields are unavailable ex ante.
  ``original_covariates``: the primary specification, using only fields known
      from the original study. This is the leakage-free deployment proxy.

The ``6`` and ``4`` aliases identify the two feature sets.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from rpp_outcomes import canonical_rpp_outcomes_for_titles  # noqa: E402

# ── Ordinal encodings (from the llm_direct analysis) ─────────────────────────
SIMILARITY_MAP = {
    'Not at all similar': 1, 'Slightly similar': 2, 'Somewhat similar': 3,
    'Moderately similar': 4, 'Very similar': 5, 'Extremely similar': 6,
    'Virtually identical': 7,
}

_BASE_FEATURES = ['Type of effect (O)', 'p_cat_O', 'Surprising result (O)', 'Discipline (O)']

FEATURES_POST_REPLICATION = ['Effect similarity (R)', 'Findings similarity (R)'] + _BASE_FEATURES
FEATURES_ORIGINAL = _BASE_FEATURES
# Backward-compatible aliases for existing scripts; prefer semantic names above.
FEATURES_6 = FEATURES_POST_REPLICATION
FEATURES_4 = FEATURES_ORIGINAL

_FEATURE_MAPS = {
    'Effect similarity (R)': SIMILARITY_MAP,
    'Findings similarity (R)': SIMILARITY_MAP,
}


def _encode_pval_cat(p):
    """Binary p-value strength: 1 if p<.001 (very strong evidence) else 0.

    ``p < .001`` marks very strong evidence.
    """
    if pd.isna(p):
        return np.nan
    return 1 if p < 0.001 else 0


def _encode_effect_type(v):
    """Ordinal: interaction=0 (lowest replication rate), main effect=2, other=1."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s == 'interaction':
        return 0
    if s == 'main effect':
        return 2
    return 1


def _encode_discipline(v):
    """Binary: Social=0 (30% replication rate), Cognitive=1 (53%)."""
    if pd.isna(v):
        return np.nan
    return 1 if str(v).strip().lower() == 'cognitive' else 0


def compute_pstar(rpp, n_predictors=6, seed=2024):
    """Full-dataset logistic-regression replication probability per study.

    Fits a single model on all available studies and returns in-sample predicted
    probabilities as the reference target p*_6 for VEA unit 6 (Scoring).  A
    full-dataset fit is preferred over cross-validated predictions here because
    p*_6 is a reference target, not a performance estimate — we want the best-
    calibrated probability for each study, which uses all the data.

    Returns DataFrame[Study Title (O), p_star]. Studies missing any required
    feature (or the label) are dropped (~2 studies lack T_pval_USE..O.).
    """
    feats = FEATURES_POST_REPLICATION if n_predictors == 6 else FEATURES_ORIGINAL
    raw_cols = ['Effect similarity (R)', 'Findings similarity (R)',
                'Type of effect (O)', 'Surprising result (O)', 'Discipline (O)',
                'T_pval_USE..O.', 'T_r..O.']
    raw_cols = list(dict.fromkeys(raw_cols))
    d = rpp[['Study Title (O)', 'Replicate (R)'] + raw_cols].copy()

    # Derived features
    d['p_cat_O']   = pd.to_numeric(d['T_pval_USE..O.'], errors='coerce').apply(_encode_pval_cat)
    d['r_abs_O']   = pd.to_numeric(d['T_r..O.'],        errors='coerce').abs()
    d['Type of effect (O)'] = d['Type of effect (O)'].apply(_encode_effect_type)
    d['Discipline (O)']     = d['Discipline (O)'].apply(_encode_discipline)

    d['y'] = canonical_rpp_outcomes_for_titles(d['Study Title (O)'])
    for col, mapping in _FEATURE_MAPS.items():
        if col in d.columns:
            d[col] = d[col].map(mapping)
    d = d.dropna(subset=['y'] + feats).reset_index(drop=True)

    X = d[feats].values.astype(float)
    y = d['y'].values.astype(int)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('lr', LogisticRegression(max_iter=500, random_state=seed))])
    pipe.fit(X, y)
    d['p_star'] = pipe.predict_proba(X)[:, 1]
    return d[['Study Title (O)', 'p_star']]


# ── Per-unit reference targets p*_{i,m} ───────────────────────────────────────
# Unit ids match the LLM's 7-step generation order (= Table 2's numbering).
#   1  Extraction          -> 1  (correct extraction is the unique target)
#   3  Statistical review  -> analytic replication power (power_reference.csv)
#   4  Researcher DoF      -> 1 - composite QRP risk from RPP expert codes
#   5  Theory and context  -> 1 - normalised surprisingness (Surprising result (O))
#   6  Scoring             -> logistic baseline p*^LR (5-fold CV on RPP features)
#   7  Verdict             -> observed outcome Y_i in {0,1}
#   8  Effect-type tag     -> 1  (auxiliary diagnostic, not one of Table 2's 7)
# Unit 2 has no RPP reference target. Unit 3 reads this stored power table.
POWER_REF_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "prediction", "LLM_Reasoning", "RPP", "power_reference.csv")
UNIT_TARGET = {
    1: "extraction_unique",
    3: "power_analytic",
    4: "qrp_risk_code",
    5: "context_rating",
    6: "logistic",
    7: "observed_outcome",
    8: "effect_type_unique",
}

# Risk maps for the two RPP expert-coded text columns.
_BIAS_RISK = {
    "no opportunity for researcher expectations to influence results": 0.00,
    "slight opportunity for researcher expectations to influence results": 0.33,
    "moderate opportunity for researcher expectations to influence results": 0.67,
    "strong opportunity for researcher expectations to influence results": 1.00,
}
_DILIG_RISK = {
    "no opportunity for lack of diligence to affect the results": 0.00,
    "slight opportunity for lack of diligence to affect the results": 0.25,
    "moderate opportunity for lack of diligence to affect the results": 0.50,
    "strong opportunity for lack of diligence to affect the results": 0.75,
    "extreme opportunity for lack of diligence to affect the results": 1.00,
}


def _const_reference(rpp, value=1.0):
    d = rpp[["Study Title (O)"]].copy()
    d["p_star"] = float(value)
    return d


def _power_reference(rpp):
    """Replication power parsed from the replication reports (see power_reference.py).
    NaN where the report is missing/unparsed -> those studies drop from unit 3."""
    base = rpp[["Study Title (O)"]].copy()
    if not os.path.exists(POWER_REF_CSV):
        base["p_star"] = np.nan
        return base
    pr = pd.read_csv(POWER_REF_CSV)[["Study Title (O)", "repl_power"]]
    return (base.merge(pr, on="Study Title (O)", how="left")
                .rename(columns={"repl_power": "p_star"})[["Study Title (O)", "p_star"]])


def _outcome_reference(rpp):
    d = rpp[["Study Title (O)"]].copy()
    d["p_star"] = canonical_rpp_outcomes_for_titles(d["Study Title (O)"]).astype(float)
    return d[["Study Title (O)", "p_star"]]


def _qrp_risk_reference(rpp):
    """p* for unit 4 (Researcher DoF concern): 1 - composite QRP risk
    from RPP expert codes.

    Composite = mean of normalised bias risk and diligence risk (both [0,1]).
    Higher p* = lower QRP concern = more replicable prior.
    Studies with both columns missing get NaN and drop from the unit.
    """
    d = rpp[["Study Title (O)",
             "Opportunity for expectancy bias (O)",
             "Opportunity for lack of diligence (O)"]].copy()
    d["bias_risk"] = (d["Opportunity for expectancy bias (O)"]
                      .astype(str).str.strip().str.lower().map(_BIAS_RISK))
    d["dilig_risk"] = (d["Opportunity for lack of diligence (O)"]
                       .astype(str).str.strip().str.lower().map(_DILIG_RISK))
    d["p_star"] = 1.0 - d[["bias_risk", "dilig_risk"]].mean(axis=1)
    return d[["Study Title (O)", "p_star"]]


def _context_rating_reference(rpp):
    """Return the unit-5 reference from the six-level surprisingness rating."""
    d = rpp[["Study Title (O)", "Surprising result (O)"]].copy()
    surp = pd.to_numeric(d["Surprising result (O)"], errors="coerce")
    d["p_star"] = 1.0 - (surp - 1.0) / 5.0
    return d[["Study Title (O)", "p_star"]]


def p_star_target(unit, rpp, n_predictors=6):
    """Return DataFrame[Study Title (O), p_star] giving the reference p*_{i,unit}."""
    spec = UNIT_TARGET.get(unit)
    if spec == "extraction_unique":
        return _const_reference(rpp, 1.0)
    if spec == "power_analytic":
        return _power_reference(rpp)
    if spec == "logistic":
        return compute_pstar(rpp, n_predictors=n_predictors)
    if spec == "observed_outcome":
        return _outcome_reference(rpp)
    if spec == "effect_type_unique":
        return _const_reference(rpp, 1.0)
    if spec == "qrp_risk_code":
        return _qrp_risk_reference(rpp)
    if spec == "context_rating":
        return _context_rating_reference(rpp)
    raise NotImplementedError(f"No reference p* defined for unit {unit} (spec={spec!r}).")
