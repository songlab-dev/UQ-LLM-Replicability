"""Per-unit scores q_hat_{i,m}^{(r)} in [0,1] for the llm_uq pipeline.

Unit ids match the LLM's own 7-step generation order, which is also Table 2's
numbering in the paper -- unit m IS prompt step m, no translation needed:
  1.  Extraction        — match indicator of extracted (N, p, d) vs the RPP record.
  2.  Credibility       — SKIPPED: narrative-only in the prompt (no numeric
      field), and RPP's 2008-2011 studies predate preregistration/open-data
      norms, so there is no reliable ground truth to score against.
  3.  Statistical review — the LLM's reported statistical-soundness score in [0,1].
  4.  Researcher DoF concern — 1 - (qrp_severity / 3); higher = fewer QRP concerns.
  5.  Theory and context — 1 - (surprise_rating - 1) / 4, matching
      pstar._context_rating_reference's scale so q_5 and p*_5 compare the
      same quantity: SURPRISE_RATING against RPP's Surprising result (O).
  6.  Scoring           — the LLM's predicted replication probability q_hat.
  7.  Verdict           — I{q_hat > 0.5}.

  8.  Effect-type tag   — 1 if extracted type matches RPP 'Type of effect (O)',
      else 0. AUXILIARY (first id past the 7 prompt steps): Table 2 folds
      EFFECT_TYPE into unit 1's extraction match, but this pipeline scores it
      separately as a diagnostic. Not part of the paper's 7 units.

The output is a long table with one row per (study, run) with columns
q_1, q_3, q_4, q_5, q_6, q_7, q_8, consumed by metrics.py.
"""
import re
import numpy as np
import pandas as pd

# Columns of the original (O) RPP record used as extraction ground truth.
REC_N_COL   = "N (O)"
REC_P_COL   = "Reported P-value (O)"
REC_D_COL   = "Effect size (O)"   # legacy: mixed metrics (eta^2, dz, ...); predict_pdf path
REC_R_COL   = "T_r..O."           # clean correlation r; preferred extraction target
REC_TYPE_COL = "Type of effect (O)"

# Match tolerances for unit 1 (configurable; user may tune).
P_ABS_TOL = 0.01    # p-value: |extracted - record| <= P_ABS_TOL
D_REL_TOL = 0.15    # effect size: relative difference <= D_REL_TOL

# Canonical QRP concerns the LLM is asked to flag (Step 4).
# Each tuple holds alternative phrasings; a flag is counted if any phrase matches.
_QRP_KEYWORDS = [
    ("multiplicity",),
    ("optional stopping", "optional-stopping"),
    ("flexible covariate", "flexible analysis", "flexible model"),
    ("p near", "p just", "p<.05", "just significant", "just under"),
    ("no preregistration", "not preregistered", "unregistered"),
    ("no open data", "data not available", "materials not available", "no open material"),
]
N_QRP_FLAGS = len(_QRP_KEYWORDS)

QCOLS = {1: "q_1", 3: "q_3", 4: "q_4", 5: "q_5", 6: "q_6", 7: "q_7",
         8: "q_8"}

# Public labels for claim-level outputs. Numeric prompt positions remain an
# internal implementation detail of the long-format score table.
STAGES = {
    1: ("extraction", "Extraction", 1),
    3: ("statistical_review", "Statistical review", 2),
    4: ("researcher_dof", "Researcher degrees of freedom", 3),
    5: ("theory_context", "Theory and context", 4),
    6: ("scoring", "Scoring", 5),
    7: ("verdict", "Verdict", 6),
    8: ("effect_type_diagnostic", "Effect-type diagnostic", 7),
}


def to_float(x):
    """Best-effort numeric parse; strips comparators/units, NA-likes -> nan.

    Handles RPP strings such as '<0.005', 'dz = 0.51', '__p^2 = 0.354', 'X'.
    """
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if s in ("", "x", "na", "nan", "none", "not reported"):
        return np.nan
    s = s.replace("<", "").replace(">", "")
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    return float(m.group()) if m else np.nan


def _norm_effect_type(v):
    """Normalise RPP or LLM effect-type string to 'main' | 'interaction' | 'other'."""
    if not v:
        return ""
    s = str(v).strip().lower()
    if s in ("main", "main effect"):
        return "main"
    if s == "interaction":
        return "interaction"
    if s:
        return "other"
    return ""


def match_effect_type(rec_type, ext_type):
    """1 if extracted effect type matches record, 0 if mismatch, nan if either missing."""
    r = _norm_effect_type(rec_type)
    e = _norm_effect_type(ext_type)
    if not r or not e:
        return np.nan
    return float(r == e)


def qrp_repl_score(flags_str):
    """Replicability signal from QRP flags: 1 - (n_matched_flags / N_QRP_FLAGS).

    'none' or empty -> 1.0 (no concerns flagged).  Missing/NA -> nan.
    """
    if flags_str is None or str(flags_str).strip() == "":
        return np.nan
    s = str(flags_str).strip().lower()
    if s in ("na", "nan"):
        return np.nan
    if s == "none":
        return 1.0
    n = sum(any(kw in s for kw in kws) for kws in _QRP_KEYWORDS)
    return 1.0 - n / N_QRP_FLAGS


def surprise_repl_score(rating):
    """Replicability signal from the LLM's own SURPRISE_RATING (1-5): rescaled to
    [0,1] on the same convention as pstar._context_rating_reference (1=not
    surprising -> 1, 5=very surprising -> 0), so q_5 and p*_5 compare the
    same quantity."""
    r = to_float(rating)
    if np.isnan(r):
        return np.nan
    return 1.0 - (r - 1.0) / 4.0


def match_indicator(rec_N, rec_p, rec_d, ext_N, ext_p, ext_d):
    """Fraction of available (N, p, d) fields where the extraction matches the record.

    Returns a value in [0, 1] averaged over the fields for which both the record
    and the extraction are present; nan if no field is comparable.
    """
    fields = []
    if not np.isnan(rec_N) and not np.isnan(ext_N):
        fields.append(float(int(round(ext_N)) == int(round(rec_N))))
    if not np.isnan(rec_p) and not np.isnan(ext_p):
        fields.append(float(abs(ext_p - rec_p) <= P_ABS_TOL))
    if not np.isnan(rec_d) and not np.isnan(ext_d):
        denom = abs(rec_d) if abs(rec_d) > 1e-9 else 1.0
        fields.append(float(abs(ext_d - rec_d) / denom <= D_REL_TOL))
    if not fields:
        return np.nan
    return float(np.mean(fields))


def compute_unit_scores(pred_df, rpp):
    """Map raw per-run predictions to per-unit scores q_hat_{i,m}^{(r)}.

    pred_df: rows from predict_*.py (one per study x run) with columns
             extract_N, extract_p, extract_r (or legacy extract_d), stat_score, q_hat.
    rpp:     data/rpp_data_cleaned.csv loaded as a DataFrame.
    """
    rec = rpp.set_index("Study Title (O)")
    rows = []
    for _, r in pred_df.iterrows():
        title = r["Study Title (O)"]
        rr = rec.loc[title]
        rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr

        # Prefer the correlation-r extraction (extract_r vs T_r..O.); fall back to the
        # legacy mixed-metric column (extract_d vs Effect size (O)) for the pdf pipeline.
        if "extract_r" in r and not np.isnan(to_float(r.get("extract_r"))):
            rec_es, ext_es = to_float(rr.get(REC_R_COL)), to_float(r.get("extract_r"))
        else:
            rec_es, ext_es = to_float(rr.get(REC_D_COL)), to_float(r.get("extract_d"))
        q1 = match_indicator(
            to_float(rr.get(REC_N_COL)), to_float(rr.get(REC_P_COL)), rec_es,
            to_float(r.get("extract_N")), to_float(r.get("extract_p")), ext_es,
        )
        q3 = to_float(r.get("stat_score"))
        # qrp_severity (0-3 int) takes priority; fall back to legacy qrp_flags string
        sev = r.get("qrp_severity")
        if sev is not None and not (isinstance(sev, float) and np.isnan(sev)):
            q4 = 1.0 - float(sev) / 3.0
        else:
            q4 = qrp_repl_score(r.get("qrp_flags"))
        q5 = surprise_repl_score(r.get("surprise_rating"))
        q6 = to_float(r.get("q_hat"))
        vrd = str(r.get("verdict", "")).strip().lower()
        q7 = 1.0 if vrd == "replicable" else (0.0 if vrd == "unreplicable" else np.nan)
        q8 = match_effect_type(rr.get(REC_TYPE_COL), r.get("effect_type"))

        rows.append({
            "altmejd_id": r.get("altmejd_id"),
            "Study Title (O)": title,
            "Study Number (R)": r.get("Study Number (R)"),
            "run_id": r.get("run_id"),
            "q_1": q1, "q_3": q3, "q_4": q4, "q_5": q5,
            "q_6": q6, "q_7": q7, "q_8": q8,
        })
    return pd.DataFrame(rows)
