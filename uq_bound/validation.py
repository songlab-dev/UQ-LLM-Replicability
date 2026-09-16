"""Validation of the upper bound and the two diagnostic outputs.

Validation (spec step 5): compare the mean per-claim upper bound
    100^-1 * sum_i UB_i(tau)
against the empirical task-failure rate
    F_hat(tau) = 100^-1 * sum_i  I{ |q_hat_i - Y_i| > tau },
where q_hat_i is the per-claim predicted replication probability (mean q_hat of
the Scoring unit) and Y_i is the binary replication outcome. The slack
(mean UB - F_hat) flags units whose V_hat+C_tilde dominates.

Diagnostics derived from the same estimators:
  * unit-attribution profile  (V_hat_{i,m} + C_tilde_{i,m})_m  — ranks units by
    contribution to the task-level upper bound.
  * variability-confidence ratio  sum_m V_hat_{i,m} / sum_m (V_hat_{i,m}+C_tilde_{i,m})
    — separates instability (repeat runs) from systematic confidence error
    (recalibrate, e.g. isotonic regression on unit outputs).
"""
import numpy as np
import pandas as pd

SCORING_UNIT = 6  # 'Scoring' unit; its per-claim mean q_bar is q_hat_i
CLAIM_ID = "altmejd_id"


def task_level_qy(metrics_df, pred_df):
    """Per-claim predicted replication probability q_hat_i and binary truth Y_i."""
    q = (metrics_df[metrics_df["unit"] == SCORING_UNIT]
         [[CLAIM_ID, "Study Title (O)", "q_bar"]].rename(columns={"q_bar": "q_hat_i"}))
    y = (pred_df.groupby(CLAIM_ID)["ground_truth"].first()
         .map(lambda s: 1.0 if str(s).strip().lower() == "yes" else 0.0)
         .rename("Y_i").reset_index())
    return q.merge(y, on=CLAIM_ID, how="inner")


def validation_summary(metrics_df, ub_df, pred_df, tau):
    """Mean upper bound vs empirical failure rate F_hat(tau); slack is the gap."""
    qy = task_level_qy(metrics_df, pred_df).dropna(subset=["q_hat_i", "Y_i"])
    f_hat = float((np.abs(qy["q_hat_i"] - qy["Y_i"]) > tau).mean()) if len(qy) else np.nan
    mean_ub = float(ub_df["UB"].dropna().mean()) if ub_df["UB"].notna().any() else np.nan
    return {"tau": tau, "n_claims": int(len(qy)),
            "mean_UB": mean_ub, "F_hat": f_hat, "slack": mean_ub - f_hat}


def unit_attribution(metrics_df):
    """Unit-attribution profile a_{i,m}=V_hat+C_tilde, ranked within each claim
    (rank 1 = largest contributor to that claim's upper bound)."""
    d = metrics_df.copy()
    d["attribution"] = d["V_hat"] + d["C_tilde"]
    d["rank"] = (d.groupby(CLAIM_ID)["attribution"]
                 .rank(ascending=False, method="min").astype("Int64"))
    return d[[CLAIM_ID, "Study Title (O)", "unit", "V_hat", "C_tilde", "attribution", "rank"]]


def variability_confidence_ratio(metrics_df):
    """Per-claim sum_m V_hat / sum_m (V_hat+C_tilde): ~1 -> instability (repeat),
    ~0 -> systematic confidence error (recalibrate)."""
    rows = []
    for (claim_id, title), g in metrics_df.dropna(subset=["C_tilde"]).groupby(
        [CLAIM_ID, "Study Title (O)"]
    ):
        s_v = float(g["V_hat"].sum())
        s_vc = float((g["V_hat"] + g["C_tilde"]).sum())
        rows.append({CLAIM_ID: claim_id, "Study Title (O)": title, "sum_V": s_v, "sum_VC": s_vc,
                     "vc_ratio": s_v / s_vc if s_vc > 0 else np.nan})
    return pd.DataFrame(rows)
