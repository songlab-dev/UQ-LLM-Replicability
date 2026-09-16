"""Monte-Carlo aggregation of per-stage scores into q_bar, V_hat, C_tilde.

For claim i, prompt stage m, over R Monte-Carlo runs r:

    q_bar_{i,m}  = R^-1 * sum_r q_hat_{i,m}^{(r)}
    V_hat_{i,m}  = (R-1)^-1 * sum_r (q_hat_{i,m}^{(r)} - q_bar_{i,m})^2
    C_tilde_{i,m} = (q_bar_{i,m} - p_star_{i,m})^2 - V_hat_{i,m} / R

C_tilde is the variance-corrected squared bias of unit m's mean score relative
to the target p_star_{i,m} (an unbiased estimate of the true squared bias).

Run from the repo root:  python llm_uq/metrics.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring     # noqa: E402
import pstar       # noqa: E402
import validation  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))  # llm_uq/uq_bound
_REPO = os.path.dirname(_HERE)  # llm_uq/
RPP_CSV = os.path.join(_REPO, "data", "rpp_data_cleaned.csv")
_RPP_DIR = os.path.join(_REPO, "prediction", "LLM_Reasoning", "RPP")
# Defaults point to the canonical RPP text-prediction run.
PRED_CSV = os.getenv("PRED_CSV",
                     os.path.join(_RPP_DIR, "text_predictions_high_temp0.7.csv"))
OUT_DIR = os.getenv("OUT_DIR", _RPP_DIR)

TAU = 0.2  # total per-claim audit budget


def monte_carlo_metrics(unit_scores, pstar_by_unit):
    """Aggregate per-run scores into q_bar, V_hat, C_tilde per (claim, stage).

    unit_scores:    long table from scoring.compute_unit_scores (cols q_1,q_3,...,q_8).
    pstar_by_unit:  {unit: DataFrame[Study Title (O), p_star]}.
    """
    pmap = {
        m: df.set_index("Study Title (O)")["p_star"].to_dict()
        for m, df in pstar_by_unit.items()
    }
    out = []
    for (claim_id, title), g in unit_scores.groupby(["altmejd_id", "Study Title (O)"]):
        for m, col in scoring.QCOLS.items():
            vals = g[col].dropna().values
            r = len(vals)
            if r == 0:
                continue
            q_bar = float(vals.mean())
            v_hat = float(vals.var(ddof=1)) if r > 1 else 0.0
            p_star = pmap.get(m, {}).get(title, np.nan)
            c_tilde = (q_bar - p_star) ** 2 - v_hat / r if not np.isnan(p_star) else np.nan
            out.append({
                "altmejd_id": claim_id,
                "Study Title (O)": title,
                "unit": m,
                "R": r,
                "q_bar": q_bar,
                "V_hat": v_hat,
                "p_star": p_star,
                "C_tilde": c_tilde,
            })
    return pd.DataFrame(out).sort_values(["altmejd_id", "unit"]).reset_index(drop=True)


def tau_star_and_ub(metrics_df, tau=TAU):
    """Optimal per-unit budget allocation tau*_{i,m} and the per-claim upper bound UB_i.

    With a_{i,m} = max(V_hat_{i,m} + C_tilde_{i,m}, 0)  (C_tilde estimates a
    non-negative squared bias, so negatives are clamped):

        tau*_{i,m}  = tau * a_{i,m}^{1/3} / sum_l a_{i,l}^{1/3}
        UB_i(tau)   = min(1, sum_m (V_hat_{i,m}+C_tilde_{i,m}) / tau*_{i,m}^2)
                    = min(1, (sum_m a_{i,m}^{1/3})^3 / tau^2)   [closed form]

    Units with no valid target p* (C_tilde is NaN) are excluded from the sum over m.
    Returns internal numeric-stage allocations and per-claim bounds. Public
    ``tau_star`` outputs are migrated by ``claim_stage_output`` to stable
    claim ids and named stages before they are written.
    """
    alloc, ub_rows = [], []
    for (claim_id, title), g in metrics_df.groupby(["altmejd_id", "Study Title (O)"]):
        gg = g.dropna(subset=["C_tilde"])
        if gg.empty:
            ub_rows.append({"altmejd_id": claim_id, "Study Title (O)": title,
                            "M_units": 0, "UB": np.nan, "UB_raw": np.nan})
            continue
        a = np.clip(gg["V_hat"].values + gg["C_tilde"].values, 0.0, None)
        cbrt = np.cbrt(a)
        S = cbrt.sum()
        if S > 0:
            tau_star = tau * cbrt / S
            ub_raw = (S ** 3) / (tau ** 2)  # = sum_m (V+C)/tau*_{i,m}^2 (unclamped)
            ub = min(1.0, ub_raw)
        else:  # zero variance and bias for every unit
            tau_star = np.zeros_like(cbrt)
            ub_raw = ub = 0.0
        for u, ts, av in zip(gg["unit"].values, tau_star, a):
            alloc.append({"altmejd_id": claim_id, "Study Title (O)": title, "unit": int(u),
                          "V_plus_C": float(av), "tau_star": float(ts)})
        ub_rows.append({"altmejd_id": claim_id, "Study Title (O)": title, "M_units": int(len(a)),
                        "UB": float(ub), "UB_raw": float(ub_raw)})
    return pd.DataFrame(alloc), pd.DataFrame(ub_rows)


def claim_summary(metrics_df, ub_df, rpp, n_predictors):
    """Per-claim overview tying the scoring prediction to its reference and the bound.

    Columns: q_hat_i (scoring unit mean), p_star_i (logistic baseline p*_{i,6}),
    q_minus_pstar (calibration gap), Y_i (observed outcome), UB / UB_raw,
    innateness = p_star_i * (1 - p_star_i) (irreducible problem difficulty;
    the squared-loss Bayes risk for a Bernoulli(p*) outcome).
    """
    q_hat = (metrics_df[metrics_df["unit"] == 6][["altmejd_id", "Study Title (O)", "q_bar"]]
             .rename(columns={"q_bar": "q_hat_i"}))
    pstar6 = pstar.p_star_target(6, rpp, n_predictors).rename(columns={"p_star": "p_star_i"})
    y = pstar.p_star_target(7, rpp).rename(columns={"p_star": "Y_i"})
    s = (q_hat.merge(pstar6, on="Study Title (O)", how="left")
              .merge(y, on="Study Title (O)", how="left")
              .merge(ub_df[["altmejd_id", "UB", "UB_raw"]], on="altmejd_id", how="left"))
    s["q_minus_pstar"] = s["q_hat_i"] - s["p_star_i"]
    s["innateness"] = s["p_star_i"] * (1.0 - s["p_star_i"])
    return s


def claim_stage_output(frame, claim_keys):
    """Replace legacy numeric unit identifiers in public claim outputs."""
    stage = pd.DataFrame.from_dict(
        scoring.STAGES, orient="index", columns=["stage_id", "stage_label", "stage_order"]
    ).rename_axis("unit").reset_index()
    result = frame.merge(stage, on="unit", how="left", validate="many_to_one")
    result = result.merge(claim_keys, on="altmejd_id", how="left", validate="many_to_one",
                          suffixes=("", "_key"))
    if "Study Title (O)_key" in result:
        result = result.drop(columns=["Study Title (O)_key"])
    if result[["altmejd_id", "stage_id"]].isna().any().any():
        raise ValueError("Claim/stage migration left unmapped output rows")
    result.insert(0, "corpus", "RPP")
    result = result.drop(columns=["unit"])
    leading = ["corpus", "altmejd_id", "Study Title (O)", "stage_id", "stage_label", "stage_order"]
    return result[leading + [column for column in result if column not in leading]]


def main():
    rpp = pd.read_csv(RPP_CSV, encoding="latin1")
    pred = pd.read_csv(PRED_CSV)

    unit_scores = scoring.compute_unit_scores(pred, rpp)
    unit_scores.to_csv(f"{OUT_DIR}/unit_scores.csv", index=False)
    print(f"Saved per-run unit scores → {OUT_DIR}/unit_scores.csv  ({len(unit_scores)} rows)")

    for n_pred in (4, 6):
        pstar_by_unit = {m: pstar.p_star_target(m, rpp, n_predictors=n_pred)
                         for m in scoring.QCOLS}
        metrics = monte_carlo_metrics(unit_scores, pstar_by_unit)
        out_path = f"{OUT_DIR}/uq_metrics_{n_pred}pred.csv"
        metrics.to_csv(out_path, index=False)
        print(f"Saved metrics ({n_pred}-predictor p*) → {out_path}  ({len(metrics)} rows)")

        claim_keys = unit_scores[["altmejd_id", "Study Title (O)"]].drop_duplicates()
        if claim_keys["altmejd_id"].isna().any() or claim_keys["altmejd_id"].duplicated().any():
            raise ValueError("Expected one non-missing altmejd_id per RPP claim")

        alloc, ub = tau_star_and_ub(metrics, tau=TAU)
        alloc = claim_stage_output(alloc, claim_keys)
        alloc.to_csv(f"{OUT_DIR}/tau_star_{n_pred}pred.csv", index=False)
        ub.to_csv(f"{OUT_DIR}/upper_bound_{n_pred}pred.csv", index=False)
        print(f"  tau*_{{i,m}} (tau={TAU}) → {OUT_DIR}/tau_star_{n_pred}pred.csv;  "
              f"UB_i → {OUT_DIR}/upper_bound_{n_pred}pred.csv  "
              f"(mean UB={ub['UB'].mean():.3f})")

        # ── Validation + diagnostics ─────────────────────────────────────────
        val = validation.validation_summary(metrics, ub, pred, TAU)
        pd.DataFrame([val]).to_csv(f"{OUT_DIR}/validation_{n_pred}pred.csv", index=False)
        attribution = claim_stage_output(validation.unit_attribution(metrics), claim_keys)
        attribution.to_csv(f"{OUT_DIR}/unit_attribution_{n_pred}pred.csv", index=False)
        validation.variability_confidence_ratio(metrics).to_csv(
            f"{OUT_DIR}/vc_ratio_{n_pred}pred.csv", index=False)
        print(f"  validation: mean_UB={val['mean_UB']:.3f}  F_hat={val['F_hat']:.3f}  "
              f"slack={val['slack']:.3f}  (n={val['n_claims']})  → "
              f"validation/unit_attribution/vc_ratio _{n_pred}pred.csv")

        cs = claim_summary(metrics, ub, rpp, n_pred)
        cs.to_csv(f"{OUT_DIR}/claim_summary_{n_pred}pred.csv", index=False)
        print(f"  claim summary (q_hat_i, p*_i, UB, innateness) → "
              f"{OUT_DIR}/claim_summary_{n_pred}pred.csv  "
              f"(mean innateness={cs['innateness'].mean():.3f})")


if __name__ == "__main__":
    main()
