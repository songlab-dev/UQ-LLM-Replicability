"""VEA decomposition figures for CB and SSRP.

The figures report Extraction, Statistics, Scoring, and Verdict. Their
reference targets are, respectively, a perfect extraction score, analytic
replication power, held-out embedding-model probability, and the observed
outcome. CB original-study statistics come from ``RP_CB_eff.csv``; SSRP
stores them in its cleaned table. Researcher-DoF and Theory/context are not
shown because these corpora lack corresponding reference targets.

Usage: rep_env/bin/python figures/make_vea_figures.py
Writes: figures/{CB,SSRP}/vea_analysis_{high,low}_temp{0.7,0.2}.{png,csv}
        (PNG only, no PDF)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
FIG_DIR = os.path.join(REPO, "figures")
sys.path.insert(0, os.path.join(REPO, "uq_bound"))
import common  # noqa: E402
from power_reference import correlation_power  # noqa: E402


def monte_carlo_metrics_by_claim(unit_scores, id_col, qcols, pstar_by_unit):
    """Same computation as uq_bound/metrics.py's monte_carlo_metrics (q_bar,
    V_hat = run-to-run variance, C_tilde = finite-sample-corrected squared
    bias), generalized to an arbitrary claim-id column instead of that
    function's hardcoded "Study Title (O)" -- CB's claims are per-effect,
    not per-paper, so a title alone isn't a unique key at full scope."""
    pmap = {m: df for m, df in pstar_by_unit.items()}
    out = []
    for claim_id, g in unit_scores.groupby(id_col):
        for m, col in qcols.items():
            vals = g[col].dropna().values
            r = len(vals)
            if r == 0:
                continue
            q_bar = float(vals.mean())
            v_hat = float(vals.var(ddof=1)) if r > 1 else 0.0
            p_star = pmap.get(m, {}).get(claim_id, np.nan)
            c_tilde = (q_bar - p_star) ** 2 - v_hat / r if not np.isnan(p_star) else np.nan
            out.append({id_col: claim_id, "unit": m, "R": r, "q_bar": q_bar,
                        "V_hat": v_hat, "p_star": p_star, "C_tilde": c_tilde})
    return pd.DataFrame(out)


def compute_vea(met, units, labels, nondegen):
    rows = []
    for m, label in zip(units, labels):
        sub = met[met["unit"] == m].dropna(subset=["V_hat", "C_tilde"])
        if sub.empty:
            rows.append(dict(unit=m, label=label, V=0.0, E_raw=0.0, E=0.0, A=0.0, n=0))
            continue
        v = sub["V_hat"].mean()
        e_raw = ((sub["q_bar"] - sub["p_star"]) ** 2).mean()
        e = max(sub["C_tilde"].mean(), 0.0)
        a = (sub["p_star"] * (1 - sub["p_star"])).mean() if m in nondegen else 0.0
        rows.append(dict(unit=m, label=label, V=v, E_raw=e_raw, E=e, A=a, n=len(sub)))
    return pd.DataFrame(rows)


def make_plot(df, title, out_path, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    x = np.arange(len(df))
    w = 0.7
    bw = w / 4
    ax.bar(x - 1.5 * bw, df["V"],     width=bw, color="#4C72B0", label=r"Variance $\hat V$")
    ax.bar(x - 0.5 * bw, df["E_raw"], width=bw, color="#C44E52", label=r"Epistemic $E$ (uncorrected)")
    ax.bar(x + 0.5 * bw, df["E"],     width=bw, color="#DD8452", label=r"Epistemic $\hat E$ (corrected)")
    ax.bar(x + 1.5 * bw, df["A"],     width=bw, color="#8C8C8C", label=r"Aleatoric $A$")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Mean error component")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def run_cb(effort="high", temp="0.7"):
    """effort="low"/temp!="0.7" scores CB's reasoning-effort/temperature
    comparison run instead of the canonical high/temp0.7 run -- built
    directly rather than via common.CB_PRED_CSV (which stays pinned to
    "high"/temp0.7)."""
    scope = common.load_scope_cb().reset_index(drop=True)
    scope["p_star_6"] = common.load_pstar4_cb(scope)
    # Unit 3 (Statistics): analytic power of the replication to detect the
    # FULL original effect size at the replication's actual N -- same
    # correlation_power formula as SSRP's unit 3, fed by common.py's own
    # CB clean-conversion r (REC_R_COL) instead of a directly-reported r column.
    scope["p_star_3"] = scope.apply(
        lambda r: correlation_power(r[common.REC_R_COL], r["Replication sample size"])
        if pd.notna(r[common.REC_R_COL]) and pd.notna(r["Replication sample size"]) else np.nan,
        axis=1)

    pred_csv = os.path.join(REPO, "prediction", "LLM_Reasoning", "CB", f"text_predictions_{effort}_temp{temp}.csv")
    pred = pd.read_csv(pred_csv)
    unit_scores = common.compute_unit_scores_cb(pred, scope)  # now includes q_1

    p1 = {cid: 1.0 for cid in scope["claim_id"]}  # Extraction target: always match
    p3 = scope.set_index("claim_id")["p_star_3"].to_dict()
    p6 = scope.set_index("claim_id")["p_star_6"].to_dict()
    p7 = scope.set_index("claim_id")["Replicate (R)"].map({"yes": 1.0, "no": 0.0}).to_dict()
    pstar_by_unit = {1: p1, 3: p3, 6: p6, 7: p7}

    qcols = {1: "q_1", 3: "q_3", 4: "q_4", 5: "q_5", 6: "q_6", 7: "q_7"}
    met = monte_carlo_metrics_by_claim(unit_scores, "claim_id", qcols, pstar_by_unit)

    units, labels = [1, 3, 6, 7], ["Extraction", "Statistics", "Scoring", "Verdict"]
    df = compute_vea(met, units, labels, nondegen={3, 6})
    n_r = scope[common.REC_R_COL].notna().sum()
    print(f"\nCB VEA, {effort} effort, temp={temp} (n={scope.shape[0]} effects; Statistics scored on the "
          f"{n_r} effects with a clean-convertible original effect size; "
          f"Researcher DoF/Theory-context omitted -- no portable reference "
          f"target, see module docstring)")
    print(df[["label", "n", "V", "E_raw", "E", "A"]].to_string(index=False, float_format="{:.4f}".format))
    return df


def save_cb(df, effort, temp="0.7", ylim=None):
    suffix = f"{effort}_temp{temp}"
    out_png = os.path.join(FIG_DIR, "CB", f"vea_analysis_{suffix}.png")
    make_plot(df, f"CB VEA Decomposition ({effort} effort)\n"
                  f"(Researcher DoF/Theory-context: no portable p*)", out_png, ylim=ylim)
    df.to_csv(os.path.join(FIG_DIR, "CB", f"vea_analysis_{suffix}.csv"), index=False)


def run_ssrp(effort="high", temp="0.7"):
    """effort="low"/temp!="0.7" scores a reasoning-effort/temperature
    comparison run (see llm_uq_table_ssrp_effort_comparison.py) instead of
    the canonical high/temp0.7 run -- built directly rather than via
    common.SSRP_PRED_CSV (which stays pinned to "high"/temp0.7, the
    canonical config every other SSRP script reads)."""
    scope = common.load_scope_ssrp().reset_index(drop=True)
    scope["p_star_6"] = common.load_pstar4_ssrp(scope)
    # Unit 3 (Statistics): analytic power of the replication to detect the
    # FULL original effect size at the replication's actual N -- power_reference.py's
    # exact Fisher-z definition, computed from SSRP's own clean columns
    # instead of parsed PDF text. NOT the same as "Power (R)" -- see module docstring.
    scope["p_star_3"] = scope.apply(
        lambda r: correlation_power(r["Effect size r (O)"], r["N (R)"]), axis=1)
    pred_csv = os.path.join(REPO, "prediction", "LLM_Reasoning", "SSRP", f"text_predictions_{effort}_temp{temp}.csv")
    pred = pd.read_csv(pred_csv)
    unit_scores = common.compute_unit_scores_ssrp(pred, scope)

    p1 = {sn: 1.0 for sn in scope["Study Num"]}
    p3 = scope.set_index("Study Num")["p_star_3"].to_dict()
    p6 = scope.set_index("Study Num")["p_star_6"].to_dict()
    p7 = scope.set_index("Study Num")["Replicate (R)"].map({"yes": 1.0, "no": 0.0}).to_dict()
    pstar_by_unit = {1: p1, 3: p3, 6: p6, 7: p7}

    qcols = {1: "q_1", 3: "q_3", 4: "q_4", 5: "q_5", 6: "q_6", 7: "q_7"}
    met = monte_carlo_metrics_by_claim(unit_scores, "Study Num", qcols, pstar_by_unit)

    units, labels = [1, 3, 6, 7], ["Extraction", "Statistics", "Scoring", "Verdict"]
    df = compute_vea(met, units, labels, nondegen={3, 6})
    print(f"\nSSRP VEA, {effort} effort, temp={temp} (n={scope.shape[0]} studies; Researcher "
          f"DoF/Theory-context omitted -- no portable reference target, see "
          f"module docstring)")
    print(df[["label", "n", "V", "E_raw", "E", "A"]].to_string(index=False, float_format="{:.4f}".format))
    return df


def save_ssrp(df, effort, temp="0.7", ylim=None):
    suffix = f"{effort}_temp{temp}"
    out_png = os.path.join(FIG_DIR, "SSRP", f"vea_analysis_{suffix}.png")
    make_plot(df, f"SSRP VEA Decomposition ({effort} effort)\n"
                  f"(Researcher DoF/Theory-context: no portable p*)", out_png, ylim=ylim)
    df.to_csv(os.path.join(FIG_DIR, "SSRP", f"vea_analysis_{suffix}.csv"), index=False)


if __name__ == "__main__":
    # Compute both effort variants before plotting either, per corpus, so
    # each corpus's high/low pair shares one y-axis scale (see
    # make_vea_score_comparison_figures.py's identical rationale for RPP's VEA_RUNS).
    cb_dfs = {(effort, temp): run_cb(effort, temp)
              for effort, temp in (("high", "0.7"), ("low", "0.7"),
                                    ("high", "0.2"), ("low", "0.2"))}
    cb_ylim = 1.05 * max(
        df[col].max() for df in cb_dfs.values() for col in ("V", "E_raw", "E", "A"))
    for (effort, temp), df in cb_dfs.items():
        save_cb(df, effort, temp, ylim=cb_ylim)

    ssrp_dfs = {(effort, temp): run_ssrp(effort, temp)
                for effort, temp in (("high", "0.7"), ("low", "0.7"),
                                      ("high", "0.2"), ("low", "0.2"))}
    ssrp_ylim = 1.05 * max(
        df[col].max() for df in ssrp_dfs.values() for col in ("V", "E_raw", "E", "A"))
    for (effort, temp), df in ssrp_dfs.items():
        save_ssrp(df, effort, temp, ylim=ssrp_ylim)
