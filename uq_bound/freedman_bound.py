"""Freedman bounds for study-level prediction-reference deviations.

    Pr(|p_hat_i - p*_i| > tau) <= 2 exp( -u_i^2 / (2(v_i + (1/3) c_i u_i)) ),
    u_i = (tau - delta_i)_+

``c_i=1`` matches the unit range used by AH. ``v_i`` is the sum of mean
squared chain increments for each claim across the five scored stages.

Usage: rep_env/bin/python uq_bound/freedman_bound.py
Writes: uq_bound/results/freedman_bound.csv (per-study, all tau columns)
        uq_bound/results/freedman_bound_summary.csv (mean UB vs AH/F_hat per tau)
        uq_bound/figures/freedman_vs_ah.pdf
"""
import os

import numpy as np
import pandas as pd

import common
from ah_bound import ub_ah, f_hat

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "results", "freedman_bound.csv")
OUT_SUMMARY_CSV = os.path.join(HERE, "results", "freedman_bound_summary.csv")
OUT_FIG = os.path.join(HERE, "figures", "freedman_vs_ah.pdf")
TAUS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]


def ub_freedman(tau, delta, v, c=1.0):
    """delta, v: per-claim arrays/Series aligned by position (v_i is a
    per-claim quantity -- see module docstring -- not a pooled scalar)."""
    d = delta.values if hasattr(delta, "values") else np.asarray(delta)
    v_ = v.values if hasattr(v, "values") else np.asarray(v)
    u = np.maximum(tau - d, 0)
    denom = 2.0 * (v_ + c * u / 3.0)
    denom = np.where(denom > 0, denom, 1.0)  # guard against division by zero
    return np.minimum(1.0, 2.0 * np.exp(-u ** 2 / denom))


def main():
    claims, chain, unit_scores, met, c_hat, C2, max_abs_D = common.build_claims_and_deviations("rpp")
    n = len(claims)
    B = common.broadcast_to_claims("rpp", claims, common.per_run_exceedance("rpp", claims, unit_scores, TAUS))
    print(f"n = {n} claims")
    print(f"per-claim v_i = sum_{{s=1}}^5 mean_r[D_i,s^2] (task-level, full chain): "
          f"mean={claims.v_i.mean():.4f}  median={claims.v_i.median():.4f}  "
          f"max={claims.v_i.max():.4f}")
    print(f"(vs. AH's fixed budget sum_m c_i,m^2 = {C2:.4f})\n")

    rows = []
    print("Task-level (Scoring delta_i, that claim's own v_i over the full 5-step chain):")
    print(f"{'tau':>6}{'UB_F':>9}{'UB_AH':>9}{'F_hat':>9}")
    for t in TAUS:
        uf = ub_freedman(t, claims["delta"], claims["v_i"]).mean()
        ua = ub_ah(t, claims["delta"], C2).mean()
        f = f_hat(t, B)
        print(f"{t:>6.2f}{uf:>9.3f}{ua:>9.3f}{f:>9.3f}")
        rows.append({"tau": t, "mean_UB_Freedman": uf, "mean_UB_AH": ua, "F_hat": f})

    per_claim = claims.copy()
    for t in TAUS:
        per_claim[f"UB_F_tau{t}"] = ub_freedman(t, claims["delta"], claims["v_i"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    per_claim.to_csv(OUT_CSV)
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY_CSV}")

    # ── plot: task-level Freedman vs AH vs empirical failure ─────────────────
    import matplotlib.pyplot as plt

    taus_fine = np.linspace(0.02, 1.0, 60)
    B_fine = common.broadcast_to_claims("rpp", claims, common.per_run_exceedance("rpp", claims, unit_scores, taus_fine))
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    f_curve = [ub_freedman(t, claims["delta"], claims["v_i"]).mean() for t in taus_fine]
    ah_curve = [ub_ah(t, claims["delta"], C2).mean() for t in taus_fine]
    fhat_curve = [f_hat(t, B_fine) for t in taus_fine]
    ax.plot(taus_fine, f_curve, lw=2.2, color="#4C72B0", label=r"mean $\widehat{UB}^F(\tau)$ (Freedman)")
    ax.plot(taus_fine, ah_curve, lw=2.2, ls="--", color="#C44E52", label=r"mean $\widehat{UB}^{AH}(\tau)$")
    ax.plot(taus_fine, fhat_curve, lw=1.6, color="grey", ls=":", label=r"$\hat F(\tau)$ empirical failure")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Task-level bounds vs. empirical failure", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
