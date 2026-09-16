"""Azuma--Hoeffding bounds for run-level prediction-reference deviations.

For tolerance ``tau`` and claim offset ``delta_i``, the script evaluates
``min(1, 2 exp(-(tau-delta_i)^2 / (2 C2)))`` when ``tau > delta_i`` and one
otherwise. ``C2=5`` follows from five unit-bounded chain increments. The
empirical comparison is the matching run-level exceedance rate.

Usage: rep_env/bin/python uq_bound/ah_bound.py
Writes: uq_bound/results/ah_bound.csv (per-claim, all tau columns)
        uq_bound/results/ah_bound_summary.csv (mean UB vs F_hat per tau)
"""
import os

import numpy as np
import pandas as pd

import common

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "results", "ah_bound.csv")
OUT_SUMMARY_CSV = os.path.join(HERE, "results", "ah_bound_summary.csv")
TAUS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]


def ub_ah(tau, delta, C2):
    d = delta.values
    out = np.ones(len(d))
    ok = tau > d
    out[ok] = np.minimum(1.0, 2 * np.exp(-((tau - d[ok]) ** 2) / (2 * C2)))
    return out


def f_hat(tau, B):
    """B: common.per_run_exceedance("rpp", claims, unit_scores, taus) table (claims
    x taus). Matched-event empirical rate F_proxy(tau) = mean_i B_i(tau)."""
    return float(B[tau].mean())


def main():
    claims, chain, unit_scores, met, c_hat, C2, max_abs_D = common.build_claims_and_deviations("rpp")
    n = len(claims)
    B = common.broadcast_to_claims("rpp", claims, common.per_run_exceedance("rpp", claims, unit_scores, TAUS))
    print(f"n = {n} claims")
    print("c_hat (fixed = 1 for every chain step)  " +
          "  ".join(f"{common.ULAB[m]}={c_hat[m]:.1f}" for m in common.CHAIN))
    print(f"sum c_hat^2 = {C2:.4f}")
    print(f"validity check -- observed max|D_i,m| across all claims/runs/steps: "
          f"{max_abs_D:.3f} (must be <= c=1)")
    print(f"delta_i: median {claims.delta.median():.3f}, max {claims.delta.max():.3f}")

    tau_req = np.sqrt(2 * np.log(2) * C2)
    print(f"UB_AH < 1 requires (tau - delta_i) > {tau_req:.3f}\n")

    rows = []
    print(f"{'tau':>6}{'UB_AH':>9}{'F_hat':>9}{'frac tau<=delta':>17}")
    for t in TAUS:
        u = ub_ah(t, claims["delta"], C2).mean()
        f = f_hat(t, B)
        frac = (t <= claims.delta).mean()
        print(f"{t:>6.2f}{u:>9.3f}{f:>9.3f}{frac:>17.3f}")
        rows.append({"tau": t, "mean_UB_AH": u, "F_hat": f, "frac_tau_le_delta": frac})

    per_claim = claims.copy()
    for t in TAUS:
        per_claim[f"UB_AH_tau{t}"] = ub_ah(t, claims["delta"], C2)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    per_claim.to_csv(OUT_CSV)
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
