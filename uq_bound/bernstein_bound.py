"""Empirical Bernstein bound for the mean run-level exceedance rate.

For per-claim exceedance fractions ``B_i(tau)``, the script bounds their
population mean with confidence ``1-delta``:

    pi(tau) <= mean(B(tau)) + sqrt( 2 * var(B(tau)) * ln(2/delta) / n )
                             + 7 ln(2/delta) / (3(n-1))

The variance term uses the sample variance across claims.

Usage: rep_env/bin/python uq_bound/bernstein_bound.py
Writes: uq_bound/results/bernstein_bound.csv
"""
import os

import numpy as np
import pandas as pd

import common

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "results", "bernstein_bound.csv")
TAUS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]  # matches ah_bound.py / freedman_bound.py
DELTA = 0.05


def ub_bernstein(B_col, n, delta=DELTA):
    """B_col: array of per-claim B_i(tau) at a single tau."""
    B_mean = float(B_col.mean())
    sigma_b2 = float(np.var(B_col, ddof=1))
    variance_term = np.sqrt(2.0 * sigma_b2 * np.log(2.0 / delta) / n)
    tail_constant = 7.0 * np.log(2.0 / delta) / (3.0 * (n - 1))
    ub = float(np.minimum(1.0, B_mean + variance_term + tail_constant))
    return ub, sigma_b2, variance_term, tail_constant


def main():
    claims, chain, unit_scores, met, c_hat, C2, max_abs_D = common.build_claims_and_deviations("rpp")
    n = len(claims)
    print(f"n = {n} claims, delta (confidence) = {DELTA}")

    B = common.broadcast_to_claims("rpp", claims, common.per_run_exceedance("rpp", claims, unit_scores, TAUS))

    rows = []
    print(f"{'tau':>6}{'F_hat':>9}{'sigma2_B':>10}{'var_term':>10}{'tail_c':>9}"
          f"{'UB_Bern':>10}{'margin':>9}")
    for t in TAUS:
        f = float(B[t].mean())
        ub, s2, vt, tc = ub_bernstein(B[t].values, n)
        margin = ub - f
        print(f"{t:>6.2f}{f:>9.4f}{s2:>10.4f}{vt:>10.4f}{tc:>9.4f}{ub:>10.4f}{margin:>9.4f}")
        rows.append({"tau": t, "F_hat": f, "sigma2_B": s2, "variance_term": vt,
                     "tail_constant": tc, "UB_Bernstein": ub, "margin": margin})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
