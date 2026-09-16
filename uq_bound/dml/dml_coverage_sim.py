"""Step 13 (dml_spec.md): "the n=90, R=100 coverage simulation" -- the
request's own words, a single validation exercise. Checks whether the two
intervals reported on the real corpora (Step 9's cluster-robust pointwise CI
and Step 10's multiplier-bootstrap simultaneous band) achieve their nominal
95% coverage at this n and R, on data with a known answer.

Data-generating process. For claim i, draw a baseline mu_i ~ U(0.3,0.7),
independent of anything else (matching Step 4's empirical finding of ~zero
R^2 for any study-level covariate). For each run r, draw iid z_1..z_4,z_res
~ N(0,1) and build, with the correctly-specified additive stage map Step 5
itself assumes (c_m = sqrt(theta_m)):

    p_{i,1}^(r) = mu_i + c_1 z_1
    p_{i,m}^(r) = p_{i,m-1}^(r) + c_m z_m,   m=2,3,4
    P^(r)_i     = p_{i,4}^(r) + sqrt(theta_res) z_res

By construction (independent mean-zero increments), h_m(W_m) = p_{i,m}^(r)
exactly and h_0(X_i) = mu_i exactly, so E[D_m^2] = theta_m exactly -- this
DGP has a known, closed-form ground truth.

Usage: rep_env/bin/python uq_bound/dml_coverage_sim.py
Writes: uq_bound/results/coverage_sim.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_uq.uq_bound.dml.dml_theta as dt  # noqa: E402

N_CLUSTERS = 90     # request: n=90
R = 100             # request: R=100
REPS = 2000         # pre-registered value; replicate count is not specified
                     # by the request, so the pre-registered default is used
B_BOOT_SIM = dt.B_BOOT
ALPHA = 0.05
TRUE_THETA = {"phi1": 0.0006, "phi2": 0.0016, "phi3": 0.0003, "phi4": 0.0002, "phi_res": 0.0020}
PHI_COLS = dt.PHI_COLS


def simulate_obs(n_clusters, R, true_theta, seed):
    """One claim per cluster (n=90 studies, matching the request), R runs
    each, correctly-specified additive stage map."""
    rng = np.random.default_rng(seed)
    sqrt_theta = {c: np.sqrt(max(true_theta[c], 0.0)) for c in PHI_COLS}
    rows = []
    for i in range(n_clusters):
        mu_i = rng.uniform(0.3, 0.7)
        z = rng.standard_normal(size=(R, 5))
        p1 = mu_i + sqrt_theta["phi1"] * z[:, 0]
        p2 = p1 + sqrt_theta["phi2"] * z[:, 1]
        p3 = p2 + sqrt_theta["phi3"] * z[:, 2]
        p4 = p3 + sqrt_theta["phi4"] * z[:, 3]
        P = p4 + sqrt_theta["phi_res"] * z[:, 4]
        for r in range(R):
            rows.append((i, i, r, p1[r], p2[r], p3[r], p4[r], P[r]))
    df = pd.DataFrame(rows, columns=["claim_id", "cluster_id", "run_id",
                                      "q_1", "q_3", "q_4", "q_5", "q_6"])
    for c in dt.PFULL:
        df[c] = np.clip(df[c], 0.0, 1.0)
    return df


def one_replicate(seed):
    obs = simulate_obs(N_CLUSTERS, R, TRUE_THETA, seed)
    obs["study_fold"] = dt.assign_study_folds(obs, K=dt.K_STUDY, seed=seed + 1)
    obs["run_fold"] = dt.assign_run_folds(obs, K=dt.K_RUN, seed=seed + 2)
    obs = dt.fit_score_chain(obs)
    dt.check_telescoping(obs, verbose=False)
    phibar = dt.per_claim_phibar(obs)
    theta_hat, theta_raw, corr, r_train, n = dt.aggregate_theta(phibar)
    se, G, _ = dt.cluster_robust_se(phibar, theta_hat)
    z = stats.norm.ppf(1 - ALPHA / 2)
    q_sim, band, _ = dt.multiplier_bootstrap(phibar, theta_hat, se, B=B_BOOT_SIM, seed=seed + 3)
    lo = theta_hat[PHI_COLS].values - z * se[PHI_COLS].values
    hi = theta_hat[PHI_COLS].values + z * se[PHI_COLS].values
    band_lo, band_hi = band["lo_sim"].values, band["hi_sim"].values
    v_target, v_total, gap_pct = dt.adding_up_check(obs, theta_hat, verbose=False)
    return {
        "theta_hat": theta_hat[PHI_COLS].values, "lo": lo, "hi": hi,
        "band_lo": band_lo, "band_hi": band_hi, "gap_pct": gap_pct,
    }


def main():
    print(f"[dml_coverage_sim] Step 13: the n={N_CLUSTERS}, R={R} coverage simulation "
          f"({REPS} replicates, B={B_BOOT_SIM} bootstrap draws/replicate)")
    truth = np.array([TRUE_THETA[c] for c in PHI_COLS])
    n_theta = len(PHI_COLS)
    t0 = time.time()
    thetas = np.empty((REPS, n_theta))
    pointwise_cover = np.zeros((REPS, n_theta), dtype=bool)
    sim_cover = np.zeros(REPS, dtype=bool)
    band_width = np.empty((REPS, n_theta))
    gaps = np.empty(REPS)
    for b in range(REPS):
        out = one_replicate(seed=7 * 100000 + b)
        thetas[b] = out["theta_hat"]
        pointwise_cover[b] = (out["lo"] <= truth) & (truth <= out["hi"])
        sim_cover[b] = np.all((out["band_lo"] <= truth) & (truth <= out["band_hi"]))
        band_width[b] = out["band_hi"] - out["band_lo"]
        gaps[b] = out["gap_pct"]
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{REPS} replicates ({time.time()-t0:.0f}s elapsed)")
    bias = thetas.mean(axis=0) - truth
    print(f"Done in {time.time()-t0:.0f}s. "
          f"pointwise cov={pointwise_cover.mean(axis=0).round(3)}  "
          f"simultaneous cov={sim_cover.mean():.3f}  mean|gap%|={np.abs(gaps).mean():.2f}")

    rows = []
    for j, c in enumerate(PHI_COLS):
        rows.append({
            "component": c, "true_theta": truth[j],
            "mean_theta_hat": thetas[:, j].mean(), "bias": bias[j],
            "pointwise_coverage": pointwise_cover[:, j].mean(),
            "mean_band_width": band_width[:, j].mean(), "reps": REPS,
        })
    out_df = pd.DataFrame(rows)
    out_df["simultaneous_coverage"] = sim_cover.mean()
    out_df["mean_abs_adding_up_gap_pct"] = np.abs(gaps).mean()
    out_df["n"] = N_CLUSTERS
    out_df["R"] = R

    out_path = os.path.join(HERE, "results", "coverage_sim.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
