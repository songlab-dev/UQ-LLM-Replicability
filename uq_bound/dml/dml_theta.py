"""Cross-fitted stage-attribution estimator theta_1..theta_4, theta_res.

Implements dml_spec.json / dml_spec.md (spec_id dml_theta_v1, SHA-256
39448bec61da415b6165ab56f7d11cb23ffe083cf33491eaf03f0103b8ca7ac0). Read
dml_spec.md's Step 1-14 walkthrough alongside this file -- function names
below match the step names.

Usage: rep_env/bin/python uq_bound/dml/dml_theta.py
Writes: uq_bound/dml/results/{RPP,CB,SSRP}/dml_theta.csv
        uq_bound/dml/results/{RPP,CB,SSRP}/dml_theta_per_claim.csv
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # uq_bound/
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import scoring          # noqa: E402
import common           # noqa: E402

SPEC_PATH = os.path.join(HERE, "dml_spec.json")
SPEC_SHA256 = "39448bec61da415b6165ab56f7d11cb23ffe083cf33491eaf03f0103b8ca7ac0"

K_STUDY = 5     # Step 2
K_RUN = 5       # Step 3
SEED = 2024
B_BOOT = 10000  # Step 10
BOOT_SEED = 0
ALPHA = 0.05
PCOLS = ["q_1", "q_3", "q_4", "q_5"]  # p_{i,1}..p_{i,4} (prefix)
PFULL = ["q_1", "q_3", "q_4", "q_5", "q_6"]  # + P = q_6


def check_spec_hash():
    h = hashlib.sha256(open(SPEC_PATH, "rb").read()).hexdigest()
    tag = "OK" if h == SPEC_SHA256 else "MISMATCH -- spec changed since freeze!"
    print(f"[dml_theta] dml_spec.json sha256: {h}  ({tag})")
    return h == SPEC_SHA256


# ── Step 1: assemble observations ────────────────────────────────────────────

def load_observations_rpp():
    scope = common.load_rpp_scope()  # id (altmejd), Study Title (O), ...
    rpp = pd.read_csv(common.RPP_CSV, encoding="latin1")
    pred = pd.read_csv(common.RPP_PRED_CSV)
    pred = pred[pred["Study Title (O)"].isin(scope["Study Title (O)"])].reset_index(drop=True)
    us = scoring.compute_unit_scores(pred, rpp)
    us["claim_id"] = pred["altmejd_id"].values          # positional: same row order
    us["cluster_id"] = us["Study Title (O)"]
    us["run_id"] = pred["run_id"].values
    obs = us[["claim_id", "cluster_id", "run_id"] + PFULL].dropna().reset_index(drop=True)
    return obs


def load_observations_cb():
    scope = common.load_scope_cb()  # claim_id, Paper #, N (O), P-value (O), Effect size r (O), ...
    pred = pd.read_csv(common.CB_PRED_CSV)
    us = common.compute_unit_scores_cb(pred, scope)      # claim_id, run_id, q_1..q_7
    paper_of = scope.set_index("claim_id")["Paper #"]
    us["cluster_id"] = us["claim_id"].map(paper_of)
    obs = us[["claim_id", "cluster_id", "run_id"] + PFULL].dropna().reset_index(drop=True)
    return obs


def load_observations_ssrp():
    ssrp = pd.read_csv(common.SSRP_CSV)
    pred = pd.read_csv(common.SSRP_PRED_CSV)
    us = common.compute_unit_scores_ssrp(pred, ssrp)     # Study Num, run_id, q_1..q_7
    us["claim_id"] = us["Study Num"]
    us["cluster_id"] = us["Study Num"]
    obs = us[["claim_id", "cluster_id", "run_id"] + PFULL].dropna().reset_index(drop=True)
    return obs


LOADERS = {"RPP": load_observations_rpp, "CB": load_observations_cb, "SSRP": load_observations_ssrp}


# ── Step 2 / 3: fold assignment ──────────────────────────────────────────────

def assign_study_folds(obs, K=K_STUDY, seed=SEED):
    clusters = obs["cluster_id"].drop_duplicates().reset_index(drop=True)
    n_splits = min(K, len(clusters))
    gkf = GroupKFold(n_splits=n_splits)
    fold_of = pd.Series(index=clusters, dtype=int)
    # GroupKFold does not shuffle groups; permute them with the fixed seed.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    shuffled = clusters.iloc[order].reset_index(drop=True)
    dummy_X = np.zeros((len(shuffled), 1))
    for k, (_, test_idx) in enumerate(gkf.split(dummy_X, groups=shuffled.values)):
        fold_of.loc[shuffled.iloc[test_idx]] = k
    return obs["cluster_id"].map(fold_of).values


def assign_run_folds(obs, K=K_RUN, seed=SEED):
    out = np.full(len(obs), -1, dtype=int)
    rng = np.random.default_rng(seed)
    for cid, idx in obs.groupby("claim_id").groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        n_splits = min(K, n)
        perm = rng.permutation(n)
        fold_assign = np.array_split(perm, n_splits)
        for k, members in enumerate(fold_assign):
            out[idx[members]] = k
    return out


# ── Step 4: baseline h0 (out-of-run-fold claim mean) ─────────────────────────

def estimate_h0(obs):
    P = obs["q_6"].values
    run_fold = obs["run_fold"].values
    claim = obs["claim_id"].values
    h0 = np.empty(len(obs), dtype=float)
    r_train = np.empty(len(obs), dtype=float)
    df = pd.DataFrame({"claim": claim, "run_fold": run_fold, "P": P})
    # per (claim, run_fold): sum and count of P in that fold, to get leave-fold-out mean fast
    grp = df.groupby(["claim", "run_fold"])["P"].agg(["sum", "count"])
    claim_tot = df.groupby("claim")["P"].agg(["sum", "count"])
    grp = grp.join(claim_tot, on="claim", rsuffix="_claim")
    grp["oof_sum"] = grp["sum_claim"] - grp["sum"]
    grp["oof_n"] = grp["count_claim"] - grp["count"]
    grp["oof_mean"] = grp["oof_sum"] / grp["oof_n"]
    lut = grp["oof_mean"]
    lut_n = grp["oof_n"]
    key = list(zip(claim, run_fold))
    h0 = np.array([lut.loc[k] for k in key])
    r_train = np.array([lut_n.loc[k] for k in key])
    return h0, r_train


# ── Step 5 / 6: pooled prefix maps + orthogonal scores ───────────────────────

def fit_score_chain(obs):
    """Steps 4-6: returns obs with columns h0,h1,h2,h3,h4,phi1..phi4,phi_res added."""
    obs = obs.copy()
    obs["h0"], obs["r_train"] = estimate_h0(obs)
    obs["dev"] = obs["q_6"] - obs["h0"]

    # centered prefix: subtract each claim's own (all-run) mean -- covariate only,
    # no target leakage (Step 5).
    claim_prefix_mean = obs.groupby("claim_id")[PCOLS].transform("mean")
    centered = obs[PCOLS].values - claim_prefix_mean.values
    for j, c in enumerate(PCOLS):
        obs[f"c_{c}"] = centered[:, j]

    h = {0: obs["h0"].values}
    for m in range(1, 5):
        h[m] = np.full(len(obs), np.nan)

    study_folds = obs["study_fold"].values
    for k in sorted(pd.unique(study_folds)):
        train = obs[study_folds != k]
        test_mask = study_folds == k
        test = obs[test_mask]
        dev_train = train["dev"].values
        for m in range(1, 5):
            cols = [f"c_{c}" for c in PCOLS[:m]]
            model = LinearRegression().fit(train[cols].values, dev_train)
            pred = model.predict(test[cols].values)
            h_m_test = np.clip(test["h0"].values + pred, 0.0, 1.0)
            h[m][test_mask] = h_m_test

    for m in range(1, 5):
        obs[f"h{m}"] = h[m]

    P = obs["q_6"].values
    for m in range(1, 5):
        hm, hm1 = obs[f"h{m}"].values, obs[f"h{m-1}"].values
        obs[f"phi{m}"] = 2.0 * (hm - hm1) * P - hm ** 2 + hm1 ** 2
    obs["phi_res"] = (P - obs["h4"].values) ** 2
    return obs


# ── Step 7: telescoping check ─────────────────────────────────────────────────

def check_telescoping(obs, tol=1e-8, verbose=True):
    lhs = obs[[f"phi{m}" for m in range(1, 5)] + ["phi_res"]].sum(axis=1).values
    rhs = (obs["q_6"].values - obs["h0"].values) ** 2
    max_err = float(np.max(np.abs(lhs - rhs)))
    ok = max_err < tol
    if verbose:
        print(f"  Step 7 telescoping check: max|LHS-RHS| = {max_err:.3e}  ({'OK' if ok else 'FAILED'})")
    if not ok:
        raise AssertionError(f"Proposition C.1 telescoping failed: max error {max_err:.3e}")
    return max_err


# ── Step 8: aggregate to five scalars + finite-R correction ─────────────────

PHI_COLS = ["phi1", "phi2", "phi3", "phi4", "phi_res"]
STAGE_NAMES = {"phi1": "theta_1 (Extraction)", "phi2": "theta_2 (Stat review)",
               "phi3": "theta_3 (Researcher DoF)", "phi4": "theta_4 (Theory/context)",
               "phi_res": "theta_res (Residual)"}


def per_claim_phibar(obs):
    """phibar_{i,m}: claim-level mean of each phi column, plus cluster id and R_i."""
    g = obs.groupby("claim_id")
    phibar = g[PHI_COLS].mean()
    phibar["cluster_id"] = g["cluster_id"].first()
    phibar["R_i"] = g.size()
    phibar["r_train_mean"] = g["r_train"].mean()
    return phibar.reset_index()


def aggregate_theta(phibar):
    """Aggregate claim-level scores and apply the residual finite-R correction."""
    n = len(phibar)
    theta_raw = phibar[PHI_COLS].mean()
    corr_i = phibar["r_train_mean"] / (phibar["r_train_mean"] + 1.0)
    theta = theta_raw.copy()
    theta["phi_res"] = (phibar["phi_res"] * corr_i).mean()
    r_train = phibar["r_train_mean"].mean()
    corr = corr_i.mean()  # reported for reference; the correction itself is per-claim
    return theta, theta_raw, corr, r_train, n


# ── Step 9: cluster-robust standard error (applied identically to all three
# corpora -- the request asks for "cluster-robust and bootstrap bands", not
# a small-G-specific variant, so none is used here) ──────────────────────────

def cluster_robust_se(phibar, theta_hat):
    """Two-level cluster-robust SE, generalizing the appendix's single-claim-
    per-cluster formula to clusters that hold multiple claims (CB)."""
    n = len(phibar)
    clusters = phibar["cluster_id"].unique()
    G = len(clusters)
    resid = phibar[PHI_COLS] - theta_hat[PHI_COLS]
    cluster_sum = resid.groupby(phibar["cluster_id"]).sum()  # S_g,m
    var = (cluster_sum ** 2).sum() / (n * (n - 1)) * (G / max(G - 1, 1))
    se = np.sqrt(var)
    return se, G, cluster_sum


def multiplier_bootstrap(phibar, theta_hat, se, B=B_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """Step 10: simultaneous band across the five scalars via a Rademacher
    multiplier bootstrap over clusters -- 'a multiplier bootstrap over
    studies', per the appendix. The same draws also give the per-scalar
    'study-cluster bootstrap intervals' the appendix says to report
    alongside the normal-based ones, for all three corpora."""
    n = len(phibar)
    resid = phibar[PHI_COLS] - theta_hat[PHI_COLS]
    cluster_sum = resid.groupby(phibar["cluster_id"]).sum().values  # (G, 5)
    G = cluster_sum.shape[0]
    rng = np.random.default_rng(seed)
    xi = rng.choice([-1.0, 1.0], size=(B, G))
    draws = (xi @ cluster_sum) / n  # (B, 5): sum_g xi_g * S_g,m / n
    se_arr = se[PHI_COLS].values
    se_arr_safe = np.where(se_arr > 0, se_arr, 1.0)
    t_stat = np.abs(draws) / se_arr_safe
    sup_t = t_stat.max(axis=1)
    q_sim = np.quantile(sup_t, 1 - alpha)
    band = pd.DataFrame({
        "lo_sim": theta_hat[PHI_COLS].values - q_sim * se_arr,
        "hi_sim": theta_hat[PHI_COLS].values + q_sim * se_arr,
    }, index=PHI_COLS)
    # per-scalar percentile CI from the same draws (used as the CB/SSRP
    # wild-cluster-bootstrap-style interval)
    lo_pct = theta_hat[PHI_COLS].values + np.quantile(draws, alpha / 2, axis=0)
    hi_pct = theta_hat[PHI_COLS].values + np.quantile(draws, 1 - alpha / 2, axis=0)
    boot_ci = pd.DataFrame({"lo_boot": lo_pct, "hi_boot": hi_pct}, index=PHI_COLS)
    return q_sim, band, boot_ci


# ── Step 11: adding-up check ─────────────────────────────────────────────────

def adding_up_check(obs, theta, verbose=True):
    """E_i[V_i] computed directly from the raw runs, no fitting at all."""
    v_i = obs.groupby("claim_id")["q_6"].var(ddof=1)
    target = v_i.mean()
    total = theta.sum()
    gap_pct = 100 * (total - target) / target
    if verbose:
        print(f"  Step 11 adding-up check: sum(theta_hat)={total:.6f}  "
              f"E_i[V_i]={target:.6f}  gap={gap_pct:+.2f}%")
    return target, total, gap_pct


# ── driver ────────────────────────────────────────────────────────────────────

MIN_RUNS_PER_CLAIM = 2  # a claim needs >=2 surviving runs for Step 3/4's
                         # leave-run-out h0 to exist at all


def run_corpus(name):
    print(f"\n=== {name} ===")
    obs = LOADERS[name]()
    counts = obs.groupby("claim_id").size()
    too_few = counts[counts < MIN_RUNS_PER_CLAIM].index
    if len(too_few) > 0:
        print(f"  Step 1 (extra scope note): dropping {len(too_few)} claim(s) with "
              f"< {MIN_RUNS_PER_CLAIM} complete runs after the q_1..q_6 dropna "
              f"(cannot support a leave-run-out h0): {list(too_few)}")
        obs = obs[~obs["claim_id"].isin(too_few)].reset_index(drop=True)
    n_obs = len(obs)
    obs["study_fold"] = assign_study_folds(obs)
    obs["run_fold"] = assign_run_folds(obs)
    n_claims = obs["claim_id"].nunique()
    n_clusters = obs["cluster_id"].nunique()
    print(f"  Step 1: {n_obs} observations, {n_claims} claims, {n_clusters} clusters")

    obs = fit_score_chain(obs)
    check_telescoping(obs)

    phibar = per_claim_phibar(obs)
    theta_hat, theta_raw, corr, r_train, n = aggregate_theta(phibar)
    print(f"  Step 8: R_train={r_train:.1f}  finite-R correction factor={corr:.5f}")
    for c in PHI_COLS:
        print(f"    {STAGE_NAMES[c]:32s} raw={theta_raw[c]:.6f}  corrected={theta_hat[c]:.6f}")

    se_cluster, G, cluster_sum = cluster_robust_se(phibar, theta_hat)
    print(f"  Step 9: n_claims={n}  G_clusters={G}  cluster-robust SE computed")

    q_sim, band, boot_ci = multiplier_bootstrap(phibar, theta_hat, se_cluster)
    print(f"  Step 10: simultaneous 95% critical value q={q_sim:.3f}")

    target, total, gap_pct = adding_up_check(obs, theta_hat)

    z = stats.norm.ppf(1 - ALPHA / 2)
    out = pd.DataFrame({
        "stage": [STAGE_NAMES[c] for c in PHI_COLS],
        "phi_col": PHI_COLS,
        "theta_hat": theta_hat[PHI_COLS].values,
        "theta_hat_raw": theta_raw[PHI_COLS].values,
        "share_of_total": theta_hat[PHI_COLS].values / theta_hat[PHI_COLS].sum(),
        "se_cluster_robust": se_cluster[PHI_COLS].values,
        "ci95_lo_normal": theta_hat[PHI_COLS].values - z * se_cluster[PHI_COLS].values,
        "ci95_hi_normal": theta_hat[PHI_COLS].values + z * se_cluster[PHI_COLS].values,
        "band95_lo_simultaneous": band["lo_sim"].values,
        "band95_hi_simultaneous": band["hi_sim"].values,
        "ci95_lo_boot": boot_ci["lo_boot"].values,
        "ci95_hi_boot": boot_ci["hi_boot"].values,
    })

    meta = pd.DataFrame([{
        "corpus": name, "n_obs": n_obs, "n_claims": n_claims, "n_clusters": n_clusters,
        "r_train": r_train, "finite_R_correction": corr,
        "adding_up_target_EV": target, "adding_up_total_theta": total,
        "adding_up_gap_pct": gap_pct, "simultaneous_q95": q_sim,
    }])
    out["corpus"] = name
    for col in meta.columns:
        if col != "corpus":
            out[col] = meta[col].iloc[0]

    out_dir = os.path.join(HERE, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "dml_theta.csv"), index=False)

    per_claim = phibar.rename(columns={c: f"theta_i_{c}" for c in PHI_COLS})
    per_claim.to_csv(os.path.join(out_dir, "dml_theta_per_claim.csv"), index=False)
    print(f"  Step 14: wrote {out_dir}/dml_theta.csv and dml_theta_per_claim.csv")
    return out, per_claim, meta


def main():
    if not check_spec_hash():
        raise RuntimeError("dml_spec.json differs from the frozen pre-registration; refusing to run")
    all_out = []
    for name in ("RPP", "CB", "SSRP"):
        out, _, _ = run_corpus(name)
        all_out.append(out)
    combined = pd.concat(all_out, ignore_index=True)
    combined.to_csv(os.path.join(HERE, "results", "dml_theta_all.csv"), index=False)
    print(f"\nWrote combined table -> {HERE}/results/dml_theta_all.csv")


if __name__ == "__main__":
    main()
