"""Single-run vs. run-averaged predictors (reviewer comment W1a).

For each corpus and each predictor type, compares
    individual predictor  p_i^(r)   (q_i^r or v_i^r, one run)
    averaged predictor    p-bar_i   (q-bar_i or v-bar_i, mean over R runs)
on AUC (the grid) -- accuracy and Brier are computed too (see compare()). Single-run metrics are computed
separately for every run r = 1..R over all units, then reported as mean +- SD
across runs. The averaged-predictor metrics are computed once.

Exact identity (population variance over runs, V_i = R^-1 sum_r (p_i^r - pbar_i)^2):
    mean_r Brier(p^r) = Brier(pbar) + mean_i V_i
which is asserted below as a sanity check (not shown in the table). For votes V_i = vbar_i (1 - vbar_i).

RPP note: the published pipeline averages RPP runs by title (88 titles for 90
claims: rpp.49/50 and rpp.148/149 share titles), so averaged RPP predictors here
use the title-pooled mean, to reproduce the published numbers; single-run RPP
metrics are per claim.

Data: prediction/results/{RPP,CB,SSRP}/text_predictions_high_temp0.7.csv
(per-run q_hat and verdict). RPP outcomes use embed_lr_study_predictions.csv's y
(canonical OSF export), matching the other tables; CB/SSRP use ground_truth.
The averaged AUC/accuracy are cross-checked against the stored metric CSVs.

Usage:  rep_env/bin/python tables/table_single_run_q_grid.py
Writes: tables/table_single_run_q_grid.tex  (AUC / accuracy / Brier / NLL panels of single-run q, temperature x effort grid)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PKG, "prediction", "results")
METRIC = os.path.join(PKG, "prediction", "llm_pred", "metric")
EMBED = os.path.join(PKG, "prediction", "embed_pred")

# corpus -> (unit id columns, slug of stored metric files)
CORPORA = {
    "RPP": (["altmejd_id"], ""),
    "CB": (["paper_num", "experiment_num", "effect_num"], "_cb_full"),
    "SSRP": (["study_num"], "_ssrp"),
}


EPS = 0.01  # = 1/R, the resolution of a run-averaged vote fraction


def metrics(p, y):
    """AUC, accuracy (threshold 0.5), Brier, NLL, plus per-unit Brier/NLL losses.
    Brier uses p as is. NLL clips p to [EPS, 1-EPS] -- the same rule for every
    predictor, because vote fractions of exactly 0 or 1 give infinite NLL. NLL of
    vote fractions is convention-dependent: a larger EPS lowers it. (Single-run
    vote NLL is computed here but not shown in the table.)"""
    pc = np.clip(p, EPS, 1 - EPS)
    brier_i = (p - y) ** 2
    nll_i = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    return (roc_auc_score(y, p), float(((p >= 0.5).astype(int) == y).mean()),
            float(brier_i.mean()), float(nll_i.mean()), brier_i, nll_i)


def load(corpus, effort="high", temp="0.7"):
    ids, _ = CORPORA[corpus]
    df = pd.read_csv(os.path.join(RES, corpus, f"text_predictions_{effort}_temp{temp}.csv"),
                     usecols=ids + ["run_id", "q_hat", "verdict", "ground_truth"]
                     + (["Study Title (O)"] if corpus == "RPP" else []))
    df["unit"] = df[ids].astype(str).agg("|".join, axis=1)
    df["v"] = (df["verdict"] == "replicable").astype(float)
    y = df.groupby("unit")["ground_truth"].first().map({"yes": 1, "no": 0})
    if corpus == "RPP":  # canonical outcomes, as in the other tables
        emb = pd.read_csv(os.path.join(EMBED, "RPP", "embed_lr_study_predictions.csv"))
        y = pd.Series(df.groupby("unit")["altmejd_id"].first()).map(
            emb.set_index("altmejd_id")["y"])
    assert y.notna().all()
    return df, y


def compare(df, y, col):
    """Returns per_run (R x 4: AUC, acc, Brier, NLL), avg (AUC, acc, Brier, NLL),
    se_brier, se_nll (averaged predictor: SD of per-unit loss / sqrt(n)), R."""
    wide = df.pivot(index="unit", columns="run_id", values=col).loc[y.index]
    assert not wide.isna().any().any(), "unbalanced runs"
    yv = y.to_numpy(float)
    P = wide.to_numpy(float)                     # units x runs
    R = P.shape[1]
    per_run = np.array([
        metrics(P[:, r], yv)[:4] for r in range(R)])
    pbar = P.mean(axis=1)
    # sanity: mean_r Brier(p^r) = Brier(pbar) + mean_i V_i (population variance)
    avg_claim = metrics(pbar, yv)
    assert abs(per_run[:, 2].mean() - avg_claim[2] - P.var(axis=1, ddof=0).mean()) < 1e-9, \
        "Brier identity violated"
    if "Study Title (O)" in df:
        # RPP: the published pipeline averages runs by TITLE (88 titles for 90
        # claims), so claims sharing a title get the title-pooled mean.
        # Single-run metrics stay claim-level; the identity above is claim-level.
        title = df.groupby("unit")["Study Title (O)"].first().loc[y.index].to_numpy()
        pbar = pd.Series(pbar, index=y.index).groupby(title).transform("mean").to_numpy()
    out = metrics(pbar, yv)
    n = len(yv)
    return (per_run, out[:4], float(out[4].std(ddof=1) / np.sqrt(n)),
            float(out[5].std(ddof=1) / np.sqrt(n)), R)


def pm(a):
    return f"${a.mean():.3f} \\pm {a.std(ddof=1):.3f}$"


def hm_sd(auc, n_pos, n_neg):
    """Hanley-McNeil (1982) SE of an AUC (same formula as predict_llm_uq.py)."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


CELLS = [("0.7", "high"), ("0.7", "low"), ("0.2", "high"), ("0.2", "low")]
OUT_GRID = os.path.join(PKG, "tables", "table_single_run_q_grid.tex")


PANELS = [("AUC", 0, "(a) AUC"), ("Acc", 1, "(b) Accuracy"), ("Brier", 2, "(c) Brier score"),
          ("NLL", 3, "(d) NLL")]


def grid_table():
    """Four panels (AUC, accuracy, Brier, NLL) over temperature x effort for the single-run
    confidence q_i^(r) (mean +- SD over runs). Vote rate and averaged predictors are
    still computed (and cross-checked against stored metrics) but not tabulated."""
    lines = [
        r"% Generated by table_single_run_q_grid.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        r"\caption{Reasoning-effort $\times$ temperature grid, $R{=}100$ runs per cell, "
        r"for (a) AUC, (b) accuracy at a 0.5 threshold ($q_i^{(r)} \ge 0.5$ predicts replicable), "
        r"(c) Brier score and (d) negative log-likelihood (NLL, nats per unit) "
        r"of the single-run confidence $q_i^{(r)}$ (one run's verbalized replication "
        r"probability). Each metric is computed for each run $r$ and reported as mean $\pm$ SD "
        r"across runs (run-to-run spread, not a standard error). For NLL, probabilities are "
        r"clipped to $[0.01, 0.99]$. "
        r"SSRP's reversal of the effort ordering rests on $n{=}21$ units and is "
        r"not interpreted.}",
        r"\label{tab:single-vs-avg-grid}", r"\small",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{Temperature $0.7$} & "
        r"\multicolumn{2}{c}{Temperature $0.2$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Dataset} & high effort & "
        r"low effort & high effort & low effort \\",
    ]
    corpora = list(CORPORA)
    # res[corpus][(kind, temp, eff)] = dict(per_run, avg, se_auc, se_brier, se_nll)
    res = {}
    for corpus in corpora:
        res[corpus] = {}
        for temp, eff in CELLS:
            df, y = load(corpus, eff, temp)
            n1, n0 = int(y.sum()), int((1 - y).sum())
            for col, kind in [("v", "verdict"), ("q_hat", "qbar")]:
                per_run, avg, se_b, se_n, R = compare(df, y, col)
                assert R == 100, f"{corpus} {eff} {temp}: R={R}"
                suffix = ("" if eff == "high" else "_low") + ("" if temp == "0.7" else "_temp0.2")
                stored = pd.read_csv(os.path.join(
                    METRIC, f"llm_{kind}{CORPORA[corpus][1]}{suffix}_metrics.csv")).iloc[0]
                if abs(stored.AUC - avg[0]) > 1e-3:
                    print(f"WARNING {corpus} {kind} {eff} T{temp}: recomputed AUC "
                          f"{avg[0]:.3f} != stored {stored.AUC:.3f}")
                res[corpus][(kind, temp, eff)] = dict(
                    per_run=per_run, avg=avg,
                    se={0: hm_sd(avg[0], n1, n0), 2: se_b, 3: se_n})
    for name, idx, title in PANELS:
        lines += [r"\midrule" if name != "AUC" else r"\midrule",
                  rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{title}}}}} \\",
                  r"\addlinespace[2pt]"]
        # Only the single-run confidence q_i^(r) is shown; vote-rate rows and the
        # run-averaged rows were dropped from the table (still computed above).
        for corpus in corpora:
            cells = [pm(res[corpus][("qbar", temp, eff)]["per_run"][:, idx]) for temp, eff in CELLS]
            lines.append(f"{corpus} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(OUT_GRID, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_GRID}")


def main():
    grid_table()


if __name__ == "__main__":
    main()
