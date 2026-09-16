"""Text-embedding Logistic Regression predictor for CB replicability --
the file_CB counterpart of predict_embed_lr.py.

RPP validates against an external table (Altmejd et al.'s 90-study RF
benchmark) and needs title-normalization to bridge the two sources. CB
has no such external benchmark and needs none of that: embed_text_batch.py --corpus cb
already writes a "ground_truth" column straight from the same
cb_data_cleaned.csv this script would otherwise reload, so index.csv +
pooled.npy + context.npy alone are sufficient -- all 158 effects are labeled
(no drop-unmatched-rows step like RPP's).

n=158 effects across only 23 papers, and a paper's `pooled` (mean chunk)
vector is IDENTICAL across all of that paper's effects -- only `context`
(the focal-effect locator vector) varies within a paper. Plain k-fold CV
would let effects from the same paper land in both train and test folds,
leaking paper identity through the shared pooled half of the feature vector
and inflating the CV score. Both the inner (hyperparameter) and outer
(evaluation) folds are therefore grouped by paper_num via
StratifiedGroupKFold, not the plain StratifiedKFold RPP uses (RPP has one
effect per paper, so it never faced this).

n=158 over 23 clusters still makes any single fold-split noisy (same issue
predict_embed_lr_cb_firsteffect.py's n=23 faced, just from paper-clustering
instead of raw sample size), so the nested CV is repeated under N_SEEDS=20
different fold-split seeds, same fix as that script and predict_embed_lr_ssrp.py
use. AUC/Accuracy on the seed-averaged probability get closed-form binomial
SDs (Hanley & McNeil 1982 / Wald); the per-effect across-seed SD is reported
as this run's label-free dispersion.

Usage: rep_env/bin/python prediction/embed_pred/predict_embed_lr_cb.py
Writes: prediction/embed_pred/CB/embed_lr_metrics.csv,
        prediction/embed_pred/CB/embed_lr_per_seed.csv,
        prediction/embed_pred/CB/embed_lr_study_predictions.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LLM_UQ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
EMB_DIR = os.path.join(LLM_UQ, "prediction", "embed_pred", "CB", "embeddings_text-embedding-3-large_query")
OUT_DIR = os.path.join(LLM_UQ, "prediction", "embed_pred", "CB")
OUT_CSV = os.path.join(OUT_DIR, "embed_lr_metrics.csv")
OUT_PERSEED_CSV = os.path.join(OUT_DIR, "embed_lr_per_seed.csv")
OUT_STUDY_CSV = os.path.join(OUT_DIR, "embed_lr_study_predictions.csv")

N = 90  # hard cap on covariates, matches RPP's "no more predictors than observations"
GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "sel__max_features": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90],
}
N_SEEDS = 20


def wald_accuracy_sd(acc, n):
    """Accuracy is a Binomial(n, acc) proportion (correct/incorrect per effect)."""
    return np.sqrt(acc * (1 - acc) / n)


def hanley_mcneil_auc_sd(auc, n_pos, n_neg):
    """Hanley & McNeil (1982): analytic SE of AUC as a Mann-Whitney U-statistic."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return np.sqrt(var)


def make_pipe():
    return Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0),
            max_features=N, threshold=-np.inf)),
        ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0)),
    ])


def run_nested_cv(X, y, groups, seed):
    """One paper-grouped nested-CV pass under a given fold-split seed: outer
    5-fold eval, inner 5-fold hyperparameter search on each outer training
    fold, both grouped by paper_num so no paper's effects split across
    train/test. Returns the length-n out-of-fold probability vector."""
    pipe = make_pipe()
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=2 * seed)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=2 * seed + 1)
    proba_cv = np.full(len(y), np.nan)
    for train_idx, test_idx in outer.split(X, y, groups):
        inner_cv = list(inner.split(X[train_idx], y[train_idx], groups[train_idx]))
        gs = GridSearchCV(pipe, GRID, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X[train_idx], y[train_idx])
        proba_cv[test_idx] = gs.predict_proba(X[test_idx])[:, 1]
    return proba_cv


def main():
    idx = pd.read_csv(os.path.join(EMB_DIR, "index.csv"))
    ctx = np.load(os.path.join(EMB_DIR, "context.npy"))
    pooled = np.load(os.path.join(EMB_DIR, "pooled.npy"))
    assert len(idx) == len(ctx) == len(pooled), "index.csv/context.npy/pooled.npy row count mismatch"

    labeled = idx["ground_truth"].isin(["yes", "no"])
    print(f"effects: {len(idx)} total, {labeled.sum()} labeled, "
          f"{idx.loc[labeled, 'paper_num'].nunique()} papers")
    idx, ctx, pooled = idx[labeled], ctx[labeled.values], pooled[labeled.values]

    X = np.hstack((ctx, pooled))
    y = idx["ground_truth"].map({"yes": 1, "no": 0}).values
    groups = idx["paper_num"].values
    n = len(y)
    print(f"n = {n} effects used for training ({y.sum()} yes / {(1 - y).sum()} no), "
          f"{len(set(groups))} paper groups")

    # in-sample: fit on all data, evaluate on the same data (optimistic; matches
    # the "best hyperparameter" cell of embeddings_LR.ipynb / predict_embed_lr.py)
    inner0 = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    grid = GridSearchCV(make_pipe(), GRID, cv=list(inner0.split(X, y, groups)),
                        scoring="roc_auc", n_jobs=-1)
    grid.fit(X, y)
    proba_in = grid.predict_proba(X)[:, 1]
    pred_in = (proba_in >= 0.5).astype(int)
    non_zero = np.where(grid.best_estimator_.named_steps["clf"].coef_[0] != 0)[0]
    print(f"\nbest params (selected on full data): {grid.best_params_}, "
          f"active features: {len(non_zero)}")

    # nested CV, paper-grouped, repeated under N_SEEDS different fold-split
    # seeds; final prediction is the per-effect average out-of-fold
    # probability across seeds.
    print(f"\nrunning paper-grouped nested CV under {N_SEEDS} seeds "
          f"(outer 5-fold, inner 5-fold hyperparameter search) ...")
    proba_by_seed = np.full((N_SEEDS, n), np.nan)
    per_seed = []
    for seed in range(N_SEEDS):
        proba_by_seed[seed] = run_nested_cv(X, y, groups, seed)
        auc_s = roc_auc_score(y, proba_by_seed[seed])
        brier_s = brier_score_loss(y, proba_by_seed[seed])
        acc_s = accuracy_score(y, (proba_by_seed[seed] >= 0.5).astype(int))
        per_seed.append({"seed": seed, "AUC": auc_s, "Brier": brier_s, "Accuracy": acc_s})
    per_seed = pd.DataFrame(per_seed)
    print(per_seed.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed.to_csv(OUT_PERSEED_CSV, index=False)
    print(f"Wrote {OUT_PERSEED_CSV}")
    print(f"\nAcross {N_SEEDS} seeds: AUC mean={per_seed['AUC'].mean():.3f} "
          f"sd={per_seed['AUC'].std():.3f}; Brier mean={per_seed['Brier'].mean():.3f} "
          f"sd={per_seed['Brier'].std():.3f}; Accuracy mean={per_seed['Accuracy'].mean():.3f} "
          f"sd={per_seed['Accuracy'].std():.3f}")

    proba_cv_avg = proba_by_seed.mean(axis=0)
    pred_cv_avg = (proba_cv_avg >= 0.5).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc_cv_avg = roc_auc_score(y, proba_cv_avg)
    acc_cv_avg = accuracy_score(y, pred_cv_avg)
    auc_cv_sd = hanley_mcneil_auc_sd(auc_cv_avg, n_pos, n_neg)
    acc_cv_sd = wald_accuracy_sd(acc_cv_avg, n)
    brier_cv_avg = brier_score_loss(y, proba_cv_avg)
    print(f"\nFinal prediction (avg over {N_SEEDS}-seed nested CV): "
          f"AUC={auc_cv_avg:.3f}+/-{auc_cv_sd:.3f}  Accuracy={acc_cv_avg:.3f}+/-{acc_cv_sd:.3f}  "
          f"Brier={brier_cv_avg:.3f}")

    rows = [
        {"model": "Text-embedding LR (CB, effect-level, paper-grouped CV)",
         "evaluation": "in-sample", "n": n,
         "AUC": roc_auc_score(y, proba_in), "Brier": brier_score_loss(y, proba_in),
         "Accuracy": accuracy_score(y, pred_in)},
        {"model": "Text-embedding LR (CB, effect-level, paper-grouped CV)",
         "evaluation": f"nested CV, {N_SEEDS}-seed avg prediction", "n": n,
         "AUC": auc_cv_avg, "AUC_SD": auc_cv_sd,
         "Brier": brier_cv_avg,
         "Accuracy": acc_cv_avg, "Accuracy_SD": acc_cv_sd,
         "dispersion_mean_proba_sd": proba_by_seed.std(axis=0).mean()},
        {"model": "Text-embedding LR (CB, effect-level, paper-grouped CV)",
         "evaluation": f"nested CV, per-seed (mean of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].mean(), "Brier": per_seed["Brier"].mean(),
         "Accuracy": per_seed["Accuracy"].mean()},
        {"model": "Text-embedding LR (CB, effect-level, paper-grouped CV)",
         "evaluation": f"nested CV, per-seed (sd of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].std(), "Brier": per_seed["Brier"].std(),
         "Accuracy": per_seed["Accuracy"].std()},
    ]
    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False))

    results.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    study_preds = pd.DataFrame({
        "title": idx["title"].values, "paper_num": idx["paper_num"].values,
        "experiment_num": idx["experiment_num"].values, "effect_num": idx["effect_num"].values,
        "y": y, "proba_in": proba_in,
        "proba_cv_avg": proba_cv_avg, "proba_cv_sd_across_seeds": proba_by_seed.std(axis=0),
    })
    study_preds.to_csv(OUT_STUDY_CSV, index=False)
    print(f"Wrote {OUT_STUDY_CSV}")


if __name__ == "__main__":
    main()
