"""Text-embedding Logistic Regression predictor for SSRP replicability --
the file_SSRP counterpart of predict_embed_lr_cb.py, minus the paper-grouped
CV: SSRP's embeddings index is already one row per STUDY (no CB-style
several-effects-per-paper clustering to protect against, since a study IS a
paper here), so plain StratifiedKFold is enough -- the same as
predict_embed_lr.py (RPP, also one row per paper) uses.

n=21 studies only, so the CB/RPP scripts' [1, 5, 10, ..., 90] max_features
grid would let the selector pick nearly as many features as there are
studies. GRID instead restricts the candidate feature counts to
[5, 10, 15, 20].

At this n, a single 5-fold split is noisy (predict_embed_lr_cb_gee.py found
+/-0.1 AUC swings from fold assignment alone), so the nested CV -- inner
hyperparameter search AND outer evaluation -- is repeated under N_SEEDS=20
different fold-split seeds, same fix as that script's paper-grouped CV. The
final per-study prediction is the average out-of-fold probability across
seeds.

Usage: rep_env/bin/python llm_uq/prediction/embed_pred/predict_embed_lr_ssrp.py
Writes: llm_uq/prediction/embed_pred/SSRP/embed_lr_metrics.csv,
        llm_uq/prediction/embed_pred/SSRP/embed_lr_per_seed.csv,
        llm_uq/prediction/embed_pred/SSRP/embed_lr_study_predictions.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def wald_accuracy_sd(acc, n):
    """Accuracy is a Binomial(n, acc) proportion (correct/incorrect per study)."""
    return np.sqrt(acc * (1 - acc) / n)


def hanley_mcneil_auc_sd(auc, n_pos, n_neg):
    """Hanley & McNeil (1982): analytic SE of AUC as a Mann-Whitney U-statistic."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return np.sqrt(var)

LLM_UQ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # llm_uq/
EMB_DIR = os.path.join(LLM_UQ, "prediction", "embed_pred", "SSRP", "embeddings_text-embedding-3-large_query")
OUT_DIR = os.path.join(LLM_UQ, "prediction", "embed_pred", "SSRP")
OUT_CSV = os.path.join(OUT_DIR, "embed_lr_metrics.csv")
OUT_PERSEED_CSV = os.path.join(OUT_DIR, "embed_lr_per_seed.csv")
OUT_STUDY_CSV = os.path.join(OUT_DIR, "embed_lr_study_predictions.csv")

N = 20  # hard cap on covariates -- matches the largest candidate in GRID below
GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "sel__max_features": [5, 10, 15, 20],
}
N_SEEDS = 20

PIPE = Pipeline([
    ("sc", StandardScaler()),
    ("sel", SelectFromModel(
        LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0),
        max_features=N, threshold=-np.inf)),
    ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0)),
])


def run_nested_cv(X, y, seed):
    """One nested-CV pass under a given fold-split seed: outer 5-fold eval,
    inner 5-fold hyperparameter search on each outer training fold. Returns
    the length-n out-of-fold probability vector. Inner/outer use different
    random states (2*seed, 2*seed+1) so their fold assignments never coincide."""
    inner = StratifiedKFold(5, shuffle=True, random_state=2 * seed)
    outer = StratifiedKFold(5, shuffle=True, random_state=2 * seed + 1)
    grid = GridSearchCV(PIPE, GRID, cv=inner, scoring="roc_auc", n_jobs=-1)
    return cross_val_predict(grid, X, y, cv=outer, method="predict_proba", n_jobs=-1)[:, 1]


def main():
    idx = pd.read_csv(os.path.join(EMB_DIR, "index.csv"))
    ctx = np.load(os.path.join(EMB_DIR, "context.npy"))
    pooled = np.load(os.path.join(EMB_DIR, "pooled.npy"))
    assert len(idx) == len(ctx) == len(pooled), "index.csv/context.npy/pooled.npy row count mismatch"

    labeled = idx["ground_truth"].isin(["yes", "no"])
    print(f"studies: {len(idx)} total, {labeled.sum()} labeled")
    idx, ctx, pooled = idx[labeled], ctx[labeled.values], pooled[labeled.values]

    X = np.hstack((ctx, pooled))
    y = idx["ground_truth"].map({"yes": 1, "no": 0}).values
    n = len(y)
    print(f"n = {n} studies used for training ({y.sum()} yes / {(1 - y).sum()} no)")

    # Primary: single reference fit on all data (optimistic; matches the
    # "best hyperparameter" reporting convention of predict_embed_lr.py / _cb.py)
    inner0 = StratifiedKFold(5, shuffle=True, random_state=0)
    grid = GridSearchCV(PIPE, GRID, cv=inner0, scoring="roc_auc", n_jobs=-1)
    grid.fit(X, y)
    proba_in = grid.predict_proba(X)[:, 1]
    pred_in = (proba_in >= 0.5).astype(int)
    non_zero = np.where(grid.best_estimator_.named_steps["clf"].coef_[0] != 0)[0]
    print(f"\nbest params (selected on full data): {grid.best_params_}, "
          f"active features: {len(non_zero)}")

    # Secondary: nested CV under N_SEEDS different fold splits; final
    # prediction is the per-study average out-of-fold probability across seeds.
    print(f"\nrunning nested CV under {N_SEEDS} seeds ...")
    proba_by_seed = np.full((N_SEEDS, n), np.nan)
    per_seed = []
    for seed in range(N_SEEDS):
        proba_by_seed[seed] = run_nested_cv(X, y, seed)
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
    brier_cv_avg = brier_score_loss(y, proba_cv_avg)
    acc_cv_avg = accuracy_score(y, pred_cv_avg)
    auc_cv_sd = hanley_mcneil_auc_sd(auc_cv_avg, n_pos, n_neg)
    acc_cv_sd = wald_accuracy_sd(acc_cv_avg, n)
    print(f"\nFinal prediction (avg over {N_SEEDS}-seed nested CV): "
          f"AUC={auc_cv_avg:.3f}+/-{auc_cv_sd:.3f}  Accuracy={acc_cv_avg:.3f}+/-{acc_cv_sd:.3f}  "
          f"Brier={brier_cv_avg:.3f}")

    rows = [
        {"model": "Text-embedding LR (SSRP, study-level)", "evaluation": "in-sample", "n": n,
         "AUC": roc_auc_score(y, proba_in), "Brier": brier_score_loss(y, proba_in),
         "Accuracy": accuracy_score(y, pred_in)},
        {"model": "Text-embedding LR (SSRP, study-level)",
         "evaluation": f"nested CV, {N_SEEDS}-seed avg prediction", "n": n,
         "AUC": auc_cv_avg, "AUC_SD": auc_cv_sd,
         "Brier": brier_cv_avg,
         "Accuracy": acc_cv_avg, "Accuracy_SD": acc_cv_sd,
         "dispersion_mean_proba_sd": proba_by_seed.std(axis=0).mean()},
        {"model": "Text-embedding LR (SSRP, study-level)",
         "evaluation": f"nested CV, per-seed (mean of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].mean(), "Brier": per_seed["Brier"].mean(),
         "Accuracy": per_seed["Accuracy"].mean()},
        {"model": "Text-embedding LR (SSRP, study-level)",
         "evaluation": f"nested CV, per-seed (sd of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].std(), "Brier": per_seed["Brier"].std(),
         "Accuracy": per_seed["Accuracy"].std()},
    ]
    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False))
    results.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    study_preds = pd.DataFrame({
        "title": idx["title"].values, "study_num": idx["study_num"].values,
        "y": y, "proba_in": proba_in,
        "proba_cv_avg": proba_cv_avg, "proba_cv_sd_across_seeds": proba_by_seed.std(axis=0),
    })
    study_preds.to_csv(OUT_STUDY_CSV, index=False)
    print(f"Wrote {OUT_STUDY_CSV}")


if __name__ == "__main__":
    main()
