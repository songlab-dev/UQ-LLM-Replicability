"""Text-embedding Logistic Regression predictor for RPP replicability.

This is the standalone implementation of the embedding-regression analysis:
it loads embeddings, fits an L1-penalized
logistic regression via GridSearchCV, report AUC/Brier/Accuracy), rewritten as
a standalone script and row-aligned to Altmejd et al.'s 90-row RPP
`drop == False` scope. Random-forest, SVM, and XGBoost generators are archived
and are not part of the current paper pipeline.

Their 90 rows are at the study-*replicated-effect* level, not one-per-paper:
2 papers ("The Best Men Are (Not Always) Already Taken", "Increasing and
decreasing motor and cognitive output") each had two different effects
independently replicated, so each contributes two rows with the same title.
The embeddings index is one row per *paper*, so those two papers' embedding
is duplicated across their two rows here (unlike the RF script, where the
two rows differ on effect_size.o/p_value.o/etc.; the text embedding can't
distinguish between two effects tested in the same paper). Both rows of each
duplicated pair share the same `replicated` value in Altmejd's data (checked
directly), so this does not create a label conflict.

Matching is by normalized paper title (llm_uq's title-cleaning strips
punctuation without inserting spaces, e.g. "1/f noise" -> "1f noise",
"action-based" -> "actionbased"; matching mirrors that). One of the 90 rows,
All 90 analysis rows have a corresponding embedded study. The
analysis scope and covariates come from Altmejd et al., while the outcome
comes from the original RPP OSF export (data/rpp_data.csv). The
Altmejd-derived outcome is retained as `replicated_altmejd` for auditing.

Like predict_embed_lr_cb.py/_ssrp.py, a single 5-fold split is noisy at this
n, so the nested CV (inner hyperparameter search AND outer evaluation) is
repeated under N_SEEDS=20 different fold-split seeds. Two aggregations of
those 20 seeds are reported: per-seed mean (mean/SD of the 20 individual
per-seed AUC/Accuracy values) and mean probability (AUC/Accuracy of the
per-study average out-of-fold probability across seeds, with the closed-form
Hanley-McNeil/Wald SD of that point estimate) -- see predict_embed_lr_ssrp.py's
docstring for the fuller rationale.

The study_predictions.csv output keeps BOTH "proba_cv" (seed 0's single-seed
out-of-fold prediction -- byte-identical to what this script produced before
the 20-seed rewrite, since seed 0's inner/outer random states, 0 and 1,
match the original RANDOM_STATE=0/RANDOM_STATE+1=1 exactly) and the new
"proba_cv_avg"/"proba_cv_sd_across_seeds": uq_bound/common.py's Freedman
bound chain reads "proba_cv" and make_roc_figures.py's load_rpp() does too,
so neither needed to change to pick up this rewrite.

Usage: rep_env/bin/python prediction/embed_pred/predict_embed_lr.py
Writes: prediction/embed_pred/RPP/embed_lr_metrics.csv,
        prediction/embed_pred/RPP/embed_lr_per_seed.csv,
        prediction/embed_pred/RPP/embed_lr_study_predictions.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def wald_accuracy_sd(acc, n):
    return np.sqrt(acc * (1 - acc) / n)


def hanley_mcneil_auc_sd(auc, n_pos, n_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return np.sqrt(var)


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
from rpp_outcomes import attach_canonical_rpp_outcomes  # noqa: E402

EMB_DIR = os.path.join(REPO, "prediction", "embed_pred", "RPP", "embeddings_text-embedding-3-large_query")
ALTMEJD_CSV = os.path.join(REPO, "data", "altmejd_osf", "data.csv")
OUT_DIR = os.path.join(REPO, "prediction", "embed_pred", "RPP")
OUT_CSV = os.path.join(OUT_DIR, "embed_lr_metrics.csv")
OUT_PERSEED_CSV = os.path.join(OUT_DIR, "embed_lr_per_seed.csv")
OUT_STUDY_CSV = os.path.join(OUT_DIR, "embed_lr_study_predictions.csv")

N = 90  # hard cap on covariates, matches n (no more predictors than observations)
GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "sel__max_features": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90],
}
N_SEEDS = 20
N_JOBS = 1  # deterministic nested CV; avoids nested joblib process pools

PIPE = Pipeline([
    ("sc", StandardScaler()),
    ("sel", SelectFromModel(
        LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0),
        max_features=N, threshold=-np.inf)),
    ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0, max_iter=5000, random_state=0)),
])


def normalize_title(s):
    """Match llm_uq's title-cleaning: lowercase, strip punctuation with no
    space inserted (so hyphens/apostrophes/slashes/colons just vanish)."""
    s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def load_rf_rpp_table():
    """Altmejd's 90-row scope with outcomes from the original RPP export."""
    alt = pd.read_csv(ALTMEJD_CSV)
    rpp = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
    rpp["norm_title"] = rpp["title"].apply(normalize_title)
    rpp = attach_canonical_rpp_outcomes(rpp)
    print(f"Filtered RPP table: {len(rpp)} study-effect rows, "
          f"{rpp['norm_title'].nunique()} unique paper titles "
          f"({(rpp['norm_title'].duplicated(keep=False)).sum()} rows from "
          f"papers with 2 replicated effects)")
    return rpp


def load_embedding_lookup():
    """One feature vector (context + pooled) per embedded paper, keyed by
    normalized title."""
    ctx = np.load(os.path.join(EMB_DIR, "context.npy"))
    pooled = np.load(os.path.join(EMB_DIR, "pooled.npy"))
    idx = pd.read_csv(os.path.join(EMB_DIR, "index.csv"))
    feats = np.hstack((ctx, pooled))
    norm_titles = idx.title.apply(normalize_title)
    assert not norm_titles.duplicated().any(), "embeddings index has duplicate titles"
    return dict(zip(norm_titles, feats))


def run_nested_cv(X, y, seed):
    """One nested-CV pass under a given fold-split seed. Inner/outer use
    random states (2*seed, 2*seed+1) -- seed 0 gives (0, 1), exactly the
    original single-seed script's RANDOM_STATE/RANDOM_STATE+1."""
    inner = StratifiedKFold(5, shuffle=True, random_state=2 * seed)
    outer = StratifiedKFold(5, shuffle=True, random_state=2 * seed + 1)
    grid = GridSearchCV(PIPE, GRID, cv=inner, scoring="roc_auc", n_jobs=N_JOBS)
    return cross_val_predict(grid, X, y, cv=outer, method="predict_proba", n_jobs=N_JOBS)[:, 1]


def main():
    rpp = load_rf_rpp_table()
    emb_lookup = load_embedding_lookup()

    matched = rpp["norm_title"].isin(emb_lookup)
    print(f"of those, matching an embedded paper: {matched.sum()} "
          f"({(~matched).sum()} dropped, no embedding: "
          f"{rpp.loc[~matched, 'title'].tolist()})")

    rpp = rpp[matched]
    X = np.vstack([emb_lookup[t] for t in rpp["norm_title"]])
    y = rpp["replicated"].astype(int).values
    n = len(y)
    print(f"n = {n} studies used for training")

    # Primary: single reference fit on all data (optimistic; matches the
    # "best hyperparameter" cell of embeddings_LR.ipynb)
    inner0 = StratifiedKFold(5, shuffle=True, random_state=0)
    grid0 = GridSearchCV(PIPE, GRID, cv=inner0, scoring="roc_auc", n_jobs=N_JOBS)
    grid0.fit(X, y)
    proba_in = grid0.predict_proba(X)[:, 1]
    pred_in = (proba_in >= 0.5).astype(int)
    non_zero = np.where(grid0.best_estimator_.named_steps["clf"].coef_[0] != 0)[0]
    print(f"\nbest params (selected on full data): {grid0.best_params_}, "
          f"active features: {len(non_zero)}")

    # Secondary: nested CV under N_SEEDS different fold splits; final
    # prediction is the per-study average out-of-fold probability across seeds.
    print(f"\nrunning nested CV under {N_SEEDS} seeds (outer 5-fold, inner 5-fold hyperparameter search) ...")
    proba_by_seed = np.full((N_SEEDS, n), np.nan)
    per_seed = []
    for seed in range(N_SEEDS):
        proba_by_seed[seed] = run_nested_cv(X, y, seed)
        auc_s = roc_auc_score(y, proba_by_seed[seed])
        brier_s = brier_score_loss(y, proba_by_seed[seed])
        acc_s = accuracy_score(y, (proba_by_seed[seed] >= 0.5).astype(int))
        per_seed.append({"seed": seed, "AUC": auc_s, "Brier": brier_s, "Accuracy": acc_s})
        print(f"  seed {seed:2d}: AUC={auc_s:.3f}  Accuracy={acc_s:.3f}")
    per_seed = pd.DataFrame(per_seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed.to_csv(OUT_PERSEED_CSV, index=False)
    print(f"Wrote {OUT_PERSEED_CSV}")
    print(f"\nAcross {N_SEEDS} seeds: AUC mean={per_seed['AUC'].mean():.3f} "
          f"sd={per_seed['AUC'].std():.3f}; Accuracy mean={per_seed['Accuracy'].mean():.3f} "
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
        {"model": "Text-embedding LR (high-quality RPP subset)", "evaluation": "in-sample", "n": n,
         "AUC": roc_auc_score(y, proba_in), "Brier": brier_score_loss(y, proba_in),
         "Accuracy": accuracy_score(y, pred_in)},
        {"model": "Text-embedding LR (high-quality RPP subset)",
         "evaluation": f"nested CV, {N_SEEDS}-seed avg prediction", "n": n,
         "AUC": auc_cv_avg, "AUC_SD": auc_cv_sd, "Brier": brier_cv_avg,
         "Accuracy": acc_cv_avg, "Accuracy_SD": acc_cv_sd},
        {"model": "Text-embedding LR (high-quality RPP subset)",
         "evaluation": f"nested CV, per-seed (mean of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].mean(), "Brier": per_seed["Brier"].mean(),
         "Accuracy": per_seed["Accuracy"].mean()},
        {"model": "Text-embedding LR (high-quality RPP subset)",
         "evaluation": f"nested CV, per-seed (sd of {N_SEEDS})", "n": n,
         "AUC": per_seed["AUC"].std(), "Brier": per_seed["Brier"].std(),
         "Accuracy": per_seed["Accuracy"].std()},
    ]
    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False))

    results.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # study-level predictions: proba_cv is seed 0's out-of-fold prediction
    # (backward-compatible with existing consumers -- see module docstring),
    # proba_cv_avg/proba_cv_sd_across_seeds are the new 20-seed aggregate.
    study_preds = pd.DataFrame({
        "altmejd_id": rpp["id"].values,
        "title": rpp["title"].values,
        "y": y,
        "proba_in": proba_in,
        "proba_cv": proba_by_seed[0],
        "proba_cv_avg": proba_cv_avg,
        "proba_cv_sd_across_seeds": proba_by_seed.std(axis=0),
    })
    study_preds.to_csv(OUT_STUDY_CSV, index=False)
    print(f"Wrote {OUT_STUDY_CSV}")


if __name__ == "__main__":
    main()
