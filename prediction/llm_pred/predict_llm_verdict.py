"""gpt-5.4-mini zero-shot verdict-rate (v-bar) predictor for RPP/CB/SSRP
replicability, generalized across corpora with one --dataset flag. v-bar is
each claim's fraction of R=100 independent zero-shot runs whose verdict was
"replicable". Unlike the embedding/RF models, this predictor is never fit on
outcomes, so there is no in-sample/nested-CV distinction -- one row,
"zero-shot", evaluated on all matched claims directly.

Formerly three separate, near-identical scripts: predict_llm_verdict.py
(RPP), predict_llm_verdict_cb_full.py (CB, all 158 completed effects), and
predict_llm_verdict_ssrp.py (SSRP, all 21 studies). RPP needs its own
scope/outcome loader: Altmejd's `drop == False` field defines the analysis
scope, while outcomes come from the original OSF `data/rpp_data.csv` via
`data/rpp_outcomes.py`, not the predictions file's own `ground_truth`.
Matching: Altmejd title -> normalized -> `data/rpp_data_cleaned.csv`
"Study Title (O)" (exact string match to the predictions CSV). CB and SSRP
read `ground_truth` directly off the predictions file and group by their own
claim keys.

SD: AUC and Accuracy use closed-form binomial-based formulas rather than a
claim-resampling bootstrap, since both are themselves proportions of
"successes" over the n matched claims. Accuracy is literally a Binomial(n,
acc) proportion, so SD(accuracy) = sqrt(acc(1-acc)/n) (Wald SE). AUC is a
Mann-Whitney U-statistic estimating P(score_pos > score_neg); its variance
has the same "successes/trials" flavor but must correct for the fact that
each of the n1*n0 comparison pairs shares an underlying claim, so it's not
i.i.d. Bernoulli -- that's the Hanley & McNeil (1982) formula, used here.
(Brier is dropped: it isn't a proportion, so it has no matching closed form.)

Usage: rep_env/bin/python prediction/llm_pred/predict_llm_verdict.py --dataset rpp
       rep_env/bin/python prediction/llm_pred/predict_llm_verdict.py --dataset cb
       rep_env/bin/python prediction/llm_pred/predict_llm_verdict.py --dataset ssrp
Writes: prediction/llm_pred/metric/llm_verdict{_cb_full,_ssrp}{suffix}_metrics.csv
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
from rpp_outcomes import attach_canonical_rpp_outcomes  # noqa: E402

ALTMEJD_CSV = os.path.join(REPO, "data", "altmejd_osf", "data.csv")
RPP_CLEANED_CSV = os.path.join(REPO, "data", "rpp_data_cleaned.csv")
MODEL_NAME = "gpt-5.4-mini"

DATASET_DIR = {"rpp": "RPP", "cb": "CB", "ssrp": "SSRP"}
OUT_SLUG = {"rpp": "", "cb": "_cb_full", "ssrp": "_ssrp"}
KEY_COLS = {
    "rpp": ["Study Title (O)"],
    "cb": ["paper_num", "experiment_num", "effect_num"],
    "ssrp": ["study_num"],
}
DATASET_LABEL = {"rpp": "high-quality RPP subset", "cb": "CB full scope", "ssrp": "SSRP"}
UNIT_NAME = {"rpp": "studies", "cb": "effects", "ssrp": "studies"}


def normalize_title(s):
    """Match llm_uq's title-cleaning: lowercase, strip punctuation with no
    space inserted (so hyphens/apostrophes/slashes/colons just vanish)."""
    s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def wald_accuracy_sd(acc, n):
    """Accuracy is a Binomial(n, acc) proportion (correct/incorrect per claim)."""
    return np.sqrt(acc * (1 - acc) / n)


def hanley_mcneil_auc_sd(auc, n_pos, n_neg):
    """Hanley & McNeil (1982): analytic SE of AUC as a Mann-Whitney U-statistic,
    P(score_pos > score_neg) over n_pos*n_neg comparison pairs. Q1/Q2 correct
    the naive binomial variance auc(1-auc)/(n_pos*n_neg) for the pairs sharing
    a claim (not i.i.d. Bernoulli trials)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return np.sqrt(var)


def load_verdict_frac_rpp(pred_csv):
    """Altmejd's 90-row scope with outcomes from the original RPP export."""
    alt = pd.read_csv(ALTMEJD_CSV)
    rpp = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
    rpp["norm_title"] = rpp["title"].apply(normalize_title)
    rpp = attach_canonical_rpp_outcomes(rpp)

    rppc = pd.read_csv(RPP_CLEANED_CSV, encoding="latin1")
    norm = rppc["Study Title (O)"].apply(normalize_title)
    assert not norm.duplicated().any(), "rpp_data_cleaned has duplicate normalized titles"
    title_map = dict(zip(norm, rppc["Study Title (O)"]))
    rpp["raw_title"] = rpp["norm_title"].map(title_map)

    pred = pd.read_csv(pred_csv)
    pred["verdict_bin"] = pred["verdict"].astype(str).str.lower().map(
        {"replicable": 1, "unreplicable": 0})
    verdict_frac = pred.groupby("Study Title (O)")["verdict_bin"].mean()

    matched = rpp["raw_title"].isin(verdict_frac.index)
    print(f"matched {matched.sum()}/{len(rpp)} RF rows to a gpt-5.4-mini prediction "
          f"({(~matched).sum()} dropped: {rpp.loc[~matched, 'title'].tolist()})")
    rpp = rpp[matched]
    y = rpp["replicated"].astype(int).values
    proba = verdict_frac.loc[rpp["raw_title"]].values
    return y, proba


def load_verdict_frac_generic(pred_csv, key_cols):
    """CB/SSRP: ground_truth lives directly on the predictions file."""
    pred = pd.read_csv(pred_csv)
    pred["verdict_bin"] = pred["verdict"].astype(str).str.lower().map(
        {"replicable": 1, "unreplicable": 0})
    pred["y"] = pred["ground_truth"].astype(str).str.lower().map({"yes": 1, "no": 0})
    per_claim = pred.groupby(key_cols).agg(v_bar=("verdict_bin", "mean"), y=("y", "first"))
    return per_claim["y"].values, per_claim["v_bar"].values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rpp", "cb", "ssrp"], default="rpp")
    dataset = parser.parse_args().dataset

    reasoning_effort = os.getenv("REASONING_EFFORT", "high")
    temp = os.getenv("TEMP", "0.7")
    suffix = ("" if reasoning_effort == "high" else f"_{reasoning_effort}") + \
             ("" if temp == "0.7" else f"_temp{temp}")
    pred_csv = os.path.join(REPO, "prediction", "LLM_Reasoning", DATASET_DIR[dataset],
                             f"text_predictions_{reasoning_effort}_temp{temp}.csv")
    out_csv = os.path.join(REPO, "prediction", "llm_pred", "metric",
                            f"llm_verdict{OUT_SLUG[dataset]}{suffix}_metrics.csv")

    if dataset == "rpp":
        y, proba = load_verdict_frac_rpp(pred_csv)
    else:
        y, proba = load_verdict_frac_generic(pred_csv, KEY_COLS[dataset])
    n = len(y)
    note = (f" (CB, full scope, "
            f"{pd.read_csv(pred_csv)['paper_num'].nunique()} papers)" if dataset == "cb" else "")
    print(f"n = {n} {UNIT_NAME[dataset]}{note}")

    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    auc_sd = hanley_mcneil_auc_sd(auc, n_pos, n_neg)
    acc_sd = wald_accuracy_sd(acc, n)

    rows = [{
        "model": rf"{MODEL_NAME} vbar (mean verdict rate, {DATASET_LABEL[dataset]})",
        "evaluation": "zero-shot",
        "n": n,
        "AUC": auc, "AUC_SD": auc_sd,
        "Accuracy": acc, "Accuracy_SD": acc_sd,
    }]
    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
