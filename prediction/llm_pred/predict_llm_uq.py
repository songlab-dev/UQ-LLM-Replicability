"""The three UQ methods computable on gpt-5.4-mini's zero-shot responses,
generalized across RPP/CB/SSRP with one --dataset flag (see
predict_llm_verdict.py's docstring for why the other UQ families in the
related-work paragraph -- white-box token/logit/hidden-state methods, Duan
et al. 2025 token->task propagation -- are NOT computable: gpt-5.4-mini is
closed-weight and the Chat Completions API rejects `logprobs` outright for
this model, confirmed empirically).

Formerly three separate, near-identical scripts: predict_llm_uq.py (RPP),
predict_llm_uq_cb_full.py (CB, all 158 completed effects), and
predict_llm_uq_ssrp.py (SSRP, all 21 studies). RPP needs its own scope/outcome
loader (Altmejd et al.'s `drop == False` subset defines the 90-claim scope;
the original OSF `data/rpp_data.csv` supplies outcomes via
`data/rpp_outcomes.py`, not the predictions file's own `ground_truth`); CB
and SSRP read `ground_truth` directly off the predictions file and group by
their own claim keys.

1. Verbalized confidence (Kadavath 2022 / Lin 2022 style): q_hat is the
   model's own stated replication probability each run; q-bar = per-claim
   mean. Scored as a point predictor: AUC + Accuracy, each with a closed-form
   binomial-based SD (Hanley & McNeil 1982 for AUC, Wald SE for Accuracy)
   rather than a claim-resampling bootstrap, same protocol as
   predict_llm_verdict.py's v-bar. (Brier is dropped: not a proportion, so it
   has no matching closed form.)

2. Black-box resampling / self-consistency (Wang 2023, SelfCheckGPT/Manakul
   2023): per-claim dispersion across the R independent runs (R=100 as of
   2026-09-09; was 25 in the pilot) -- q_hat SD, binary entropy of the
   verdict split, and finite-sample pairwise verdict-agreement rate.
   Reported as their distribution across claims, plus the standard
   SelfCheckGPT-style validation: does higher agreement (lower disagreement)
   actually track whether the aggregated prediction is correct? (AUC of
   pairwise agreement predicting correctness.)

3. Calibration (Guo et al. 2017): reliability table + Expected Calibration
   Error (ECE) for both q-bar and v-bar against the observed outcome.
   Quantile-based bins rather than equal-width: 5 for RPP/CB (n=90/158), 3
   for SSRP (n=21 is too small for 5 to stay non-degenerate).

Usage: rep_env/bin/python prediction/llm_pred/predict_llm_uq.py --dataset rpp
       rep_env/bin/python prediction/llm_pred/predict_llm_uq.py --dataset cb
       rep_env/bin/python prediction/llm_pred/predict_llm_uq.py --dataset ssrp
       REASONING_EFFORT=low TEMP=0.2 rep_env/bin/python .../predict_llm_uq.py --dataset ssrp
           -- reads text_predictions_{effort}_temp{temp}.csv and suffixes
           output filenames accordingly (never overwrites the default cell)
Writes: prediction/llm_pred/metric/llm_qbar{_cb_full,_ssrp}{suffix}_metrics.csv
        prediction/llm_pred/metric/llm_selfconsistency{_cb_full,_ssrp}{suffix}_dispersion.csv
        prediction/llm_pred/metric/llm_calibration{_cb_full,_ssrp}{suffix}.csv
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
N_CAL_BINS = {"rpp": 5, "cb": 5, "ssrp": 3}
DATASET_LABEL = {"rpp": "high-quality RPP subset", "cb": "CB full scope", "ssrp": "SSRP"}
UNIT_NAME = {"rpp": "studies", "cb": "effects", "ssrp": "studies"}


def normalize_title(s):
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


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def pairwise_agreement(n1, r):
    """Finite-sample fraction of run pairs (without replacement) that agree,
    given n1 'replicable' votes out of r runs: [C(n1,2)+C(r-n1,2)] / C(r,2)."""
    n0 = r - n1
    same = n1 * (n1 - 1) / 2 + n0 * (n0 - 1) / 2
    total = r * (r - 1) / 2
    return same / total


def ece(y, proba, n_bins):
    """Quantile-binned ECE using the recorded deterministic sort convention.

    Scores are ordered with NumPy quicksort, then fixed-size bins are formed
    with ``array_split``. This explicitly preserves the ordering convention
    used for the stored calibration tables: tied discrete vote rates may span
    bins according to quicksort's deterministic order for this fixed NumPy
    environment and input table.
    """
    order = np.argsort(proba, kind="quicksort")
    y_s, p_s = y[order], proba[order]
    bin_edges = np.array_split(np.arange(len(y_s)), n_bins)
    rows = []
    ece_val = 0.0
    for b in bin_edges:
        conf = p_s[b].mean()
        acc = y_s[b].mean()
        n_b = len(b)
        ece_val += (n_b / len(y_s)) * abs(acc - conf)
        rows.append({"n": n_b, "mean_predicted": conf, "observed_rate": acc,
                     "gap": acc - conf})
    return ece_val, pd.DataFrame(rows)


def load_per_claim_rpp(pred_csv):
    """RPP's scope/outcome come from Altmejd + the canonical OSF export, not
    the predictions file's own ground_truth -- see data/rpp_outcomes.py."""
    alt = pd.read_csv(ALTMEJD_CSV)
    rpp = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
    rpp["norm_title"] = rpp["title"].apply(normalize_title)
    rpp = attach_canonical_rpp_outcomes(rpp)

    rppc = pd.read_csv(RPP_CLEANED_CSV, encoding="latin1")
    norm = rppc["Study Title (O)"].apply(normalize_title)
    assert not norm.duplicated().any()
    title_map = dict(zip(norm, rppc["Study Title (O)"]))
    rpp["raw_title"] = rpp["norm_title"].map(title_map)

    pred = pd.read_csv(pred_csv)
    pred["verdict_bin"] = pred["verdict"].astype(str).str.lower().map(
        {"replicable": 1, "unreplicable": 0})
    per_study = pred.groupby("Study Title (O)").agg(
        q_bar=("q_hat", "mean"), q_sd=("q_hat", "std"), v_bar=("verdict_bin", "mean"),
        r=("verdict_bin", "size"), n1=("verdict_bin", "sum"),
    )
    matched = rpp["raw_title"].isin(per_study.index)
    print(f"matched {matched.sum()}/{len(rpp)} RF rows to a gpt-5.4-mini prediction "
          f"({(~matched).sum()} dropped: {rpp.loc[~matched, 'title'].tolist()})")
    rpp = rpp[matched]
    per_claim = per_study.loc[rpp["raw_title"]].reset_index()
    per_claim["y"] = rpp["replicated"].astype(int).values
    return per_claim


def load_per_claim_generic(pred_csv, key_cols):
    """CB/SSRP: ground_truth lives directly on the predictions file."""
    pred = pd.read_csv(pred_csv)
    pred["verdict_bin"] = pred["verdict"].astype(str).str.lower().map(
        {"replicable": 1, "unreplicable": 0})
    pred["y"] = pred["ground_truth"].astype(str).str.lower().map({"yes": 1, "no": 0})
    return pred.groupby(key_cols).agg(
        q_bar=("q_hat", "mean"), q_sd=("q_hat", "std"), v_bar=("verdict_bin", "mean"),
        r=("verdict_bin", "size"), n1=("verdict_bin", "sum"), y=("y", "first"),
    ).reset_index()


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
    slug = OUT_SLUG[dataset]
    metric_dir = os.path.join(REPO, "prediction", "llm_pred", "metric")
    out_qbar = os.path.join(metric_dir, f"llm_qbar{slug}{suffix}_metrics.csv")
    out_dispersion = os.path.join(metric_dir, f"llm_selfconsistency{slug}{suffix}_dispersion.csv")
    out_cal = os.path.join(metric_dir, f"llm_calibration{slug}{suffix}.csv")
    key_cols = KEY_COLS[dataset]

    if dataset == "rpp":
        per_claim = load_per_claim_rpp(pred_csv)
    else:
        per_claim = load_per_claim_generic(pred_csv, key_cols)
    n = len(per_claim)
    note = f" ({per_claim['paper_num'].nunique()} papers)" if dataset == "cb" else ""
    print(f"n = {n} {UNIT_NAME[dataset]}{note}\n")

    y = per_claim["y"].values
    n_pos = int(y.sum())
    n_neg = n - n_pos

    # ── 1. Verbalized confidence (q-bar) ─────────────────────────────────────
    q_bar = per_claim["q_bar"].values
    auc_q = roc_auc_score(y, q_bar)
    acc_q = accuracy_score(y, (q_bar >= 0.5).astype(int))
    auc_q_sd = hanley_mcneil_auc_sd(auc_q, n_pos, n_neg)
    acc_q_sd = wald_accuracy_sd(acc_q, n)
    qbar_row = pd.DataFrame([{
        "model": rf"{MODEL_NAME} qbar (verbalized confidence, {DATASET_LABEL[dataset]})",
        "evaluation": "zero-shot", "n": n,
        "AUC": auc_q, "AUC_SD": auc_q_sd,
        "Accuracy": acc_q, "Accuracy_SD": acc_q_sd,
    }])
    os.makedirs(os.path.dirname(out_qbar), exist_ok=True)
    qbar_row.to_csv(out_qbar, index=False)
    print("1. Verbalized confidence (q-bar)")
    print(qbar_row.to_string(index=False))
    print(f"   Wrote {out_qbar}\n")

    # ── 2. Self-consistency / resampling dispersion ─────────────────────────
    v_bar = per_claim["v_bar"].values
    entropy = binary_entropy(v_bar)
    agree = np.array([pairwise_agreement(n1, r) for n1, r in
                       zip(per_claim["n1"], per_claim["r"])])
    q_sd = per_claim["q_sd"].values
    pred_v = (v_bar >= 0.5).astype(int)
    correct = (pred_v == y).astype(int)

    disp = per_claim[key_cols].copy()
    disp["q_bar"] = q_bar
    disp["q_sd"] = q_sd
    disp["v_bar"] = v_bar
    disp["verdict_entropy_bits"] = entropy
    disp["pairwise_agreement"] = agree
    disp["correct"] = correct
    disp.to_csv(out_dispersion, index=False)

    r_count = pd.read_csv(pred_csv, usecols=["run_id"])["run_id"].nunique()
    auc_agree_correct = roc_auc_score(correct, agree) if len(np.unique(correct)) > 1 else np.nan
    print(f"2. Self-consistency / resampling dispersion "
          f"(R={r_count} runs/{UNIT_NAME[dataset].rstrip('s')})")
    print(f"   q_hat SD:            mean={q_sd.mean():.3f}  median={np.median(q_sd):.3f}")
    print(f"   verdict entropy:     mean={entropy.mean():.3f} bits  median={np.median(entropy):.3f} bits")
    print(f"   pairwise agreement:  mean={agree.mean():.3f}  median={np.median(agree):.3f}")
    if dataset == "rpp":
        print(f"   agreement by correctness: correct={agree[correct==1].mean():.3f} "
              f"(n={int((correct==1).sum())})  incorrect={agree[correct==0].mean():.3f} "
              f"(n={int((correct==0).sum())})")
    print(f"   AUC(pairwise agreement -> is-correct): {auc_agree_correct:.3f}")
    print(f"   Wrote {out_dispersion}\n")

    # ── 3. Calibration (ECE) for q-bar and v-bar ─────────────────────────────
    n_bins = N_CAL_BINS[dataset]
    ece_q, tbl_q = ece(y, q_bar, n_bins)
    ece_v, tbl_v = ece(y, v_bar, n_bins)
    tbl_q.insert(0, "predictor", "qbar")
    tbl_v.insert(0, "predictor", "vbar")
    tbl_q.insert(1, "bin", range(1, len(tbl_q) + 1))
    tbl_v.insert(1, "bin", range(1, len(tbl_v) + 1))
    cal = pd.concat([tbl_q, tbl_v], ignore_index=True)
    cal.to_csv(out_cal, index=False)
    print(f"3. Calibration ({n_bins} quantile bins)")
    print(f"   ECE(qbar) = {ece_q:.4f}")
    print(f"   ECE(vbar) = {ece_v:.4f}")
    print("   " + cal.to_string(index=False).replace("\n", "\n   "))
    print(f"   Wrote {out_cal}")


if __name__ == "__main__":
    main()
