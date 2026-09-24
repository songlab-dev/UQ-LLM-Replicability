"""Three related tables built from the same per-corpus q-bar/v-bar metrics,
selected with --table:

  main     (default) All three datasets (RPP, CB, SSRP) side by side, two
           rows each: gpt-5.4-mini's two computable *probability estimators*
           -- verbalized confidence (q-bar) and self-consistency vote rate
           (v-bar) -- no baselines. -> table_llm_uq.tex

  cb       CB only, three rows: q-bar, v-bar, and the text-embedding Lasso
           LR baseline (20-seed nested-CV average). -> table_llm_uq_cb.tex

  cross_corpus  CB + SSRP side by side, two rows each (q-bar, v-bar only -- no
           embedding-LR row here; that's covered by table_embed_cross_corpus.tex
           and table_predictor_comparison_cross_corpus.tex). CB uses the full
           158-effect scope; SSRP uses all 21 studies (one row per paper).
           -> table_llm_uq_cross_corpus.tex

Row label is "elicitation method", not "UQ signal": q-bar and v-bar are both
just point estimates of p(replicates), scored identically to any other
model's predicted probability. What makes them "UQ" is only how they're
extracted from the model (verbalization vs. resampling), not the number
itself. To keep that distinction visible instead of collapsing it into one
undifferentiated row of numbers, columns are split into two groups:

    vs. ground truth   AUC / Accuracy / ECE -- needs the replication outcome
                        to compute. AUC and Accuracy SDs are closed-form
                        binomial-based formulas (Hanley & McNeil 1982 for
                        AUC, Wald SE for Accuracy), not a bootstrap -- see
                        predict_llm_uq.py / predict_llm_verdict.py. Brier is
                        dropped: it isn't a proportion, so it has no matching
                        closed form.
    label-free (UQ)     run-to-run dispersion across the R resamples --
                        computable with no ground truth at all, which is
                        the actual uncertainty-quantification quantity.
                        q-bar's is q_hat SD; v-bar's is verdict entropy +
                        pairwise agreement (different statistics because
                        one elicits a continuous number, the other a vote).

R is read from each corpus's predictions file rather than hardcoded (R=100
for all three). ECE is recomputed from the stored
reliability rows: RPP and CB use five fixed-size quantile bins; SSRP uses
three; scores are ordered with explicit NumPy quicksort before
``array_split`` assigns the bins, preserving the stored RPP tie convention.
CB uses the "_cb_full" metric files (158 effects).

Reads the four CSVs per corpus that predict_llm_uq.py/predict_llm_verdict.py
already wrote (plus the embedding-LR CSVs for --table cb):

    prediction/llm_pred/metric/llm_qbar{_c}_metrics.csv              AUC/Acc + binomial-based SD for q-bar
    prediction/llm_pred/metric/llm_verdict{_c}_metrics.csv           AUC/Acc + binomial-based SD for v-bar
    prediction/llm_pred/metric/llm_calibration{_c}.csv               quantile-binned reliability table -> ECE
    prediction/llm_pred/metric/llm_selfconsistency{_c}_dispersion.csv  per-study dispersion -> mean q_hat SD /
                                                    agreement/entropy, and AUC(agreement -> is-correct)
    prediction/embed_pred/CB/embed_lr_metrics.csv, embed_lr_study_predictions.csv  (--table cb only)

Usage: rep_env/bin/python tables/llm_uq_table.py --table main
       rep_env/bin/python tables/llm_uq_table.py --table cb
       rep_env/bin/python tables/llm_uq_table.py --table cross_corpus
Writes: tables/table_llm_uq.tex          (--table main)
        tables/table_llm_uq_cb.tex       (--table cb)
        tables/table_llm_uq_cross_corpus.tex  (--table cross_corpus)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PACKAGE_DIR, "prediction", "llm_pred", "metric")
REASONING = os.path.join(PACKAGE_DIR, "prediction", "LLM_Reasoning")
EMBED_DIR = os.path.join(PACKAGE_DIR, "prediction", "embed_pred")

MODEL_NAME = "gpt-5.4-mini"
N_CAL_BINS_CB = 5
N_SEEDS = 20

# One entry per corpus. "slug" is the infix predict_llm_uq.py /
# predict_llm_verdict.py give their output files ("" for RPP).
CORPORA = [
    dict(key="RPP", label="RPP", slug="",
         pred=os.path.join(REASONING, "RPP", "text_predictions_high_temp0.7.csv"),
         unit="studies"),
    dict(key="CB", label="CB", slug="_cb_full",
         pred=os.path.join(REASONING, "CB", "text_predictions_high_temp0.7.csv"),
         unit="effects"),
    dict(key="SSRP", label="SSRP", slug="_ssrp",
         pred=os.path.join(REASONING, "SSRP", "text_predictions_high_temp0.7.csv"),
         unit="studies"),
]
CORPUS_BY_KEY = {c["key"]: c for c in CORPORA}


def ece_from_table(cal, predictor):
    t = cal[cal.predictor == predictor]
    n_tot = t["n"].sum()
    return float((t["n"] / n_tot * (t["observed_rate"] - t["mean_predicted"]).abs()).sum())


def ece_from_proba(y, proba, n_bins):
    """ECE under the recorded fixed-size, quicksort tie convention."""
    order = np.argsort(proba, kind="quicksort")
    y_sorted, proba_sorted = y[order], proba[order]
    bins = np.array_split(np.arange(len(y_sorted)), n_bins)
    return sum(
        (len(indices) / len(y_sorted))
        * abs(y_sorted[indices].mean() - proba_sorted[indices].mean())
        for indices in bins
    )


def runs_per_unit(pred_csv):
    """R, read from the predictions file itself -- see module docstring."""
    return int(pd.read_csv(pred_csv, usecols=["run_id"])["run_id"].nunique())


def load_corpus(cfg):
    s = cfg["slug"]
    qbar = pd.read_csv(os.path.join(RES, f"llm_qbar{s}_metrics.csv")).iloc[0]
    vbar = pd.read_csv(os.path.join(RES, f"llm_verdict{s}_metrics.csv")).iloc[0]
    cal = pd.read_csv(os.path.join(RES, f"llm_calibration{s}.csv"))
    disp = pd.read_csv(os.path.join(RES, f"llm_selfconsistency{s}_dispersion.csv"))

    assert qbar.n == vbar.n == len(disp), (
        f"{cfg['key']}: n mismatch across metric files "
        f"(qbar={qbar.n}, vbar={vbar.n}, dispersion={len(disp)}) -- rerun "
        f"predict_llm_uq.py/predict_llm_verdict.py --dataset {cfg['key'].lower()} together.")

    return dict(
        cfg=cfg,
        n=int(qbar.n),
        R=runs_per_unit(cfg["pred"]),
        auc_agree_correct=float(roc_auc_score(disp["correct"], disp["pairwise_agreement"])),
        rows=[
            {
                "method": r"Verbalized confidence ($\bar q$)",
                "AUC": f"{qbar.AUC:.3f} $\\pm$ {qbar.AUC_SD:.3f}",
                "Accuracy": f"{qbar.Accuracy:.3f} $\\pm$ {qbar.Accuracy_SD:.3f}",
                "ECE": f"{ece_from_table(cal, 'qbar'):.3f}",
                "dispersion": rf"SD($\hat q$) $= {disp.q_sd.mean():.3f}$",
            },
            {
                "method": r"Self-consistency vote rate ($\bar v$)",
                "AUC": f"{vbar.AUC:.3f} $\\pm$ {vbar.AUC_SD:.3f}",
                "Accuracy": f"{vbar.Accuracy:.3f} $\\pm$ {vbar.Accuracy_SD:.3f}",
                "ECE": f"{ece_from_table(cal, 'vbar'):.3f}",
                "dispersion": (rf"entropy $= {disp.verdict_entropy_bits.mean():.3f}$ bits, "
                               rf"agree.\ $= {disp.pairwise_agreement.mean():.3f}$"),
            },
        ],
    )


def build_main():
    corpora = [load_corpus(c) for c in CORPORA]

    for c in corpora:
        print(f"\n{c['cfg']['label']}: n = {c['n']} {c['cfg']['unit']}, R = {c['R']}")
        print(pd.DataFrame(c["rows"]).to_string(index=False))
        print(f"AUC(pairwise agreement -> aggregated verdict is correct) = "
              f"{c['auc_agree_correct']:.3f}")

    Rs = sorted({c["R"] for c in corpora})
    r_txt = rf"$R{{=}}{Rs[0]}$" if len(Rs) == 1 else \
        ", ".join(rf"{c['cfg']['label']} $R{{=}}{c['R']}$" for c in corpora)
    scope = "; ".join(rf"{c['cfg']['label']} $n={c['n']}$" for c in corpora)
    agree = ", ".join(rf"{c['cfg']['label']} ${c['auc_agree_correct']:.3f}$" for c in corpora)

    tex = [
        r"% Generated by llm_uq_table.py --table main -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Two label-free ways to elicit a probability estimate from "
        rf"{MODEL_NAME}, across all three corpora ({scope}): verbalized "
        rf"confidence $\bar q$ (mean stated replication probability across "
        rf"{r_txt} zero-shot runs) and self-consistency vote rate $\bar v$ "
        rf"(fraction of those runs verdicting `replicable'). The left block "
        rf"scores each estimate against the true replication outcome "
        rf"(AUC $\pm$ Hanley\textendash McNeil (1982) SE; Accuracy $\pm$ Wald "
        rf"binomial SE; ECE from fixed-size quantile bins (5 for RPP/CB, "
        rf"3 for SSRP), with tied scores ordered by the recorded NumPy quicksort "
        rf"convention); the right "
        rf"block is the estimate's own run-to-run dispersion, computable with "
        rf"no outcome label at all. In every corpus $\bar v$ separates classes "
        rf"slightly better (AUC) but is worse calibrated (ECE) than $\bar q$ "
        rf"-- the coarse vote fraction is pushed toward 0/1 by the "
        rf"{Rs[0] if len(Rs) == 1 else 'R'}-run majority. "
        rf"Pairwise run-to-run agreement tracks whether the aggregated verdict "
        rf"is later correct only weakly, AUC(agreement $\rightarrow$ correct): "
        rf"{agree}.}}",
        r"\label{tab:llm-uq}", r"\small",
        r"\begin{tabular}{@{}l cc c c@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{vs.\ ground truth} & \multicolumn{1}{c}{label-free} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-5}",
        r"\textbf{Elicitation method} & \textbf{AUC} & \textbf{Accuracy} & "
        r"\textbf{ECE} & \textbf{Run-to-run dispersion} \\",
    ]
    for i, c in enumerate(corpora):
        tex.append(r"\midrule" if i == 0 else r"\addlinespace")
        tex.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{c['cfg']['label']}}} "
                   rf"($n={c['n']}$ {c['cfg']['unit']}, $R={c['R']}$)}} \\")
        for r_ in c["rows"]:
            tex.append(rf"\quad {r_['method']} & {r_['AUC']} & {r_['Accuracy']} & "
                       rf"{r_['ECE']} & {r_['dispersion']} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_tex = os.path.join(PACKAGE_DIR, "tables", "table_llm_uq.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"\nWrote {out_tex}")


def build_cb():
    corpus = load_corpus(CORPUS_BY_KEY["CB"])
    n = corpus["n"]

    embed_metrics = pd.read_csv(os.path.join(EMBED_DIR, "CB", "embed_lr_metrics.csv"))
    embed = embed_metrics[embed_metrics["evaluation"].str.contains(f"{N_SEEDS}-seed avg")].iloc[0]
    embed_predictions = pd.read_csv(os.path.join(EMBED_DIR, "CB", "embed_lr_study_predictions.csv"))
    assert n == embed.n == len(embed_predictions)

    rows = corpus["rows"] + [{
        "method": r"Text-embedding Lasso LR (20-seed nested-CV average)",
        "AUC": f"{embed.AUC:.3f} $\\pm$ {embed.AUC_SD:.3f}",
        "Accuracy": f"{embed.Accuracy:.3f} $\\pm$ {embed.Accuracy_SD:.3f}",
        "ECE": f"{ece_from_proba(embed_predictions.y.values, embed_predictions.proba_cv_avg.values, N_CAL_BINS_CB):.3f}",
        "dispersion": rf"SD($\hat p$) $= {embed_predictions.proba_cv_sd_across_seeds.mean():.3f}$",
    }]

    tex = [
        r"% Generated by llm_uq_table.py --table cb -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Replication-prediction performance on all {n} completed CB effects: "
        rf"{MODEL_NAME}'s verbalized confidence $\bar q$ and self-consistency vote rate "
        rf"$\bar v$ (each averaged over $R{{=}}100$ zero-shot runs), and text-embedding "
        rf"Lasso LR (the mean held-out probability over {N_SEEDS} nested-CV seeds). "
        rf"AUC $\pm$ Hanley\textendash McNeil (1982) SE; Accuracy $\pm$ Wald binomial SE; "
        rf"ECE uses {N_CAL_BINS_CB} fixed-size quantile bins with the recorded NumPy quicksort "
        rf"tie convention.}}",
        r"\label{tab:llm-uq-cb}", r"\small",
        r"\begin{tabular}{@{}l cc c c@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{vs.\ ground truth} & \multicolumn{1}{c}{label-free} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-5}",
        r"\textbf{Predictor} & \textbf{AUC} & \textbf{Accuracy} & "
        r"\textbf{ECE} & \textbf{Run-to-run dispersion} \\", r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['method']} & {row['AUC']} & {row['Accuracy']} & "
            f"{row['ECE']} & {row['dispersion']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_tex = os.path.join(PACKAGE_DIR, "tables", "table_llm_uq_cb.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out_tex}")


def build_cross_corpus():
    cb = load_corpus(CORPUS_BY_KEY["CB"])
    ssrp = load_corpus(CORPUS_BY_KEY["SSRP"])

    print(f"CB (n={cb['n']}, full scope)")
    print(pd.DataFrame(cb["rows"]).to_string(index=False))
    print(f"\nSSRP (n={ssrp['n']})")
    print(pd.DataFrame(ssrp["rows"]).to_string(index=False))

    tex = [
        r"% Generated by llm_uq_table.py --table cross_corpus -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Two label-free ways to elicit a probability estimate from "
        rf"{MODEL_NAME}, on all {cb['n']} completed CB effects and all "
        rf"{ssrp['n']} SSRP studies: verbalized confidence $\bar q$ (mean stated "
        rf"replication probability across $R$ zero-shot runs per effect/study, "
        rf"$R{{=}}{cb['R']}$ for CB and $R{{=}}{ssrp['R']}$ for SSRP) and "
        rf"self-consistency vote rate $\bar v$ (fraction of those runs verdicting "
        rf"`replicable'). The left block scores each estimate against the true "
        rf"replication outcome (AUC $\pm$ Hanley\textendash McNeil (1982) SE; "
        rf"Accuracy $\pm$ Wald binomial SE; ECE from quantile-binned reliability "
        rf"bins, 5 for CB and 3 for SSRP, scaled to $n$); the right block is "
        rf"each estimate's own run-to-run dispersion. Both corpora's rows are "
        rf"run against corrected (post-anonymization-fix) text; $n$ is small for "
        rf"SSRP, so its point estimates should be read as noisier than CB's. A "
        rf"text-embedding Lasso LR baseline for both corpora is in "
        rf"table\_embed\_cross\_corpus.tex / table\_predictor\_comparison\_cross\_corpus.tex, "
        rf"not repeated here.}}",
        r"\label{tab:llm-uq-cross-corpus}", r"\small",
        r"\begin{tabular}{@{}ll cc c c@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{vs.\ ground truth} & \multicolumn{1}{c}{label-free} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-6}",
        r"\textbf{Corpus} & \textbf{Elicitation method} & \textbf{AUC} & \textbf{Accuracy} & "
        r"\textbf{ECE} & \textbf{Run-to-run dispersion} \\", r"\midrule",
    ]
    for corpus_label, rows in [(r"CB", cb["rows"]), (r"SSRP", ssrp["rows"])]:
        for i, r_ in enumerate(rows):
            corpus_cell = corpus_label if i == 0 else ""
            tex.append(f"{corpus_cell} & {r_['method']} & {r_['AUC']} & {r_['Accuracy']} & "
                        f"{r_['ECE']} & {r_['dispersion']} \\\\")
        if corpus_label != "SSRP":
            tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_tex = os.path.join(PACKAGE_DIR, "tables", "table_llm_uq_cross_corpus.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"\nWrote {out_tex}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["main", "cb", "cross_corpus"], default="main")
    table = parser.parse_args().table
    {"main": build_main, "cb": build_cb, "cross_corpus": build_cross_corpus}[table]()


if __name__ == "__main__":
    main()
