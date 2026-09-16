"""Data-specific classification-metrics tables (AUC/Accuracy) for the two
prediction sources this repo scores against RPP/CB/SSRP -- the LLM zero-shot
pipeline ("llm_uq") and the embedding-LR baseline ("embed_pred"). One folder
per corpus under tables/{llm_uq,embed_pred}/{RPP,CB,SSRP}/, two tables per
corpus per source. Written as .md only -- not used in the paper, so the
.tex half of the pair was dropped (see write_pair()). This script also owns
the publication-level CB/SSRP embedding summary formerly split across the
corpus-specific llm_uq_table_embed_*.py generators:

  table_embed_cross_corpus.tex: embedding-LR uncertainty metrics for CB and SSRP.

  llm_uq:   q-bar (verbalized confidence) and v-bar (self-consistency vote
            rate) -- the two label-free elicitation methods used throughout
            this repo's other table_*.tex files. Each at both reasoning_effort
            variants (temp=0.7 both; CB uses the full 158-effect scope, not
            the 23-paper first-effect subset). RPP/CB's low-effort metrics
            needed the RPP and CB predict_llm_uq.py / predict_llm_verdict.py
            variants (since consolidated into one script per task, taking
            --corpus rpp/cb/ssrp) to gain REASONING_EFFORT parameterization
            first (2026-08-20, matching the SSRP variant's pre-existing
            pattern) -- before that only SSRP had a low-effort variant to
            report.

  embed_pred: per-seed mean (mean +/- SD of the 20 individual per-seed
            AUC/Accuracy values -- what one run of the pipeline looks like)
            and mean probability (AUC/Accuracy of the per-effect/study
            average out-of-fold probability across the 20 seeds, +/- the
            closed-form Hanley-McNeil/Wald SD of that single point estimate
            -- a 20-way ensemble, a different quantity from the per-seed
            mean). Same distinction table_predictor_comparison_cross_corpus.tex
            already draws for CB/SSRP; RPP gets the same treatment here. No
            effort variant here -- embed_pred doesn't depend on the LLM at all.

Usage: rep_env/bin/python tables/make_classification_tables.py
Writes: tables/llm_uq/{RPP,CB,SSRP}/{qbar,vbar,qbar_low,vbar_low}.md
        tables/llm_uq/{RPP,CB,SSRP}/{qbar,vbar}_[low_]temp0.2.md --
            each corpus's temp=0.2 comparison pair (predict_text_batch{,_cb,
            _ssrp}.py: RPP 2026-08-21, SSRP 2026-08-26, CB 2026-09-06)
        tables/llm_uq/summary_{qbar,vbar}.md -- all corpora/efforts for
            one method, effort and temp as explicit columns instead of baked
            into the filename/caption (q-bar and v-bar kept as separate
            tables, not merged into one with a Method column); every
            corpus's temp0.2 pair is included here too, grouped with that
            corpus's other rows
        tables/embed_pred/{RPP,CB,SSRP}/{per_seed_mean,mean_probability}.md
        tables/table_embed_cross_corpus.tex
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PACKAGE_DIR, "prediction", "llm_pred", "metric")  # predict_llm_uq*.py output
OUT_DIR = os.path.join(PACKAGE_DIR, "tables")
DOC_DIR = OUT_DIR
N_SEEDS = 20

LLM_QBAR_CSV = {
    "RPP": os.path.join(RESULTS, "llm_qbar_metrics.csv"),
    "CB": os.path.join(RESULTS, "llm_qbar_cb_full_metrics.csv"),
    "SSRP": os.path.join(RESULTS, "llm_qbar_ssrp_metrics.csv"),
}
LLM_VBAR_CSV = {
    "RPP": os.path.join(RESULTS, "llm_verdict_metrics.csv"),
    "CB": os.path.join(RESULTS, "llm_verdict_cb_full_metrics.csv"),
    "SSRP": os.path.join(RESULTS, "llm_verdict_ssrp_metrics.csv"),
}
# Low-effort comparison variant -- now computed for all three corpora
# (the per-corpus predict_llm_uq.py/predict_llm_verdict.py variants, since
# consolidated into one script per task with --corpus, gained
# REASONING_EFFORT parameterization for RPP/CB 2026-08-20, matching SSRP's
# existing pattern).
LLM_QBAR_LOW_CSV = {
    "RPP": os.path.join(RESULTS, "llm_qbar_low_metrics.csv"),
    "CB": os.path.join(RESULTS, "llm_qbar_cb_full_low_metrics.csv"),
    "SSRP": os.path.join(RESULTS, "llm_qbar_ssrp_low_metrics.csv"),
}
LLM_VBAR_LOW_CSV = {
    "RPP": os.path.join(RESULTS, "llm_verdict_low_metrics.csv"),
    "CB": os.path.join(RESULTS, "llm_verdict_cb_full_low_metrics.csv"),
    "SSRP": os.path.join(RESULTS, "llm_verdict_ssrp_low_metrics.csv"),
}
# RPP temp=0.2 comparison pair (predict_text_batch.py submitted both efforts
# alongside the canonical temp=0.7 runs, 2026-08-21; boosted R=25->100
# 2026-08-25).
LLM_QBAR_RPP_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_qbar_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_qbar_low_temp0.2_metrics.csv"),
}
LLM_VBAR_RPP_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_verdict_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_verdict_low_temp0.2_metrics.csv"),
}
# SSRP temp=0.2 comparison pair (predict_text_batch.py --corpus ssrp, 2026-08-26).
LLM_QBAR_SSRP_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_qbar_ssrp_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_qbar_ssrp_low_temp0.2_metrics.csv"),
}
LLM_VBAR_SSRP_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_verdict_ssrp_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_verdict_ssrp_low_temp0.2_metrics.csv"),
}
# CB temp=0.2 comparison pair (predict_text_batch.py --corpus cb, 2026-09-06).
LLM_QBAR_CB_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_qbar_cb_full_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_qbar_cb_full_low_temp0.2_metrics.csv"),
}
LLM_VBAR_CB_TEMP02_CSV = {
    "high": os.path.join(RESULTS, "llm_verdict_cb_full_temp0.2_metrics.csv"),
    "low": os.path.join(RESULTS, "llm_verdict_cb_full_low_temp0.2_metrics.csv"),
}
EMBED_LR_CSV = {
    "RPP": os.path.join(PACKAGE_DIR, "prediction", "embed_pred", "RPP", "embed_lr_metrics.csv"),
    "CB": os.path.join(PACKAGE_DIR, "prediction", "embed_pred", "CB", "embed_lr_metrics.csv"),
    "SSRP": os.path.join(PACKAGE_DIR, "prediction", "embed_pred", "SSRP", "embed_lr_metrics.csv"),
}
EMBED_STUDY_CSV = {
    "RPP": os.path.join(
        PACKAGE_DIR, "prediction", "embed_pred", "RPP", "embed_lr_study_predictions.csv"
    ),
    "CB": os.path.join(PACKAGE_DIR, "prediction", "embed_pred", "CB", "embed_lr_study_predictions.csv"),
    "SSRP": os.path.join(PACKAGE_DIR, "prediction", "embed_pred", "SSRP", "embed_lr_study_predictions.csv"),
}
CORPORA = ["RPP", "CB", "SSRP"]
FULL_NAME = {"RPP": "RPP", "CB": "CB", "SSRP": "SSRP"}


def write_pair(basename, out_dir, tex_lines, md_lines):
    # .md only: these per-corpus breakdown tables aren't used in the paper,
    # so the .tex half of the pair is dropped rather than kept alongside a
    # redundant .md. tex_lines stays a parameter so callers don't change.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{basename}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def write_doc_table(filename, tex_lines):
    os.makedirs(DOC_DIR, exist_ok=True)
    out_path = os.path.join(DOC_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Wrote {out_path}")


def cv_rows(metrics_csv):
    """Return the averaged-probability, per-seed mean, and per-seed SD rows."""
    metrics = pd.read_csv(metrics_csv)
    avg = metrics[metrics["evaluation"].str.contains(f"{N_SEEDS}-seed avg")].iloc[0]
    mean = metrics[
        metrics["evaluation"].str.contains(rf"per-seed \(mean of {N_SEEDS}\)")
    ].iloc[0]
    sd = metrics[
        metrics["evaluation"].str.contains(rf"per-seed \(sd of {N_SEEDS}\)")
    ].iloc[0]
    return avg, mean, sd


def llm_table(corpus, kind, csv_path, method_label, method_symbol, effort="high", temp="0.7"):
    row = pd.read_csv(csv_path).iloc[0]
    n, auc, auc_sd = int(row["n"]), row["AUC"], row["AUC_SD"]
    acc, acc_sd = row["Accuracy"], row["Accuracy_SD"]
    full = FULL_NAME[corpus]
    basename = kind
    if effort != "high":
        basename += f"_{effort}"
    if temp != "0.7":
        basename += f"_temp{temp}"

    tex = [
        r"% Generated by make_classification_tables.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{gpt-5.4-mini zero-shot {method_label} (${method_symbol}$) on "
        rf"{full} ($n{{=}}{n}$), reasoning\_effort={effort}, temp={temp}. AUC $\pm$ "
        rf"Hanley\textendash McNeil (1982) SE; Accuracy $\pm$ Wald binomial SE.}}",
        rf"\label{{tab:{basename}-{corpus.lower()}}}", r"\small",
        r"\begin{tabular}{@{}l c c@{}}", r"\toprule",
        r"\textbf{AUC} & \textbf{Accuracy} & \textbf{$n$} \\", r"\midrule",
        rf"{auc:.3f} $\pm$ {auc_sd:.3f} & {acc:.3f} $\pm$ {acc_sd:.3f} & {n} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    md = [
        r"<!-- Generated by make_classification_tables.py -- do not edit by hand. -->",
        f"# {full}: {method_label} ({method_symbol}), {effort} effort, temp={temp}",
        "",
        f"gpt-5.4-mini zero-shot, reasoning_effort={effort}, temp={temp}, n={n}. "
        "AUC ± Hanley-McNeil (1982) SE; Accuracy ± Wald binomial SE.",
        "",
        "| AUC | Accuracy | n |",
        "|---|---|---:|",
        f"| {auc:.3f} ± {auc_sd:.3f} | {acc:.3f} ± {acc_sd:.3f} | {n} |",
    ]
    write_pair(basename, os.path.join(OUT_DIR, "llm_uq", corpus), tex, md)


def embed_table(corpus, kind, csv_path):
    avg_row, mean_row, sd_row = cv_rows(csv_path)
    full = FULL_NAME[corpus]

    if kind == "per_seed_mean":
        n = int(mean_row["n"])
        auc, auc_sd = mean_row["AUC"], sd_row["AUC"]
        acc, acc_sd = mean_row["Accuracy"], sd_row["Accuracy"]
        spread_label = f"SD across {N_SEEDS} seeds"
        caption_kind = (f"per-seed mean -- mean and SD of the {N_SEEDS} individual "
                         "per-seed AUC/Accuracy values, what one run looks like")
    else:  # mean_probability
        n = int(avg_row["n"])
        auc, auc_sd = avg_row["AUC"], avg_row["AUC_SD"]
        acc, acc_sd = avg_row["Accuracy"], avg_row["Accuracy_SD"]
        spread_label = "closed-form SD of the point estimate"
        caption_kind = ("mean probability -- AUC/Accuracy of the per-study average "
                         f"out-of-fold probability across the {N_SEEDS} seeds (a "
                         f"{N_SEEDS}-way "
                         "ensemble, a different quantity from the per-seed mean)")

    tex = [
        r"% Generated by make_classification_tables.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Text-embedding Lasso LR, {full} ($n{{=}}{n}$), nested CV "
        rf"($N{{=}}{N_SEEDS}$ independent outer/inner StratifiedKFold splits): "
        rf"{caption_kind}.}}",
        rf"\label{{tab:embed-{kind.replace('_', '-')}-{corpus.lower()}}}", r"\small",
        r"\begin{tabular}{@{}l c c@{}}", r"\toprule",
        r"\textbf{AUC} & \textbf{Accuracy} & \textbf{$n$} \\", r"\midrule",
        rf"{auc:.3f} $\pm$ {auc_sd:.3f} & {acc:.3f} $\pm$ {acc_sd:.3f} & {n} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    md = [
        r"<!-- Generated by make_classification_tables.py -- do not edit by hand. -->",
        f"# {full}: {caption_kind.split(' -- ')[0]}",
        "",
        f"Text-embedding Lasso LR, nested CV (N={N_SEEDS} independent outer/inner "
        f"StratifiedKFold splits), n={n}. AUC/Accuracy ± {spread_label}.",
        "",
        "| AUC | Accuracy | n |",
        "|---|---|---:|",
        f"| {auc:.3f} ± {auc_sd:.3f} | {acc:.3f} ± {acc_sd:.3f} | {n} |",
    ]
    write_pair(kind, os.path.join(OUT_DIR, "embed_pred", corpus), tex, md)


def ece_from_proba(y, proba, n_bins):
    """Quantile-binned expected calibration error."""
    order = np.argsort(proba)
    y_sorted, proba_sorted = y[order], proba[order]
    bins = np.array_split(np.arange(len(y_sorted)), n_bins)
    return sum(
        (len(bin_indices) / len(y_sorted))
        * abs(y_sorted[bin_indices].mean() - proba_sorted[bin_indices].mean())
        for bin_indices in bins
    )


def embed_cross_corpus_summary():
    """Write the CB/SSRP embedding-LR uncertainty summary."""
    rows = []
    configs = [
        ("CB", r"CB (effect level, paper-grouped CV)", 5),
        ("SSRP", r"SSRP (study level)", 3),
    ]
    for corpus, label, n_bins in configs:
        avg, _, _ = cv_rows(EMBED_LR_CSV[corpus])
        predictions = pd.read_csv(EMBED_STUDY_CSV[corpus])
        ece = ece_from_proba(
            predictions["y"].values,
            predictions["proba_cv_avg"].values,
            n_bins,
        )
        rows.append({
            "corpus": label,
            "n": int(avg.n),
            "AUC": f"{avg.AUC:.3f} $\\pm$ {avg.AUC_SD:.3f}",
            "Accuracy": f"{avg.Accuracy:.3f} $\\pm$ {avg.Accuracy_SD:.3f}",
            "ECE": f"{ece:.3f}",
            "dispersion": (
                rf"SD($\hat p$) $= "
                rf"{predictions['proba_cv_sd_across_seeds'].mean():.3f}$"
            ),
        })

    print(pd.DataFrame(rows).to_string(index=False))
    tex = [
        r"% Generated by make_classification_tables.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        r"\caption{Text-embedding Lasso logistic regression, scored by repeated "
        rf"nested CV ($N{{=}}{N_SEEDS}$ independent outer/inner StratifiedKFold "
        r"splits; final prediction is the per-effect/per-study average "
        r"out-of-fold probability across splits), on CB's 158 completed "
        r"effects (paper-grouped CV, 23 papers) and SSRP's 21 studies (one per "
        r"paper). AUC $\pm$ Hanley\textendash McNeil (1982) SE; Accuracy $\pm$ "
        r"Wald binomial SE; ECE from quantile-binned reliability bins (5 for "
        r"CB, 3 for SSRP, scaled to $n$); dispersion is the SD of each "
        rf"effect/study's predicted probability across the {N_SEEDS} CV-split "
        r"seeds -- the run-to-run variability of the estimate itself, needing "
        r"no ground truth. The two corpora differ in $n$, granularity, and "
        r"base rate, so this is a side-by-side comparison, not a "
        r"head-to-head one.}",
        r"\label{tab:embed-cb-ssrp}", r"\small",
        r"\begin{tabular}{@{}l c cc c c@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{vs.\ ground truth} & \multicolumn{1}{c}{label-free} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-6}",
        r"\textbf{Corpus} & \textbf{$n$} & \textbf{AUC} & \textbf{Accuracy} & "
        r"\textbf{ECE} & \textbf{Run-to-run dispersion} \\", r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['corpus']} & {row['n']} & {row['AUC']} & {row['Accuracy']} & "
            f"{row['ECE']} & {row['dispersion']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_doc_table("table_embed_cross_corpus.tex", tex)


def rpp_predictor_comparison():
    """Replace the retired notebook-generated RPP comparison table."""
    embed_metrics = pd.read_csv(EMBED_LR_CSV["RPP"])
    embed = embed_metrics[
        embed_metrics["evaluation"].str.contains(f"{N_SEEDS}-seed avg")
    ].iloc[0]
    embed_predictions = pd.read_csv(EMBED_STUDY_CSV["RPP"])
    predictions = pd.read_csv(
        os.path.join(
            PACKAGE_DIR, "prediction", "LLM_Reasoning", "RPP", "text_predictions_high_temp0.7.csv"
        )
    )
    predictions["verdict_binary"] = predictions["verdict"].map(
        {"replicable": 1, "unreplicable": 0}
    )
    llm = predictions.groupby("altmejd_id").agg(
        qbar=("q_hat", "mean"), vbar=("verdict_binary", "mean"),
        ground_truth=("ground_truth", "first"),
    ).reset_index()
    joined = embed_predictions[["altmejd_id", "y", "proba_cv_avg"]].merge(
        llm, on="altmejd_id", how="inner", validate="one_to_one"
    )
    if len(joined) != len(embed_predictions):
        raise ValueError("RPP predictor comparison lost analysis units in its join.")
    if not (joined["y"] == joined["ground_truth"].map({"yes": 1, "no": 0})).all():
        raise ValueError("RPP LLM and embedding outcomes disagree.")

    qbar = pd.read_csv(LLM_QBAR_CSV["RPP"]).iloc[0]
    vbar = pd.read_csv(LLM_VBAR_CSV["RPP"]).iloc[0]
    y = joined["y"].to_numpy()
    baseline = np.repeat(y.mean(), len(y))
    rows = [
        ("Text-embedding LR", "20-seed nested-CV mean", embed.AUC, embed.Brier,
         embed.Accuracy),
        (r"gpt-5.4-mini $\bar q$", "zero-shot", qbar.AUC,
         brier_score_loss(y, joined["qbar"]), qbar.Accuracy),
        (r"gpt-5.4-mini $\bar v$", "zero-shot", vbar.AUC,
         brier_score_loss(y, joined["vbar"]), vbar.Accuracy),
        ("Base-rate probability", "constant", 0.5,
         brier_score_loss(y, baseline), max(y.mean(), 1 - y.mean())),
    ]
    tex = [
        r"% Generated by make_classification_tables.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Replication-prediction performance on RPP's {len(y)} analysis claims. "
        rf"The embedding baseline is the mean held-out probability across {N_SEEDS} nested-CV "
        rf"seeds; the LLM estimates use $R{{=}}100$ zero-shot runs. Accuracy thresholds "
        rf"predicted probability at 0.5. The base-rate row always predicts the observed "
        rf"sample replication rate.}}",
        r"\label{tab:predictor-comparison}", r"\small",
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"\textbf{Predictor} & \textbf{Evaluation} & \textbf{AUC} & "
        r"\textbf{Brier} & \textbf{Accuracy} \\", r"\midrule",
    ]
    for predictor, evaluation, auc, brier, accuracy in rows:
        tex.append(
            f"{predictor} & {evaluation} & {auc:.3f} & {brier:.3f} & {accuracy:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_doc_table("table_predictor_comparison.tex", tex)


def llm_summary_table(basename, method_label, method_symbol, method_name, high_csv, low_csv,
                       temp02_csv=None):
    """One table across all (corpus, effort) combinations for a single
    elicitation method, with effort and temperature as explicit columns --
    unlike the per-file tables under tables/llm_uq/{corpus}/, which bake
    effort into the filename and caption instead. temp was always 0.7 when
    this table was first built, included as a literal column anyway since a
    future low-temp/high-temp comparison would slot in the same way effort
    did -- RPP's (2026-08-21), SSRP's (2026-08-26), and CB's (2026-09-06)
    temp=0.2 comparison pairs are exactly that case, passed via temp02_csv=
    {corpus: {"high": ..., "low": ...}} and each corpus's pair inserted
    right after that corpus's canonical temp=0.7 rows so all of a corpus's
    rows stay grouped; a corpus absent from temp02_csv would be unaffected
    (all three now have an entry). q-bar and v-bar are kept as two separate
    tables/files rather than one combined table with a Method column, since
    they're different quantities, not different rows of the same measurement."""
    temp02_csv = temp02_csv or {}
    rows = []
    for corpus in CORPORA:
        for effort, csv_map in [("high", high_csv), ("low", low_csv)]:
            r = pd.read_csv(csv_map[corpus]).iloc[0]
            rows.append(dict(
                corpus=FULL_NAME[corpus], effort=effort, temp=0.7,
                n=int(r["n"]), auc=r["AUC"], auc_sd=r["AUC_SD"],
                acc=r["Accuracy"], acc_sd=r["Accuracy_SD"]))
        if corpus in temp02_csv:
            for effort in ("high", "low"):
                r = pd.read_csv(temp02_csv[corpus][effort]).iloc[0]
                rows.append(dict(
                    corpus=FULL_NAME[corpus], effort=effort, temp=0.2,
                    n=int(r["n"]), auc=r["AUC"], auc_sd=r["AUC_SD"],
                    acc=r["Accuracy"], acc_sd=r["Accuracy_SD"]))

    tex = [
        r"% Generated by make_classification_tables.py -- do not edit by hand.",
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{gpt-5.4-mini zero-shot {method_name} (${method_symbol}$) "
        r"classification metrics, all corpora, both reasoning-effort variants "
        r"(each corpus also includes a temp=0.2 comparison pair). "
        r"AUC $\pm$ Hanley\textendash McNeil (1982) SE; Accuracy $\pm$ Wald "
        r"binomial SE. Per-corpus/effort tables with the same numbers, one "
        rf"file each, are under table/llm\_uq/\{{corpus\}}/{basename}"
        r"[_low].}",
        rf"\label{{tab:llm-uq-summary-{basename}}}", r"\small",
        r"\begin{tabular}{@{}l c r c c r@{}}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Effort} & \textbf{Temp} & "
        r"\textbf{AUC} & \textbf{Accuracy} & \textbf{$n$} \\",
        r"\midrule",
    ]
    prev_corpus = None
    for r in rows:
        corpus_cell = r["corpus"] if r["corpus"] != prev_corpus else ""
        prev_corpus = r["corpus"]
        tex.append(
            f"{corpus_cell} & {r['effort']} & {r['temp']:g} & "
            f"{r['auc']:.3f} $\\pm$ {r['auc_sd']:.3f} & "
            f"{r['acc']:.3f} $\\pm$ {r['acc_sd']:.3f} & {r['n']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    md = [
        r"<!-- Generated by make_classification_tables.py -- do not edit by hand. -->",
        f"# LLM zero-shot {method_name} ({method_label}) — all corpora, effort variants",
        "",
        "gpt-5.4-mini. AUC ± Hanley-McNeil (1982) SE; Accuracy ± Wald binomial SE. "
        f"Per-corpus/effort tables with the same numbers, one file each, are "
        f"under `tables/llm_uq/{{corpus}}/{basename}[_low]`.",
        "",
        "| Corpus | Effort | Temp | AUC | Accuracy | n |",
        "|---|---|---:|---|---|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['corpus']} | {r['effort']} | {r['temp']:g} | "
            f"{r['auc']:.3f} ± {r['auc_sd']:.3f} | {r['acc']:.3f} ± {r['acc_sd']:.3f} | {r['n']} |"
        )

    write_pair(f"summary_{basename}", os.path.join(OUT_DIR, "llm_uq"), tex, md)


def llm_summary():
    llm_summary_table("qbar", "q̄", r"\bar q", "verbalized confidence",
                       LLM_QBAR_CSV, LLM_QBAR_LOW_CSV,
                       temp02_csv={"RPP": LLM_QBAR_RPP_TEMP02_CSV, "CB": LLM_QBAR_CB_TEMP02_CSV,
                                   "SSRP": LLM_QBAR_SSRP_TEMP02_CSV})
    llm_summary_table("vbar", "v̄", r"\bar v", "self-consistency vote rate",
                       LLM_VBAR_CSV, LLM_VBAR_LOW_CSV,
                       temp02_csv={"RPP": LLM_VBAR_RPP_TEMP02_CSV, "CB": LLM_VBAR_CB_TEMP02_CSV,
                                   "SSRP": LLM_VBAR_SSRP_TEMP02_CSV})


def main():
    for corpus in CORPORA:
        llm_table(corpus, "qbar", LLM_QBAR_CSV[corpus],
                  "verbalized confidence", r"\bar q", effort="high")
        llm_table(corpus, "vbar", LLM_VBAR_CSV[corpus],
                  "self-consistency vote rate", r"\bar v", effort="high")
        llm_table(corpus, "qbar", LLM_QBAR_LOW_CSV[corpus],
                  "verbalized confidence", r"\bar q", effort="low")
        llm_table(corpus, "vbar", LLM_VBAR_LOW_CSV[corpus],
                  "self-consistency vote rate", r"\bar v", effort="low")
        embed_table(corpus, "per_seed_mean", EMBED_LR_CSV[corpus])
        embed_table(corpus, "mean_probability", EMBED_LR_CSV[corpus])
        print(f"{corpus}: wrote qbar/vbar x {{high,low}} (llm_uq) + "
              f"per_seed_mean/mean_probability (embed_pred)")

    for effort in ("high", "low"):
        llm_table("RPP", "qbar", LLM_QBAR_RPP_TEMP02_CSV[effort],
                  "verbalized confidence", r"\bar q", effort=effort, temp="0.2")
        llm_table("RPP", "vbar", LLM_VBAR_RPP_TEMP02_CSV[effort],
                  "self-consistency vote rate", r"\bar v", effort=effort, temp="0.2")
    print("RPP: wrote qbar/vbar x {high,low} temp0.2 (llm_uq)")

    for effort in ("high", "low"):
        llm_table("SSRP", "qbar", LLM_QBAR_SSRP_TEMP02_CSV[effort],
                  "verbalized confidence", r"\bar q", effort=effort, temp="0.2")
        llm_table("SSRP", "vbar", LLM_VBAR_SSRP_TEMP02_CSV[effort],
                  "self-consistency vote rate", r"\bar v", effort=effort, temp="0.2")
    print("SSRP: wrote qbar/vbar x {high,low} temp0.2 (llm_uq)")

    for effort in ("high", "low"):
        llm_table("CB", "qbar", LLM_QBAR_CB_TEMP02_CSV[effort],
                  "verbalized confidence", r"\bar q", effort=effort, temp="0.2")
        llm_table("CB", "vbar", LLM_VBAR_CB_TEMP02_CSV[effort],
                  "self-consistency vote rate", r"\bar v", effort=effort, temp="0.2")
    print("CB: wrote qbar/vbar x {high,low} temp0.2 (llm_uq)")

    llm_summary()
    print("wrote tables/llm_uq/summary_{qbar,vbar}.md")

    embed_cross_corpus_summary()
    rpp_predictor_comparison()


if __name__ == "__main__":
    main()
