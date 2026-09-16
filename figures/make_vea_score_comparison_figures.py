"""Regenerate vea_tau_analysis.png (RPP only) and score_comparison.png (all
three corpora), originally built for archive/old_writeup/slides.tex, written
under llm_uq/figures/{RPP,CB,SSRP}/ -- the same per-corpus output location
make_roc_figures.py's
roc_comparison.png and make_vea_figures.py's (CB/SSRP) vea_analysis.png
already use, so all of a corpus's figures live in one place. Each VEA
figure is written alongside a vea_tau_analysis_{effort}_temp{temp}.csv with
the same per-unit V/E_raw/E/A/n rows the plot draws, matching what
make_vea_figures.py already writes for CB/SSRP -- *.png is gitignored, so
the CSV is the committed, quotable copy of these numbers.

vea_tau_analysis.png stays RPP-only: it's built on scoring.py/pstar.py,
which are hard-coded to RPP-only ground-truth columns from the original OSC
coding project (a 6-level "Surprising result (O)" coder rating, "Opportunity
for expectancy bias/lack of diligence (O)" risk codes, a precomputed
statistical-power reference) that CB/SSRP's own source data has no
equivalent for -- of the 6 scored units, only Verdict (ground_truth) and
Scoring (embedding-LR reference) have usable ground truth in CB/SSRP, so a
6-unit chart there would be mostly zeros, not a real comparison. Porting it
would mean inventing new ground-truth definitions per corpus, not a refactor.

score_comparison.png generalizes cleanly: every corpus's
text_predictions_high_temp0.7.csv already carries its own "ground_truth"
column (yes/no), and every corpus already has an embed_lr_study_predictions.csv
(CB: llm_uq/prediction/embed_pred/CB/, SSRP: llm_uq/prediction/embed_pred/SSRP/, RPP:
llm_uq/prediction/embed_pred/RPP/) with a "proba_cv_avg" column keyed by the same unit
columns as its predictions file. CB/SSRP key on their numeric ids; RPP keys
on `altmejd_id`, preserving all 90 claims even though two pairs share paper
titles.

RPP, CB, and SSRP each have a temp=0.2 comparison pair (high/low) alongside
the canonical temp=0.7 high/low -- both VEA_RUNS and CORPORA carry a "temp"
field so output filenames (_{effort}_temp{temp}) never collide. (VEA_RUNS
itself stays RPP-only regardless -- see above.)

Run from the repo root:
    python llm_uq/figures/make_vea_score_comparison_figures.py
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_title(s):
    s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

ROOT = Path(__file__).resolve().parent.parent  # llm_uq/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "uq_bound"))
import scoring, pstar, metrics as uq_metrics  # noqa: E402

FIG_DIR = ROOT / "figures"

# RPP-only inputs for compute_vea() / make_plot() (see module docstring for
# why VEA doesn't generalize to CB/SSRP). RPP now has a high/low effort run
# at BOTH temp=0.7 (canonical) and temp=0.2 (comparison), same as SSRP's
# effort comparison -- VEA_RUNS drives all four; PRED_CSV/OUT_PNG stay as
# the "high"/"0.7" aliases since embed_pstar6()/other code below still
# references them directly for the canonical run.
PRED_CSV = ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_high_temp0.7.csv"
RPP_CSV  = ROOT / "data" / "rpp_data_cleaned.csv"
OUT_PNG  = FIG_DIR / "RPP" / "vea_tau_analysis_high_temp0.7.png"
EMB_DIR  = ROOT / "prediction" / "embed_pred" / "RPP" / "embeddings_text-embedding-3-large_query"
VEA_RUNS = [
    ("high", "0.7", PRED_CSV, OUT_PNG),
    ("low", "0.7", ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_low_temp0.7.csv",
     FIG_DIR / "RPP" / "vea_tau_analysis_low_temp0.7.png"),
    ("high", "0.2", ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_high_temp0.2.csv",
     FIG_DIR / "RPP" / "vea_tau_analysis_high_temp0.2.png"),
    ("low", "0.2", ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_low_temp0.2.csv",
     FIG_DIR / "RPP" / "vea_tau_analysis_low_temp0.2.png"),
]

# Per-(corpus, effort, temp) run configs for make_score_comparison_plot().
# key_cols must match columns present in BOTH the predictions CSV and the
# embed_lr_study_predictions.csv (verified: identical dtypes, int64, across
# all three corpora). CB/SSRP only ever run at temp=0.7; RPP additionally
# has the temp=0.2 comparison pair -- output filenames carry the
# _{effort}_temp{temp} suffix throughout so no variant collides.
CORPORA = [
    dict(
        corpus="RPP", effort="high", temp="0.7",
        pred_csv=PRED_CSV,
        embed_csv=ROOT / "prediction" / "embed_pred" / "RPP" / "embed_lr_study_predictions.csv",
        key_cols=["altmejd_id"],
        out_png=FIG_DIR / "RPP" / "score_comparison_high_temp0.7.png",
    ),
    dict(
        corpus="RPP", effort="low", temp="0.7",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_low_temp0.7.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "RPP" / "embed_lr_study_predictions.csv",
        key_cols=["altmejd_id"],
        out_png=FIG_DIR / "RPP" / "score_comparison_low_temp0.7.png",
    ),
    dict(
        corpus="RPP", effort="high", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_high_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "RPP" / "embed_lr_study_predictions.csv",
        key_cols=["altmejd_id"],
        out_png=FIG_DIR / "RPP" / "score_comparison_high_temp0.2.png",
    ),
    dict(
        corpus="RPP", effort="low", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "RPP" / "text_predictions_low_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "RPP" / "embed_lr_study_predictions.csv",
        key_cols=["altmejd_id"],
        out_png=FIG_DIR / "RPP" / "score_comparison_low_temp0.2.png",
    ),
    dict(
        corpus="CB", effort="high", temp="0.7",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "CB" / "text_predictions_high_temp0.7.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "CB" / "embed_lr_study_predictions.csv",
        key_cols=["paper_num", "experiment_num", "effect_num"],
        out_png=FIG_DIR / "CB" / "score_comparison_high_temp0.7.png",
    ),
    dict(
        corpus="CB", effort="low", temp="0.7",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "CB" / "text_predictions_low_temp0.7.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "CB" / "embed_lr_study_predictions.csv",
        key_cols=["paper_num", "experiment_num", "effect_num"],
        out_png=FIG_DIR / "CB" / "score_comparison_low_temp0.7.png",
    ),
    dict(
        corpus="CB", effort="high", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "CB" / "text_predictions_high_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "CB" / "embed_lr_study_predictions.csv",
        key_cols=["paper_num", "experiment_num", "effect_num"],
        out_png=FIG_DIR / "CB" / "score_comparison_high_temp0.2.png",
    ),
    dict(
        corpus="CB", effort="low", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "CB" / "text_predictions_low_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "CB" / "embed_lr_study_predictions.csv",
        key_cols=["paper_num", "experiment_num", "effect_num"],
        out_png=FIG_DIR / "CB" / "score_comparison_low_temp0.2.png",
    ),
    dict(
        corpus="SSRP", effort="high", temp="0.7",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "SSRP" / "text_predictions_high_temp0.7.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "SSRP" / "embed_lr_study_predictions.csv",
        key_cols=["study_num"],
        out_png=FIG_DIR / "SSRP" / "score_comparison_high_temp0.7.png",
    ),
    dict(
        corpus="SSRP", effort="low", temp="0.7",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "SSRP" / "text_predictions_low_temp0.7.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "SSRP" / "embed_lr_study_predictions.csv",
        key_cols=["study_num"],
        out_png=FIG_DIR / "SSRP" / "score_comparison_low_temp0.7.png",
    ),
    dict(
        corpus="SSRP", effort="high", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "SSRP" / "text_predictions_high_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "SSRP" / "embed_lr_study_predictions.csv",
        key_cols=["study_num"],
        out_png=FIG_DIR / "SSRP" / "score_comparison_high_temp0.2.png",
    ),
    dict(
        corpus="SSRP", effort="low", temp="0.2",
        pred_csv=ROOT / "prediction" / "LLM_Reasoning" / "SSRP" / "text_predictions_low_temp0.2.csv",
        embed_csv=ROOT / "prediction" / "embed_pred" / "SSRP" / "embed_lr_study_predictions.csv",
        key_cols=["study_num"],
        out_png=FIG_DIR / "SSRP" / "score_comparison_low_temp0.2.png",
    ),
]

UNITS  = [1, 3, 4, 5, 6, 7]  # Table 2 / prompt-step order; 2 (Credibility) unscored
LABELS = ["Extraction", "Statistics", "Researcher DoF\nconcern", "Theory\nand context", "Scoring", "Verdict"]
NONDEGEN = {3, 4, 5, 6}   # aleatoric = p*(1-p*) > 0 wherever p* is a
                    # genuine constructed index that varies continuously
                    # across studies (analytic power; QRP/diligence risk
                    # composite; rescaled surprise rating; logistic-CV
                    # prediction) -- the aleatoric interpretation only fails
                    # for a p* that is structurally degenerate BY
                    # CONSTRUCTION, not merely because it targets a
                    # different quantity than Scoring. Only two units are
                    # structurally degenerate: Verdict's p*=Y_i is a
                    # REALIZED observed outcome, not a probability -- Y_i is
                    # already exactly 0 or 1, so Y_i(1-Y_i)=0 identically; a
                    # "Bernoulli success probability" (per Table 2's
                    # caption) means the parameter p, not a draw from it.
                    # Extraction's p*=1 is likewise a fixed constant target,
                    # not a probability. Every other unit's reference is a
                    # bona fide [0,1]-valued parameter that varies by study,
                    # so it stays in NONDEGEN.


def embed_pstar6():
    """p*_6 (Scoring) from the text-embedding logistic model's nested-CV
    prediction -- embed_pred/RPP/embed_lr_study_predictions.csv (built by
    predict_embed_lr.py), the same up-to-date n=90 reference used throughout
    uq_bound/. Uses "proba_cv_avg" (the N_SEEDS=20 average out-of-fold
    probability across independent fold-split seeds, predict_embed_lr.py's
    "agg prob" aggregation) rather than the single-seed "proba_cv" this used
    to read -- the same variance-reduction upgrade already applied to CB/
    SSRP's embed-LR reference throughout uq_bound/common.py's
    load_pstar4_cb/load_pstar4_ssrp (proba_cv_avg). Matched to
    rpp_data_cleaned's "Study Title (O)" via normalized title, since
    embed_lr_study_predictions.csv carries Altmejd's raw titles."""
    emb = pd.read_csv(ROOT / "prediction" / "embed_pred" / "RPP" / "embed_lr_study_predictions.csv")
    emb["norm_title"] = emb["title"].apply(normalize_title)
    emb = emb.drop_duplicates("norm_title")  # multi-effect papers share one embedding

    rpp_titles = pd.read_csv(RPP_CSV, encoding="latin1")[["Study Title (O)"]].copy()
    rpp_titles["norm_title"] = rpp_titles["Study Title (O)"].apply(normalize_title)

    merged = rpp_titles.merge(emb[["norm_title", "proba_cv_avg"]], on="norm_title", how="inner")
    return merged[["Study Title (O)", "proba_cv_avg"]].rename(columns={"proba_cv_avg": "p_star"})


def compute_vea(rpp, pred, n_predictors=4):
    unit_scores = scoring.compute_unit_scores(pred, rpp)
    pstar_by_unit = {m: pstar.p_star_target(m, rpp, n_predictors=n_predictors)
                     for m in UNITS}
    pstar_by_unit[6] = embed_pstar6()   # Scoring unit target: text-embedding logistic
    met = uq_metrics.monte_carlo_metrics(unit_scores, pstar_by_unit)

    rows = []
    for m, label in zip(UNITS, LABELS):
        sub = met[met["unit"] == m].dropna(subset=["V_hat", "C_tilde"])
        if sub.empty:
            rows.append(dict(unit=m, label=label, V=0.0, E_raw=0.0, E=0.0, A=0.0, n=0))
            continue
        v = sub["V_hat"].mean()
        e_raw = ((sub["q_bar"] - sub["p_star"]) ** 2).mean()  # uncorrected squared bias
        e = max(sub["C_tilde"].mean(), 0.0)   # finite-sample corrected; E[C_tilde] >= 0
        a = (sub["p_star"] * (1 - sub["p_star"])).mean() if m in NONDEGEN else 0.0
        rows.append(dict(unit=m, label=label, V=v, E_raw=e_raw, E=e, A=a, n=len(sub)))

    return pd.DataFrame(rows)


def make_plot(df, out_path, ylim=None):
    fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 4.2))
    x = np.arange(len(UNITS))
    w = 0.7

    bw = w / 4
    ax1.bar(x - 1.5 * bw, df["V"],     width=bw, color="#4C72B0", label=r"Variance $\hat V$")
    ax1.bar(x - 0.5 * bw, df["E_raw"], width=bw, color="#C44E52", label=r"Epistemic $E$ (uncorrected)")
    ax1.bar(x + 0.5 * bw, df["E"],     width=bw, color="#DD8452", label=r"Epistemic $\hat E$ (corrected)")
    ax1.bar(x + 1.5 * bw, df["A"],     width=bw, color="#8C8C8C", label=r"Aleatoric $A$")
    ax1.set_xticks(x); ax1.set_xticklabels(df["label"], fontsize=8)
    ax1.set_ylabel("Mean error component")
    ax1.set_title("VEA Decomposition per Unit\n", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    if ylim is not None:
        ax1.set_ylim(0, ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return df


def make_score_comparison_plot(pred, embed_df, key_cols, out_path, corpus_label="",
                                normalize_keys=False):
    """Predicted probability distributions by true outcome, LLM vs. embedding
    LR -- generic across corpora. Y comes from the predictions file's own
    "ground_truth" column (yes/no), matching what predict_llm_verdict_*.py
    already uses; embed_df supplies "proba_cv_avg" keyed by key_cols, which
    must be shared columns between pred and embed_df (see CORPORA table).

    normalize_keys=True remains available for a future corpus keyed only by
    free text; all current configurations use stable claim identifiers.
    """
    pred2 = pred.copy()
    pred2["Y"] = pred2["ground_truth"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0})
    pred2["verdict_bin"] = pred2["verdict"].astype(str).str.lower().map(
        {"replicable": 1, "unreplicable": 0})
    pred2 = pred2.dropna(subset=["Y", "q_hat"])

    embed_df = embed_df.copy()
    if normalize_keys:
        assert len(key_cols) == 1, "normalize_keys only supports a single text key column"
        col = key_cols[0]
        pred2[col] = pred2[col].apply(normalize_title)
        embed_df[col] = embed_df[col].apply(normalize_title)

    agg = (pred2.groupby(key_cols)
               .agg(q_hat_mean=("q_hat", "mean"), verdict_frac=("verdict_bin", "mean"),
                    Y=("Y", "first"))
               .reset_index()
               .set_index(key_cols))

    # Embedding-LR probabilities are proba_cv_avg from predict_embed_lr*.py:
    # nested CV (outer 5-fold, inner 5-fold hyperparameter search), genuinely
    # out-of-sample for every unit -- same out-of-sample status as the LLM's
    # zero-shot panel, so both panels are shown over all units with no
    # train/test split needed. Y reused from agg (predictions' own
    # ground_truth) rather than embed_df's own "y" column, so both panels
    # score against the identical outcome definition.
    # Stable claim keys are unique; refuse accidental duplicates by retaining
    # only the first and checking the expected join size in corpus loaders.
    embed_df = embed_df.drop_duplicates(subset=key_cols, keep="first")
    emb_proba = embed_df.set_index(key_cols)["proba_cv_avg"]
    emb_y = agg["Y"].reindex(emb_proba.index).dropna()
    emb_proba = emb_proba.reindex(emb_y.index)

    models = {
        "Embedding LR (nested CV)": (emb_proba, emb_y),
        r"LLM: $\bar{v}$ (mean verdict rate)": (agg["verdict_frac"], agg["Y"]),
    }
    colors = {1: "#2196F3", 0: "#F44336"}
    labels = {1: "Replicated", 0: "Not replicated"}

    fig, axes = plt.subplots(1, len(models), figsize=(7.5, 3.8), sharey=True)
    rng = np.random.default_rng(42)
    for ax, (name, (proba, outcome)) in zip(axes, models.items()):
        proba   = proba.reindex(outcome.index).dropna()
        outcome = outcome.reindex(proba.index)
        for y_val in [1, 0]:
            vals = proba[outcome == y_val].values
            ax.boxplot(vals, positions=[y_val], widths=0.3, showfliers=False,
                       patch_artist=True, zorder=1,
                       boxprops=dict(facecolor=colors[y_val], alpha=0.18,
                                     edgecolor=colors[y_val], linewidth=1.3),
                       medianprops=dict(color=colors[y_val], linewidth=2.2),
                       whiskerprops=dict(color=colors[y_val], linewidth=1.1),
                       capprops=dict(color=colors[y_val], linewidth=1.1))
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), y_val) + jitter, vals,
                       alpha=0.5, s=7, color=colors[y_val],
                       label=labels[y_val], zorder=2, edgecolors="none")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not\nreplicated", "Replicated"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Predicted probability", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

    suffix = f" ({corpus_label})" if corpus_label else ""
    fig.suptitle(f"Predicted Scores by True Outcome{suffix} (both panels out-of-sample)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    rpp = pd.read_csv(RPP_CSV, encoding="latin1")
    rpp_pred_cache = {}

    # Compute every VEA_RUNS variant first so all of them share one y-axis
    # scale -- plotting each independently (matplotlib's default autoscale)
    # would let a smaller run's bars look artificially "bigger" relative to
    # its own axis, defeating a cross-run visual comparison.
    vea_dfs = {}
    for effort, temp, pred_csv, out_png in VEA_RUNS:
        rpp_pred = pd.read_csv(pred_csv)
        rpp_pred_cache[(effort, temp)] = rpp_pred
        vea_dfs[(effort, temp)] = (compute_vea(rpp, rpp_pred, n_predictors=4), out_png)

    shared_ylim = 1.05 * max(
        df[col].max() for df, _ in vea_dfs.values() for col in ("V", "E_raw", "E", "A"))
    for (effort, temp), (df, out_png) in vea_dfs.items():
        print(f"\nVEA summary ({effort} effort, temp={temp}):")
        print(df[["label", "n", "V", "E_raw", "E", "A"]].to_string(index=False, float_format="{:.4f}".format))
        make_plot(df, out_png, ylim=shared_ylim)
        # Same PNG+CSV pair make_vea_figures.py writes for CB/SSRP: the plot
        # alone can't be quoted from, and *.png is gitignored, so the numbers
        # behind each figure need a committed CSV next to it.
        df.to_csv(out_png.with_suffix(".csv"), index=False)

    for cfg in CORPORA:
        cfg["out_png"].parent.mkdir(parents=True, exist_ok=True)
        pred = (rpp_pred_cache[(cfg["effort"], cfg["temp"])] if cfg["corpus"] == "RPP"
                else pd.read_csv(cfg["pred_csv"]))
        embed_df = pd.read_csv(cfg["embed_csv"]).rename(columns=cfg.get("embed_rename", {}))
        label = cfg["corpus"] if cfg["effort"] == "high" and cfg["temp"] == "0.7" \
            else f"{cfg['corpus']}, {cfg['effort']} effort, temp={cfg['temp']}"
        make_score_comparison_plot(pred, embed_df, cfg["key_cols"], cfg["out_png"],
                                    corpus_label=label,
                                    normalize_keys=cfg.get("normalize_keys", False))
