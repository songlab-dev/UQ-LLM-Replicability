"""ROC comparison of three label-free replication-probability estimators,
one figure per corpus (RPP, CB, SSRP):

  - Text-embedding LR   -- the embed_pred nested-CV baseline (proba_cv /
                           proba_cv_avg), an external predictor with no LLM
                           involvement at all.
  - Self-consistency vote rate (v_bar) -- fraction of a study's R zero-shot
                           runs whose VERDICT field said "replicable".
  - Verbalized confidence (q_bar) -- mean Q_HAT (the model's own stated
                           replication probability) across those same runs.

All three are scored against the same observed replication outcome Y, so a
single figure's three curves are directly comparable. Study sets and join
keys differ by corpus (Altmejd claim id for RPP,
(paper_num, experiment_num, effect_num) for CB, study_num for SSRP) because
that is how the prediction and embed_pred files key their rows; see each
load_* function.

Output filenames carry the reasoning_effort/temperature suffix (matching
the text_predictions_{effort}_temp{temp}.csv convention) since SSRP now has
both a high and a low effort run; RPP/CB only ever have "high" but are
suffixed too for naming consistency. roc_auc_summary.csv stays scoped to
the "high" row per corpus (each corpus's canonical config) -- SSRP's low
variant is a standalone reference point, not folded into the cross-corpus
summary (matches table_ssrp_effort_comparison.tex's own framing).

Usage: rep_env/bin/python figures/make_roc_figures.py
Writes: figures/{RPP,CB,SSRP}/roc_comparison_high_temp0.7.png,
        roc_auc_high_temp0.7.csv (PNG only, no PDF)
        figures/SSRP/roc_comparison_low_temp0.7.png,
        roc_auc_low_temp0.7.csv
        figures/roc_auc_summary.csv
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
FIG_DIR = os.path.join(REPO, "figures")

METHODS = [
    ("embed_proba", "Text-embedding LR"),
    ("v_bar", "Self-consistency vote rate"),
    ("q_bar", "Verbalized confidence"),
]
# dataviz skill's default categorical palette, slots 1-3 (validated all-pairs
# CVD-safe in both light and dark modes).
COLORS = {
    "Text-embedding LR": "#2a78d6",
    "Self-consistency vote rate": "#eb6834",
    "Verbalized confidence": "#1baf7a",
}


def verdict_to_binary(verdict):
    return (verdict == "replicable").astype(int)


def per_study_llm_scores(pred, key):
    pred = pred.dropna(subset=["q_hat", "verdict"]).copy()
    pred["verdict_bin"] = verdict_to_binary(pred["verdict"])
    return pred.groupby(key).agg(q_bar=("q_hat", "mean"), v_bar=("verdict_bin", "mean"))


def load_rpp(effort="high", temp="0.7"):
    """All 90 filtered RPP claims, including two claims for each of two papers."""
    pred = pd.read_csv(os.path.join(
        REPO, "prediction", "LLM_Reasoning", "RPP", f"text_predictions_{effort}_temp{temp}.csv"))
    embed = pd.read_csv(os.path.join(REPO, "prediction", "embed_pred", "RPP", "embed_lr_study_predictions.csv"))

    per_study = per_study_llm_scores(pred, "altmejd_id")
    embed = embed.set_index("altmejd_id")[["y", "proba_cv"]]

    df = embed.join(per_study, how="inner")
    assert len(df) == 90
    assert not df.isna().any().any()
    return df.rename(columns={"proba_cv": "embed_proba"})


def load_cb(effort="high", temp="0.7"):
    """158 CB effects, keyed by (paper_num, experiment_num, effect_num) --
    the canonical run used by common.py (CB section) / the paper's table (40% prior,
    reasoning_effort/temperature encoded in the filename; R is not, since it's
    about to move from 25 to 100 and prior is now always 40%)."""
    pred = pd.read_csv(os.path.join(REPO, "prediction", "LLM_Reasoning", "CB", f"text_predictions_{effort}_temp{temp}.csv"))
    embed = pd.read_csv(os.path.join(REPO, "prediction", "embed_pred", "CB", "embed_lr_study_predictions.csv"))

    key = ["paper_num", "experiment_num", "effect_num"]
    per_study = per_study_llm_scores(pred, key)
    embed = embed.set_index(key)[["y", "proba_cv_avg"]]

    df = embed.join(per_study, how="inner")
    assert not df.isna().any().any()
    return df.rename(columns={"proba_cv_avg": "embed_proba"})


def load_ssrp(effort="high", temp="0.7"):
    """21 SSRP studies, keyed by study_num. text_predictions_{effort}_temp0.7.csv
    is the calibrated-prior (40%, matching RPP/CB) rerun -- renamed from the
    bare text_predictions.csv to the same reasoning-effort/temperature-
    encoding convention CB's file uses (prior and R are deliberately NOT in
    the name: prior is always 40% now, and R is about to move from 25 to 100,
    so the name only tracks what actually varies between runs); the pre-rerun
    file is archived at archive/text_predictions_uncalibrated_prior_OLD.csv
    for reference only. effort="low" reads the reasoning-effort comparison run instead
    (see llm_uq_table_ssrp_effort_comparison.py) -- same 21 studies, same
    embed-LR reference, only the LLM predictions differ."""
    pred = pd.read_csv(os.path.join(
        REPO, "prediction", "LLM_Reasoning", "SSRP", f"text_predictions_{effort}_temp{temp}.csv"))
    embed = pd.read_csv(os.path.join(REPO, "prediction", "embed_pred", "SSRP", "embed_lr_study_predictions.csv"))

    per_study = per_study_llm_scores(pred, "study_num")
    embed = embed.set_index("study_num")[["y", "proba_cv_avg"]]

    df = embed.join(per_study, how="inner")
    assert not df.isna().any().any()
    return df.rename(columns={"proba_cv_avg": "embed_proba"})


# (corpus, effort, temp) triples to render. All three corpora now have a
# temp=0.2 comparison pair alongside the canonical temp=0.7 high/low, via
# predict_text_batch{,_cb,_ssrp}.py's reasoning-effort/temperature runs.
RUNS = [("RPP", "high", "0.7", load_rpp), ("RPP", "low", "0.7", load_rpp),
        ("RPP", "high", "0.2", load_rpp), ("RPP", "low", "0.2", load_rpp),
        ("CB", "high", "0.7", load_cb), ("CB", "low", "0.7", load_cb),
        ("CB", "high", "0.2", load_cb), ("CB", "low", "0.2", load_cb),
        ("SSRP", "high", "0.7", load_ssrp), ("SSRP", "low", "0.7", load_ssrp),
        ("SSRP", "high", "0.2", load_ssrp), ("SSRP", "low", "0.2", load_ssrp)]


def make_roc_plot(name, df, out_dir, suffix):
    y = df["y"].astype(int).values
    fig, ax = plt.subplots(figsize=(4.6, 4.3))

    rows = []
    for col, label in METHODS:
        fpr, tpr, _ = roc_curve(y, df[col].values)
        auc = roc_auc_score(y, df[col].values)
        ax.plot(fpr, tpr, color=COLORS[label], lw=2, label=f"{label} (AUC={auc:.3f})")
        rows.append({"dataset": name, "method": label, "auc": auc, "n": len(df)})

    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} (n={len(df)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"roc_comparison_{suffix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    summary_rows = []
    for name, effort, temp, loader in RUNS:
        suffix = f"{effort}_temp{temp}"
        df = loader(effort, temp)
        out_dir = os.path.join(FIG_DIR, name)
        rows = make_roc_plot(name, df, out_dir, suffix)
        rows.to_csv(os.path.join(out_dir, f"roc_auc_{suffix}.csv"), index=False)
        print(f"{name} ({effort}, temp={temp}): n={len(df)}")
        print(rows.to_string(index=False))
        print()
        if effort == "high" and temp == "0.7":
            summary_rows.append(rows)

    pd.concat(summary_rows, ignore_index=True).to_csv(
        os.path.join(FIG_DIR, "roc_auc_summary.csv"), index=False)


if __name__ == "__main__":
    main()
