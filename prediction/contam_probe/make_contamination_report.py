"""Analyze the replication-outcome contamination probe across three corpora.

Each effect/claim has 10 responses from predict_contamination_probe.py. Recall
is outcome-conditional and any-hit: a unit is recalled if at least one run
jointly reports FAMILIARITY=recognized and the observed directional outcome.
The report also retains a stricter six-of-ten consensus diagnostic and links
the outcome-concordant recognition count K_i to default-cell prediction
accuracy.

The report distinguishes:
  * any-hit recall rate: recalled effects or claims / all effects or claims;
  * consensus-recall rate: six-of-ten directional consensus / all effects or
    claims;
  * conditional consensus accuracy: correct / consensus recalls, tested against the
    selected subset's majority-class baseline with an exact binomial test; and
  * correlation with self-consistency vote-rate accuracy across all effects or
    claims: phi and Fisher's exact p for correct-claimed-recall vs \bar v_i
    accuracy.

"Claimed" is deliberate. Self-reported familiarity is necessary but not
sufficient to establish memorization. Verified recall would additionally
require blinded validation of NOTES against the replication report; that
annotation is not present in the current data.

Usage: rep_env/bin/python prediction/contam_probe/make_contamination_report.py
Writes: prediction/contam_probe/contamination_probe.{md,tex,png}
        prediction/contam_probe/per_study_{RPP,CB,SSRP}_high_temp0.7.csv
        prediction/contam_probe/contamination_recognition_{accuracy,spearman}.csv
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, pearsonr, spearmanr
import statsmodels.api as sm

SYNTH_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_UQ = os.path.dirname(os.path.dirname(SYNTH_DIR))
OUT_DIR = SYNTH_DIR

KEY_COLS = {
    "RPP": ["unit_id"],
    "CB": ["paper_num", "experiment_num", "effect_num"],
    "SSRP": ["study_num"],
}
MAIN_KEY_COLS = {
    "RPP": ["altmejd_id"],
    "CB": ["paper_num", "experiment_num", "effect_num"],
    "SSRP": ["study_num"],
}
FULL_NAME = {"RPP": "RPP", "CB": "CB", "SSRP": "SSRP"}
CORPUS_COLOR = {"RPP": "#2a78d6", "CB": "#eb6834", "SSRP": "#1baf7a"}
N_PROBE_RUNS = 10
N_VERDICT_RUNS = 100
RECALL_THRESHOLD = N_PROBE_RUNS // 2 + 1
FIVE_HIT_THRESHOLD = 5
FAMILIARITY_LEVELS = ("recognized", "unsure", "unfamiliar", "missing")
VERDICT_LEVELS = ("replicable", "unreplicable", "unknown", "missing")
MIN_STRICT_STRATUM_N = 5
EMBED_PRED_CSV = {
    "RPP": os.path.join(LLM_UQ, "prediction", "embed_pred", "RPP", "embed_lr_study_predictions.csv"),
    "CB": os.path.join(LLM_UQ, "prediction", "embed_pred", "CB", "embed_lr_study_predictions.csv"),
    "SSRP": os.path.join(LLM_UQ, "prediction", "embed_pred", "SSRP", "embed_lr_study_predictions.csv"),
}


def majority_vote(series):
    """Return a unique categorical mode, or None for empty data/an exact tie."""
    counts = series.value_counts()
    if counts.empty:
        return None
    winners = counts[counts == counts.iloc[0]].index.tolist()
    return winners[0] if len(winners) == 1 else None


def exact_error(successes, n):
    """Return two-sided 95% Clopper-Pearson error distances for a plotted rate."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    ci = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="exact")
    return (p - ci.low, ci.high - p)


def probe_per_unit(corpus):
    key_cols = KEY_COLS[corpus]
    path = os.path.join(
        LLM_UQ, "prediction", "LLM_Reasoning", corpus, "contamination_probe_high_temp0.7.csv"
    )
    df = pd.read_csv(path)
    rows = []
    for key_vals, group in df.groupby(key_cols):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        if len(group) != N_PROBE_RUNS:
            raise ValueError(
                f"{corpus} unit {key_vals} has {len(group)} probe rows; "
                f"expected {N_PROBE_RUNS}."
            )
        recognized_replicable = int(
            ((group["familiarity"] == "recognized")
             & (group["verdict"] == "replicable")).sum()
        )
        recognized_unreplicable = int(
            ((group["familiarity"] == "recognized")
             & (group["verdict"] == "unreplicable")).sum()
        )
        if recognized_replicable >= RECALL_THRESHOLD:
            verdict = "replicable"
        elif recognized_unreplicable >= RECALL_THRESHOLD:
            verdict = "unreplicable"
        else:
            verdict = None
        rows.append(
            dict(
                zip(key_cols, key_vals),
                probe_verdict=verdict,
                recognized_replicable=recognized_replicable,
                recognized_unreplicable=recognized_unreplicable,
                ground_truth=str(group["ground_truth"].iloc[0]).strip().lower(),
            )
        )

    out = pd.DataFrame(rows)
    out["claimed_recall"] = out["probe_verdict"].notna()
    observed_replicable = out["ground_truth"].eq("yes")
    out["any_hit_recall"] = (
        ((out["recognized_replicable"] > 0) & observed_replicable)
        | ((out["recognized_unreplicable"] > 0) & ~observed_replicable)
    )
    out["correct_recognized_runs"] = np.where(
        observed_replicable,
        out["recognized_replicable"],
        out["recognized_unreplicable"],
    )
    out["K_i"] = out["correct_recognized_runs"]
    out["five_hit_recall"] = out["correct_recognized_runs"] >= FIVE_HIT_THRESHOLD
    out["recall_score"] = (
        out["recognized_replicable"] - out["recognized_unreplicable"]
    ) / N_PROBE_RUNS
    # Neutral 0.5 encodes no net recognized directional recall evidence.
    out["recall_vbar"] = 0.5 * (1 + out["recall_score"])
    out["claim_correct"] = np.where(
        out["claimed_recall"],
        (out["probe_verdict"] == "replicable") == (out["ground_truth"] == "yes"),
        np.nan,
    )
    out["correct_recall"] = out["claim_correct"].fillna(False).astype(bool)
    return out


def response_distribution(corpus, column, levels):
    """Counts and proportions of raw probe responses for one corpus/field."""
    path = os.path.join(
        LLM_UQ, "prediction", "LLM_Reasoning", corpus, "contamination_probe_high_temp0.7.csv"
    )
    values = pd.read_csv(path)[column].fillna("missing").astype(str).str.strip().str.lower()
    counts = values.value_counts().reindex(levels, fill_value=0)
    return {"corpus": corpus, "n_responses": len(values), **counts.to_dict()}


def response_distributions(column, levels):
    return pd.DataFrame(
        [response_distribution(corpus, column, levels) for corpus in ("RPP", "CB", "SSRP")]
    )


def recognition_verdict_by_outcome():
    """Raw recognition/verdict combinations, stratified by observed outcome."""
    rows = []
    for corpus in ("RPP", "CB", "SSRP"):
        path = os.path.join(
            LLM_UQ, "prediction", "LLM_Reasoning", corpus, "contamination_probe_high_temp0.7.csv"
        )
        df = pd.read_csv(path)
        df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.lower()
        for ground_truth, outcome in (("yes", "replicable"), ("no", "unreplicable")):
            group = df[df["ground_truth"] == ground_truth]
            recognized = group["familiarity"].eq("recognized")
            rows.append(
                {
                    "corpus": corpus,
                    "outcome": outcome,
                    "n_responses": len(group),
                    "recognized": int(recognized.sum()),
                    "recognized_replicable": int((recognized & group["verdict"].eq("replicable")).sum()),
                    "recognized_unreplicable": int(
                        (recognized & group["verdict"].eq("unreplicable")).sum()
                    ),
                    "recognized_unknown": int((recognized & group["verdict"].eq("unknown")).sum()),
                }
            )
    return pd.DataFrame(rows)


def outcome_verdict_tests():
    """Association tests for recognized directional verdicts and observed outcomes."""
    rows = []
    for corpus in ("RPP", "CB", "SSRP"):
        path = os.path.join(
            LLM_UQ, "prediction", "LLM_Reasoning", corpus, "contamination_probe_high_temp0.7.csv"
        )
        df = pd.read_csv(path)
        observed_replicable = df["ground_truth"].astype(str).str.strip().str.lower().eq("yes")
        eligible = df[
            df["familiarity"].eq("recognized")
            & df["verdict"].isin(["replicable", "unreplicable"])
        ].copy()
        eligible["outcome_replicable"] = observed_replicable.loc[eligible.index].astype(int)
        eligible["verdict_replicable"] = eligible["verdict"].eq("replicable").astype(int)
        eligible["cluster"] = eligible[KEY_COLS[corpus]].astype(str).agg(":".join, axis=1)

        design = sm.add_constant(eligible["outcome_replicable"])
        gee = sm.GEE(
            eligible["verdict_replicable"],
            design,
            groups=eligible["cluster"],
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        log_odds = gee.params["outcome_replicable"]
        rows.append(
            {
                "corpus": corpus,
                "n_directional": len(eligible),
                "n_units": eligible["cluster"].nunique(),
                "gee_odds_ratio": np.exp(log_odds),
                "gee_p": gee.pvalues["outcome_replicable"],
            }
        )
    return pd.DataFrame(rows)


def main_predictions_per_unit(corpus):
    """Aggregate the 100-run main pipeline to one row per study/effect."""
    key_cols = MAIN_KEY_COLS[corpus]
    path = os.path.join(
        LLM_UQ, "prediction", "LLM_Reasoning", corpus, "text_predictions_high_temp0.7.csv"
    )
    df = pd.read_csv(path)
    rows = []
    for key_vals, group in df.groupby(key_cols):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        if len(group) != N_VERDICT_RUNS:
            raise ValueError(
                f"{corpus} unit {key_vals} has {len(group)} verdict rows; "
                f"expected {N_VERDICT_RUNS}."
            )
        verdict_bin = group["verdict"].astype(str).str.lower().map(
            {"replicable": 1, "unreplicable": 0}
        )
        if verdict_bin.isna().any():
            raise ValueError(f"{corpus} unit {key_vals} has an unparsable main-task verdict.")
        v_bar = verdict_bin.mean()
        outcomes = group["ground_truth"].astype(str).str.strip().str.lower().unique()
        if len(outcomes) != 1 or outcomes[0] not in {"yes", "no"}:
            raise ValueError(f"{corpus} unit {key_vals} has inconsistent ground truth.")
        ground_truth = outcomes[0]
        title_col = "Study Title (O)" if corpus == "RPP" else "title"
        rows.append(
            dict(
                zip(KEY_COLS[corpus], key_vals),
                title=str(group[title_col].iloc[0]),
                q_bar=pd.to_numeric(group["q_hat"], errors="coerce").mean(),
                v_bar=v_bar,
                outcome=ground_truth,
            )
        )
    out = pd.DataFrame(rows)
    out["majority_correct"] = np.where(
        out["v_bar"].notna(),
        (out["v_bar"] >= 0.5) == (out["outcome"] == "yes"),
        np.nan,
    )
    return out


def embedding_predictions_per_unit(corpus):
    """Load the held-out, seed-averaged embedding-LR prediction per unit."""
    key_cols = KEY_COLS[corpus]
    embed_key_cols = MAIN_KEY_COLS[corpus]
    embed = pd.read_csv(EMBED_PRED_CSV[corpus]).copy()
    if corpus == "RPP":
        embed = embed.rename(columns={"altmejd_id": "unit_id"})
    elif embed_key_cols != key_cols:
        raise ValueError(f"Unexpected key mapping for {corpus}.")
    if embed.duplicated(key_cols).any():
        raise ValueError(f"{corpus}: duplicate embedding-LR unit keys.")
    embed["embed_lr_probability"] = pd.to_numeric(embed["proba_cv_avg"], errors="coerce")
    embed["embed_lr_correct"] = (
        embed["embed_lr_probability"].ge(0.5) == embed["y"].astype(int).eq(1)
    )
    return embed[key_cols + ["y", "embed_lr_probability", "embed_lr_correct"]]


def recognition_accuracy_per_unit(corpus):
    """Merge K_i, default-cell predictions, and embedding LR by stable unit id."""
    key_cols = KEY_COLS[corpus]
    probe = probe_per_unit(corpus)
    main = main_predictions_per_unit(corpus)
    embed = embedding_predictions_per_unit(corpus)
    joined = probe.merge(main, on=key_cols, how="inner", validate="one_to_one")
    joined = joined.merge(embed, on=key_cols, how="inner", validate="one_to_one")
    if len(joined) != len(probe):
        raise ValueError(f"{corpus}: joined {len(joined)} of {len(probe)} probe units.")
    if not (joined["ground_truth"] == joined["outcome"]).all():
        raise ValueError(f"{corpus}: probe and main-pipeline outcomes disagree.")
    if not (joined["ground_truth"].map({"yes": 1, "no": 0}) == joined["y"]).all():
        raise ValueError(f"{corpus}: probe and embedding-LR outcomes disagree.")
    return joined


def recognition_accuracy_results():
    """Return per-stratum accuracies, corpus-level Spearman rho, and unit data."""
    accuracy_rows = []
    spearman_rows = []
    per_unit = {}
    for corpus in ("RPP", "CB", "SSRP"):
        joined = recognition_accuracy_per_unit(corpus)
        per_unit[corpus] = joined
        rho = spearmanr(joined["K_i"], joined["majority_correct"].astype(int)).statistic
        spearman_rows.append(
            {
                "corpus": corpus,
                "corpus_label": FULL_NAME[corpus],
                "n": len(joined),
                "spearman_rho": rho,
            }
        )
        strata = [
            ("K=0", joined["K_i"].eq(0)),
            ("K>=1", joined["K_i"].ge(1)),
            ("K>=5", joined["K_i"].ge(5)),
        ]
        for stratum, mask in strata:
            subset = joined.loc[mask]
            if stratum == "K>=5" and len(subset) < MIN_STRICT_STRATUM_N:
                continue
            accuracy_rows.append(
                {
                    "corpus": corpus,
                    "corpus_label": FULL_NAME[corpus],
                    "stratum": stratum,
                    "n": len(subset),
                    "majority_correct_n": int(subset["majority_correct"].sum()),
                    "majority_accuracy": subset["majority_correct"].mean(),
                    "embed_lr_correct_n": int(subset["embed_lr_correct"].sum()),
                    "embed_lr_accuracy": subset["embed_lr_correct"].mean(),
                }
            )
    return pd.DataFrame(accuracy_rows), pd.DataFrame(spearman_rows), per_unit


def write_recognition_accuracy_outputs(accuracy, correlations, per_unit):
    accuracy.to_csv(os.path.join(OUT_DIR, "contamination_recognition_accuracy.csv"), index=False)
    correlations.to_csv(
        os.path.join(OUT_DIR, "contamination_recognition_spearman.csv"), index=False
    )
    for corpus, frame in per_unit.items():
        frame.to_csv(
            os.path.join(OUT_DIR, f"per_study_{corpus}_high_temp0.7.csv"), index=False
        )


def corpus_summary(corpus):
    key_cols = KEY_COLS[corpus]
    probe = probe_per_unit(corpus)
    vbar = main_predictions_per_unit(corpus)
    n_total = len(probe)

    claimed = probe[probe["claimed_recall"]].copy()
    n_claimed = len(claimed)
    n_correct = int(claimed["claim_correct"].sum())
    n_recalled = int(probe["any_hit_recall"].sum())
    recall_rate = n_recalled / n_total
    n_five_hit = int(probe["five_hit_recall"].sum())
    five_hit_rate = n_five_hit / n_total
    claimed_rate = n_claimed / n_total
    correct_recall_rate = n_correct / n_total
    claim_accuracy = n_correct / n_claimed if n_claimed else np.nan

    if n_claimed:
        modal_outcome = claimed["ground_truth"].mode().iloc[0]
        baseline = (claimed["ground_truth"] == modal_outcome).mean()
        chance_p = binomtest(n_correct, n_claimed, baseline).pvalue
    else:
        baseline = chance_p = np.nan

    joined = probe.merge(vbar, on=key_cols, how="inner")
    joined = joined.dropna(subset=["majority_correct"])
    if len(joined) != n_total:
        raise ValueError(
            f"{corpus}: joined {len(joined)} of {n_total} probe effects/claims to predictions."
        )
    if not (joined["ground_truth"] == joined["outcome"]).all():
        raise ValueError(f"{corpus}: probe and v-bar ground truths disagree.")
    pearson_r, pearson_p = pearsonr(joined["recall_vbar"], joined["v_bar"])
    a = int((joined["correct_recall"] & (joined["majority_correct"] == True)).sum())  # noqa: E712
    b = int((joined["correct_recall"] & (joined["majority_correct"] == False)).sum())  # noqa: E712
    c = int((~joined["correct_recall"] & (joined["majority_correct"] == True)).sum())  # noqa: E712
    d = int((~joined["correct_recall"] & (joined["majority_correct"] == False)).sum())  # noqa: E712
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denominator if denominator > 0 else np.nan
    fisher_p = fisher_exact([[a, b], [c, d]]).pvalue if denominator > 0 else np.nan

    return {
        "corpus": corpus,
        "corpus_label": FULL_NAME[corpus],
        "n_total": n_total,
        "n_claimed": n_claimed,
        "n_correct": n_correct,
        "n_recalled": n_recalled,
        "recall_rate": recall_rate,
        "n_five_hit": n_five_hit,
        "five_hit_rate": five_hit_rate,
        "claimed_rate": claimed_rate,
        "correct_recall_rate": correct_recall_rate,
        "claim_accuracy": claim_accuracy,
        "baseline": baseline,
        "chance_p": chance_p,
        "n_join": len(joined),
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "phi": phi,
        "fisher_p": fisher_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


def all_rows():
    return pd.DataFrame([corpus_summary(corpus) for corpus in ("RPP", "CB", "SSRP")])


def fmt(value, spec=".3f"):
    """Render undefined statistics as an em dash rather than literal NaN."""
    return "—" if pd.isna(value) else format(value, spec)


def fmt_p(value, prefix=False):
    """Use a conventional lower-bound display for extremely small p-values."""
    if pd.isna(value):
        return "—"
    if value < 0.0001:
        return "p<0.0001" if prefix else "<0.0001"
    return f"p={value:.3g}" if prefix else format(value, ".3g")


def chance_note(df):
    above_chance = df[df["chance_p"] < 0.05]
    if above_chance.empty:
        return (
            "No corpus's conditional claim accuracy exceeds its majority-class "
            "baseline at alpha=.05 by the exact binomial test."
        )
    return (
        f"{', '.join(above_chance['corpus_label'])} exceed the relevant "
        "majority-class baseline at alpha=.05."
    )


def correlation_note(df, plain=False):
    correlated = df[df["pearson_p"] < 0.05]
    if correlated.empty:
        return (
            "No corpus has a significant Pearson correlation between the averaged "
            "recall verdict and vote rate at alpha=.05."
        )
    verb = "has" if len(correlated) == 1 else "have"
    return (
        f"{', '.join(correlated['corpus_label'])} {verb} a significant Pearson correlation "
        "between the averaged recall verdict and vote rate at alpha=.05."
    )


def inference_note(df, plain=False):
    """Summarize the chance and correlation tests without hardcoding outcomes."""
    return f"{chance_note(df)} {correlation_note(df, plain=plain)}"


def recognition_accuracy_note(accuracy, correlations):
    """Concise interpretation of recognition-stratified main-task accuracy."""
    cb = accuracy[accuracy["corpus"] == "CB"].set_index("stratum")
    unrecognized = cb.loc["K=0"]
    recognized = cb.loc["K>=1"]
    strict = cb.loc["K>=5"]
    delta = unrecognized["majority_accuracy"] - unrecognized["embed_lr_accuracy"]
    rho = correlations.set_index("corpus").loc["CB", "spearman_rho"]
    return (
        "In CB, main-pipeline accuracy rose from "
        f"{unrecognized['majority_correct_n']}/{unrecognized['n']} "
        f"({unrecognized['majority_accuracy']:.1%}) for $K_i=0$ to "
        f"{recognized['majority_correct_n']}/{recognized['n']} "
        f"({recognized['majority_accuracy']:.1%}) for $K_i\\ge1$ and "
        f"{strict['majority_correct_n']}/{strict['n']} "
        f"({strict['majority_accuracy']:.1%}) for $K_i\\ge5$; "
        f"Spearman $\\rho={rho:.3f}$. Among the $K_i=0$ effects, the main pipeline "
        f"still exceeded embedding LR ({unrecognized['majority_accuracy']:.1%} versus "
        f"{unrecognized['embed_lr_accuracy']:.1%}, a {100 * delta:.1f} percentage-point difference), so its "
        "transfer advantage is not confined to effects with an elicited concordant recall, "
        "although it is markedly smaller in that stratum. Because $K_i=0$ means that this "
        "probe did not elicit a concordant recognized response, it is not proof that the "
        "model never encountered the study during pretraining."
    )


def tex_escape(text):
    return (text.replace("alpha", r"$\alpha$")
                .replace("α", r"$\alpha$")
                .replace("φ", r"$\phi$")
                .replace("%", r"\%"))


def count_pct(count, total, tex=False):
    """Compact count-and-share cell for distribution tables."""
    pct = 100 * count / total if total else np.nan
    suffix = r"\%" if tex else "%"
    return f"{int(count)} ({pct:.1f}{suffix})"


def highlight(text, tex=False):
    return rf"\textbf{{{text}}}" if tex else f"**{text}**"


def write_table(df, accuracy, correlations):
    cb = df.loc[df["corpus"] == "CB"].iloc[0]
    familiarity = response_distributions("familiarity", FAMILIARITY_LEVELS)
    verdicts = response_distributions("verdict", VERDICT_LEVELS)
    outcome_distribution = recognition_verdict_by_outcome()
    outcome_tests = outcome_verdict_tests()
    outcome_tests = outcome_tests.set_index("corpus")
    tex = [
        r"% Generated by make_contamination_report.py -- do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Contamination probe: strict consensus diagnostic for recall of replication outcomes from a title + "
        r"one-line locator for the specific finding (RPP/CB: effect description; SSRP: year "
        r"plus the null-distribution family of the original test statistic, e.g.\ ``an F "
        r"distribution''), "
        rf"$R{{=}}{N_PROBE_RUNS}$ queries per effect/claim at reasoning\_effort=high. A consensus "
        rf"recall requires at least {RECALL_THRESHOLD}/{N_PROBE_RUNS} responses to jointly report "
        r"familiarity=recognized and the same directional verdict (equivalently, a 10-run "
        r"recognized-directional mean of at least 0.6). Consensus accuracy and the "
        r"majority-class baseline use only consensus recalls; Chance $p$ is their two-sided exact "
        r"binomial comparison. Formally, $n_R=\sum_i R_i$, $n_C=\sum_i C_i$, "
        r"Consensus acc. $=n_C/n_R$, and $p_0=\max\{\sum_iR_iY_i,\sum_iR_i(1-Y_i)\}/n_R$. "
        r"Chance $p$ is the two-sided exact "
        r"test for $X\sim\mathrm{Binomial}(n_R,p_0)$ at $X=n_C$. "
        rf"{tex_escape(chance_note(df))}}}",
        r"\label{tab:contamination-probe}",
        r"\scriptsize",
        r"\begin{tabular}{@{}l rrr c c c@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{$N$} & \textbf{Consensus $n$} & \textbf{Correct $n$} & "
        r"\textbf{Consensus acc.} & \textbf{Baseline} & \textbf{Chance $p$} \\",
        r"\midrule",
    ]
    md = [
        "<!-- Generated by make_contamination_report.py -- do not edit by hand. -->",
        "# Contamination probe — recall of replication outcomes",
        "",
        f"R={N_PROBE_RUNS} repeated queries per effect/claim with reasoning_effort=high. Each query "
        "contains the study title plus a one-line locator for the specific finding "
        "(RPP/CB: effect description; SSRP: year plus the null-distribution family of the "
        "original test statistic, e.g. \"an F distribution\").",
        "",
        "A **recall** is an outcome-conditional any hit: at least one of the 10 responses must "
        "jointly report `FAMILIARITY: recognized` and the observed directional outcome. Thus the "
        "required response is `recognized & replicable` for an observed replicable outcome and "
        "`recognized & unreplicable` for an observed unreplicable outcome.",
        "",
        "The strict **consensus diagnostic** separately requires at least 6 of 10 responses to "
        "jointly report `FAMILIARITY: recognized` and the same directional verdict. For direction "
        "$d\\in\\{\\mathrm{rep},\\mathrm{unrep}\\}$, define the "
        "10-run average $\\bar q_i^{(d)}=10^{-1}\\sum_{r=1}^{10}"
        "\\mathbf{1}\\{F_{ir}=\\mathrm{recognized},D_{ir}=d\\}$. Effect/claim $i$ is a "
        "consensus recall when $\\max_d\\bar q_i^{(d)}\\ge0.6$; its consensus prediction is the "
        "qualifying direction. Here $F_{ir}$ is familiarity and $D_{ir}$ is the probe verdict.",
        "",
        f"The correlation uses all effects/claims. The 10 probe runs yield "
        "$\\bar v_i^{\\mathrm{recall}}=\\left[1+\\bar q_i^{(\\mathrm{rep})}-"
        "\\bar q_i^{(\\mathrm{unrep})}\\right]/2$, and the 100 main-task verdicts yield "
        "$\\bar v_i$. The separate recall table reports Pearson "
        "$r=\\mathrm{corr}(\\bar v_i^{\\mathrm{recall}},\\bar v_i)$. "
        f"{correlation_note(df, plain=True)}",
        "",
        "## Strict consensus diagnostic",
        "",
        "| Corpus | N | Consensus n | Correct n | Consensus acc. | Baseline | Chance p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        tex.append(
            f"{row['corpus_label']} & {row['n_total']} & {row['n_claimed']} & "
            f"{row['n_correct']} & {fmt(row['claim_accuracy'])} & "
            f"{fmt(row['baseline'])} & {fmt(row['chance_p'])} \\\\"
        )
        md.append(
            f"| {row['corpus_label']} | {row['n_total']} | {row['n_claimed']} | "
            f"{row['n_correct']} | {fmt(row['claim_accuracy'])} | "
            f"{fmt(row['baseline'])} | {fmt(row['chance_p'])} |"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Effect/claim-level recall and correlation with verdict scores. "
        r"Any-hit recall requires at least one probe response that is recognized and matches the "
        r"observed directional outcome. "
        r"Five-hit recall requires at least five such responses. "
        r"The 10 probe responses are averaged within effect/claim, where $\bar q_i^{(d)}=10^{-1}\sum_r "
        r"\mathbf{1}\{F_{ir}=\mathrm{recognized},D_{ir}=d\}$. "
        r"The 100 main-task verdicts are likewise averaged within effect/claim, "
        r"$\bar v_i=100^{-1}\sum_{r=1}^{100}V_{ir}$. We report Pearson "
        r"$r=\mathrm{corr}(\bar v_i^{\mathrm{recall}},\bar v_i)$, where "
        r"$\bar v_i^{\mathrm{recall}}=\left[1+\bar q_i^{(\mathrm{rep})}-"
        r"\bar q_i^{(\mathrm{unrep})}\right]/2$. "
        rf"{tex_escape(correlation_note(df))}}}",
        r"\label{tab:contamination-probe-recall-association}",
        r"\small",
        r"\begin{tabular}{@{}l r c c c c@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Effects/claims} & \textbf{Any-hit recall rate} & "
        r"\textbf{Five-hit recall rate} & "
        r"\textbf{$\mathrm{corr}(\bar v_i^{\mathrm{recall}},\bar v_i)$} & "
        r"\textbf{Pearson $p$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        tex.append(
            f"{row['corpus_label']} & {row['n_total']} & {row['recall_rate']:.3f} & "
            f"{row['five_hit_rate']:.3f} & "
            f"{fmt(row['pearson_r'])} & {fmt_p(row['pearson_p'])} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += [
        "",
        "## Effect/claim-level recall and verdict-score correlation",
        "",
        "Any-hit recall requires at least one recognized response matching the observed outcome; "
        "five-hit recall requires at least five such responses.",
        "",
        "| Corpus | Effects/claims | Any-hit recall rate | Five-hit recall rate | "
        "$\\mathrm{corr}(\\bar v_i^{\\mathrm{recall}},\\bar v_i)$ | Pearson $p$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        md.append(
            f"| {row['corpus_label']} | {row['n_total']} | {row['recall_rate']:.3f} | "
            f"{row['five_hit_rate']:.3f} | "
            f"{fmt(row['pearson_r'])} | {fmt_p(row['pearson_p'])} |"
        )
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Main-pipeline accuracy stratified by outcome-concordant recognition. "
        r"For unit $i$, $K_i=\sum_{r=1}^{10}\mathbf{1}\{F_{ir}=\mathrm{recognized},"
        r"D_{ir}=Y_i\}$. Main accuracy is the accuracy of the majority verdict from the "
        r"100 high-effort, temperature-0.7 main-task runs; embedding-LR accuracy thresholds "
        r"the seed-averaged held-out probability at 0.5 on the same units. The $K_i\ge5$ "
        rf"stratum is shown only when it contains at least {MIN_STRICT_STRATUM_N} units; "
        r"the $K_i\ge5$ and $K_i\ge1$ strata overlap.}",
        r"\label{tab:contamination-recognition-accuracy}",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Recall stratum} & \textbf{$n$} & "
        r"\textbf{Correct $n$} & \textbf{Main $\bar v$ acc.} & "
        r"\textbf{Embedding-LR acc.} \\",
        r"\midrule",
    ]
    stratum_tex = {"K=0": r"$K_i=0$", "K>=1": r"$K_i\ge1$", "K>=5": r"$K_i\ge5$"}
    for _, row in accuracy.iterrows():
        tex.append(
            f"{row['corpus_label']} & {stratum_tex[row['stratum']]} & {row['n']} & "
            f"{row['majority_correct_n']} & {row['majority_accuracy']:.3f} & "
            f"{row['embed_lr_accuracy']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Rank correlation between the number of outcome-concordant recognized probe "
        r"responses and main-pipeline correctness. Spearman's $\rho$ is computed across all "
        r"units in each corpus between $K_i$ and "
        r"$M_i=\mathbf{1}\{\mathbf{1}(\bar v_i\ge0.5)=Y_i\}$.}",
        r"\label{tab:contamination-recognition-spearman}",
        r"\small",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{$N$} & \textbf{$\rho_S=\mathrm{corr}_{\rm rank}(K_i,M_i)$} \\",
        r"\midrule",
    ]
    for _, row in correlations.iterrows():
        tex.append(
            f"{row['corpus_label']} & {row['n']} & {row['spearman_rho']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex += [r"\noindent " + tex_escape(recognition_accuracy_note(accuracy, correlations))]
    md += [
        "",
        "## Main-pipeline accuracy by outcome-concordant recognition",
        "",
        "$K_i=\\sum_{r=1}^{10}\\mathbf{1}\\{F_{ir}=\\mathrm{recognized},D_{ir}=Y_i\\}$ "
        "counts probe runs that jointly claim recognition and give the observed directional "
        "outcome. Main accuracy uses the majority verdict from the 100 high-effort, "
        "temperature-0.7 main-task runs. Embedding-LR accuracy thresholds the seed-averaged "
        "held-out probability at 0.5 on the same units. The $K_i\\ge5$ row is shown only for "
        f"strata with at least {MIN_STRICT_STRATUM_N} units and is a subset of $K_i\\ge1$.",
        "",
        "| Corpus | Recall stratum | n | Correct n | Main $\\bar v$ acc. | Embedding-LR acc. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    stratum_md = {"K=0": "$K_i=0$", "K>=1": "$K_i\\ge1$", "K>=5": "$K_i\\ge5$"}
    for _, row in accuracy.iterrows():
        md.append(
            f"| {row['corpus_label']} | {stratum_md[row['stratum']]} | {row['n']} | "
            f"{row['majority_correct_n']} | {row['majority_accuracy']:.3f} | "
            f"{row['embed_lr_accuracy']:.3f} |"
        )
    md += [
        "",
        "| Corpus | N | $\\rho_S=\\mathrm{corr}_{\\rm rank}(K_i,M_i)$ |",
        "|---|---:|---:|",
    ]
    for _, row in correlations.iterrows():
        md.append(
            f"| {row['corpus_label']} | {row['n']} | {row['spearman_rho']:.3f} |"
        )
    md += ["", recognition_accuracy_note(accuracy, correlations)]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Raw recognition and verdict responses by corpus and observed replication outcome. "
        r"Cells are count (share) among raw responses in the row's corpus--outcome stratum. "
        r"GEE values are the corpus-level clustered logistic association test described below.}",
        r"\label{tab:contamination-probe-outcome-distribution}",
        r"\scriptsize",
        r"\begin{tabular}{@{}l l r r r r r c c@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Observed outcome} & \textbf{Responses} & "
        r"\textbf{Recognized} & \textbf{Rec. \& rep.} & "
        r"\textbf{Rec. \& unrep.} & \textbf{Rec. \& unknown} & "
        r"\textbf{GEE OR} & \textbf{GEE $p$} \\",
        r"\midrule",
    ]
    for _, row in outcome_distribution.iterrows():
        test = outcome_tests.loc[row["corpus"]]
        recognized_replicable = count_pct(
            row["recognized_replicable"], row["n_responses"], tex=True
        )
        recognized_unreplicable = count_pct(
            row["recognized_unreplicable"], row["n_responses"], tex=True
        )
        if row["outcome"] == "replicable":
            recognized_replicable = highlight(recognized_replicable, tex=True)
        else:
            recognized_unreplicable = highlight(recognized_unreplicable, tex=True)
        tex.append(
            f"{FULL_NAME[row['corpus']]} & {row['outcome']} & {row['n_responses']} & "
            f"{count_pct(row['recognized'], row['n_responses'], tex=True)} & "
            f"{recognized_replicable} & {recognized_unreplicable} & "
            f"{count_pct(row['recognized_unknown'], row['n_responses'], tex=True)} & "
            f"{test['gee_odds_ratio']:.3f} & {test['gee_p']:.3g} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex += [
        r"\noindent For recognized directional response $r$ in unit $i$, let $V_{ir}=1$ "
        r"for a replicable verdict and let $Y_i=1$ for a replicable observed outcome. We fit, "
        r"separately by corpus,",
        r"\[ \mathrm{logit}\{\Pr(V_{ir}=1\mid Y_i)\}=\beta_0+\beta_1Y_i, \]",
        r"using a binomial GEE with unit-clustered, exchangeable working correlation. "
        r"The table reports $\exp(\beta_1)$ and its robust Wald-test $p$-value; values repeat "
        r"within a corpus because the model is fit across both observed-outcome rows.",
    ]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{CB secondary binary contingency table underlying $\phi$ and Fisher's exact test. "
        r"Rows indicate whether an effect has a correct qualified recall claim; "
        r"columns indicate correctness of the self-consistency vote-rate ($\bar v_i$) classifier.}",
        r"\label{tab:contamination-probe-cb-contingency}",
        r"\small",
        r"\begin{tabular}{@{}l rr@{}}",
        r"\toprule",
        r" & \textbf{$\bar v_i$ correct} & \textbf{$\bar v_i$ incorrect} \\",
        r"\midrule",
        f"Correct qualified claim & {cb['a']} & {cb['b']} \\\\",
        f"No correct qualified claim & {cb['c']} & {cb['d']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Distribution of raw contamination-probe familiarity responses. "
        r"Cells are count (share) among all $R=10$ responses per corpus unit.}",
        r"\label{tab:contamination-probe-familiarity}",
        r"\small",
        r"\begin{tabular}{@{}l r r r r r@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Responses} & \textbf{Recognized} & "
        r"\textbf{Unsure} & \textbf{Unfamiliar} & \textbf{Missing} \\",
        r"\midrule",
    ]
    for _, row in familiarity.iterrows():
        tex.append(
            f"{FULL_NAME[row['corpus']]} & {row['n_responses']} & "
            f"{count_pct(row['recognized'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['unsure'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['unfamiliar'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['missing'], row['n_responses'], tex=True)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Distribution of raw contamination-probe verdict responses. "
        r"Cells are count (share) among all $R=10$ responses per corpus unit.}",
        r"\label{tab:contamination-probe-verdict}",
        r"\small",
        r"\begin{tabular}{@{}l r r r r r@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Responses} & \textbf{Replicable} & "
        r"\textbf{Unreplicable} & \textbf{Unknown} & \textbf{Missing} \\",
        r"\midrule",
    ]
    for _, row in verdicts.iterrows():
        tex.append(
            f"{FULL_NAME[row['corpus']]} & {row['n_responses']} & "
            f"{count_pct(row['replicable'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['unreplicable'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['unknown'], row['n_responses'], tex=True)} & "
            f"{count_pct(row['missing'], row['n_responses'], tex=True)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += [
        "",
        "## CB secondary binary contingency table",
        "",
        "This all-effect table is the secondary binary diagnostic underlying CB's φ and Fisher p.",
        "",
        "| Correct qualified claim | $\\bar v_i$ correct | $\\bar v_i$ incorrect |",
        "|---|---:|---:|",
        f"| Yes | {cb['a']} | {cb['b']} |",
        f"| No | {cb['c']} | {cb['d']} |",
        "",
        "## Familiarity distribution",
        "",
        "Counts and shares are over all raw probe responses ($R=10$ per corpus unit), "
        "before the qualified-unit filter.",
        "",
        "| Corpus | Responses | Recognized | Unsure | Unfamiliar | Missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in familiarity.iterrows():
        md.append(
            f"| {FULL_NAME[row['corpus']]} | {row['n_responses']} | "
            f"{count_pct(row['recognized'], row['n_responses'])} | "
            f"{count_pct(row['unsure'], row['n_responses'])} | "
            f"{count_pct(row['unfamiliar'], row['n_responses'])} | "
            f"{count_pct(row['missing'], row['n_responses'])} |"
        )
    md += [
        "",
        "## Verdict distribution",
        "",
        "Counts and shares are over all raw probe responses ($R=10$ per corpus unit), "
        "before the qualified-unit filter.",
        "",
        "| Corpus | Responses | Replicable | Unreplicable | Unknown | Missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in verdicts.iterrows():
        md.append(
            f"| {FULL_NAME[row['corpus']]} | {row['n_responses']} | "
            f"{count_pct(row['replicable'], row['n_responses'])} | "
            f"{count_pct(row['unreplicable'], row['n_responses'])} | "
            f"{count_pct(row['unknown'], row['n_responses'])} | "
            f"{count_pct(row['missing'], row['n_responses'])} |"
        )
    md += [
        "",
        "## Recognition and verdict by observed outcome",
        "",
        "Each row is a corpus--observed-outcome stratum. Counts and shares use all raw "
        "responses in that row, before the qualified-unit filter.",
        "",
        "| Corpus | Observed outcome | Responses | Recognized | Recognized & replicable | Recognized & unreplicable | Recognized & unknown | GEE OR | GEE p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in outcome_distribution.iterrows():
        test = outcome_tests.loc[row["corpus"]]
        recognized_replicable = count_pct(
            row["recognized_replicable"], row["n_responses"]
        )
        recognized_unreplicable = count_pct(
            row["recognized_unreplicable"], row["n_responses"]
        )
        if row["outcome"] == "replicable":
            recognized_replicable = highlight(recognized_replicable)
        else:
            recognized_unreplicable = highlight(recognized_unreplicable)
        md.append(
            f"| {FULL_NAME[row['corpus']]} | {row['outcome']} | {row['n_responses']} | "
            f"{count_pct(row['recognized'], row['n_responses'])} | "
            f"{recognized_replicable} | {recognized_unreplicable} | "
            f"{count_pct(row['recognized_unknown'], row['n_responses'])} | "
            f"{test['gee_odds_ratio']:.3f} | {test['gee_p']:.3g} |"
        )
    md += [
        "",
        "The GEE values repeat within each corpus because the model is fit across both "
        "observed-outcome rows. ANOVA is not appropriate for this categorical outcome. "
        "For recognized directional response $r$ in unit $i$, let $V_{ir}=1$ denote a "
        "replicable verdict and let $Y_i=1$ denote a replicable observed outcome. Separately "
        "by corpus, the binomial GEE is:",
        "",
        "$$",
        "\\mathrm{logit}\\{\\Pr(V_{ir}=1\\mid Y_i)\\}=\\beta_0+\\beta_1Y_i.",
        "$$",
        "",
        "The model uses binomial variance and an exchangeable working correlation for repeated "
        "responses in the same unit. GEE OR is $\\exp(\\beta_1)$; GEE p is the robust Wald-test "
        "p-value for $H_0:\\beta_1=0$.",
        "",
        "## How the statistics are derived",
        "",
        "For effect/claim $i$, let $H_i=1$ when at least one of 10 probe responses is "
        "recognized and has the observed directional verdict; otherwise let $H_i=0$. "
        "For the strict consensus diagnostic and direction $d\\in\\{\\mathrm{rep},\\mathrm{unrep}\\}$, first "
        "average the 10 probe responses: $\\bar q_i^{(d)}=10^{-1}\\sum_{r=1}^{10}"
        "\\mathbf{1}\\{F_{ir}=\\mathrm{recognized},D_{ir}=d\\}$. Let $R_i=1$ when "
        "$\\max_d\\bar q_i^{(d)}\\ge0.6$; the qualifying direction is the recall prediction. "
        "Here $F_{ir}$ is familiarity and $D_{ir}$ is the probe verdict. "
        "Separately average the 100 binary main-task verdicts: "
        "$\\bar v_i=100^{-1}\\sum_{r=1}^{100}V_{ir}$, where $V_{ir}=1$ denotes replicable. "
        "Let $C_i=1$ when $R_i=1$ and the recall prediction equals the observed "
        "replication outcome $Y_i$; otherwise let $C_i=0$. Let $M_i=1$ when the classifier based "
        "on $\\bar v_i$ is correct, where $\\bar v_i$ is classified as replicable at "
        "$\\bar v_i\\ge0.5$.",
        "",
        "- **Any-hit recall rate** is $n_H/N$, where $n_H=\\sum_{i=1}^{N}H_i$.",
        "- **Five-hit recall rate** is $n_5/N$, where $n_5=\\sum_{i=1}^{N}"
        "\\mathbf{1}\\{K_i\\ge5\\}$ and $K_i$ is the number of recognized probe responses "
        "that match the observed directional outcome.",
        "- **Recognition-stratified accuracy** divides units by $K_i=0$ versus "
        "$K_i\\ge1$, with $K_i\\ge5$ as a stricter overlapping stratum when it contains at "
        f"least {MIN_STRICT_STRATUM_N} units. Main accuracy is $\\sum_i M_i/n$ within the "
        "stratum. The embedding-LR comparison uses held-out, seed-averaged probabilities "
        "thresholded at 0.5 for exactly the same units.",
        "- **Spearman correlation** is "
        "$\\rho_S=\\mathrm{corr}_{\\rm rank}(K_i,M_i)$ across all $N$ units in a corpus; "
        "ties in the count and binary correctness variables receive average ranks.",
        "- **Consensus n** is $n_R=\\sum_{i=1}^{N}R_i$. **Correct n** is "
        "$n_C=\\sum_{i=1}^{N}C_i$. Thus Consensus acc. is $n_C/n_R$.",
        "- **Baseline** is the modal-outcome accuracy among consensus recalls: "
        "$p_0=\\max\\{\\sum_iR_iY_i,\\;\\sum_iR_i(1-Y_i)\\}/n_R$. It is the accuracy of always "
        "predicting the more common observed outcome in that subset.",
        "- **Chance p** is the two-sided exact binomial-test p-value for observing $n_C$ correct "
        "consensus claims under $X\\sim\\mathrm{Binomial}(n_R,p_0)$. It tests whether Consensus acc. differs "
        "from the modal-outcome baseline; it is not the probability that a response was guessed.",
        "- **Pearson correlation** uses all $N$ effects/claims: "
        "$\\bar v_i^{\\mathrm{recall}}=\\left[1+\\bar q_i^{(\\mathrm{rep})}-"
        "\\bar q_i^{(\\mathrm{unrep})}\\right]/2$, which is 0.5 for no net recognized directional "
        "recall evidence, and the main-task rate is $\\bar v_i$. The recall table reports "
        "$r=\\mathrm{corr}(\\bar v_i^{\\mathrm{recall}},\\bar v_i)$ with its two-sided Pearson p-value.",
        "- **$\\phi$ and Fisher p** remain a secondary binary diagnostic. Define $a=\\sum_i C_iM_i$, "
        "$b=\\sum_i C_i(1-M_i)$, $c=\\sum_i(1-C_i)M_i$, and "
        "$d=\\sum_i(1-C_i)(1-M_i)$. Then "
        "$\\phi=(ad-bc)/\\sqrt{(a+b)(c+d)(a+c)(b+d)}$. Fisher p is the two-sided exact-test "
        "p-value for independence in this same $2\\times2$ table, conditional on its margins.",
        "",
        "Self-reported familiarity is a necessary filter, not proof of memorization. These "
        "are therefore *claimed*, not verified, recalls. Verification requires blinded coding "
        "of each `NOTES` response for an accurate replication-specific detail. CB also "
        "contains effects nested within 23 papers, so its effect-level Pearson and Fisher p-values "
        "should be treated as descriptive pending paper-clustered inference.",
        "",
        "## Recognition-accuracy result",
        "",
        recognition_accuracy_note(accuracy, correlations),
    ]

    with open(os.path.join(OUT_DIR, "contamination_probe.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex) + "\n")
    with open(os.path.join(OUT_DIR, "contamination_probe.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")
    print(f"Wrote {os.path.join(OUT_DIR, 'contamination_probe.tex')}")
    print(f"Wrote {os.path.join(OUT_DIR, 'contamination_probe.md')}")


def make_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    x = np.arange(len(df))
    width = 0.32
    colors = [CORPUS_COLOR[corpus] for corpus in df["corpus"]]

    ax = axes[0]
    for i, (column, label) in enumerate(
        [("recall_rate", "Any-hit recall"), ("claimed_rate", "6/10 consensus")]
    ):
        successes = df["n_recalled"] if column == "recall_rate" else df["n_claimed"]
        errors = np.array(
            [exact_error(int(k), int(n)) for k, n in zip(successes, df["n_total"])]
        ).T
        ax.bar(
            x + (i - 0.5) * width,
            df[column],
            width=width,
            yerr=errors,
            capsize=3,
            color=colors if i == 0 else "0.72",
            label=label,
        )
    ax.set_ylabel("Fraction of all effects/claims")
    ax.set_title("Any-hit recall and consensus")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              fontsize=7.5, frameon=False)

    ax = axes[1]
    for i, (column, label) in enumerate(
        [("claim_accuracy", "Consensus accuracy"), ("baseline", "Majority baseline")]
    ):
        kwargs = {}
        if column == "claim_accuracy":
            kwargs["yerr"] = np.array(
                [exact_error(int(k), int(n)) for k, n in zip(df["n_correct"], df["n_claimed"])]
            ).T
            kwargs["capsize"] = 3
        ax.bar(
            x + (i - 0.5) * width,
            df[column],
            width=width,
            color=colors if i == 0 else "0.72",
            label=label,
            **kwargs,
        )
    ax.set_ylabel("Accuracy among consensus recalls")
    ax.set_title("Consensus accuracy vs. baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              fontsize=7.5, frameon=False)

    ax = axes[2]
    ax.bar(x, df["pearson_r"], color=colors, width=0.55)
    for i, row in df.iterrows():
        ax.text(
            i,
            row["pearson_r"] + 0.06,
            fmt_p(row["pearson_p"], prefix=True),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_ylabel(r"Pearson $r$")
    ax.set_title(r"$\mathrm{corr}(\bar v_i^{\mathrm{recall}}, \bar v_i)$")
    ax.set_ylim(-1.05, 1.05)

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(df["corpus_label"])
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax = axes[2]
    ax.set_xticks(x)
    ax.set_xticklabels(df["corpus_label"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(r"Contamination probe: title + locator recall and $\bar v_i$ accuracy", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out_png = os.path.join(OUT_DIR, "contamination_probe.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    summary = all_rows()
    accuracy, correlations, per_unit = recognition_accuracy_results()
    write_recognition_accuracy_outputs(accuracy, correlations, per_unit)
    print(
        summary[
            [
                "corpus_label",
                "n_total",
                "n_claimed",
                "n_correct",
                "claimed_rate",
                "correct_recall_rate",
                "claim_accuracy",
                "baseline",
                "chance_p",
                "phi",
                "fisher_p",
            ]
        ].to_string(index=False, float_format="{:.3f}".format)
    )
    print("\nRecognition-stratified accuracy:")
    print(accuracy.to_string(index=False, float_format="{:.3f}".format))
    print("\nSpearman correlations:")
    print(correlations.to_string(index=False, float_format="{:.3f}".format))
    write_table(summary, accuracy, correlations)
    make_figure(summary)
