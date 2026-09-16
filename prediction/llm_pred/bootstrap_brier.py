"""Study-cluster bootstrap metrics from the stored default/grid predictions.

This is the repository-native counterpart to the external audit script.  It
uses the canonical RPP OSF outcomes (rather than the legacy embedding-file
label) and the ECE convention used by the active calibration generators:
scores are ordered with NumPy quicksort and split into fixed-size quantile
bins.  No API requests are made.

The bootstrap resamples paper clusters with replacement: normalized RPP study
title (88 clusters for 90 claims), CB paper number (23 clusters), or SSRP
study number (21 clusters).  It reports percentile 95% intervals from
B=10,000 draws with seed 20260913.

Usage: rep_env/bin/python prediction/llm_pred/bootstrap_brier.py
Writes: prediction/llm_pred/metric/bootstrap_brier_metrics.csv
        prediction/llm_pred/metric/bootstrap_brier_per_unit_{RPP,CB,SSRP}.csv
        prediction/llm_pred/table_bootstrap_ci.tex
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data"))
from rpp_outcomes import canonical_rpp_outcomes_for_titles  # noqa: E402


B = 10_000
SEED = 20_260_913
CELLS = ("high_temp0.7", "low_temp0.7", "high_temp0.2", "low_temp0.2")
N_BINS = {"RPP": 5, "CB": 5, "SSRP": 3}
FOLDERS = {"RPP": "RPP", "CB": "CB", "SSRP": "SSRP"}
UNIT_COLUMNS = {
    "RPP": ("altmejd_id",),
    "CB": ("paper_num", "experiment_num", "effect_num"),
    "SSRP": ("study_num",),
}
CLUSTER_COLUMNS = {
    "RPP": ("Study Title (O)",),
    "CB": ("paper_num",),
    "SSRP": ("study_num",),
}
OUTCOME_MAP = {
    "yes": 1, "no": 0, "replicable": 1, "unreplicable": 0,
    "1": 1, "0": 0,
}


def _join_columns(frame, columns):
    return frame.loc[:, columns].astype(str).agg("|".join, axis=1)


def _binary_outcome(values):
    outcome = values.astype(str).str.strip().str.lower().map(OUTCOME_MAP)
    if outcome.isna().any():
        bad = values.loc[outcome.isna()].unique().tolist()
        raise ValueError(f"Unexpected outcome labels: {bad}")
    return outcome.astype(int)


def load_cell(corpus, cell):
    """Load one stored prediction cell as one row per analysis unit."""
    path = REPO / "prediction" / "LLM_Reasoning" / FOLDERS[corpus] / f"text_predictions_{cell}.csv"
    frame = pd.read_csv(path)
    unit_cols = UNIT_COLUMNS[corpus]
    cluster_cols = CLUSTER_COLUMNS[corpus]
    missing = set((*unit_cols, *cluster_cols, "q_hat", "verdict")) - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["unit"] = _join_columns(frame, unit_cols)
    frame["cluster"] = _join_columns(frame, cluster_cols)
    frame["verdict_binary"] = _binary_outcome(frame["verdict"])
    grouped = frame.groupby("unit", sort=False)
    per_unit = grouped.agg(
        qbar=("q_hat", "mean"),
        vbar=("verdict_binary", "mean"),
        cluster=("cluster", "first"),
        runs=("verdict_binary", "size"),
    ).reset_index()
    if not (per_unit["runs"] == 100).all():
        bad = per_unit.loc[per_unit["runs"] != 100, ["unit", "runs"]]
        raise ValueError(f"Expected 100 runs per {corpus} unit in {path}; got\n{bad}")

    if corpus == "RPP":
        titles = grouped["Study Title (O)"].first().reindex(per_unit["unit"])
        per_unit["outcome"] = canonical_rpp_outcomes_for_titles(titles).to_numpy()
    else:
        if "ground_truth" not in frame:
            raise KeyError(f"{path} is missing ground_truth")
        outcomes = grouped["ground_truth"].first().reindex(per_unit["unit"])
        per_unit["outcome"] = _binary_outcome(outcomes).to_numpy()
    if per_unit["outcome"].isna().any():
        raise ValueError(f"Missing outcomes after aggregating {path}")
    per_unit["outcome"] = per_unit["outcome"].astype(int)
    return per_unit


def ece(y, probability, n_bins):
    """Weighted fixed-count ECE under the recorded quicksort tie convention."""
    order = np.argsort(probability, kind="quicksort")
    result = 0.0
    for bin_index in np.array_split(order, n_bins):
        weight = len(bin_index) / len(order)
        result += weight * abs(probability[bin_index].mean() - y[bin_index].mean())
    return float(result)


def point_metrics(y, probability, n_bins):
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float((predicted == y).mean()),
        "brier": float(np.mean((probability - y) ** 2)),
        "ece": ece(y, probability, n_bins),
    }


def bootstrap_metrics(per_unit, predictor, n_bins, seed):
    """Return point metrics and paired base-rate Brier contrasts."""
    y = per_unit["outcome"].to_numpy(dtype=int)
    probability = per_unit[predictor].to_numpy(dtype=float)
    clusters = per_unit["cluster"].to_numpy()
    unique_clusters = pd.unique(clusters)
    members = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    rng = np.random.default_rng(seed)
    draws = {
        name: np.empty(B)
        for name in ("accuracy", "brier", "ece", "brier_excess_vs_base_rate")
    }

    for draw in range(B):
        sampled_clusters = rng.integers(0, len(members), size=len(members))
        index = np.concatenate([members[position] for position in sampled_clusters])
        sampled_y = y[index]
        metrics = point_metrics(sampled_y, probability[index], n_bins)
        sampled_base_rate = sampled_y.mean()
        metrics["brier_excess_vs_base_rate"] = (
            metrics["brier"] - sampled_base_rate * (1 - sampled_base_rate)
        )
        for name, value in metrics.items():
            draws[name][draw] = value

    result = point_metrics(y, probability, n_bins)
    base_rate = y.mean()
    result["brier_excess_vs_base_rate"] = result["brier"] - base_rate * (1 - base_rate)
    for name, values in draws.items():
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.percentile(
            values, [2.5, 97.5]
        )
    return result


def latex_table(results):
    """Render the default-cell LLM metrics and Brier base-rate benchmark."""
    default = results.loc[results["cell"] == "high_temp0.7"].copy()
    labels = {"qbar": r"$\bar q$", "vbar": r"$\bar v$"}
    lines = [
        r"% Generated by llm_pred/bootstrap_brier.py -- do not edit by hand.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Study-cluster bootstrap estimates for the default LLM cell "
        r"(high effort, temperature $0.7$, $R=100$). Values are point estimates "
        r"with percentile 95\% intervals from $B=10{,}000$ paper-cluster "
        r"resamples (seed 20260913). RPP outcomes use the original OSF export. "
        r"ECE uses fixed-count bins (five for RPP/CB; three for SSRP) and "
        r"the recorded NumPy quicksort tie convention. The final two columns "
        r"give the Brier score of a constant predictor set to the observed "
        r"corpus base rate and the paired Brier difference from that benchmark "
        r"(positive values are worse).}",
        r"\label{tab:bootstrap-brier}",
        r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
        r"Corpus & Predictor & Accuracy & Brier & ECE & Base-rate Brier & "
        r"$\Delta$ Brier " + r"\\\\",
        r"\midrule",
    ]
    for corpus in ("RPP", "CB", "SSRP"):
        rows = default.loc[default["corpus"] == corpus]
        for position, (_, row) in enumerate(rows.iterrows()):
            interval = lambda name: (
                f"{row[name]:.3f} {{\\scriptsize$[{row[f'{name}_ci_low']:.3f}, "
                f"{row[f'{name}_ci_high']:.3f}]$}}"
            )
            lines.append(
                f"{corpus if position == 0 else ''} & {labels[row['predictor']]} & "
                f"{interval('accuracy')} & {interval('brier')} & {interval('ece')} & "
                f"{row['base_rate_brier']:.3f} & "
                f"{interval('brier_excess_vs_base_rate')} \\\\"
            )
        if corpus != "SSRP":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    rows = []
    per_unit_outputs = {}
    run_number = 0
    for corpus in ("RPP", "CB", "SSRP"):
        for cell in CELLS:
            per_unit = load_cell(corpus, cell)
            if cell == "high_temp0.7":
                per_unit_outputs[corpus] = per_unit
            for predictor in ("qbar", "vbar"):
                metrics = bootstrap_metrics(
                    per_unit, predictor, N_BINS[corpus], SEED + run_number
                )
                run_number += 1
                base_rate = per_unit["outcome"].mean()
                rows.append({
                    "corpus": corpus,
                    "cell": cell,
                    "predictor": predictor,
                    "n": len(per_unit),
                    "clusters": per_unit["cluster"].nunique(),
                    "runs_per_unit": int(per_unit["runs"].iloc[0]),
                    "base_rate": base_rate,
                    "base_rate_brier": base_rate * (1 - base_rate),
                    **metrics,
                })
            print(f"{corpus} {cell}: completed qbar/vbar bootstrap metrics")

    metric_dir = REPO / "prediction" / "llm_pred" / "metric"
    metric_dir.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(metric_dir / "bootstrap_brier_metrics.csv", index=False, encoding="utf-8")
    for corpus, per_unit in per_unit_outputs.items():
        per_unit.to_csv(
            metric_dir / f"bootstrap_brier_per_unit_{FOLDERS[corpus]}.csv", index=False,
            encoding="utf-8"
        )
    (REPO / "prediction" / "llm_pred" / "table_bootstrap_ci.tex").write_text(
        latex_table(results), encoding="utf-8"
    )
    print(f"Wrote {metric_dir / 'bootstrap_brier_metrics.csv'}")
    print(f"Wrote {REPO / 'doc' / 'table_bootstrap_ci.tex'}")


if __name__ == "__main__":
    main()
