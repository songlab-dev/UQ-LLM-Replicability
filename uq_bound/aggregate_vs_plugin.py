"""Tier-3 diagnostic: plug-in versus aggregate aleatoric estimator.

For each corpus x reasoning-effort x temperature cell, compare

    A_plugin = n^-1 sum_i p_i^* (1 - p_i^*)

with

    A_agg = n^-1 sum_i R_i^-1 sum_r (p_ir - Y_i)^2
            - n^-1 sum_i (V_i + E_i),

where V_i is the sample variance of p_ir and

    E_i = (pbar_i - p_i^*)^2 - V_i / R_i

is the finite-R corrected squared-bias estimate.

Usage: rep_env/bin/python llm_uq/uq_bound/aggregate_vs_plugin.py
Writes: llm_uq/uq_bound/results/tier3_aggregate_vs_plugin.csv
        llm_uq/uq_bound/figures/tier3_aggregate_vs_plugin.pdf/.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent  # uq_bound/
REPO_ROOT = HERE.parent  # llm_uq/
sys.path.insert(0, str(HERE))
import common  # noqa: E402


PRED_ROOT = REPO_ROOT / "prediction" / "LLM_Reasoning"
OUT_CSV = HERE / "results" / "tier3_aggregate_vs_plugin.csv"
OUT_FIG = HERE / "figures" / "tier3_aggregate_vs_plugin"
CELLS = [("high", "0.7"), ("low", "0.7"),
         ("high", "0.2"), ("low", "0.2")]
CELL_LABELS = {
    ("high", "0.7"): "High / 0.7",
    ("low", "0.7"): "Low / 0.7",
    ("high", "0.2"): "High / 0.2",
    ("low", "0.2"): "Low / 0.2",
}


def parse_outcome(values):
    mapping = {"yes": 1.0, "no": 0.0, "replicable": 1.0,
               "unreplicable": 0.0, "1": 1.0, "0": 0.0}
    return values.astype(str).str.strip().str.lower().map(mapping)


def load_reference(corpus):
    """Return one row per effect/claim with its p* and resampling cluster."""
    if corpus == "RPP":
        scope = common.load_rpp_scope().copy()
        # VEA uses the 20-seed nested-CV average.
        embed = pd.read_csv(common.RPP_EMBED_LR_CSV)
        if len(embed) != len(scope) or not np.array_equal(embed["title"], scope["title"]):
            raise ValueError("RPP embedding-LR rows no longer align with the 90-effect scope")
        scope["p_star"] = embed["proba_cv_avg"].to_numpy()
        return scope.rename(columns={"id": "claim_id",
                                     "Study Title (O)": "cluster_id"})[
            ["claim_id", "cluster_id", "p_star"]]

    if corpus == "CB":
        scope = common.load_scope_cb().copy()
        scope["p_star"] = common.load_pstar4_cb(scope)
        return scope.rename(columns={"Paper #": "cluster_id"})[
            ["claim_id", "cluster_id", "p_star"]]

    if corpus == "SSRP":
        scope = common.load_scope_ssrp().copy()
        scope["p_star"] = common.load_pstar4_ssrp(scope)
        return scope.rename(columns={"Study Num": "claim_id"}).assign(
            cluster_id=lambda d: d["claim_id"]
        )[["claim_id", "cluster_id", "p_star"]]

    raise ValueError(f"Unknown corpus: {corpus}")


def load_predictions(corpus, effort, temp):
    directory = {"RPP": "RPP", "CB": "CB", "SSRP": "SSRP"}[corpus]
    path = PRED_ROOT / directory / f"text_predictions_{effort}_temp{temp}.csv"
    if corpus == "RPP":
        pred = pd.read_csv(path, usecols=["altmejd_id", "q_hat", "ground_truth", "run_id"])
        pred = pred.rename(columns={"altmejd_id": "claim_id"})
    elif corpus == "CB":
        pred = pd.read_csv(path, usecols=["paper_num", "experiment_num", "effect_num",
                                          "q_hat", "ground_truth", "run_id"])
        pred["claim_id"] = [common._claim_key(p, e, f) for p, e, f in
                            zip(pred["paper_num"], pred["experiment_num"], pred["effect_num"])]
    else:
        pred = pd.read_csv(path, usecols=["study_num", "q_hat", "ground_truth", "run_id"])
        pred = pred.rename(columns={"study_num": "claim_id"})

    pred["q_hat"] = pd.to_numeric(pred["q_hat"], errors="coerce")
    pred["Y_run"] = parse_outcome(pred["ground_truth"])
    if pred[["q_hat", "Y_run"]].isna().any().any():
        missing = pred[["q_hat", "Y_run"]].isna().sum().to_dict()
        raise ValueError(f"{path} has missing diagnostic inputs: {missing}")
    pred["run_sq_error"] = (pred["q_hat"] - pred["Y_run"]) ** 2
    return pred


def load_claims(corpus, effort, temp):
    reference = load_reference(corpus)
    pred = load_predictions(corpus, effort, temp)
    grouped = pred.groupby("claim_id", sort=False, dropna=False)
    if grouped["Y_run"].nunique().max() != 1:
        raise ValueError(f"{corpus} {effort}/{temp}: outcome varies within claim")

    claims = grouped.agg(
        q_bar=("q_hat", "mean"),
        V_i=("q_hat", "var"),
        run_mse_i=("run_sq_error", "mean"),
        Y_run=("Y_run", "first"),
        R=("q_hat", "size"),
    ).reset_index().merge(reference, on="claim_id", how="left", validate="one_to_one")

    if claims[["cluster_id", "p_star", "V_i"]].isna().any().any():
        missing = claims[["cluster_id", "p_star", "V_i"]].isna().sum().to_dict()
        raise ValueError(f"{corpus} {effort}/{temp}: unmatched/missing reference inputs: {missing}")
    if len(claims) != len(reference):
        raise ValueError(f"{corpus} {effort}/{temp}: predictions do not match the reference scope")

    claims["E_i"] = (claims["q_bar"] - claims["p_star"]) ** 2 - claims["V_i"] / claims["R"]
    claims["plugin_i"] = claims["p_star"] * (1.0 - claims["p_star"])
    claims["aggregate_i"] = claims["run_mse_i"] - claims["V_i"] - claims["E_i"]
    return claims


def compute_results():
    rows = []
    corpora = ["RPP", "CB", "SSRP"]
    for corpus in corpora:
        for effort, temp in CELLS:
            claims = load_claims(corpus, effort, temp)
            plugin = claims["plugin_i"].mean()
            aggregate = claims["run_mse_i"].mean() - claims[["V_i", "E_i"]].sum(axis=1).mean()
            rows.append({
                "corpus": corpus,
                "effort": effort,
                "temperature": float(temp),
                "n_claims": len(claims),
                "n_clusters": claims["cluster_id"].nunique(),
                "R_min": int(claims["R"].min()),
                "R_max": int(claims["R"].max()),
                "plugin_estimator": plugin,
                "aggregate_estimator": aggregate,
                "mean_run_squared_error": claims["run_mse_i"].mean(),
                "mean_V": claims["V_i"].mean(),
                "mean_E": claims["E_i"].mean(),
                "gap_aggregate_minus_plugin": aggregate - plugin,
                "ratio_aggregate_to_plugin": aggregate / plugin,
            })
    return pd.DataFrame(rows)


def build_figure(results):
    corpus_colors = {"RPP": "#0072B2", "CB": "#D55E00", "SSRP": "#009E73"}
    cell_markers = {
        ("high", 0.7): "o",
        ("low", 0.7): "s",
        ("high", 0.2): "D",
        ("low", 0.2): "^",
    }
    fig, ax = plt.subplots(figsize=(7.1, 5.8))

    lo = min(results["plugin_estimator"].min(), results["aggregate_estimator"].min())
    hi = max(results["plugin_estimator"].max(), results["aggregate_estimator"].max())
    pad = 0.04 * (hi - lo)
    limits = (lo - pad, hi + pad)
    ax.plot(limits, limits, color="#666666", linewidth=1.1, linestyle="--", zorder=1)

    for _, row in results.iterrows():
        x = row["plugin_estimator"]
        y = row["aggregate_estimator"]
        ax.plot(x, y, marker=cell_markers[(row["effort"], row["temperature"])],
                linestyle="none", color=corpus_colors[row["corpus"]],
                markersize=9, markeredgecolor="white", markeredgewidth=0.7,
                zorder=3)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Plug-in aleatoric estimate")
    ax.set_ylabel("Aggregate aleatoric estimate")
    ax.set_title("Plug-in versus aggregate aleatoric uncertainty")
    ax.grid(color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    corpus_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                             markerfacecolor=color, markeredgecolor=color, label=corpus)
                      for corpus, color in corpus_colors.items()]
    cell_handles = [Line2D([0], [0], marker=cell_markers[(effort, float(temp))],
                           linestyle="none", markersize=7, markerfacecolor="#666666",
                           markeredgecolor="#666666", label=CELL_LABELS[(effort, temp)])
                    for effort, temp in CELLS]
    equality = Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.1,
                      label="Equality")
    legend1 = ax.legend(handles=corpus_handles, title="Corpus", frameon=False,
                        loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=cell_handles + [equality], title="Effort / temperature",
              frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def main():
    results = compute_results()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(results[["corpus", "effort", "temperature", "plugin_estimator",
                   "aggregate_estimator", "gap_aggregate_minus_plugin"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {OUT_CSV}")

    fig = build_figure(results)
    fig.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_FIG.with_suffix('.pdf')}")
    print(f"Wrote {OUT_FIG.with_suffix('.png')}")


if __name__ == "__main__":
    main()
