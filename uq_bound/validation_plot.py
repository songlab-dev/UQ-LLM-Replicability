"""Validation plots for RPP, CB, and SSRP uncertainty bounds.

The three former validation-plot scripts shared Panels A--C and differed in
their corpus adapter and Panel B design.  This entry point keeps those
differences explicit while sharing the plotting, bootstrap, and bound logic.

Examples (run from any directory)::

    python uq_bound/validation_plot.py --corpus rpp
    python uq_bound/validation_plot.py --corpus cb --effort high
    python uq_bound/validation_plot.py --corpus all

By default, all four existing effort/temperature combinations are rendered,
matching the former scripts. Outputs remain under uq_bound/results/{RPP,CB,SSRP}.
"""

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
from ah_bound import ub_ah  # noqa: E402
from bernstein_bound import ub_bernstein  # noqa: E402
from freedman_bound import ub_freedman  # noqa: E402

TAUS_FINE = np.linspace(0.01, 0.6, 60)
TAU_LEVELS_B = [0.2, 0.4, 0.6]
N_BOOT = 1000
BOOT_ALPHA = 0.05
BERN_DELTA = 0.05
C_TIGHT_RPP = 0.4
C_TIGHT_GRID = 0.05
N_C_GRID = 60
LO_PCT, HI_PCT = 0.05, 0.95
STEP_LABELS = ["Extraction", "Stat review", "Researcher DoF\nconcern",
               "Theory\nand context", "Scoring"]


def data_driven_c_tight(max_abs_d, grid=C_TIGHT_GRID):
    return math.ceil(max_abs_d / grid) * grid


def per_study_curves(delta, variance, clip, taus):
    return np.array([ub_freedman(tau, delta, variance, clip) for tau in taus]).T


def cluster_bootstrap_band(delta, variance, clip, taus, n_boot=N_BOOT,
                           seed=0, alpha=BOOT_ALPHA):
    rng = np.random.default_rng(seed)
    delta, variance = np.asarray(delta), np.asarray(variance)
    curves = np.empty((n_boot, len(taus)))
    for draw in range(n_boot):
        indices = rng.integers(0, len(delta), len(delta))
        curves[draw] = [ub_freedman(tau, delta[indices], variance[indices], clip).mean()
                        for tau in taus]
    return (np.percentile(curves, 100 * alpha / 2, axis=0),
            np.percentile(curves, 100 * (1 - alpha / 2), axis=0))


def run_bootstrap_ci(corpus, claim_id, chain, n_boot=N_BOOT,
                     seed=0, alpha=BOOT_ALPHA):
    key_column = {"rpp": "altmejd_id", "ssrp": "Study Num", "cb": "claim_id"}[corpus]
    values = chain[chain[key_column] == claim_id][["D1", "D2", "D3", "D4", "D5"]].to_numpy() ** 2
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, values.shape[1]))
    for draw in range(n_boot):
        indices = rng.integers(0, len(values), len(values))
        draws[draw] = values[indices].mean(axis=0)
    mean = values.mean(axis=0)
    return mean, np.percentile(draws, 100 * alpha / 2, axis=0), np.percentile(
        draws, 100 * (1 - alpha / 2), axis=0
    )


def claim_label(corpus, claims, claim_id, maxlen=60):
    row = claims.loc[claim_id]
    title = str(row["Study Title (O)"])[:maxlen]
    if corpus == "cb":
        return f"{title} (Exp {int(row['Experiment #'])}, Effect {int(row['Effect #'])})"
    return title


def build_data(corpus, pred_csv):
    result = common.build_claims_and_deviations(corpus, pred_csv)
    if corpus == "rpp":
        claims, chain, unit_scores, _met, _c_hat, c2, max_abs_d = result
    else:
        claims, chain, unit_scores, _c_hat, c2, max_abs_d = result
    return claims, chain, unit_scores, c2, max_abs_d


def plot_panel_a(corpus, claims, unit_scores, c2, max_abs_d, c_tight, label, outputs):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(claims)
    empirical = common.broadcast_to_claims(
        corpus, claims, common.per_run_exceedance(corpus, claims, unit_scores, TAUS_FINE)
    ).mean(axis=0).values
    ah = np.array([ub_ah(tau, claims["delta"], c2).mean() for tau in TAUS_FINE])
    c2_tight = len(common.CHAIN) * c_tight ** 2
    ah_tight = np.array([ub_ah(tau, claims["delta"], c2_tight).mean() for tau in TAUS_FINE])
    freedman = np.array([
        ub_freedman(tau, claims["delta"], claims["v_i"], c=1.0).mean()
        for tau in TAUS_FINE
    ])
    freedman_tight = np.array([
        ub_freedman(tau, claims["delta"], claims["v_i"], c=c_tight).mean()
        for tau in TAUS_FINE
    ])
    bernstein = np.array([
        ub_bernstein(common.broadcast_to_claims(
            corpus, claims, common.per_run_exceedance(corpus, claims, unit_scores, [tau])
        )[tau].values, n, delta=BERN_DELTA)[0]
        for tau in TAUS_FINE
    ])
    print(f"Bootstrapping study-cluster bands (B={N_BOOT}, n={n})...")
    f_lo, f_hi = cluster_bootstrap_band(claims["delta"], claims["v_i"], 1.0, TAUS_FINE)
    ft_lo, ft_hi = cluster_bootstrap_band(claims["delta"], claims["v_i"], c_tight, TAUS_FINE)
    individual = per_study_curves(claims["delta"], claims["v_i"], 1.0, TAUS_FINE)
    individual_tight = per_study_curves(claims["delta"], claims["v_i"], c_tight, TAUS_FINE)

    def plot_freedman(ax, individual_curves, lo, hi, curve, color, clip, azuma):
        for row in individual_curves:
            ax.plot(TAUS_FINE, row, color=color, lw=0.6, alpha=0.35, zorder=1)
        ax.fill_between(TAUS_FINE, lo, hi, color=color, alpha=0.25, lw=0, zorder=2)
        ax.plot(TAUS_FINE, curve, color=color, lw=2.2, zorder=3,
                label=rf"Freedman ($c_i={clip:g}$)")
        ax.plot(TAUS_FINE, empirical, color="grey", lw=1.8, ls=":", zorder=3,
                label=r"$\widehat F_{\mathrm{proxy}}(\tau)$ empirical")
        ax.plot(TAUS_FINE, azuma, color="#C44E52", lw=2.0, ls="--", zorder=3,
                label=rf"Azuma, $c_{{i,m}}={clip:g}$")
        ax.plot(TAUS_FINE, bernstein, color="#7B68EE", lw=2.2, ls="-.", zorder=3,
                label=r"empirical-Bernstein, $1{-}\delta=0.95$")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylim(-0.03, 1.08)

    fig = plt.figure(figsize=(7.5, 4.6))
    grid = GridSpec(2, 2, height_ratios=[1, 0.32], hspace=0.4, wspace=0.3, figure=fig)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1], sharey=None)
    legend_ax = fig.add_subplot(grid[1, :]); legend_ax.axis("off")
    plot_freedman(ax1, individual, f_lo, f_hi, freedman, "#4C72B0", 1.0, ah)
    plot_freedman(ax2, individual_tight, ft_lo, ft_hi, freedman_tight, "#55A868", c_tight, ah_tight)
    ax1.set_title(r"$c_i=1$", fontsize=9); ax2.set_title(rf"$c_i={c_tight:g}$", fontsize=9)
    ax1.set_ylabel("rate"); plt.setp(ax2.get_yticklabels(), visible=False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    legend_ax.legend([h1[0], h2[0], h1[2], h1[1], h1[3]],
                     [l1[0], l2[0], "Azuma", l1[1], l1[3]],
                     fontsize=6.5, loc="center", ncol=3, frameon=True)
    fig.suptitle(f"{label}: aggregated Freedman vs. empirical Bernstein upper bound",
                 fontsize=12, fontweight="bold")
    fig.savefig(outputs["a"], bbox_inches="tight", dpi=200)
    plt.close(fig)
    return empirical, bernstein


def choose_claims(corpus, claims):
    informative = claims
    if corpus in {"cb", "ssrp"}:
        informative = claims[claims["delta"] < TAU_LEVELS_B[0]]
        print(f"Panel B candidate pool: {len(informative)}/{len(claims)} claims with "
              f"delta < {TAU_LEVELS_B[0]:g}")
    lo = (informative["v_i"] - informative["v_i"].quantile(LO_PCT)).abs().idxmin()
    hi = (informative["v_i"] - informative["v_i"].quantile(HI_PCT)).abs().idxmin()
    return lo, hi


def plot_panel_b(corpus, claims, unit_scores, c_tight, lo_claim, hi_claim,
                 empirical, outputs, label):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lo_label = rf"low-variance {'effect' if corpus == 'cb' else 'study'}, $v_i$ p{LO_PCT:.0%} ($\hat v_i={claims.loc[lo_claim, 'v_i']:.4f}$)"
    hi_label = rf"high-variance {'effect' if corpus == 'cb' else 'study'}, $v_i$ p{HI_PCT:.0%} ($\hat v_i={claims.loc[hi_claim, 'v_i']:.4f}$)"
    lo_color, hi_color = "#4C72B0", "#DD8452"

    if corpus == "rpp":
        def study_curve(claim_id, clip):
            row = claims.loc[claim_id]
            return np.array([float(np.asarray(ub_freedman(
                t, row["delta"], row["v_i"], clip
            )).reshape(-1)[0]) for t in TAUS_FINE])
        event = common.broadcast_to_claims(corpus, claims, common.per_run_exceedance(
            corpus, claims, unit_scores, TAUS_FINE))
        lo_emp, hi_emp = event.loc[lo_claim].values, event.loc[hi_claim].values
        fig = plt.figure(figsize=(7.5, 3.6)); grid = GridSpec(2, 2, height_ratios=[1, .3], figure=fig)
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        legend = fig.add_subplot(grid[1, :]); legend.axis("off")
        for ax, clip in zip(axes, [1.0, c_tight]):
            ax.plot(TAUS_FINE, lo_emp, color=lo_color, lw=1.3, ls=":")
            ax.plot(TAUS_FINE, hi_emp, color=hi_color, lw=1.3, ls=":")
            ax.plot(TAUS_FINE, study_curve(lo_claim, clip), color=lo_color, lw=2.2, label=lo_label)
            ax.plot(TAUS_FINE, study_curve(hi_claim, clip), color=hi_color, lw=2.2, label=hi_label)
            ax.set_title(rf"$c_i={clip:g}$", fontsize=9); ax.set_xlabel(r"$\tau$"); ax.set_ylim(-.03, 1.08)
        axes[0].set_ylabel("rate"); h, l = axes[0].get_legend_handles_labels()
        legend.legend(h, l, fontsize=7, loc="center", ncol=2, frameon=True)
        fig.suptitle("Freedman bound for study with low and high variance", fontsize=12, fontweight="bold")
    else:
        c_grid = np.linspace(c_tight, 1.0, N_C_GRID)
        levels = common.broadcast_to_claims(corpus, claims, common.per_run_exceedance(
            corpus, claims, unit_scores, TAU_LEVELS_B))
        fig = plt.figure(figsize=(7.5, 4.0)); grid = GridSpec(2, 3, height_ratios=[1, .28], figure=fig)
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        legend = fig.add_subplot(grid[1, :]); legend.axis("off")
        for ax, tau in zip(axes, TAU_LEVELS_B):
            for claim_id, color, text in [(lo_claim, lo_color, lo_label), (hi_claim, hi_color, hi_label)]:
                row = claims.loc[claim_id]
                ax.plot(c_grid, [float(np.asarray(ub_freedman(
                    tau, row["delta"], row["v_i"], c
                )).reshape(-1)[0]) for c in c_grid],
                        color=color, lw=2.2, label=text)
                ax.axhline(levels.loc[claim_id, tau], color=color, lw=1.3, ls=":")
            ax.set_title(rf"$\tau={tau:g}$", fontsize=9); ax.set_xlabel(r"$c_i$")
            ax.set_xlim(c_grid[0], c_grid[-1]); ax.set_ylim(-.03, 1.08)
        axes[0].set_ylabel("rate"); h, l = axes[0].get_legend_handles_labels()
        legend.legend(h, l, fontsize=7, loc="center", ncol=2, frameon=True)
        suffix = "effect" if corpus == "cb" else "study"
        fig.suptitle(rf"Freedman bound vs. clip $c_i$ at three fixed $\tau$, low- vs. high-variance {suffix}",
                     fontsize=12, fontweight="bold")
    fig.savefig(outputs["b"], bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_panel_c(corpus, claims, chain, lo_claim, hi_claim, outputs):
    import matplotlib.pyplot as plt

    lo_key = lo_claim
    hi_key = hi_claim
    lo, lo_low, lo_high = run_bootstrap_ci(corpus, lo_key, chain)
    hi, hi_low, hi_high = run_bootstrap_ci(corpus, hi_key, chain)
    x = np.arange(len(STEP_LABELS)); width = .35
    lo_err = np.clip(np.vstack([lo - lo_low, lo_high - lo]), 0, None)
    hi_err = np.clip(np.vstack([hi - hi_low, hi_high - hi]), 0, None)
    lo_color, hi_color = "#4C72B0", "#DD8452"
    kind = "effect" if corpus == "cb" else "study"
    lo_label = rf"low-variance {kind}, $v_i$ p{LO_PCT:.0%}"
    hi_label = rf"high-variance {kind}, $v_i$ p{HI_PCT:.0%}"
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - width / 2, lo, width, yerr=lo_err, capsize=3, color=lo_color, label=lo_label)
    ax.bar(x + width / 2, hi, width, yerr=hi_err, capsize=3, color=hi_color, label=hi_label)
    for xi, value, upper in zip(x - width / 2, lo, lo_high):
        ax.text(xi, upper, f"{value:.4f}", ha="center", va="bottom", fontsize=7.5)
    for xi, value, upper in zip(x + width / 2, hi, hi_high):
        ax.text(xi, upper, f"{value:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(STEP_LABELS)
    ax.set_ylim(0, max(lo_high.max(), hi_high.max()) * 1.2)
    ax.set_ylabel(r"mean$_r[D_{i,m}^2]$")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
    fig.suptitle("Sub-output attribution", fontsize=12, fontweight="bold")
    fig.savefig(outputs["c"], bbox_inches="tight", dpi=200)
    plt.close(fig)
    return lo, hi


def run(corpus, effort="high", temp="0.7"):
    suffix = f"{effort}_temp{temp}"
    llm_uq_dir = Path(common.REPO)
    pred_csv = llm_uq_dir / "prediction" / "LLM_Reasoning" / {
        "rpp": "RPP", "cb": "CB", "ssrp": "SSRP"
    }[corpus] / f"text_predictions_{suffix}.csv"
    output_dir = HERE / "results" / {"rpp": "RPP", "cb": "CB", "ssrp": "SSRP"}[corpus]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: output_dir / f"panel_{key}_{suffix}.png" for key in ("a", "b", "c")}
    outputs["notes"] = output_dir / f"notes_{suffix}.txt"

    claims, chain, unit_scores, c2, max_abs_d = build_data(corpus, str(pred_csv))
    label = {"rpp": "RPP", "cb": "CB", "ssrp": "SSRP"}[corpus]
    print(f"n = {len(claims)} claims ({label}, {effort} effort, temp={temp})")
    c_tight = C_TIGHT_RPP if corpus == "rpp" else data_driven_c_tight(max_abs_d)
    assert c_tight >= max_abs_d, f"C_TIGHT={c_tight} < observed max|D|={max_abs_d:.4f}"
    empirical, _ = plot_panel_a(corpus, claims, unit_scores, c2, max_abs_d, c_tight, label, outputs)
    lo_claim, hi_claim = choose_claims(corpus, claims)
    print(f"Panel B: low id={lo_claim}, high id={hi_claim}")
    plot_panel_b(corpus, claims, unit_scores, c_tight, lo_claim, hi_claim,
                 empirical, outputs, label)
    lo, hi = plot_panel_c(corpus, claims, chain, lo_claim, hi_claim, outputs)
    with outputs["notes"].open("w") as handle:
        handle.write(
            f"Panels B and C select claims closest to the {LO_PCT:.0%}/{HI_PCT:.0%} "
            f"v_i percentiles for {label} ({effort}, temp={temp}).\n"
            f"Low id={lo_claim}, v_i={claims.loc[lo_claim, 'v_i']:.6f}, "
            f"delta={claims.loc[lo_claim, 'delta']:.6f}\n"
            f"High id={hi_claim}, v_i={claims.loc[hi_claim, 'v_i']:.6f}, "
            f"delta={claims.loc[hi_claim, 'delta']:.6f}\n"
        )
    print(f"Wrote {outputs['a']}, {outputs['b']}, {outputs['c']}, and {outputs['notes']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["rpp", "cb", "ssrp", "all"], default="rpp")
    parser.add_argument("--effort", choices=["high", "low", "all"], default="all")
    parser.add_argument("--temp", choices=["0.7", "0.2", "all"], default="all")
    args = parser.parse_args()
    corpora = ["rpp", "cb", "ssrp"] if args.corpus == "all" else [args.corpus]
    efforts = ["high", "low"] if args.effort == "all" else [args.effort]
    temps = ["0.7", "0.2"] if args.temp == "all" else [args.temp]
    for corpus in corpora:
        for effort in efforts:
            for temp in temps:
                run(corpus, effort, temp)


if __name__ == "__main__":
    main()
