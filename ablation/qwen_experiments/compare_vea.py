"""VEA decomposition and Freedman / empirical-Bernstein bounds on RPP: Qwen3.8-Flash vs.
gpt-5.4-mini, matched on temperature (0.7), reasoning effort (high; see RESULTS.md for the
Qwen label) and run ids (0..runs-1).

    python ablation/qwen_experiments/compare_vea.py [--set models|qwen_arms|qwen_spurious] [--runs 10]

Reuses ablation/action_intervention/make_vea_bound_figures (compute_vea reproduces the
published RPP VEA exactly) and compare_vea_bound (bound curves, figure style).
Writes to results/figures/, with STEM = gpt-5.4-mini_vs_qwen3.8-flash_<runs>runs:
  compare_vea_STEM.{png,csv}        VEA only (V-hat, E-hat)
  compare_vea_bound_STEM.png        VEA (top) + bounds (bottom)
  compare_bound_curves_STEM.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "action_intervention"))
import compare_vea_bound as cvb  # noqa: E402
import make_vea_bound_figures as mv  # noqa: E402

vp = mv.vp
QWEN = HERE / "results/Qwen3.8-Flash"
SETS = {  # --set -> (name -> (predictions CSV, colour)), file stem, title
    "models": ({"gpt-5.4-mini": (mv.BASELINE_CSV, "#2a78d6"),
                "Qwen3.8-Flash": (QWEN / "RPP/text_predictions_high_temp0.7.csv", "#eb6834")},
               "gpt-5.4-mini_vs_qwen3.8-flash", "RPP: gpt-5.4-mini vs. Qwen3.8-Flash"),
    "qwen_arms": ({"Full extraction (N, r, p)": (QWEN / "RPP/text_predictions_high_temp0.7.csv", "#2a78d6"),
                   "N only": (HERE / "results/archive/Qwen3.8-Flash/n_only/text_predictions_high_temp0.7.csv", "#eb6834")},
                  "qwen3.8-flash_full_vs_n_only", "RPP, Qwen3.8-Flash: full extraction vs. N only"),
    "qwen_spurious": ({"Baseline": (QWEN / "RPP/text_predictions_high_temp0.7.csv", "#2a78d6"),
                        "Spurious stats (Step 3)": (QWEN / "spurious_stats/text_predictions_high_temp0.7.csv", "#1baf7a")},
                       "qwen3.8-flash_baseline_vs_spurious_stats", "RPP, Qwen3.8-Flash: baseline vs. spurious stats"),
}
MODELS = {}  # filled from --set in main()
CHAIN = [1, 3, 4, 5, 6]
# Freedman's c must bound every increment |D|. The repo fixes c = 0.4 for RPP, but Qwen N-only has
# one pooled-OLS increment of 0.40003. For that set, c follows the repo's rule for CB/SSRP
# (validation_plot.data_driven_c_tight: max|D| rounded up to a 0.05 grid), taken over both arms.
DATA_DRIVEN_C = {"qwen_arms"}


def plot_vea(axs, vea):
    units = cvb.UNITS
    x, w = np.arange(len(units)), 0.8 / len(MODELS)
    for ax, col, title in [(axs[0], "V", r"Variance $\hat V$"),
                           (axs[1], "E", r"Epistemic $\hat E$ (corrected)")]:
        for k, (name, (_, color)) in enumerate(MODELS.items()):
            vals = [vea[name][col].get(u, np.nan) for u in units]
            ax.bar(x + (k - (len(MODELS) - 1) / 2) * w, vals, w * 0.92, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([cvb.TICKS[u] for u in units], fontsize=8, color=cvb.INK)
        ax.set_title(title, fontsize=10, color=cvb.INK, loc="left")
        cvb.style(ax)
    axs[0].set_ylabel("mean error component", fontsize=9, color=cvb.INK)
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")


def plot_bounds(axs, curves):
    for ax, key, title in [(axs[0], "freedman_c1", r"Freedman bound, $c_i=1$"),
                           (axs[1], f"freedman_c{vp.C_TIGHT_RPP:g}", rf"Freedman bound, $c_i={vp.C_TIGHT_RPP:g}$")]:
        for name, (_, color) in MODELS.items():
            c = curves[name]
            ax.plot(c["tau"], c[key], color=color, lw=2.2)
            ax.plot(c["tau"], c["bernstein"], color=color, lw=1.6, ls="--")
            ax.plot(c["tau"], c["empirical"], color=color, lw=1.6, ls=":")
        ax.set_xlabel(r"$\tau$", color=cvb.INK)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title(title, fontsize=10, color=cvb.INK, loc="left")
        cvb.style(ax)
    axs[0].set_ylabel("rate", fontsize=9, color=cvb.INK)
    keys = [Line2D([], [], color=cvb.MUTE, lw=2.2), Line2D([], [], color=cvb.MUTE, lw=1.6, ls="--"),
            Line2D([], [], color=cvb.MUTE, lw=1.6, ls=":")]
    axs[1].legend(keys, ["Freedman", "empirical-Bernstein (95%)", "observed exceedance"],
                  fontsize=8, frameon=False, loc="lower left")


def finish(fig, title, path):
    fig.suptitle(title, fontsize=12, fontweight="bold", color=cvb.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=cvb.SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=SETS, default="models",
                        help="models: gpt-5.4-mini vs Qwen (baseline prompt); "
                             "qwen_arms: Qwen baseline prompt vs N-only prompt")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "figures")
    args = parser.parse_args()
    models, set_stem, set_title = SETS[args.set]
    MODELS.update(models)

    vea, inputs, curves = {}, {}, {}
    repo_c = vp.C_TIGHT_RPP
    vp.C_TIGHT_RPP = 1.0  # first pass only needs max|D|; c is fixed below
    for name, (csv, _) in MODELS.items():
        pred = pd.read_csv(csv)
        pred = pred[pred["run_id"] < args.runs]
        assert pred.groupby("altmejd_id")["run_id"].nunique().eq(args.runs).all(), f"{name}: missing runs"
        print(f"{name}: {len(pred)} rows, {pred['altmejd_id'].nunique()} claims")
        vea[name] = mv.compute_vea(pred, CHAIN).set_index("unit")
        claims, us, _c2, max_d = mv.compute_bound_inputs(pred, CHAIN)
        inputs[name] = (claims, us, max_d)
        print(f"  max|D| = {max_d:.5f}")
    max_d = max(v[2] for v in inputs.values())
    vp.C_TIGHT_RPP = vp.data_driven_c_tight(max_d) if args.set in DATA_DRIVEN_C else repo_c
    assert vp.C_TIGHT_RPP >= max_d, f"c={vp.C_TIGHT_RPP} < max|D|={max_d}"
    print(f"Freedman c = {vp.C_TIGHT_RPP:g}")
    for name, (claims, us, _) in inputs.items():
        curves[name] = cvb.bound_curves(claims, us)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{set_stem}_{args.runs}runs"
    title = f"{set_title} (high effort, T=0.7, R={args.runs})"

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=cvb.SURFACE)
    plot_vea(axs, vea)
    finish(fig, title, args.out_dir / f"compare_vea_{stem}.png")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=cvb.SURFACE)
    plot_vea(axs[0], vea)
    plot_bounds(axs[1], curves)
    finish(fig, title, args.out_dir / f"compare_vea_bound_{stem}.png")

    table = pd.concat(vea, names=["model"])
    table.to_csv(args.out_dir / f"compare_vea_{stem}.csv")
    cols = ["tau", "empirical", "freedman_c1", f"freedman_c{vp.C_TIGHT_RPP:g}", "bernstein"]
    pd.concat([pd.DataFrame({"model": name, **{k: curves[name][k] for k in cols}}) for name in MODELS]
              ).to_csv(args.out_dir / f"compare_bound_curves_{stem}.csv", index=False)
    print(table[["n", "V", "E_raw", "E", "A"]].to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
