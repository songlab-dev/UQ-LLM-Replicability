"""One figure comparing Step-1 extraction variants on matched runs: VEA components (top) and
Freedman / empirical-Bernstein bounds against the observed exceedance rate (bottom).

    python ablation/action_intervention/compare_vea_bound.py --runs 10 [--arms baseline n_only claim_only]

Reuses make_vea_bound_figures.py (itself a thin layer over the repo's VEA, chain
and bound code). Writes results/figures/compare_vea_bound_<runs>runs.{png,csv}.

With exactly two arms, it also writes the 1x3 paper figure
vea_panels_<tag>_<runs>runs.png (V-hat | E-hat | stage attribution theta-hat) via
plot_vea_panels.py, provided compare_theta.py has already written
compare_theta_<tag>_<runs>runs.csv to the same directory; otherwise it prints the
command to run first. Paper figure (baseline vs. spurious stats):

    python ablation/action_intervention/compare_theta.py --runs 100 --arms baseline spurious_stats --panels levels
    python ablation/action_intervention/compare_vea_bound.py --runs 100 --arms baseline spurious_stats
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_vea_bound_figures as mv
from ah_bound import ub_ah  # noqa: F401  (imported via mv's sys.path setup)
from bernstein_bound import ub_bernstein
from freedman_bound import ub_freedman

common, vp = mv.common, mv.vp
ORDER = ["baseline", "spurious_stats", "n_only", "n_only_nohint", "claim_only"]
NAMES = {"baseline": "Baseline", "spurious_stats": "Spurious stats (Step 3)", "n_only": "N only", "n_only_nohint": "N only, no hints",
         "claim_only": "Claim only"}
SHORT = {"baseline": "baseline", "spurious_stats": "spurious stats", "n_only": "N only", "n_only_nohint": "N only (no hints)",
         "claim_only": "claim only"}  # title
TAGS = {"baseline": "baseline", "spurious_stats": "spurious_stats", "n_only": "n_only", "n_only_nohint": "n_only_nohint",
        "claim_only": "claim_only"}  # file name
COLORS = {"baseline": "#2a78d6", "spurious_stats": "#1baf7a", "n_only": "#eb6834", "n_only_nohint": "#8e5bd9",
          "claim_only": "#1baf7a"}
INK, MUTE, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
UNITS = [1, 3, 4, 5, 6, 7]
TICKS = {1: "Extraction", 3: "Statistics", 4: "Researcher\nDoF", 5: "Theory/\ncontext", 6: "Scoring", 7: "Verdict"}


def bound_curves(claims, us):
    taus = vp.TAUS_FINE
    n = len(claims)
    emp = common.broadcast_to_claims("rpp", claims, common.per_run_exceedance("rpp", claims, us, taus))
    out = {"tau": taus, "empirical": emp.mean(axis=0).values}
    for c in (1.0, vp.C_TIGHT_RPP):
        out[f"freedman_c{c:g}"] = np.array([ub_freedman(t, claims["delta"], claims["v_i"], c=c).mean() for t in taus])
    out["bernstein"] = np.array([
        ub_bernstein(common.broadcast_to_claims(
            "rpp", claims, common.per_run_exceedance("rpp", claims, us, [t]))[t].values,
            n, delta=vp.BERN_DELTA)[0] for t in taus])
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=["baseline", "n_only"], choices=ORDER,
                        help="arms to compare, in plotting order (default: baseline and n_only)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=mv.HERE / "results" / "figures")
    args = parser.parse_args()
    arms = args.arms
    tag = "_vs_".join(TAGS[x] for x in arms)

    vea, inputs, curves = {}, {}, {}
    repo_c = vp.C_TIGHT_RPP
    vp.C_TIGHT_RPP = 1.0  # first pass only needs max|D|; the clip c is fixed below
    max_all = 0.0
    for arm in arms:
        pred, _name, chain = mv.load_arm(arm, args.runs)
        vea[arm] = mv.compute_vea(pred, chain).set_index("unit")
        claims, us, _c2, max_d = mv.compute_bound_inputs(pred, chain)
        inputs[arm] = (claims, us)
        print(f"  max|D| = {max_d:.5f}")
        max_all = max(max_all, max_d)
    # Freedman's c must bound every increment. Keep the repo's RPP value (0.4) when it does;
    # otherwise use the repo's data-driven rule (max|D| rounded up to a 0.05 grid), shared by all arms.
    vp.C_TIGHT_RPP = repo_c if max_all <= repo_c else vp.data_driven_c_tight(max_all)
    print(f"Freedman c = {vp.C_TIGHT_RPP:g}")
    for arm, (claims, us) in inputs.items():
        curves[arm] = bound_curves(claims, us)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE)
    # ── VEA: variance and corrected epistemic error, grouped by unit ─────────
    x = np.arange(len(UNITS))
    w = 0.8 / len(arms)
    for ax, col, title in [(axs[0, 0], "V", r"Variance $\hat V$"),
                           (axs[0, 1], "E", r"Epistemic $\hat E$ (corrected)")]:
        for k, arm in enumerate(arms):
            vals = [vea[arm][col].get(u, np.nan) for u in UNITS]
            ax.bar(x + (k - (len(arms) - 1) / 2) * w, vals, w * 0.92, color=COLORS[arm], label=NAMES[arm])
        ax.set_xticks(x)
        ax.set_xticklabels([TICKS[u] for u in UNITS], fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axs[0, 0].set_ylabel("mean error component", fontsize=9, color=INK)
    # ── Bounds ───────────────────────────────────────────────────────────────
    for ax, key, title in [(axs[1, 0], "freedman_c1", r"Freedman bound, $c_i=1$"),
                           (axs[1, 1], f"freedman_c{vp.C_TIGHT_RPP:g}", rf"Freedman bound, $c_i={vp.C_TIGHT_RPP:g}$")]:
        for arm in arms:
            c = curves[arm]
            ax.plot(c["tau"], c[key], color=COLORS[arm], lw=2.2)
            ax.plot(c["tau"], c["bernstein"], color=COLORS[arm], lw=1.6, ls="--")
            ax.plot(c["tau"], c["empirical"], color=COLORS[arm], lw=1.6, ls=":")
        ax.set_xlabel(r"$\tau$", color=INK)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axs[1, 0].set_ylabel("rate", fontsize=9, color=INK)
    # ── Legends ──────────────────────────────────────────────────────────────
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(h, l, fontsize=8, frameon=False, loc="upper left")
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=MUTE, lw=2.2), Line2D([], [], color=MUTE, lw=1.6, ls="--"),
            Line2D([], [], color=MUTE, lw=1.6, ls=":")]
    axs[1, 1].legend(keys, ["Freedman", "empirical-Bernstein (95%)", "observed exceedance"],
                     fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"RPP: {' vs. '.join(SHORT[a] for a in arms)} (R={args.runs})",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"compare_vea_bound_{tag}_{args.runs}runs.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    rows = [{"arm": arm, **{k: v for k, v in zip(["tau", "empirical", "freedman_c1", f"freedman_c{vp.C_TIGHT_RPP:g}",
                                                    "bernstein"],
                                                   [curves[arm][c] for c in ["tau", "empirical", "freedman_c1",
                                                                              f"freedman_c{vp.C_TIGHT_RPP:g}",
                                                                              "bernstein"]])}}
            for arm in arms]
    pd.concat([pd.DataFrame(r) for r in rows]).to_csv(args.out_dir / f"compare_bound_curves_{tag}_{args.runs}runs.csv", index=False)
    vea_csv = args.out_dir / f"compare_vea_{tag}_{args.runs}runs.csv"
    pd.concat({a: vea[a] for a in arms}, names=["arm"]).to_csv(vea_csv)
    print(f"Wrote {out}")
    if len(arms) == 2:
        write_panels(vea_csv, arms, tag, args.runs, args.out_dir)


def write_panels(vea_csv, arms, tag, runs, out_dir):
    """1x3 paper figure (V-hat, E-hat, theta-hat) from the CSVs just written."""
    theta_csv = out_dir / f"compare_theta_{tag}_{runs}runs.csv"
    if not theta_csv.exists():
        print(f"Skipped vea_panels: {theta_csv.name} not found -- run compare_theta.py "
              f"for the same arms and --runs {runs}, then rerun this script.")
        return
    subprocess.run([sys.executable, str(mv.HERE / "plot_vea_panels.py"), str(vea_csv),
                    "--labels", NAMES[arms[0]], NAMES[arms[1]],
                    "--theta", str(theta_csv), "--font-scale", "1.1",
                    "--out", str(out_dir / f"vea_panels_{tag}_{runs}runs.png")], check=True)


if __name__ == "__main__":
    main()
