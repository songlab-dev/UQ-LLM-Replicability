"""VEA figure from a saved compare_vea_*.csv: V-hat in blues, E-hat in reds.
The first group in the CSV (reference arm/model) is the light shade, the second the dark shade.
Optional: --theta adds a stage-attribution panel (1x3); --bounds adds a row of Freedman bound panels (2x2),
with the reference in amber and the comparison in purple.

    python ablation/action_intervention/plot_vea_panels.py CSV --labels "Baseline" "Spurious stats (Step 3)" \
        [--title "..."] [--sharey] [--theta THETA_CSV | --bounds BOUND_CURVES_CSV] --out OUT.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTE, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES, REDS = ["#8fbcef", "#1f5fae"], ["#f2a196", "#b8322a"]  # light = reference, dark = comparison
BOUND_COLORS = ["#d98a0b", "#5b3a9e"]  # bound panels (--bounds): amber = reference, purple = comparison
GREENS = ["#5fb98a", "#16653c"]  # theta panel (--theta); markers: circle = reference, square = comparison
MARKERS = ["o", "s"]
FS = 1.0  # font scale; 1.15 for the --bounds (2x2) layout
STAGES = ["phi1", "phi2", "phi3", "phi4", "phi_res"]
STAGE_TICKS = [r"$\theta_1$" "\nExtraction", r"$\theta_2$" "\nStatistics", r"$\theta_3$" "\nResearcher\nDoF",
               r"$\theta_4$" "\nTheory/\ncontext", r"$\theta_{\mathrm{res}}$" "\nResidual"]
UNITS = [1, 3, 4, 5, 6, 7]
TICKS = {1: "Extraction", 3: "Statistics", 4: "Researcher\nDoF", 5: "Theory/\ncontext", 6: "Scoring", 7: "Verdict"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=10 * FS, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path)
    p.add_argument("--labels", nargs=2, default=None, help="legend names (default: group names in the CSV)")
    p.add_argument("--title", default=None, help="figure title (default: none)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sharey", action="store_true", help="same y-axis scale for the V and E panels")
    p.add_argument("--theta", type=Path, default=None,
                   help="compare_theta_*.csv: adds a third panel with theta levels and 95%% intervals (1x3 layout)")
    p.add_argument("--bounds", type=Path, default=None,
                   help="compare_bound_curves_*.csv: adds a second row with Freedman bound panels (2x2 layout)")
    p.add_argument("--split", action="store_true",
                   help="with --bounds: also save the two rows as 1x2 PNGs (_vea, _bound) and each panel alone (_V, _E, _bound_c1, _bound_c0.4); no panel subtitles")
    p.add_argument("--font-scale", type=float, default=None,
                   help="multiply all font sizes (default: 1.0, or 1.15 with --bounds)")
    a = p.parse_args()
    assert not (a.theta and a.bounds), "use either --theta or --bounds"
    global FS
    FS = a.font_scale or (1.15 if a.bounds else 1.0)
    df = pd.read_csv(a.csv)
    gcol = df.columns[0]
    groups = list(dict.fromkeys(df[gcol]))
    assert len(groups) == 2, groups
    labels = a.labels or groups
    x, w = np.arange(len(UNITS)), 0.4
    ncol = 3 if a.theta else 2
    if a.bounds:
        fig, grid = plt.subplots(2, 2, figsize=(12.5, 8.4), facecolor=SURFACE)
        axs, baxs = grid[0], grid[1]
    else:
        fig, axs = plt.subplots(1, ncol, figsize=((5.5 + 4 * (FS - 1)) * ncol, 4.8), facecolor=SURFACE)
    if a.sharey:
        axs[1].sharey(axs[0])
    for ax, col, shades, title in [(axs[0], "V", BLUES, r"Variance $\hat V$"),
                                   (axs[1], "E", REDS, r"Epistemic $\hat E$ (corrected)")]:
        for k, (g, lab) in enumerate(zip(groups, labels)):
            s = df[df[gcol] == g].set_index("unit")[col]
            ax.bar(x + (k - 0.5) * w, [s.get(u, np.nan) for u in UNITS], w * 0.92, color=shades[k], label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([TICKS[u] for u in UNITS], fontsize=9.5 - 5 * (FS - 1), color=INK)  # shrinks slightly as FS grows: 6 one-word labels per panel collide otherwise
        ax.set_title(title, fontsize=12 * FS, color=INK, loc="left", pad=24)
        ax.legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1, handlelength=1.2, columnspacing=1.2)
        style(ax)
    axs[0].set_ylabel("mean error component", fontsize=11 * FS, color=INK)
    if a.theta:
        th = pd.read_csv(a.theta)
        th = th[th["kind"] == "level"]
        arms = list(dict.fromkeys(th["arm"]))
        assert len(arms) == 2, arms
        # point + 95% interval, as in the main-text stage-attribution figure:
        # filled marker = interval excludes 0, hollow = interval includes 0
        from matplotlib.lines import Line2D
        ax, xs, off = axs[2], np.arange(len(STAGES)), 0.16
        ax.axhline(0, color=INK, lw=0.9, zorder=1)
        for k, (arm, lab) in enumerate(zip(arms, labels)):
            t = th[th["arm"] == arm].set_index("stage").loc[STAGES, ["theta", "ci_lo", "ci_hi"]] * 1e3
            pos = xs + (k - 0.5) * 2 * off
            ax.errorbar(pos, t["theta"], yerr=np.vstack([t["theta"] - t["ci_lo"], t["ci_hi"] - t["theta"]]),
                        fmt="none", ecolor=GREENS[k], elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
            excl = ((t["ci_lo"] > 0) | (t["ci_hi"] < 0)).values
            for xi, yi, e in zip(pos, t["theta"], excl):
                ax.plot(xi, yi, MARKERS[k], ms=8, mew=1.8, mec=GREENS[k], mfc=GREENS[k] if e else SURFACE, zorder=3)
            ax.plot([], [], MARKERS[k], ms=8, color=GREENS[k], label=lab)
        key = [Line2D([], [], marker="o", ls="", ms=8, color=MUTE, label="95% CI excludes 0"),
               Line2D([], [], marker="o", ls="", ms=8, mew=1.8, mec=MUTE, mfc=SURFACE, label="95% CI includes 0")]
        ax.add_artist(ax.legend(handles=key, fontsize=9 * FS, frameon=False, loc="upper left"))
        ax.set_xticks(xs)
        ax.set_xticklabels(STAGE_TICKS, fontsize=9 - 5 * (FS - 1), color=INK)
        ax.set_title(r"Stage attribution $\hat\theta$ ($\times10^{-3}$, 95% CI)", fontsize=12 * FS, color=INK, loc="left", pad=24)
        ax.legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1,
                  handlelength=1.2, columnspacing=1.2)
        style(ax)
        ax.set_xlim(-0.5, len(STAGES) - 0.5)
    if a.bounds:
        from matplotlib.lines import Line2D
        bc = pd.read_csv(a.bounds)
        bcol = bc.columns[0]
        bgroups = list(dict.fromkeys(bc[bcol]))
        assert bgroups == groups, (bgroups, groups)
        fkeys = [c for c in bc.columns if c.startswith("freedman_c")]
        fkeys.sort(key=lambda c: -float(c[len("freedman_c"):]))  # c=1 first, then the tight c
        for ax, key in zip(baxs, fkeys):
            for k, (g, lab) in enumerate(zip(groups, labels)):
                c = bc[bc[bcol] == g]
                ax.plot(c["tau"], c[key], color=BOUND_COLORS[k], lw=2.2, label=lab)
                ax.plot(c["tau"], c["bernstein"], color=BOUND_COLORS[k], lw=1.6, ls="--")
                ax.plot(c["tau"], c["empirical"], color=BOUND_COLORS[k], lw=1.6, ls=":")
            ax.set_xlabel(r"$\tau$", fontsize=11 * FS, color=INK)
            ax.set_ylim(-0.03, 1.08)
            ax.set_title(rf"Freedman bound, $c_i={float(key[len('freedman_c'):]):g}$", fontsize=12 * FS, color=INK,
                         loc="left", pad=24)
            ax.legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                      borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
            style(ax)
        baxs[0].set_ylabel("rate", fontsize=11 * FS, color=INK)
        keys = [Line2D([], [], color=MUTE, lw=2.2), Line2D([], [], color=MUTE, lw=1.6, ls="--"),
                Line2D([], [], color=MUTE, lw=1.6, ls=":")]
        baxs[1].add_artist(baxs[1].legend(keys, ["Freedman", "empirical-Bernstein (95%)", "observed exceedance"],
                                          fontsize=10 * FS, frameon=False, loc="lower left"))
        baxs[1].legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                       borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)  # restore model legend
    if a.title:
        fig.suptitle(a.title, fontsize=12 * FS, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if a.title else 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print(f"Wrote {a.out}")
    if a.split:  # crop the --bounds layout into two 1x2 figures (rows) and four single panels
        assert a.bounds, "--split needs --bounds"
        from matplotlib.transforms import Bbox
        r = fig.canvas.get_renderer()
        c_names = [f"bound_c{float(k[len('freedman_c'):]):g}" for k in fkeys]
        jobs = [("vea", list(axs)), ("bound", list(baxs)), ("V", [axs[0]]), ("E", [axs[1]]),
                (c_names[0], [baxs[0]]), (c_names[1], [baxs[1]])]
        for nm, row in jobs:
            others = [o for o in fig.axes if o not in row]
            for o in others:
                o.set_visible(False)
            saved, extra = [(ax.get_title(loc="left"), ax.get_ylabel()) for ax in row], None
            for ax in row:  # split outputs carry no panel subtitles
                ax.set_title("", loc="left")
            if len(row) == 1:  # single panel: own y-label, and the line-style key on the c=1 panel
                ax = row[0]
                if not ax.get_ylabel():
                    ax.set_ylabel("rate" if ax in baxs else "mean error component", fontsize=11 * FS, color=INK)
                if ax is baxs[0]:
                    extra = ax.legend(keys, ["Freedman", "empirical-Bernstein (95%)", "observed exceedance"],
                                      fontsize=10 * FS, frameon=False, loc="lower left")
                    ax.add_artist(extra)
                    ax.legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                              borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
            bb = Bbox.union([ax.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted()) for ax in row])
            bb = Bbox.from_extents(bb.x0 - 0.08, bb.y0 - 0.08, bb.x1 + 0.08, bb.y1 + 0.08)
            out = a.out.with_name(f"{a.out.stem}_{nm}.png")
            fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches=bb)
            print(f"Wrote {out}")
            for ax, (t, yl) in zip(row, saved):
                ax.set_title(t, loc="left")
                ax.set_ylabel(yl)
            if extra is not None:
                extra.remove()
                baxs[0].legend(fontsize=10 * FS, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                               borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
            for o in others:
                o.set_visible(True)

if __name__ == "__main__":
    main()
