"""Cross-fitted stage attribution (theta_1..theta_4, theta_res) for the extraction-ablation arms on RPP,
plus paired contrasts against the baseline, in one figure.

    python ablation/action_intervention/compare_theta.py --runs 10 [--set gpt_arms|models|qwen_arms|qwen_spurious] [--panels levels]
    python ablation/action_intervention/compare_theta.py --validate     # baseline, all 100 runs vs. published

Reuses uq_bound/dml/dml_theta.py's step functions unchanged (fold assignment, cross-fitted prefix
maps, telescoping check, finite-R correction, cluster-robust SE, multiplier bootstrap). It never
calls run_corpus(), which would overwrite uq_bound/dml/results/. Paired contrasts apply the same
SE and multiplier bootstrap to the per-claim difference of the corrected phi-bars.

Sets: gpt_arms (gpt-5.4-mini: baseline, spurious stats (spurious_stats), and the archived N-only arms; pick with --arms; writes to action_intervention/results/figures),
models (gpt-5.4-mini baseline vs. Qwen3.8-Flash baseline) and qwen_arms (Qwen baseline vs. N-only), both written to
qwen_experiments/results/figures. Files: compare_theta_<arms>_<runs>runs[_levels].{png,csv}.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_vea_bound_figures as mv  # sets sys.path for uq_bound/ and figures/
import compare_vea_bound as cvb      # palette, names, style

sys.path.insert(0, str(mv.PKG / "uq_bound" / "dml"))
import dml_theta as dml  # noqa: E402

common, scoring = mv.common, mv.scoring
PHI = dml.PHI_COLS
STAGES = ["Extraction", "Statistics", "Researcher\nDoF", "Theory/\ncontext", "Residual"]
QWEN = mv.PKG / "ablation" / "qwen_experiments" / "results" / "Qwen3.8-Flash"
GPT_FIG = mv.HERE / "results" / "figures"
QWEN_FIG = mv.PKG / "ablation" / "qwen_experiments" / "results" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _arm(arm):  # an extraction-ablation arm of gpt-5.4-mini, styled as in compare_vea_bound
    return dict(csv=mv.ARMS[arm][0], color=cvb.COLORS[arm], name=cvb.NAMES[arm], short=cvb.SHORT[arm], tag=cvb.TAGS[arm])


# set -> (out dir, arm -> csv/colour/legend name/title name/file tag); the first arm is the reference
SETS = {
    "gpt_arms": (GPT_FIG, {a: _arm(a) for a in ["baseline", "spurious_stats", "n_only", "n_only_nohint"]}),
    "models": (QWEN_FIG, {
        "baseline": dict(csv=mv.BASELINE_CSV, color=BLUE, name="gpt-5.4-mini", short="gpt-5.4-mini", tag="gpt-5.4-mini"),
        "qwen": dict(csv=QWEN / "RPP/text_predictions_high_temp0.7.csv", color=ORANGE, name="Qwen3.8-Flash",
                     short="Qwen3.8-Flash", tag="qwen3.8-flash")}),
    "qwen_arms": (QWEN_FIG, {
        "baseline": dict(csv=QWEN / "RPP/text_predictions_high_temp0.7.csv", color=BLUE, name="Full extraction (N, r, p)",
                         short="full extraction", tag="qwen3.8-flash_full"),
        "n_only": dict(csv=QWEN.parent / "archive/Qwen3.8-Flash/n_only/text_predictions_high_temp0.7.csv", color=ORANGE, name="N only",
                       short="N only", tag="n_only")}),
    "qwen_spurious": (QWEN_FIG, {
        "baseline": dict(csv=QWEN / "RPP/text_predictions_high_temp0.7.csv", color=BLUE, name="Baseline",
                         short="Qwen3.8-Flash baseline", tag="qwen3.8-flash_baseline"),
        "spurious_stats": dict(csv=QWEN / "spurious_stats/text_predictions_high_temp0.7.csv", color="#1baf7a",
                                name="Spurious stats (Step 3)", short="spurious stats", tag="spurious_stats")}),
}
PUBLISHED = mv.PKG / "uq_bound" / "dml" / "results" / "RPP" / "dml_theta.csv"
MARGIN = 0.2e-3  # PLAN.md: negligible if the CI for the theta_2 contrast lies within +/-0.2e-3


def load_obs(csv, runs):
    """dml_theta.load_observations_rpp, but for an arm's predictions CSV and the first `runs` runs."""
    scope = common.load_rpp_scope()
    rpp = pd.read_csv(common.RPP_CSV, encoding="latin1")
    pred = pd.read_csv(csv)
    if runs:
        pred = pred[pred["run_id"] < runs]
    pred = pred[pred["Study Title (O)"].isin(scope["Study Title (O)"])].reset_index(drop=True)
    us = scoring.compute_unit_scores(pred, rpp)
    us["claim_id"] = pred["altmejd_id"].values  # positional: same row order
    us["cluster_id"] = us["Study Title (O)"]
    us["run_id"] = pred["run_id"].values
    return us[["claim_id", "cluster_id", "run_id"] + dml.PFULL].dropna().reset_index(drop=True)


def corrected(phibar):
    """Claim-level phi-bars with the finite-R residual correction applied claim by claim."""
    c = phibar.copy()
    c["phi_res"] = c["phi_res"] * c["r_train_mean"] / (c["r_train_mean"] + 1.0)
    return c


def fit_arm(csv, runs, keep=None):
    obs = load_obs(csv, runs)
    counts = obs.groupby("claim_id").size()
    obs = obs[~obs["claim_id"].isin(counts[counts < dml.MIN_RUNS_PER_CLAIM].index)]
    if keep is not None:
        obs = obs[obs["claim_id"].isin(keep)]
    obs = obs.reset_index(drop=True)
    obs["study_fold"] = dml.assign_study_folds(obs)
    obs["run_fold"] = dml.assign_run_folds(obs)
    obs = dml.fit_score_chain(obs)
    dml.check_telescoping(obs, verbose=False)
    phibar = dml.per_claim_phibar(obs)
    theta, theta_raw, _corr, r_train, n = dml.aggregate_theta(phibar)
    se, G, _ = dml.cluster_robust_se(phibar, theta)
    _q, _band, boot = dml.multiplier_bootstrap(phibar, theta, se)
    target, total, gap = dml.adding_up_check(obs, theta, verbose=False)
    return dict(obs=obs, phibar=phibar, theta=theta, se=se, boot=boot, n=n, G=G, r_train=r_train, gap=gap)


def paired(a, b):
    """theta_b - theta_a with cluster-robust SE and multiplier-bootstrap interval, on shared claims."""
    ca = corrected(a["phibar"]).set_index("claim_id")
    cb = corrected(b["phibar"]).set_index("claim_id")
    ids = ca.index.intersection(cb.index)
    diff = (cb.loc[ids, PHI] - ca.loc[ids, PHI]).reset_index()
    diff["cluster_id"] = ca.loc[ids, "cluster_id"].values
    diff["r_train_mean"] = 1e9  # correction already applied; keeps the SE code path identical
    theta = diff[PHI].mean()
    se, _G, _ = dml.cluster_robust_se(diff, theta)
    _q, _band, boot = dml.multiplier_bootstrap(diff, theta, se)
    return theta, se, boot


def validate(runs):
    """Baseline with all runs must reproduce the published RPP table."""
    r = fit_arm(mv.BASELINE_CSV, runs)
    pub = pd.read_csv(PUBLISHED).set_index("phi_col")["theta_hat"]
    ours = r["theta"]
    print("validation vs. uq_bound/dml/results/RPP/dml_theta.csv (theta x 1e-3):")
    for c in PHI:
        print(f"  {c:8s} ours {ours[c] * 1e3:9.4f}  published {pub[c] * 1e3:9.4f}  diff {(ours[c] - pub[c]) * 1e3:+.5f}")
    assert np.allclose(ours[PHI].values, pub[PHI].values, atol=1e-9), "does not reproduce the published table"
    print("  reproduces the published table")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=SETS, default="gpt_arms", dest="set_name")
    parser.add_argument("--arms", nargs="+", default=None, help="subset of the set's arms (default: all, in order)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--panels", choices=["both", "levels"], default="both",
                        help="both: levels and paired change vs. baseline; levels: the left panel only")
    parser.add_argument("--validate", action="store_true", help="check the pipeline against the published baseline")
    parser.add_argument("--out-dir", type=Path, default=None, help="default depends on --set")
    args = parser.parse_args()
    dml.check_spec_hash()
    if args.validate:
        return validate(None)
    out_dir, spec = SETS[args.set_name]
    args.out_dir = args.out_dir or out_dir
    arms = args.arms or list(spec)
    assert all(a in spec for a in arms), f"arms must come from {list(spec)}"
    assert arms[0] == "baseline", "contrasts are taken against the baseline, which must come first"

    # Same claim set in every arm, so the contrasts are paired on identical claims.
    keep = None
    for arm in arms:
        ids = set(load_obs(spec[arm]["csv"], args.runs).groupby("claim_id").filter(
            lambda g: len(g) >= dml.MIN_RUNS_PER_CLAIM)["claim_id"])
        keep = ids if keep is None else keep & ids
    fits = {arm: fit_arm(spec[arm]["csv"], args.runs, keep) for arm in arms}
    for arm, f in fits.items():
        print(f"{arm}: {f['n']} claims, {f['G']} studies, R_train={f['r_train']:.1f}, adding-up gap {f['gap']:+.2f}%")
    contrasts = {arm: paired(fits["baseline"], fits[arm]) for arm in arms[1:]}

    # ── table ────────────────────────────────────────────────────────────────
    rows = []
    for arm, f in fits.items():
        tot = f["theta"][PHI].sum()
        for c in PHI:
            rows.append(dict(arm=arm, stage=c, theta=f["theta"][c], se=f["se"][c], ci_lo=f["boot"].loc[c, "lo_boot"],
                             ci_hi=f["boot"].loc[c, "hi_boot"], share=f["theta"][c] / tot, kind="level"))
    for arm, (th, se, boot) in contrasts.items():
        for c in PHI:
            rows.append(dict(arm=arm, stage=c, theta=th[c], se=se[c], ci_lo=boot.loc[c, "lo_boot"],
                             ci_hi=boot.loc[c, "hi_boot"], share=np.nan, kind="minus_baseline"))
    table = pd.DataFrame(rows)
    tag = "_vs_".join(spec[a]["tag"] for a in arms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"compare_theta_{tag}_{args.runs}runs.csv", index=False)
    show = table.assign(theta=lambda d: d.theta * 1e3, se=lambda d: d.se * 1e3, ci_lo=lambda d: d.ci_lo * 1e3,
                        ci_hi=lambda d: d.ci_hi * 1e3)
    print("\ntheta x 1e-3 (bootstrap 95% interval); 'minus_baseline' rows are paired contrasts")
    print(show.to_string(index=False, float_format="{:.3f}".format))

    # ── figure ───────────────────────────────────────────────────────────────
    both = args.panels == "both"
    fig, axs = plt.subplots(1, 2 if both else 1, figsize=(11 if both else 6.6, 4.3), facecolor=cvb.SURFACE)
    axs = np.atleast_1d(axs)
    x = np.arange(len(PHI))

    ax = axs[0]
    w = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        f = fits[arm]
        v = f["theta"][PHI].values * 1e3
        lo = f["boot"]["lo_boot"].values * 1e3
        hi = f["boot"]["hi_boot"].values * 1e3
        pos = x + (k - (len(arms) - 1) / 2) * w
        ax.bar(pos, v, w * 0.92, color=spec[arm]["color"], label=spec[arm]["name"])
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=cvb.INK, elinewidth=1, capsize=2)
    ax.axhline(0, color=cvb.GRID, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=8, color=cvb.INK)
    ax.set_ylabel(r"$\hat\theta$  ($\times10^{-3}$)", fontsize=9, color=cvb.INK)
    ax.set_title("Stage contribution to run-to-run variance", fontsize=10, color=cvb.INK, loc="left")
    cvb.style(ax)

    if both:
        ax = axs[1]
        others = arms[1:]
        w = 0.8 / max(len(others), 1)
        for k, arm in enumerate(others):
            th, _se, boot = contrasts[arm]
            v = th[PHI].values * 1e3
            lo = boot["lo_boot"].values * 1e3
            hi = boot["hi_boot"].values * 1e3
            pos = x + (k - (len(others) - 1) / 2) * w
            ax.bar(pos, v, w * 0.92, color=spec[arm]["color"])
            ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=cvb.INK, elinewidth=1, capsize=2)
        ax.axhline(0, color=cvb.MUTE, lw=0.8)
        ax.plot([0.6, 1.4], [MARGIN * 1e3] * 2, color=cvb.MUTE, lw=1.2, ls=":")
        ax.plot([0.6, 1.4], [-MARGIN * 1e3] * 2, color=cvb.MUTE, lw=1.2, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=8, color=cvb.INK)
        ax.set_ylabel(r"$\Delta\hat\theta$  ($\times10^{-3}$)", fontsize=9, color=cvb.INK)
        ax.set_title(f"Paired change vs. {spec[arms[0]]['short']}", fontsize=10, color=cvb.INK, loc="left")
        cvb.style(ax)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=spec[a]["color"], label=spec[a]["name"]) for a in arms]
    if both:
        handles.append(Line2D([], [], color=cvb.MUTE, lw=1.2, ls=":", label=r"$\pm0.2\times10^{-3}$ margin (Statistics)"))
    handles.append(Line2D([], [], color=cvb.INK, lw=1, marker="_", ms=6, label="95% study-cluster bootstrap interval"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles) if both else 2, fontsize=8, frameon=False)

    fig.suptitle(f"RPP stage attribution: {' vs. '.join(spec[a]["short"] for a in arms)} (R={args.runs})" if both
                 else f"RPP stage attribution (R={args.runs})",
                 fontsize=12, fontweight="bold", color=cvb.INK)
    fig.tight_layout(rect=(0, 0.07 if both else 0.12, 1, 0.95))
    out = args.out_dir / f"compare_theta_{tag}_{args.runs}runs{'' if both else '_levels'}.png"
    fig.savefig(out, dpi=200, facecolor=cvb.SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
