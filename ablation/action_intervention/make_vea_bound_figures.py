"""VEA decomposition and Freedman/Azuma/Bernstein bound figures for an ablation arm.

Reuses figures/make_vea_figures.py (compute_vea, make_plot),
uq_bound/validation_plot.py (plot_panel_a) and uq_bound/common.py unchanged.
The only new logic is the chain of conditional-expectation increments for A2,
whose Step 1 extracts no numbers: with q_1 unscorable, the chain drops
Extraction (4 increments instead of 5).

    python ablation/action_intervention/make_vea_bound_figures.py --arm n_only
    python ablation/action_intervention/make_vea_bound_figures.py --arm claim_only
    python ablation/action_intervention/make_vea_bound_figures.py --arm baseline --runs 100 --check

Outputs go to results/figures/ (or --out-dir).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent
sys.path.insert(0, str(PKG / "uq_bound"))
sys.path.insert(0, str(PKG / "figures"))
import common  # noqa: E402
import pstar  # noqa: E402
import scoring  # noqa: E402
import validation_plot as vp  # noqa: E402
from make_vea_figures import compute_vea as vea_table, make_plot  # noqa: E402
from metrics import monte_carlo_metrics  # noqa: E402

BASELINE_CSV = PKG / "prediction" / "results" / "RPP" / "text_predictions_high_temp0.7.csv"
ARMS = {  # arm -> (predictions CSV, Step-1 extraction used for the label, chain)
    "baseline": (BASELINE_CSV, "A0 (N, r, p)", [1, 3, 4, 5, 6]),
    "spurious_stats": (HERE / "results/spurious_stats/text_predictions_high_temp0.7.csv", "Spurious stats (full extraction)", [1, 3, 4, 5, 6]),
    "n_only": (HERE / "results/archive/n_only/text_predictions_high_temp0.7.csv", "A1 (N only)", [1, 3, 4, 5, 6]),
    "claim_only": (HERE / "results/claim_only/text_predictions_high_temp0.7.csv", "A2 (claim only)", [3, 4, 5, 6]),
    "n_only_nohint": (HERE / "results/archive/n_only_nohint/text_predictions_high_temp0.7.csv", "A3 (N only, no hints)", [1, 3, 4, 5, 6]),
}
VEA_LABELS = {1: "Extraction", 3: "Statistics", 4: "Researcher DoF\nconcern",
              5: "Theory\nand context", 6: "Scoring", 7: "Verdict"}


def estimate_chain(unit_scores, key_col, chain):
    """Same as common.estimate_chain_of_action, for an arbitrary chain.

    Pooled OLS of q_6 on each growing prefix of the chain's scores gives the
    conditional expectations; the first is the claim mean, the last is q_6.
    With chain = [1, 3, 4, 5, 6] this reproduces the original exactly.
    """
    cols = [f"q_{m}" for m in chain]
    d = unit_scores[[key_col, "run_id"] + cols].dropna().copy()
    d["h0"] = d.groupby(key_col)["q_6"].transform("mean")
    prev = "h0"
    for k in range(1, len(chain)):
        prefix = cols[:k]
        fit = LinearRegression().fit(d[prefix], d["q_6"])
        d[f"h{k}"] = np.clip(fit.predict(d[prefix]), 0.0, 1.0)
        d[f"D{k}"] = d[f"h{k}"] - d[prev]
        prev = f"h{k}"
    last = len(chain)
    d[f"D{last}"] = d["q_6"] - d[prev]
    dcols = [f"D{k}" for k in range(1, last + 1)]
    v_i = d.groupby(key_col)[dcols].apply(lambda g: (g ** 2).mean().sum())
    v_i.name = "v_i"
    return d[[key_col, "run_id"] + dcols], v_i, float(d[dcols].abs().to_numpy().max())


def compute_vea(pred, chain):
    rpp = pd.read_csv(common.RPP_CSV, encoding="latin1")
    unit_scores = scoring.compute_unit_scores(pred, rpp)
    # Reference targets as in the published RPP VEA figure: unit 3 = replication
    # power (power_reference.csv, now under prediction/results/); unit 6 = the 20-seed
    # nested-CV average embedding-model probability (VEA's p*; the bound uses the
    # single-seed proba_cv instead).
    pstar.POWER_REF_CSV = str(PKG / "prediction" / "results" / "RPP" / "power_reference.csv")
    pstar_by_unit = {m: pstar.p_star_target(m, rpp, n_predictors=6) for m in scoring.QCOLS}
    scope = common.load_rpp_scope()
    scope["p_star_6"] = pd.read_csv(common.RPP_EMBED_LR_CSV)["proba_cv_avg"].to_numpy()
    pstar_by_unit[6] = (scope.drop_duplicates("Study Title (O)").set_index("Study Title (O)")["p_star_6"]
                        .rename("p_star").reset_index())
    met = monte_carlo_metrics(unit_scores, pstar_by_unit)
    units = [1, 3, 4, 5, 6, 7]
    vea = vea_table(met, units, [VEA_LABELS[m] for m in units], nondegen={3, 4, 5, 6})
    if 1 not in chain:  # no Extraction scores in this arm
        vea = vea[vea["unit"] != 1].reset_index(drop=True)
    return vea


def compute_bound_inputs(pred, chain):
    # build_claims_and_deviations_rpp resolves these names at call time, so the
    # arm's chain can be swapped in without editing common.py.
    common.CHAIN = chain
    common.estimate_chain_of_action = lambda us, key: estimate_chain(us, key, chain)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        pred.to_csv(tmp.name, index=False)
    claims, _chain, us, _met, _c_hat, c2, max_abs_d = common.build_claims_and_deviations_rpp(tmp.name)
    Path(tmp.name).unlink()
    assert vp.C_TIGHT_RPP >= max_abs_d, f"c_tight={vp.C_TIGHT_RPP} < observed max|D|={max_abs_d:.3f}"
    return claims, us, c2, max_abs_d


def load_arm(arm, runs):
    csv, name, chain = ARMS[arm]
    pred = pd.read_csv(csv)
    pred = pred[pred["run_id"] < runs]
    print(f"RPP {name}, {runs} runs: {len(pred)} rows, {pred['altmejd_id'].nunique()} claims, chain={chain}")
    return pred, name, chain


def run(arm, runs, out_dir, check=False):
    pred, name, chain = load_arm(arm, runs)
    tag, label = f"{arm}_{runs}runs", f"RPP {name}, {runs} runs"
    vea = compute_vea(pred, chain)
    print(vea[["unit", "n", "V", "E_raw", "E", "A"]].to_string(index=False, float_format="{:.4f}".format))
    out_dir.mkdir(parents=True, exist_ok=True)
    vea.to_csv(out_dir / f"vea_{tag}.csv", index=False)
    if check:
        return vea, None
    make_plot(vea, f"VEA Decomposition per Unit: {label}", str(out_dir / f"vea_{tag}.png"), ylim=0.35)

    claims, us, c2, max_abs_d = compute_bound_inputs(pred, chain)
    outputs = {"a": out_dir / f"bound_{tag}.png"}
    vp.plot_panel_a("rpp", claims, us, c2, max_abs_d, vp.C_TIGHT_RPP, label, outputs)
    pd.DataFrame([{"arm": arm, "runs": runs, "n_claims": len(claims), "chain_steps": len(chain),
                   "mean_delta": claims["delta"].mean(), "mean_v_i": claims["v_i"].mean(),
                   "max_abs_D": max_abs_d}]).to_csv(out_dir / f"bound_summary_{tag}.csv", index=False)
    print(f"Wrote {outputs['a']}")
    return vea, claims


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--runs", type=int, default=10, help="use run ids 0..runs-1 (match across arms)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "figures")
    parser.add_argument("--check", action="store_true", help="VEA only, no figures (for validation)")
    args = parser.parse_args()
    run(args.arm, args.runs, args.out_dir, args.check)


if __name__ == "__main__":
    main()
