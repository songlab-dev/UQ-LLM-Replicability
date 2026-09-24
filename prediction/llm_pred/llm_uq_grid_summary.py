"""Summarize the 12 corpus, effort, and temperature analysis cells.

The table combines q-bar, v-bar, calibration, and dispersion outputs. ECE is
computed from each cell's reliability bins and run counts are read from the
prediction files. Missing or inconsistent inputs are recorded in ``status``.

Usage: rep_env/bin/python prediction/llm_pred/llm_uq_grid_summary.py
Writes: prediction/llm_pred/metric/llm_uq_grid_summary.csv
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "prediction", "llm_pred", "metric")
REASONING = os.path.join(REPO, "prediction", "LLM_Reasoning")
OUT_CSV = os.path.join(RES, "llm_uq_grid_summary.csv")

# slug is the infix the predict_* scripts give their per-corpus files; CB uses
# the 158-effect "_cb_full" scope.
CORPORA = [("RPP", "", "RPP", "studies"),
           ("CB", "_cb_full", "CB", "effects"),
           ("SSRP", "_ssrp", "SSRP", "studies")]
EFFORTS = ("high", "low")
TEMPS = ("0.7", "0.2")


def cell_suffix(slug, effort, temp):
    """Same _{effort}_temp{temp} convention predict_llm_uq*.py builds, where
    the canonical high/0.7 cell carries no suffix at all."""
    return slug + ("" if effort == "high" else f"_{effort}") + \
        ("" if temp == "0.7" else f"_temp{temp}")


def ece_from_table(cal, predictor):
    t = cal[cal.predictor == predictor]
    return float((t["n"] / t["n"].sum() * (t["observed_rate"] - t["mean_predicted"]).abs()).sum())


def main():
    rows = []
    for label, slug, pred_dir, unit in CORPORA:
        for effort in EFFORTS:
            for temp in TEMPS:
                sfx = cell_suffix(slug, effort, temp)
                base = dict(corpus=label, effort=effort, temp=temp, unit=unit)
                try:
                    qbar = pd.read_csv(os.path.join(RES, f"llm_qbar{sfx}_metrics.csv")).iloc[0]
                    vbar = pd.read_csv(os.path.join(RES, f"llm_verdict{sfx}_metrics.csv")).iloc[0]
                    cal = pd.read_csv(os.path.join(RES, f"llm_calibration{sfx}.csv"))
                    disp = pd.read_csv(os.path.join(RES,
                                       f"llm_selfconsistency{sfx}_dispersion.csv"))
                    pred = pd.read_csv(os.path.join(
                        REASONING, pred_dir,
                        f"text_predictions_{effort}_temp{temp}.csv"), usecols=["run_id"])
                except FileNotFoundError as e:
                    rows.append(dict(base, status=f"MISSING: {os.path.basename(e.filename)}"))
                    continue

                if not (qbar.n == vbar.n == len(disp)):
                    rows.append(dict(base, status=f"INCONSISTENT: qbar n={qbar.n}, "
                                                  f"vbar n={vbar.n}, dispersion n={len(disp)}"))
                    continue

                rows.append(dict(
                    base, status="ok", n=int(qbar.n), R=int(pred["run_id"].nunique()),
                    AUC_qbar=qbar.AUC, AUC_qbar_SD=qbar.AUC_SD,
                    AUC_vbar=vbar.AUC, AUC_vbar_SD=vbar.AUC_SD,
                    Acc_qbar=qbar.Accuracy, Acc_qbar_SD=qbar.Accuracy_SD,
                    Acc_vbar=vbar.Accuracy, Acc_vbar_SD=vbar.Accuracy_SD,
                    ECE_qbar=ece_from_table(cal, "qbar"),
                    ECE_vbar=ece_from_table(cal, "vbar"),
                    q_sd_mean=disp.q_sd.mean(),
                    verdict_entropy_bits_mean=disp.verdict_entropy_bits.mean(),
                    pairwise_agreement_mean=disp.pairwise_agreement.mean(),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    show = ["corpus", "effort", "temp", "n", "R", "AUC_qbar", "AUC_vbar",
            "ECE_qbar", "ECE_vbar", "q_sd_mean", "verdict_entropy_bits_mean",
            "pairwise_agreement_mean"]
    print(df[[c for c in show if c in df.columns]].to_string(
        index=False, float_format="{:.3f}".format))
    bad = df[df["status"] != "ok"]
    if len(bad):
        print(f"\n{len(bad)} cell(s) not summarized:")
        print(bad[["corpus", "effort", "temp", "status"]].to_string(index=False))
    print(f"\nWrote {OUT_CSV}  ({len(df)} cells)")


if __name__ == "__main__":
    main()
