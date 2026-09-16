"""Build the effect-level CB analysis table.

Rows are keyed by paper, experiment, and effect. Internal replications of an
effect are collapsed with a majority outcome; the first effect description is
retained as its locator. The resulting table contains 158 completed effects.

Run from repo root: python prompt/build_cb_data.py
Writes: data/cb_data_cleaned.csv
"""
import os

import pandas as pd

PAPER_CSV = "data/RP_CB_paper.csv"
EFF_CSV = "data/RP_CB_eff.csv"
MANIFEST_CSV = "file_CB/_manifest.csv"
OUT_CSV = "data/cb_data_cleaned.csv"


def main():
    paper = pd.read_csv(PAPER_CSV)
    eff = pd.read_csv(EFF_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    paper["Study Title (O)"] = paper["Original paper title"].astype(str).str.strip().str.rstrip(".")
    titles = paper.set_index("Paper #")["Study Title (O)"]
    dirs = manifest.set_index("paper")["dir"]

    eff = eff.sort_values(["Paper #", "Experiment #", "Effect #", "Internal replication #"])
    eff["is_positive"] = eff["Observed difference in replication?"] == "Positive"

    key = ["Paper #", "Experiment #", "Effect #"]
    grouped = eff.groupby(key)
    vote = grouped["is_positive"].agg(n_internal_reps="count", n_internal_reps_positive="sum")
    vote["frac_internal_reps_positive"] = vote["n_internal_reps_positive"] / vote["n_internal_reps"]
    vote["Replicate (R)"] = (vote["n_internal_reps_positive"] > vote["n_internal_reps"] / 2).map(
        {True: "yes", False: "no"})
    first_desc = grouped["Effect description"].first()

    out = vote.join(first_desc).rename(columns={"Effect description": "Description of effect (O)"})
    out = out.reset_index()
    out["Study Title (O)"] = out["Paper #"].map(titles)
    out["dir"] = out["Paper #"].map(dirs)

    out = out[["Paper #", "Study Title (O)", "dir", "Experiment #", "Effect #",
               "Description of effect (O)", "Replicate (R)",
               "n_internal_reps", "frac_internal_reps_positive"]]
    out = out.sort_values(["Paper #", "Experiment #", "Effect #"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"{OUT_CSV}: {len(out)} effects across {out['Paper #'].nunique()} papers "
          f"({(out['Replicate (R)'] == 'yes').sum()} yes / {(out['Replicate (R)'] == 'no').sum()} no); "
          f"{(out['n_internal_reps'] > 1).sum()} collapsed from multiple internal replications")


if __name__ == "__main__":
    main()
