"""Summarize response delivery and parsing quality for each analysis cell.

The output reports expected and collected requests, null fields, refusal
phrases, and API errors. Contamination probes are excluded. ``missing`` is
the current post-backfill gap; API errors include archived batch-error files.

Usage: rep_env/bin/python prediction/llm_pred/llm_uq_batch_quality.py
Writes: prediction/llm_pred/metric/llm_batch_quality.csv
        prediction/llm_pred/metric/llm_batch_api_errors.csv
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REASONING = os.path.join(REPO, "prediction", "LLM_Reasoning")
OUT_CSV = os.path.join(REPO, "prediction", "llm_pred", "metric", "llm_batch_quality.csv")
OUT_ERR_CSV = os.path.join(REPO, "prediction", "llm_pred", "metric", "llm_batch_api_errors.csv")

# Unit key per corpus: what one "claim" is. CB's claims are per-effect, so a
# paper number alone is not unique.
CORPORA = [("RPP", "RPP", ["altmejd_id"]),
           ("CB", "CB", ["paper_num", "experiment_num", "effect_num"]),
           ("SSRP", "SSRP", ["study_num"])]
EFFORTS = ("high", "low")
TEMPS = ("0.7", "0.2")

CONTAMINATION = "contamination"

# Fields the prompt asks the model to report, in prompt order. q_hat/verdict
# are handled separately (they gate skip-exist); the rest are the "important
# info" whose parse failures survive into the final data.
PARSED_FIELDS = ["extract_N", "extract_p", "extract_r", "effect_type",
                 "stat_score", "qrp_severity", "surprise_rating", "context"]

# Refusal language used for the delivery-quality count.
REFUSAL_RE = (r"(?:i (?:can'?t|cannot|am unable to|won'?t)|"
              r"unable to (?:assess|determine|evaluate|provide)|"
              r"i'?m sorry|as an ai|cannot provide|refuse to)")


def api_errors_by_cell():
    """{(corpus, effort, temp) or (corpus, None, None): Counter(code)} from the
    batch_errors_*.jsonl files, skipping the contamination probes."""
    per = defaultdict(Counter)
    paths = (glob.glob(os.path.join(REASONING, "*", "batch_errors*.jsonl"))
             + glob.glob(os.path.join(REASONING, "*", "archive", "batch_errors*.jsonl")))
    for path in sorted(paths):
        if CONTAMINATION in os.path.basename(path):
            continue
        # The first path segment identifies the corpus for archived files.
        corpus_dir = os.path.relpath(path, REASONING).split(os.sep)[0]
        m = re.search(r"batch_errors_(high|low)_temp([\d.]+)", os.path.basename(path))
        key = (corpus_dir, m.group(1), m.group(2)) if m else (corpus_dir, None, None)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    per[key]["unparseable_error_line"] += 1
                    continue
                body = (rec.get("response") or {}).get("body") or {}
                code = ((body.get("error") or {}).get("code")
                        or (rec.get("error") or {}).get("code") or "unknown")
                per[key][code] += 1
    return per


def main():
    errs = api_errors_by_cell()
    rows = []
    for label, corpus_dir, key_cols in CORPORA:
        for effort in EFFORTS:
            for temp in TEMPS:
                pred_csv = os.path.join(REASONING, corpus_dir,
                                        f"text_predictions_{effort}_temp{temp}.csv")
                base = dict(corpus=label, effort=effort, temp=temp)
                if not os.path.exists(pred_csv):
                    rows.append(dict(base, status="MISSING predictions file"))
                    continue

                d = pd.read_csv(pred_csv)
                units = int(d.groupby(key_cols).ngroups)
                R = int(d["run_id"].max()) + 1
                expected = units * R
                txt = d["prediction"].astype(str).str.strip()
                empty = int(txt.isin(["", "nan", "None"]).sum())
                cell_err = errs.get((corpus_dir, effort, temp), Counter())

                text_blob = txt.str.cat(d["reasoning"].astype(str), sep=" ")
                refusal = text_blob.str.contains(REFUSAL_RE, case=False, regex=True, na=False)

                row = dict(
                    base, status="ok", units=units, R=R, expected=expected,
                    collected=len(d), missing=expected - len(d),
                    missing_pct=round(100 * (expected - len(d)) / expected, 3),
                    empty_completion=empty,
                    q_hat_unparsed=int(d["q_hat"].isna().sum()),
                    verdict_unparsed=int(d["verdict"].isna().sum()),
                    refusal_phrase=int(refusal.sum()),
                    refusal_phrase_no_answer=int(d.loc[refusal, "q_hat"].isna().sum()),
                )
                for col in PARSED_FIELDS:
                    if col in d.columns:
                        n_null = int(d[col].isna().sum())
                        row[f"{col}_null"] = n_null
                        row[f"{col}_null_pct"] = round(100 * n_null / len(d), 2)
                # "Nothing came back that we could use" -- every prompted field
                # null at once. This is the refusal-shaped statistic: a single
                # null field is usually the paper not reporting that value
                # (see the caveat above), but an all-null row means the
                # completion yielded no parsable content at all.
                tracked = [c for c in PARSED_FIELDS if c in d.columns] + ["q_hat", "verdict"]
                none_parsed = int(d[tracked].isna().all(axis=1).sum())
                row["none_fields_parsed"] = none_parsed
                row["none_fields_parsed_pct"] = round(100 * none_parsed / len(d), 3)
                row["api_errors"] = int(sum(cell_err.values()))
                row["api_error_codes"] = ";".join(f"{k}={v}" for k, v in sorted(cell_err.items()))
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    err_rows = [dict(corpus=k[0], effort=k[1] or "(untagged)", temp=k[2] or "(untagged)",
                     code=code, count=n)
                for k, c in sorted(errs.items(), key=lambda kv: str(kv[0]))
                for code, n in sorted(c.items())]
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(OUT_ERR_CSV, index=False)

    show = ["corpus", "effort", "temp", "collected", "missing", "empty_completion",
            "q_hat_unparsed", "verdict_unparsed", "refusal_phrase",
            "refusal_phrase_no_answer", "extract_N_null_pct", "extract_p_null_pct",
            "extract_r_null_pct", "none_fields_parsed", "none_fields_parsed_pct"]
    print(df[[c for c in show if c in df.columns]].to_string(index=False))
    print(f"\nAPI-level rejections recorded in batch_errors_*.jsonl "
          f"(not refusals -- see module docstring):")
    print(err_df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_ERR_CSV}")


if __name__ == "__main__":
    main()
