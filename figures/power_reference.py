"""Unit-3 reference p*_{i,3}: replication power, parsed from each study's
replication report (file/<title>/replication.{pdf,docx}).

Definition (per spec): the power of the *replication* to detect the *original*
effect size at the replication's planned sample size.

  - Original effect: parse the original test statistic from the report
    (t(df)=... or F(1,df2)=...) and convert to a correlation r.
  - Replication N: parse the planned sample size from the report.
  - Power: correlation power via Fisher's z at alpha=0.05 (two-tailed).

Studies whose report is missing/unparseable get NaN (dropped from unit 3).

Run: rep_env/bin/python figures/power_reference.py
Writes prediction/LLM_Reasoning/RPP/power_reference.csv, read by pstar.py.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None
try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None

# Paths are anchored to the repository root, rather than the working directory.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
ROOT = HERE
RPP_CSV = os.path.join(HERE, "data", "rpp_data_cleaned.csv")
OUT_CSV = os.path.join(HERE, "prediction", "LLM_Reasoning", "RPP", "power_reference.csv")
FILE_DIR = os.path.join(ROOT, "file")
ALPHA = 0.05

# Original test statistic: t(df)=v  or  F(1,df2)=v
STAT_RE = re.compile(r"\b([tF])\s*\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\)\s*=\s*(-?\d+(?:\.\d+)?)")
# Planned replication sample size (several phrasings seen in the reports)
PLAN_RES = [
    re.compile(r"[Pp]lanned [Ss]ample.{0,200}?\b(\d{2,4})\b"),
    re.compile(r"\b(\d{2,4})\s*(?:participants|subjects|students)[^.]{0,40}?"
               r"(?:for|achiev|provide)[^.]{0,20}?9[05]%\s*power", re.I),
    re.compile(r"(?:total|final|collect|recruit)[^.]{0,40}?\bN\s*=\s*(\d{2,4})", re.I),
]


def read_replication_text(title):
    """Return (text, source_filename) for a study's replication report, or (None, None)."""
    base = os.path.join(FILE_DIR, title)
    if fitz is not None and os.path.exists(os.path.join(base, "replication.pdf")):
        try:
            d = fitz.open(os.path.join(base, "replication.pdf"))
            t = " ".join(pg.get_text() for pg in d); d.close()
            return re.sub(r"\s+", " ", t), "replication.pdf"
        except Exception:
            pass
    if docx is not None and os.path.exists(os.path.join(base, "replication.docx")):
        try:
            t = " ".join(p.text for p in docx.Document(os.path.join(base, "replication.docx")).paragraphs)
            return re.sub(r"\s+", " ", t), "replication.docx"
        except Exception:
            pass
    return None, None


def stat_to_r(kind, df1, df2, val):
    """Convert a reported t/F statistic to a correlation effect size r in [0,1)."""
    if kind == "t":
        t, df = abs(val), df1
        return t / np.sqrt(t * t + df) if df > 0 else np.nan
    if kind == "F" and df1 == 1 and df2:  # only single-df-numerator F reduces to one r
        t = np.sqrt(abs(val))
        return t / np.sqrt(t * t + df2) if df2 > 0 else np.nan
    return np.nan  # multi-df F, chi-square, etc. — not reducible to a single r here


def correlation_power(r, n, alpha=ALPHA):
    """Power of a two-tailed test of rho=0 for effect r at sample size n (Fisher z)."""
    if r is None or n is None or not np.isfinite(r) or n <= 3:
        return np.nan
    z = np.arctanh(min(abs(r), 0.999)) * np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z - zc) + norm.cdf(-z - zc))


def extract(title):
    """Parse one study's report -> dict with stat, r_orig, N_rep, repl_power."""
    text, src = read_replication_text(title)
    rec = {"Study Title (O)": title, "src": src, "stat_type": None, "df1": np.nan,
           "df2": np.nan, "stat_val": np.nan, "r_orig": np.nan, "N_rep": np.nan,
           "repl_power": np.nan}
    if not text:
        return rec
    sm = STAT_RE.search(text)
    if sm:
        kind = sm.group(1)
        df1 = int(sm.group(2))
        df2 = int(sm.group(3)) if sm.group(3) else None
        val = float(sm.group(4))
        rec.update(stat_type=kind, df1=df1, df2=(df2 if df2 else np.nan), stat_val=val,
                   r_orig=stat_to_r(kind, df1, df2, val))
    for pat in PLAN_RES:
        pm = pat.search(text)
        if pm:
            rec["N_rep"] = int(pm.group(1))
            break
    rec["repl_power"] = correlation_power(rec["r_orig"], rec["N_rep"])
    return rec


def main():
    rpp = pd.read_csv(RPP_CSV, encoding="latin1")
    rows = [extract(t) for t in rpp["Study Title (O)"]]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    ok = df["repl_power"].notna().sum()
    print(f"Saved {OUT_CSV}: {len(df)} studies, {df['src'].notna().sum()} with a report, "
          f"{ok} with a computed replication power "
          f"(mean={df['repl_power'].mean():.3f})")


if __name__ == "__main__":
    main()
