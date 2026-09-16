"""Canonical RPP replication outcomes from the original OSF data export."""

from pathlib import Path
import re

import pandas as pd


DATA_CSV = Path(__file__).resolve().parent / "rpp_data.csv"


def normalize_title(value):
    """Normalize titles using the convention used throughout this repository."""
    value = re.sub(r"[^a-z0-9\s]", "", str(value).lower())
    return re.sub(r"\s+", " ", value).strip()


def load_canonical_rpp_outcomes(path=DATA_CSV):
    """Return normalized title -> binary outcome from the original RPP export."""
    raw = pd.read_csv(path, encoding="latin1")
    labeled = raw.loc[
        raw["Study Title (O)"].notna() & raw["Replicate (R)"].notna(),
        ["Study Title (O)", "Replicate (R)"],
    ].copy()
    labeled["norm_title"] = labeled["Study Title (O)"].map(normalize_title)
    labeled["canonical_replicated"] = (
        labeled["Replicate (R)"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    if labeled["canonical_replicated"].isna().any():
        values = labeled.loc[labeled["canonical_replicated"].isna(), "Replicate (R)"].unique()
        raise ValueError(f"Unexpected Replicate (R) values: {values.tolist()}")

    conflicts = labeled.groupby("norm_title")["canonical_replicated"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting outcomes for a normalized title in rpp_data.csv")

    return (
        labeled.drop_duplicates("norm_title")
        .set_index("norm_title")["canonical_replicated"]
        .astype(int)
    )


def canonical_rpp_outcomes_for_titles(titles):
    """Return canonical binary outcomes aligned to an iterable of RPP titles."""
    title_series = pd.Series(titles, copy=False)
    return title_series.map(normalize_title).map(load_canonical_rpp_outcomes())


def attach_canonical_rpp_outcomes(frame, title_col="norm_title"):
    """Replace ``replicated`` with the original RPP outcome, retaining Altmejd's."""
    result = frame.copy()
    if title_col not in result:
        raise KeyError(f"Missing title key column: {title_col}")
    if "replicated" in result:
        result["replicated_altmejd"] = result["replicated"]

    result["replicated"] = result[title_col].map(load_canonical_rpp_outcomes())
    if result["replicated"].isna().any():
        missing = result.loc[result["replicated"].isna(), title_col].unique()
        raise ValueError(f"RPP outcomes missing for normalized titles: {missing.tolist()}")
    result["replicated"] = result["replicated"].astype(int)
    return result
