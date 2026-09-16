"""Prompt and corpus loaders for the title-only contamination probe.

Each query contains a title and a short locator for the focal finding, but
not paper text. RPP and CB use the effect description; SSRP uses publication
year and test family.
"""
import re
from pathlib import Path

import pandas as pd

from rpp_outcomes import canonical_rpp_outcomes_for_titles

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Test-family labels used in SSRP locators.
TEST_FAMILY_ARTICLE_NAME = {"F": "an F", "t": "a t", "z": "a z", "Chi2": "a chi-squared"}

SYSTEM_PROMPT = (
    "You will be given the title of a peer-reviewed research finding that was "
    "later the subject of a formal, independent replication attempt{effect_clause}.\n\n"
    "Answer ONLY from what you already know from training about this SPECIFIC "
    "study and its SPECIFIC replication attempt. Do not reason about what seems "
    "plausible, and do not fall back on general priors about replication rates "
    "in the field. If you do not specifically recognize this study, say so "
    "honestly rather than guessing.\n\n"
    "Respond in exactly this format:\n"
    "FAMILIARITY: <recognized|unsure|unfamiliar>\n"
    "VERDICT: <replicable|unreplicable|unknown>\n"
    "NOTES: <one sentence: what you recall about this study/replication, or "
    "why you don't recognize it>"
)


def build_messages(title, locator=None):
    """Build chat messages for one title and optional finding locator."""
    effect_clause = ", along with a detail identifying the specific finding being replicated" \
        if locator else ""
    system = SYSTEM_PROMPT.format(effect_clause=effect_clause)
    user = title if not locator else f"{title}\n{locator}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_title(s):
    s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def load_rpp_units():
    """Load the 90 RPP claims with title, effect locator, and outcome."""
    rpp = pd.read_csv(DATA_DIR / "rpp_data_cleaned.csv", encoding="latin1")
    rpp["_norm_title"] = rpp["Study Title (O)"].apply(_normalize_title)
    assert not rpp["_norm_title"].duplicated().any()

    alt = pd.read_csv(DATA_DIR / "altmejd_osf" / "data.csv")
    alt90 = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
    alt90["_norm_title"] = alt90["title"].apply(_normalize_title)

    claims = alt90.merge(
        rpp[["Study Title (O)", "_norm_title", "Description of effect (O)"]],
        on="_norm_title", how="inner")
    assert len(claims) == len(alt90)
    out = claims[["id", "Study Title (O)", "Description of effect (O)"]].rename(
        columns={"id": "unit_id", "Study Title (O)": "title",
                 "Description of effect (O)": "locator"})
    out["locator"] = "Effect: " + out["locator"].astype(str).str.strip()
    outcome = canonical_rpp_outcomes_for_titles(out["title"])
    if outcome.isna().any():
        raise ValueError("Original rpp_data.csv lacks an outcome for a filtered RPP title")
    out["ground_truth"] = outcome.map({1: "yes", 0: "no"})
    return out


def load_cb_units():
    """Load the 158 CB effects with title, locator, and outcome."""
    cb = pd.read_csv(DATA_DIR / "cb_data_cleaned.csv")
    cb = cb.sort_values(["Paper #", "Experiment #", "Effect #"]).reset_index(drop=True)
    out = cb[["Paper #", "Experiment #", "Effect #", "Study Title (O)",
              "Description of effect (O)", "Replicate (R)"]].rename(columns={
        "Paper #": "paper_num", "Experiment #": "experiment_num", "Effect #": "effect_num",
        "Study Title (O)": "title", "Description of effect (O)": "locator",
        "Replicate (R)": "ground_truth"})
    out["locator"] = "Effect: " + out["locator"].astype(str).str.strip()
    out["ground_truth"] = out["ground_truth"].astype(str).str.strip().str.lower()
    return out


def load_ssrp_units():
    """Load the 21 SSRP studies with a year-and-test-family locator."""
    ssrp = pd.read_csv(DATA_DIR / "ssrp_data_cleaned.csv")
    out = ssrp[["Study Num", "Study Title (O)", "Year (O)",
                "Test statistic type (O)", "Replicate (R)"]].rename(columns={
        "Study Num": "study_num", "Study Title (O)": "title", "Replicate (R)": "ground_truth"})

    family = out["Test statistic type (O)"].astype(str).str.extract(r"^([A-Za-z0-9]+)")[0]
    unmapped = set(family) - set(TEST_FAMILY_ARTICLE_NAME)
    assert not unmapped, f"TEST_FAMILY_ARTICLE_NAME has no entry for: {unmapped!r}"
    out["locator"] = (
        "Year: " + out["Year (O)"].astype(int).astype(str)
        + "; the null distribution of test statistics was "
        + family.map(TEST_FAMILY_ARTICLE_NAME) + " distribution.")
    out = out[["study_num", "title", "locator", "ground_truth"]]
    out["ground_truth"] = out["ground_truth"].astype(str).str.strip().str.lower()
    return out
