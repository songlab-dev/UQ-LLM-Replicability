"""Single source of truth for the SSRP text-condition prompt.

Imported by the SSRP modes of predict_text_batch.py and embed_text_batch.py, so the text that gets embedded
is exactly the text the model saw. Side-effect free -- no argparse, no API
client, no file I/O at import time.

RECOVERED 2026-09-09 after the .py source was lost. SYSTEM_PROMPT is verified
byte-identical to the sent prompt (2982 chars), matching the length recorded from the
original bytecode before it was lost. The function bodies were
rebuilt from the bytecode's disassembly and reproduce the original's output on
every probe case. Original comments and formatting did not survive.
"""
import os
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "file_SSRP"

SYSTEM_PROMPT = (
    'You assess whether a focal effect in a scientific study will replicate. Important '
    'calibration: across large-scale systematic replication attempts in the social and '
    'behavioral sciences, roughly 40% of tested effects replicated successfully. Use this '
    'as your prior — do not assume replication is the default.\n'
    '\n'
    'Work through these seven steps in order, each in 1-3 sentences.\n'
    '\n'
    '1. Extraction: report only facts stated in the paper — focal claim, sample size N, the '
    'focal effect size expressed as a correlation r (convert from d, eta-squared, or f if '
    'the paper reports one of those; write NA if no conversion is possible), p-value, study '
    'design (within- or between-subjects), primary endpoint, and effect type (main effect, '
    "interaction, or other). Write 'not reported' for anything absent.\n"
    '2. Credibility: rate transparency, preregistration, open data/code, measurement '
    'validity, and design strength.\n'
    '3. Statistics: judge statistical power, multiplicity, model appropriateness, and '
    'effect-size plausibility; summarize as a single [0,1] statistical-soundness score. '
    'Small N (<50), large effect sizes (r>.5), or p just above .01 should lower this score.\n'
    '4. QRP severity: score researcher-degrees-of-freedom concerns on a 0–3 scale. 0 = no '
    'concerns (preregistered, open data, single primary outcome, adequate N). 1 = minor '
    '(one or two of: no preregistration, no open data, modest N). 2 = moderate (multiple '
    'concerns: no preregistration AND no open data AND either many outcomes/conditions or p '
    'just under .05 or flexible covariates). 3 = severe (most of the above, or evidence of '
    'optional stopping or HARKing). State the specific concerns that drove your score, not '
    'a generic list.\n'
    '5. Theory and context: give a surprise rating from 1 to 5 (1 = theoretically expected, '
    '5 = highly surprising or counterintuitive); then judge contextual or temporal '
    'fragility (effects tied to a specific population, cultural moment, or sensitive topic) '
    'as low, medium, or high — and state the specific reason if medium or high.\n'
    '6. Score: start from the 40% base rate. Adjust upward for high statistical soundness '
    '(step 3), low QRP severity (step 4 = 0 or 1), and low surprise/fragility (step 5). '
    'Adjust downward for low soundness, QRP severity >= 2, or high fragility. Give your '
    'final replication probability in [0,1].\n'
    '7. Verdict: replicable or unreplicable, consistent with whether Q_HAT is above or '
    'below 0.5, with a one-sentence rationale referencing the dominant factor.\n'
    '\n'
    'After the reasoning, end with EXACTLY these ten lines and nothing after them. Use NA '
    'if a value is not reported in the paper; numbers only (no units):\n'
    'EXTRACT_N: <integer sample size>\n'
    'EXTRACT_P: <p-value as a number, e.g. 0.03>\n'
    'EXTRACT_R: <effect size as a correlation r, e.g. 0.30>\n'
    'EFFECT_TYPE: <main|interaction|other>\n'
    'STAT_SCORE: <0-1 statistical-soundness score>\n'
    'QRP_SEVERITY: <integer 0-3>\n'
    'SURPRISE_RATING: <integer 1-5>\n'
    'CONTEXT_FRAGILITY: <low|medium|high>\n'
    'Q_HAT: <0-1 replication probability>\n'
    'VERDICT: <replicable|unreplicable>'
)

def _study_dir(study_num):
    matches = glob.glob(os.path.join(ROOT, f"Study {int(study_num)} - *"))
    return matches[0] if matches else None


def read_paper_text(study_num, max_chars=0):
    """OCR text for an SSRP study, or None if the file/dir is missing."""
    d = _study_dir(study_num)
    if not d:
        return None
    p = os.path.join(d, "OriginalAnony.txt")
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8", errors="ignore").read()
    return txt[:max_chars] if max_chars else txt


def build_locator(ssrp, study_num):
    """The focal-study context line - the part of the prompt that varies by
    study. SSRP has no free-text "Description of effect (O)" the way RPP/CB
    do, so there's no effect description to add here either -- and the
    original study's own test statistic/N/p-value/effect size are
    deliberately left out too, not just absent. Those numbers are themselves
    known predictors of replication (weaker original evidence -- higher p,
    smaller r -- replicates less often), so embedding them would let the model
    recover a hint about the answer instead of testing what the paper's text
    alone supports."""
    row = ssrp[ssrp["Study Num"] == study_num].iloc[0]
    return [f"Focal study to evaluate: {row['Study Title (O)']} (SSRP study {int(study_num)})."]


def build_user_message(ssrp, study_num, paper_text):
    """The exact user-turn string sent to the model."""
    loc = build_locator(ssrp, study_num)
    return (
        "Here is the original study paper (OCR text; author/affiliation names are masked):\n\n"
        f"{paper_text}\n\n"
        f"{chr(10).join(loc)}\n\n"
        "Using the text above as the source material, locate this focal study/effect "
        "and complete the seven-step replicability assessment."
    )


def build_messages(ssrp, study_num, paper_text):
    """The exact chat messages sent to the model."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(ssrp, study_num, paper_text)}]
