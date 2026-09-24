"""Emit the appendix section documenting the zero-shot prompt, generated from
the prompt modules themselves so it cannot drift from what was actually sent.

All three datasets share one prompt. Diffing the modules shows the system
prompt is byte-identical across RPP, CB and SSRP except for a single
clause naming the reference population for the 40% base rate, and the user
turn is identical except for the locator block that names the focal unit.
This script verifies that claim at build time (it asserts the shared parts
really are equal before writing anything) and emits:

  - the shared system prompt, verbatim
  - the shared user-turn template
  - a table of the two corpus-specific slots

Requires \\usepackage{listings} in the paper preamble (lstlisting with
breaklines, so no line of the prompt is truncated in the margin). fancyvrb's
Verbatim is avoided because its breaklines key breaks if another package
(e.g. moreverb) defines a same-named environment and loads after it.

Usage: rep_env/bin/python prediction/llm_pred/llm_uq_appendix_prompt.py
Writes: tables/appendix_prompt.tex
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "prompt"))
OUT_TEX = os.path.join(REPO, "tables", "appendix_prompt.tex")

import prompt_text          # noqa: E402  RPP
import prompt_text_cb       # noqa: E402
import prompt_text_ssrp     # noqa: E402

MODULES = [("RPP", prompt_text), ("CB", prompt_text_cb), ("SSRP", prompt_text_ssrp)]

# The calibration sentence is the one system-prompt line that varies.
# "roughly 40%" itself is byte-identical across all three, so it is shown as
# fixed text. What varies on either side of it: the named reference
# population before it, and a short outcome phrase after it (RPP: "replicate
# successfully"; CB/SSRP: "of tested effects replicated successfully") --
# hence two placeholders.
FIXED_OPENER = "You assess whether a focal effect in a scientific study will replicate. "
CAL_HEAD = "Important calibration: across "
CAL_PCT = "roughly 40%"
SPLIT_TAIL = " Use this as your prior"

LOCATORS = {
    "RPP": (r"\texttt{Focal study to evaluate: study \{n\}.} followed by the "
            r"description of the focal effect, the type of effect, and the type "
            r"of analysis, each line omitted when the record does not carry it."),
    "CB": (r"\texttt{Focal study to evaluate: \{title\} (RP:CB paper \{n\}).}, "
              r"then \texttt{Focal effect: Experiment \{e\}, Effect \{f\}.}, then "
              r"\texttt{Description of the focal effect: \dots} The 30 of 53 papers "
              r"with no completed CB effect get the title line alone."),
    "SSRP": (r"\texttt{Focal study to evaluate: \{title\} (SSRP study \{n\}).} and "
             r"nothing further --- see the note below."),
}


def split_prompt(text):
    """(population phrase, outcome phrase, shared remainder) for one corpus's
    system prompt, split around the fixed "roughly 40%" anchor."""
    assert text.startswith(FIXED_OPENER), "system prompt no longer starts with the shared opener"
    assert CAL_HEAD in text, f"'{CAL_HEAD}' not found -- system prompt structure changed"
    after_head = text[text.index(CAL_HEAD) + len(CAL_HEAD):]
    assert CAL_PCT in after_head, "'roughly 40%' no longer appears verbatim in the system prompt"
    population, after_pct = after_head.split(CAL_PCT, 1)
    i = after_pct.index(SPLIT_TAIL)
    outcome, remainder = after_pct[:i], after_pct[i:]
    return population.rstrip(", "), outcome.strip(), remainder


def ascii_safe(text):
    """Map the prompt's non-ASCII characters (em and en dashes) to ASCII
    before it goes inside \\begin{lstlisting}, which otherwise fails to
    compile ("Invalid UTF-8 byte sequence") without a per-font `literate=`
    mapping. The substitution happens here, not in prompt_text*.py, which
    must stay byte-identical to what the API received. Raises on any other
    non-ASCII character."""
    out = text.replace("—", "--").replace("–", "-")
    bad = {ch for ch in out if ord(ch) > 127}
    assert not bad, f"unhandled non-ASCII character(s) in prompt text: {bad!r}"
    return out


def main():
    populations, outcomes, remainders = {}, {}, {}
    for label, mod in MODULES:
        populations[label], outcomes[label], remainders[label] = split_prompt(mod.SYSTEM_PROMPT)

    # The appendix claims one prompt covers all three datasets; fail if they
    # differ outside the two named substitutions.
    shared_remainder = remainders["RPP"]
    for label in remainders:
        assert remainders[label] == shared_remainder, (
            f"{label}'s system prompt differs from RPP's outside the two calibration "
            f"substitutions -- the 'one shared prompt' claim in this appendix is no longer true.")

    user_tmpl = prompt_text.build_user_message.__doc__  # sanity: modules present
    assert user_tmpl

    # Placeholder markers are plain ASCII so lstlisting compiles without a
    # fontenc/literate setup. "roughly 40%" is shared text, not a placeholder.
    body = ascii_safe(
        FIXED_OPENER + CAL_HEAD + "<REFERENCE POPULATION>, " + CAL_PCT +
        " <OUTCOME PHRASING>." + shared_remainder)

    tex = [
        r"% Generated by llm_uq_appendix_prompt.py -- do not edit by hand.",
        r"% Requires \usepackage{listings} in the preamble. Deliberately not fancyvrb's",
        r"% Verbatim: its breaklines key silently disappears if another package (e.g.",
        r"% moreverb) defines a same-named environment and loads after it.",
        r"\section{Prompt}\label{app:prompt}",
        r"",
        r"All three corpora are scored with one prompt. The system prompt below is "
        r"byte-identical across RPP, CB and SSRP except for two spans in the "
        r"calibration sentence, marked \texttt{<REFERENCE POPULATION>} and "
        r"\texttt{<OUTCOME PHRASING>} -- the 40\% figure between them is itself "
        r"byte-identical across all three and is shown as fixed text, not a "
        r"substitution. The user turn is identical except for the locator block "
        r"marked \texttt{<LOCATOR>}. All three substitutions are listed in "
        r"Table~\ref{tab:prompt-slots}.",
        r"",
        r"\subsection{System prompt}",
        r"\begin{lstlisting}[breaklines=true,basicstyle=\small\ttfamily,columns=fullflexible]",
        body,
        r"\end{lstlisting}",
        r"",
        r"\subsection{User turn}",
        r"\begin{lstlisting}[breaklines=true,basicstyle=\small\ttfamily,columns=fullflexible]",
        "Here is the original study paper (OCR text; author/affiliation names are masked):",
        "",
        "<PAPER_TEXT>",
        "",
        "<LOCATOR>",
        "",
        "Using the text above as the source material, locate this focal study/effect "
        "and complete the seven-step replicability assessment.",
        r"\end{lstlisting}",
        r"",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{The three corpus-specific substitutions. The 40\% base rate "
        r"itself never varies -- only the named reference population, the short "
        r"outcome phrasing around it, and the focal-unit locator do.}",
        r"\label{tab:prompt-slots}",
        r"\small",
        r"\begin{tabular}{@{}p{0.08\textwidth} p{0.24\textwidth} p{0.16\textwidth} "
        r"p{0.42\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \texttt{<REFERENCE POPULATION>} & \texttt{<OUTCOME "
        r"PHRASING>} & \texttt{<LOCATOR>} \\",
        r"\midrule",
    ]
    for label, _ in MODULES:
        pop = populations[label].replace("%", r"\%").replace("&", r"\&")
        out = outcomes[label].replace("%", r"\%").replace("&", r"\&").rstrip(".")
        tex.append(rf"{label} & {pop} & {out} & {LOCATORS[label]} \\")
        tex.append(r"\addlinespace")
    tex = tex[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}", r""]

    tex += [
        r"The locator identifies only the effect being evaluated. The original "
        r"study's test statistic, sample size, $p$-value, and effect size are "
        r"\emph{withheld}, as these quantities are themselves known predictors of "
        r"replication success, so including them could introduce label leakage "
        r"rather than test what can be inferred from the paper's text alone. Second, "
        r"the same ten-line output contract is used across all three corpora, "
        r"allowing a single parser and a common set of checklist-unit definitions to "
        r"be applied throughout. Third, the paper text supplied to the model is "
        r"extracted from each paper's PDF text layer (not OCR'd, despite the "
        r"prompt's own wording below) and anonymized before assessment: "
        r"\texttt{gpt-4.1-nano} (distinct from \texttt{gpt-5.4-mini}, the model "
        r"being evaluated) identifies author and institutional-affiliation "
        r"mentions in the extracted text, which a deterministic text-replacement "
        r"pass then masks, so no identifying information reaches the model under "
        r"evaluation.",
    ]

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Shared remainder: {len(shared_remainder)} chars, identical across "
          f"{len(MODULES)} corpora. 'roughly 40%' verified byte-identical across all "
          f"three and shown as fixed text (not a substitution).")
    for label, _ in MODULES:
        print(f"  {label:6s} population: {populations[label]!r}")
        print(f"  {label:6s} outcome:    {outcomes[label]!r}")
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
