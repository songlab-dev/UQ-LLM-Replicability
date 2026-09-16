"""Emit the appendix section documenting the zero-shot prompt, generated from
the prompt modules themselves so it cannot drift from what was actually sent.

All three corpora share one prompt. Diffing the modules shows the system
prompt is byte-identical across RPP, CB and SSRP except for a single
clause naming the reference population for the 40% base rate, and the user
turn is identical except for the locator block that names the focal unit.
This script verifies that claim at build time (it asserts the shared parts
really are equal before writing anything) and emits:

  - the shared system prompt, verbatim
  - the shared user-turn template
  - a table of the two corpus-specific slots

Hand-copying a prompt into a paper is exactly how an appendix ends up
describing a prompt nobody ran; generating it keeps the appendix and the
sent text on one source.

Requires \\usepackage{listings} in the paper preamble (lstlisting with
breaklines, so no line of the prompt is silently truncated in the margin).
Deliberately not fancyvrb's Verbatim: that environment's "breaklines" key
only exists if fancyvrb is loaded LAST among any package defining a
same-named environment (moreverb's Verbatim shadows it silently, and the
resulting "Package keyval Error: breaklines undefined" gives no hint that
the fix is a load-order issue rather than a typo) -- hit exactly this when
pasting the generated .tex into the paper. listings is already loaded in
nearly every ML-paper template for code blocks, so this needs one less new
package dependency and one less thing to get the load order right for.

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

# The calibration sentence is the one system-prompt line that varies -- but
# not uniformly: "roughly 40%" itself is BYTE-IDENTICAL across all three
# (verified, not just numerically equal), so it's shown as fixed text rather
# than folded into a per-corpus substitution. What actually varies on either
# side of it: the named reference population before it, and a short outcome
# phrase after it (RPP: "replicate successfully"; CB/SSRP, identical to each
# other: "of tested effects replicated successfully") -- two real
# differences (a population choice and a tense/wording one), not one, so
# they get two placeholders rather than pretending the whole clause is a
# single opaque swap the way the previous version of this appendix did.
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
    """Map the prompt's two non-ASCII characters (em dash x9, en dash x3 --
    checked, nothing else) to ASCII before it goes inside \\begin{lstlisting}.
    listings renders each byte literally with no font-encoding layer of its
    own, so a real em/en dash there needs an explicit `literate=...` mapping
    per engine/font or it fails to compile at all ("Invalid UTF-8 byte
    sequence") -- hit exactly this after switching Verbatim->lstlisting for
    the fancyvrb load-order issue. Substituting here, not in prompt_text*.py:
    those modules must stay byte-identical to what the API actually received,
    but the appendix's typeset copy owes the reader an accurate prompt, not
    an identical byte stream -- and in a monospaced listing "--"/"-" already
    render indistinguishably from the dash glyphs they replace.
    Checked against every prompt_text*.py module (see the docstring above) --
    if a new non-ASCII character shows up later, this raises rather than
    silently drop it."""
    out = text.replace("—", "--").replace("–", "-")
    bad = {ch for ch in out if ord(ch) > 127}
    assert not bad, f"unhandled non-ASCII character(s) in prompt text: {bad!r}"
    return out


def main():
    populations, outcomes, remainders = {}, {}, {}
    for label, mod in MODULES:
        populations[label], outcomes[label], remainders[label] = split_prompt(mod.SYSTEM_PROMPT)

    # The whole point of the appendix is that one prompt covers three corpora;
    # if that stops being true outside the two named substitutions, say so
    # loudly rather than print a false claim.
    shared_remainder = remainders["RPP"]
    for label in remainders:
        assert remainders[label] == shared_remainder, (
            f"{label}'s system prompt differs from RPP's outside the two calibration "
            f"substitutions -- the 'one shared prompt' claim in this appendix is no longer true.")

    user_tmpl = prompt_text.build_user_message.__doc__  # sanity: modules present
    assert user_tmpl

    # ASCII placeholder markers, not <angle-bracket Unicode>: these three
    # tokens are my own authorial addition to mark substitution points, not
    # part of the sent prompt, so there's no accuracy cost to keeping them
    # plain-ASCII -- and lstlisting/pdflatex can choke on non-ASCII glyphs
    # inside a verbatim-like environment without a matching fontenc/literate
    # setup, which is exactly the kind of silent portability trap this
    # appendix should not add on top of the prompt text's own real em-dash.
    # "roughly 40%" is shown as literal fixed text, not a third placeholder --
    # it's byte-identical across all three corpora (verified in split_prompt's
    # caller data, not just numerically equal), so there's nothing to
    # substitute there. What genuinely varies is on either side of it.
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
