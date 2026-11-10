"""Statistics-step ablation on RPP: Step 3 scores a spurious feature (arm spurious_stats).

Reuses prediction/llm_pred/predict_text_batch.py unchanged. The only
differences from the main run are the system prompt (one passage swapped)
and the output location (results/<arm>/). baseline_rerun is the unchanged
baseline prompt (same-week drift control).

    MODEL=gpt-5.4-mini REASONING_EFFORT=high \
      python ablation/action_intervention/run_ablation.py --arm spurious_stats --mode submit
    python ablation/action_intervention/run_ablation.py --arm spurious_stats --mode status
    python ablation/action_intervention/run_ablation.py --arm spurious_stats --mode collect
    python ablation/action_intervention/run_ablation.py --arm spurious_stats --print-prompt
"""
import argparse
import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "prediction" / "llm_pred"))
import predict_text_batch as ptb  # noqa: E402  (also puts prompt/ on sys.path)

BASE_PROMPT = ptb.prompt_text.SYSTEM_PROMPT
A0_MODEL, A0_EFFORT, A0_RUNS = "gpt-5.4-mini", "high", 100

# ── Prompt variant: swap one passage of the baseline prompt ──

# Baseline Step 3, the passage the variant replaces. Step 1, the output block and every other step stay
# byte-identical.
STEP3_BASE = (
    "3. Statistics: judge statistical power, multiplicity, model appropriateness, and "
    "effect-size plausibility; summarize as a single [0,1] statistical-soundness score. "
    "Small N (<50), large effect sizes (r>.5), or p just above .01 should lower this score.\n"
)
# spurious_stats: Step 3 scores an arbitrary feature (last digit of N) and is told the usual
# statistical cues are uninformative, so the score carries no information about soundness.
STEP3_SPURIOUS = (
    "3. Statistics: base the [0,1] statistical-soundness score on the last digit of the sample "
    "size N (0 → 0.0, 9 → 0.9); power, multiplicity, model appropriateness, effect sizes, and "
    "p-values are not informative "
    "for replication in this task.\n"
)
# arm -> [(baseline passage, replacement), ...]. The earlier arms (n_only, n_only_nohint, claim_only,
# p_only, no_stats, stats_jargon) are retired; their code is in archive/archive/run_ablation_retired_arms.py.
VARIANTS = {
    "spurious_stats": [(STEP3_BASE, STEP3_SPURIOUS)],
}



def swap(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Passage must occur exactly once in the prompt: {old[:60]!r}")
    return text.replace(old, new)


ARMS = ["baseline_rerun", *sorted(VARIANTS)]  # baseline_rerun = unchanged A0 prompt (drift control)


def swaps(arm):
    """(baseline passage, replacement) pairs that turn the baseline prompt into this arm's."""
    return [] if arm == "baseline_rerun" else VARIANTS[arm]


def build_prompt(arm):
    pairs = swaps(arm)
    prompt = BASE_PROMPT
    for old, new in pairs:
        prompt = swap(prompt, old, new)
    # Undoing every swap must give back the baseline: nothing else changed.
    undone = prompt
    for old, new in reversed(pairs):
        undone = swap(undone, new, old)
    assert undone == BASE_PROMPT
    return prompt


# ── Run ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--print-prompt", action="store_true",
                        help="print the prompt diff vs. the baseline and exit")
    parser.add_argument("--allow-other-model", action="store_true",
                        help=f"skip the check that MODEL/REASONING_EFFORT match A0 "
                             f"({A0_MODEL}, {A0_EFFORT})")
    args, rest = parser.parse_known_args()
    if "--corpus" in rest:
        parser.error("this ablation is RPP-only; do not pass --corpus")
    return args, rest


def parse_batch_args(rest):
    """predict_text_batch's own CLI, forced to RPP with A0's run count by default."""
    if not any(a == "--runs" or a.startswith("--runs=") for a in rest):
        rest = [*rest, "--runs", str(A0_RUNS)]
    sys.argv = [sys.argv[0], "--corpus", "rpp", *rest]
    return ptb.parse_args()


def print_prompt_diff(prompt):
    diff = difflib.unified_diff(BASE_PROMPT.splitlines(), prompt.splitlines(),
                                "baseline", "ablation", lineterm="", n=1)
    print("\n".join(diff))


def make_job(arm, batch_args):
    # BatchJob.__init__ creates prediction/LLM_Reasoning/RPP as a side effect;
    # remove it again if it was empty and did not exist before.
    default_dir = ptb.PACKAGE_DIR / "prediction" / "LLM_Reasoning" / "RPP"
    existed = default_dir.exists()
    job = ptb.BatchJob(batch_args)
    if not existed and default_dir.is_dir() and not any(default_dir.iterdir()):
        default_dir.rmdir()
        if not any(default_dir.parent.iterdir()):
            default_dir.parent.rmdir()

    # Redirect BOTH outputs: the default meta_file would overwrite the main run's.
    job.out_dir = HERE / "results" / arm
    job.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{job.effort if job.reasoning else 'openai'}_temp{batch_args.temperature}"
    job.out_csv = job.out_dir / f"text_predictions_{tag}.csv"
    job.meta_file = job.out_dir / f"batch_meta_{tag}.json"
    return job


def main():
    args, rest = parse_args()
    prompt = build_prompt(args.arm)
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if args.print_prompt:
        print_prompt_diff(prompt)
        print(f"\nsha256: {sha}")
        return

    batch_args = parse_batch_args(rest)
    ptb.prompt_text.SYSTEM_PROMPT = prompt  # read by build_messages at call time
    job = make_job(args.arm, batch_args)
    if not args.allow_other_model and (job.model, job.effort) != (A0_MODEL, A0_EFFORT):
        raise SystemExit(f"MODEL={job.model}, REASONING_EFFORT={job.effort} do not match A0 "
                         f"({A0_MODEL}, {A0_EFFORT}). Set them, or pass --allow-other-model.")

    if batch_args.mode == "submit":
        (job.out_dir / "system_prompt.txt").write_text(prompt)
        print(f"[run_ablation] arm={args.arm} prompt sha256={sha}")
    job.run()
    if batch_args.mode == "submit":
        print(f"(Use run_ablation.py --arm {args.arm} for status/collect, "
              "not the command printed above.)")


if __name__ == "__main__":
    main()
