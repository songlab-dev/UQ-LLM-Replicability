# Ablation experiments

Two RPP experiments (90 claims, R = 100), each addressing one concern:

- **Action-intervened chain of actions**: Step 3 of the prompt is replaced so that it scores the last digit
  of N ("Spurious stats", gpt-5.4-mini), showing how intervening on one action changes its stage attribution.
- **Generalizability to an open-weight model**: baseline and Spurious stats rerun with Qwen3.8-Flash
  (Together AI).

| Path | Contents |
| --- | --- |
| `action_intervention/` | gpt-5.4-mini runs (`run_ablation.py`) and VEA, bound and θ̂ comparisons. |
| `qwen_experiments/` | Qwen runs (`run_qwen.py`) and gpt vs Qwen comparisons (`compare_vea.py`). |
| `doc/` | LaTeX classification tables. |

Outputs are in each folder's `results/`. API keys come from the repo-root `.env`. gpt-5.4-mini runs at its
fixed default temperature (1); `temp0.7` in its file names is only a label.
