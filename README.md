# Uncertainty in LLM replication prediction

This repository contains the empirical and methodological analyses for
predicting replication outcomes with a zero-shot LLM pipeline, evaluated
against a supervised embedding baseline, and analyzed for calibration,
pretraining contamination, and stage-level uncertainty attribution.

## Experiment overview

Three replication corpora are evaluated: **RPP** (Reproducibility Project:
Psychology, 90 study-effect claims), **CB** (Replication Project: Cancer
Biology, 158 effects), and **SSRP** (21 studies). For each claim, an LLM
(gpt-5.4-mini) reasons zero-shot through a fixed seven-step chain —
extraction, credibility, statistical review, researcher-degrees-of-freedom
concern, theory/context, scoring, and verdict — and emits two label-free
replication-probability estimates: verbalized confidence (**q̄**, the mean
stated probability across repeated runs) and self-consistency vote rate
(**v̄**, the fraction of runs voting "replicable"). Each cell is repeated
R=100 times so run-to-run dispersion can itself serve as an uncertainty
signal, across a reasoning-effort × temperature grid.

This zero-shot pipeline is compared against a supervised **text-embedding
logistic-regression baseline** (nested cross-validation, held-out
predictions) on the same claims. Both are scored for discrimination (AUC),
calibration (ECE), and bootstrap confidence intervals.

Two further analyses probe the pipeline's reliability: a **contamination
probe** that asks the model to recall a study's outcome from its title alone
(no paper text), as a check on pretraining memorization risk — a null result
here is a diagnostic association, not proof the model never saw the study
during training; and a
**stage-attribution analysis** using a cross-fitted, doubly-robust (DML)
estimator that decomposes how much each reasoning stage contributes to
overall prediction uncertainty, validated against theoretical bounds
(Freedman/Bernstein/Anderson-Hoeffding) and simulation.

## Data and outcome definitions

RPP has two distinct roles for source data. Use Altmejd et al.'s export only
to select the 90 `project == "rpp"` and `drop == False` claims and to supply
model covariates. Use the original OSF RPP export, `data/rpp_data.csv`, for
the observed replication outcome — `data/rpp_outcomes.py` enforces this rule.
The sources differ for only `rpp.37`, where Altmejd's derived label
disagrees with the original OSF outcome.

The stable analysis keys are `altmejd_id` for RPP, `(paper_num,
experiment_num, effect_num)` for CB, and `study_num` for SSRP — do not join
across corpora by title alone.

The repository includes the Camerer et al. SSRP D1--D8 source tables in
`data/`, alongside `ssrp_data_cleaned.csv`.  `prompt/build_ssrp_data.py`
rebuilds the cleaned table from the D2 sample-size CSV and D3 results Stata
file; [the SSRP source note](data/ssrp_source/README_SOURCE.md) identifies
the complete source bundle and pins its checksums.

## License and source materials

Original code and documentation are released under the [MIT License](LICENSE).
The license does not grant rights in third-party data, source PDFs, exports,
or other source materials; see [NOTICE](NOTICE) and the source terms for those
materials.

## Repository structure

The package is organized around the pipeline's actual stages:

| Path | Contents |
| --- | --- |
| `data/` | Source exports, cleaned corpus records, and canonical outcome resolution. |
| `prediction/` | Everything about the LLM's zero-shot output and its embedding-baseline comparison: raw stored responses (`LLM_Reasoning/`), LLM score/verdict predictors and reporting (`llm_pred/`), the embedding baseline (`embed_pred/`), and the contamination probe (`contam_probe/`). |
| `uq_bound/` | Theoretical UQ bounds, cross-fitted DML stage attribution, and simulation, plus the shared per-unit scoring/Monte-Carlo utilities the bound calculations use. |
| `tables/` | LaTeX/Markdown table generators and their output. |
| `figures/` | Cross-corpus figure generators and their output. |
| `prompt/` | Shared prompt definitions (used by both the LLM and embedding pipelines) and the CB/SSRP data builders. |

## Reproducing results

### Environment

The checked environment is CPython 3.12.12. Create it with `uv`:

```sh
uv venv --python 3.12.12 rep_env
uv pip install --python rep_env/bin/python -r requirements-lock.txt
```

`requirements.txt` lists direct dependencies. Matplotlib
rendering isn't guaranteed byte-identical across platforms; for figure
generation, set:

```sh
MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mplconfig" MPL_IGNORE_SYSTEM_FONTS=1
```

### Regenerating reported artifacts

All commands below run from the parent repository directory using
`rep_env/bin/python`, read only stored inputs, and make no API requests.
Scripts that submit new model requests (`predict_text_batch.py`,
`predict_embed_lr*.py`'s embedding step via `embed_text_batch.py`,
`predict_contamination_probe.py`) require an `OPENAI_API_KEY` and are not
needed to reproduce the stored analyses.

```bash
for dataset in rpp cb ssrp; do
  rep_env/bin/python prediction/llm_pred/predict_llm_uq.py --dataset "$dataset"
  rep_env/bin/python prediction/llm_pred/predict_llm_verdict.py --dataset "$dataset"
done

for table in main cb cross_corpus; do
  rep_env/bin/python tables/llm_uq_table.py --table "$table"
done

rep_env/bin/python figures/make_roc_figures.py
rep_env/bin/python figures/make_vea_figures.py

rep_env/bin/python uq_bound/dml/dml_theta.py

rep_env/bin/python prediction/contam_probe/make_contamination_report.py
```

See `uq_bound/dml/dml_spec.md` for the DML estimand, folds, nuisance model,
bootstrap, and simulation mechanism.

The commands above cover the main cross-corpus results. A number of other
scripts under `tables/`, `figures/`, `uq_bound/`, and `prediction/` regenerate
additional reported tables/figures (appendix grids, per-corpus predictor
comparisons, the DML attribution table/figure, etc.) the same way -- each is
self-documented with a `Usage:`/`Writes:` docstring header, so its command
line and output path are in the file itself. For example:

```bash
rep_env/bin/python uq_bound/dml/dml_theta_outputs.py
```

writes `uq_bound/dml/table_dml_theta.tex` and the DML attribution figures
from the `dml_theta.py` output above.

### Conventions

The default majority classifier treats `v_bar >= 0.5` as replicable. Most
aggregate outputs use 100 main-task runs per unit and 10 contamination-probe
runs per unit; scripts validate those counts before reporting results.
Batch metadata, error JSONL files, local environment files, and API
credentials are intentionally excluded from version control.
