# Source

Downloaded from the OSF project for Altmejd et al. (2019), "Predicting the
replicability of social science lab experiments" (PLOS ONE), on 2026-07-30:

- Project: https://osf.io/4fn73/  (DOI: 10.17605/OSF.IO/4fn73)
- `data.csv` <- https://osf.io/download/vfhcw/  (their `data/data.csv`)
- `variables.tsv` <- https://osf.io/download/6rqtc/  (their `data/variables.tsv`)

`data.csv` pools all four of their replication projects (one row per
experiment-replication pair; `project` column is one of `rpp`, `ml1`, `ml3`,
`ee`). `variables.tsv` is their variable dictionary (name, display name,
description) -- the machine-readable equivalent of the paper's S1 Table.

Not re-downloaded here: `author_data.csv`, `authors.csv` (per-author detail,
not needed once `data.csv`'s `authors_male.*` / `author_citations_max.*`
columns are used), and their `code/` folder (R, using `caret`+`randomForest`;
see their `nested_cv_prepare_data.R` and `nested_cv_class.R` for the exact
formula and hyperparameter grid this repo's `predict_rf_replicability.py`
mirrors in Python).
