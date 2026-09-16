# Reproducibility notes

Run all commands from the repository root.  The commands in `README.md` read
only committed inputs and stored model responses; they do not make API
requests.  `uq_bound/dml/dml_theta.py` verifies the SHA-256 of the committed
pre-registered specification, `uq_bound/dml/dml_spec.json`, before fitting.

The checked environment is CPython 3.12.12 with the package versions in
`requirements-lock.txt`.  Create it with:

```sh
uv venv --python 3.12.12 rep_env
uv pip install --python rep_env/bin/python -r requirements-lock.txt
```

For non-interactive Matplotlib rendering, set:

```sh
MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mplconfig" MPL_IGNORE_SYSTEM_FONTS=1
```

Text, CSV, and LaTeX outputs are the reproducibility targets.  PNG/PDF bytes
can differ across operating systems because Matplotlib, FreeType, and the
available fonts participate in rendering; compare the underlying data and
visual appearance rather than image hashes across hosts.

The repository redistributes third-party source data. See the README's
“Data and outcome definitions” section and `NOTICE` for provenance and
licensing scope.
