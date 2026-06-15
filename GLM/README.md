# GLM Benchmark

This folder contains the current GLM simulation pipeline for comparing R and
Python implementations of Poisson regression.

## Current Design Grid

The active simulation settings are defined in `01_generate_data.R`.

| Regime | Settings |
| --- | --- |
| `dense` | `(n, p) = (200, 20), (1000, 50), (3000, 50)` |
| `sparse` | `(n, p, delta) = (500, 100, 0.01), (2000, 100, 0.01), (500, 100, 0.05), (2000, 100, 0.05)` |
| `highdim` | `(n, p) = (300, 1000), (500, 500)` |
| `sparse_highdim` | `(n, p, delta) = (300, 1000, 0.01), (500, 500, 0.01)` |

`delta` is the nonzero-entry probability for sparse design matrices.

## Main Files

| File | Purpose |
| --- | --- |
| `01_generate_data.R` | Generate datasets and `data_glm/manifest.csv` |
| `02_run_R_models.R` | Run R methods: `stats::glm`, `glmnet` |
| `03_run_python_models.py` | Run Python methods: sklearn, statsmodels, glum, skglm, spglm |
| `04_summarize_results.R` | Merge R/Python results and compute summaries |
| `05_generate_paper_tables.R` | Generate LaTeX tables in `paper_tables/` |
| `install_dependencies.sh` | Install R and Python dependencies |
| `run_all.sh` | Run the full pipeline |

Old outputs, old notes, and legacy scripts are archived under `deprecated/`.

## Install Dependencies

```bash
python3 -m venv .venv-glm
./install_dependencies.sh
```

The script installs R packages from CRAN and Python packages into `.venv-glm/`.
You can override defaults with environment variables:

```bash
VENV_DIR=.venv-glm CRAN_REPO=https://cloud.r-project.org ./install_dependencies.sh
```

## Run

Quick test:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=1 ./run_all.sh
```

Full run:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=5 ./run_all.sh
```

Parallel run:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=5 GEN_WORKERS=4 R_WORKERS=4 PY_WORKERS=4 ./run_all.sh
```

## Outputs

After a run, the active outputs are:

| Path | Content |
| --- | --- |
| `data_glm/` | Generated `.rds` and `.csv` datasets |
| `results_R.csv` | Raw R method results |
| `results_python.csv` | Raw Python method results |
| `results_all_raw.csv` | Combined raw results |
| `results_all_summary.csv` | Summary statistics |
| `paper_tables/` | Paper-ready LaTeX tables |

