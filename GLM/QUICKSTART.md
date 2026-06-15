# Quickstart

## 1. Install dependencies

```bash
python3 -m venv .venv-glm
./install_dependencies.sh
```

## 2. Run the pipeline

From this folder:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=1 ./run_all.sh
```

Use `REPS=5` for the full run:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=5 ./run_all.sh
```

Use workers if you want a faster run:

```bash
PYTHON_BIN=.venv-glm/bin/python REPS=5 GEN_WORKERS=4 R_WORKERS=4 PY_WORKERS=4 ./run_all.sh
```

## 3. What runs

The pipeline has six steps:

1. Clean active output folders/files.
2. Generate datasets in `data_glm/`.
3. Run R methods.
4. Run Python methods.
5. Merge and summarize results.
6. Generate LaTeX tables.

The active regimes are:

| Regime | Settings |
| --- | --- |
| `dense` | `(200, 20)`, `(1000, 50)`, `(3000, 50)` |
| `sparse` | `(500, 100, 0.01)`, `(2000, 100, 0.01)`, `(500, 100, 0.05)`, `(2000, 100, 0.05)` |
| `highdim` | `(300, 1000)`, `(500, 500)` |
| `sparse_highdim` | `(300, 1000, 0.01)`, `(500, 500, 0.01)` |

## 4. Outputs

| File or folder | Meaning |
| --- | --- |
| `data_glm/manifest.csv` | Dataset index used by R and Python scripts |
| `results_R.csv` | R results |
| `results_python.csv` | Python results |
| `results_all_raw.csv` | Combined raw results |
| `results_all_summary.csv` | Aggregated results |
| `paper_tables/all_tables.tex` | Master LaTeX file |

Individual table files:

```text
paper_tables/table_dense_regime.tex
paper_tables/table_sparse_regime.tex
paper_tables/table_highdim_regime.tex
paper_tables/table_sparse_highdim_regime.tex
```

## 5. Run one regime after data generation

```bash
Rscript 01_generate_data.R --reps 1
Rscript 02_run_R_models.R --regime dense
PYTHON_BIN=.venv-glm/bin/python
$PYTHON_BIN 03_run_python_models.py --regime dense
```

