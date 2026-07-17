# Cross-Language Survival-Analysis Benchmark

Reproduction code and checked outputs for the survival-analysis simulation comparing representative R and Python implementations. R and Python use the same generated datasets and held-out test observations in every replication.

## Quick Start

Run all commands from the repository root. Two Conda environments are used because the non-neural models and DeepSurv require different compatible dependency versions.

```bash
conda env create -f environment/python_models.yml
conda env create -f environment/deepsurv.yml
Rscript environment/install_r_packages.R
```

Verify the complete pipeline with one replication:

```bash
REPS=1 ./run_all.sh main
```

Run the 50-replication main and sensitivity analyses:

```bash
./run_all.sh main
./run_all.sh sensitivity
```

New runs are written under `outputs/`; the checked manuscript results under `results/` are not overwritten. The generated datasets are deterministic from the recorded seeds and are not included because a full run requires several gigabytes of storage.

## Run on Slurm

For a full cluster run, use the Slurm launcher instead of running on a login node:

```bash
export SLURM_ACCOUNT=your_allocation
REPS=1 ./slurm/submit.sh main       # one-replication check
./slurm/submit.sh main              # n=500
./slurm/submit.sh sensitivity       # n=1000 and n=1500
```

The launcher loads the recorded Great Lakes modules, runs the four model-fitting jobs in parallel, and starts summarization only after all four succeed. No Jupyter or interactive job is required. Set `PYTHON_ENV`, `DEEPSURV_ENV`, `PYTHON_MODULE`, or `R_MODULE` only when using different local names.

## Study Design

- 50 independent replications.
- Main analysis: `n=500`; sensitivity analysis: `n=1000` and `n=1500`.
- Approximately 30% independent censoring.
- Gaussian AR(1) covariates with correlation `rho=0.30`.
- Weibull proportional hazards with cumulative baseline hazard `H0(t)=0.1*t^1.5`.
- A shared 60%/20%/20% training/validation/test split.
- Evaluation by IPCW-integrated Brier score (IBS), C-index, and fitting time.

The two simulation settings are:

1. **High-dimensional sparse Cox:** `p=500`, `s=50`; nonzero coefficient signs alternate and their magnitudes are proportional to `0.93^(j-1)`.
2. **Smooth nonlinear survival:** `p=50`, with standardized risk score

```text
eta(X) = 0.4 X1 + 0.8 sin(X2) + 0.8 (X3^2 - 1)
         + 0.7 tanh(X4 + X5) + 0.5 exp(-X6^2).
```

The high-dimensional setting compares penalized Cox, random survival forest (RSF), boosting, and DeepSurv. The nonlinear setting compares CoxPH, RSF, boosting, and DeepSurv.

Model settings are fixed across replications: penalized Cox selects from 100 L1-penalized solutions by validation partial likelihood; RSF uses 200 trees and terminal-node size 15; boosting uses 300 trees, learning rate 0.05, depth 3, and minimum node size 15; DeepSurv uses one 32-unit ReLU layer, learning rate 0.001, batch size 128, at most 150 epochs, and early-stopping patience 20.

The predefined validation split is used directly for penalized Cox and Python DeepSurv. The R `survivalmodels` interface instead creates an internal 25% validation subset from the combined training and validation pool, preserving the same 60%/20% effective proportions. This package-interface difference is retained intentionally.

All implementations use the original training split to estimate the censoring distribution and define the follow-up support for IBS. Both DeepSurv implementations set Python, NumPy, and PyTorch seeds for every replication; the R interface also sets the corresponding R seed.

## Pipeline

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `01_generate_data.py` | Generate shared datasets, splits, seeds, and metadata |
| 2 | `02_run_r_models.R` | Fit R Cox/Coxnet, RSF, and boosting models |
| 3 | `03_run_python_models.py` | Fit Python Cox/Coxnet, RSF, and boosting models |
| 4 | `04_run_r_deepsurv.R` | Fit DeepSurv through R `survivalmodels` |
| 5 | `05_run_python_deepsurv.py` | Fit DeepSurv through Python `pycox` |
| 6 | `06_summarize_results.py` | Require successful fits, combine results, and compute summaries |
| 7 | `07_generate_paper_table.py` | Generate the manuscript LaTeX table |

`run_all.sh` executes these steps sequentially on one machine. `slurm/submit.sh` submits the same dependency-aware workflow to a Slurm cluster.

## Environments

The environment files record the completed Great Lakes run:

- `surv_py`: Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.2, scikit-learn 1.7.2, scikit-survival 0.25.0, and lifelines 0.30.0.
- `deepsurv_env`: Python 3.9.25, NumPy 1.20.3, pandas 1.5.3, SciPy 1.10.1, scikit-survival 0.23.1, PyTorch 2.8.0, torchtuples 0.2.2, and pycox 0.3.0.
- R 4.5.1: survival 3.8-6, glmnet 4.1-10, randomForestSRC 3.6.2, gbm 2.2.2, survivalmodels 0.1.191, and reticulate 1.46.0.

Great Lakes used modules `python3.11-anaconda/2024.02` and `Rtidyverse/4.5.1`. The Conda environments supply their own Python versions, and R DeepSurv is directed to `deepsurv_env` through `RETICULATE_PYTHON`.

## Outputs

Each run creates four implementation-level result files, then produces:

| File | Content |
| --- | --- |
| `all_results.csv` | Replication-level results from all implementations |
| `summary.csv` | Mean and standard deviation by setting, model, and language |
| `manuscript_table.tex` | LaTeX table generated from `summary.csv` |

Checked reference outputs are stored in `results/main_n500/` and `results/sensitivity_n1000_n1500/`. The main table contains 800 successful fits; the sensitivity results contain 1,600 successful fits. Small numerical and runtime differences can still occur across hardware and numerical-library versions, although seeds and model settings are fixed.
