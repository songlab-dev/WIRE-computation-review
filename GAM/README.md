# Generalized Additive Models

Code to reproduce the simulation and benchmark results from the paper for GAM section.

All scripts read/write relative to the **project root** (`data/`, `results/`).

## Requirements

**R packages** (mgcv, gamlss, gamlss.dist, bench):
```bash
Rscript public/install_packages.R
```

**Python packages** (statsmodels, pyGAM, gamdist; Python >= 3.11)
```bash
pip install -r public/requirements.txt
```

## Accuracy comparison

Run from the project root:

```bash
# 1. Generate datasets (500 replicates each)
Rscript public/generate_data.R
Rscript public/generate_logistic_data.R

# 2. Run simulations (R and Python can run in parallel)
Rscript public/simulate_r.R
python  public/simulate_python.py
Rscript public/simulate_logistic_r.R
python  public/simulate_logistic_python.py

# 3. Build the table
python public/make_accuracy_table.py
```

Each simulation script runs every (task, n) cell by default, writes one chunk
per replicate to `results/sim_chunks/` (finished reps are skipped on rerun),
and averages them into `results/{r,python}_<task>_n<n>.csv`. Cells whose summary
CSV already exists are skipped. To run a single cell, or to split one across
jobs:

```bash
python public/simulate_python.py global 500                        # one cell
SIM_REP_START=1 SIM_REP_END=100 python public/simulate_python.py additive_p50 500   # one shard
python public/simulate_python.py --merge                           # average chunks into the CSVs
```

Sharded runs only write chunks, so run `--merge` afterwards (any of the four
scripts; `submit_simulations.sh` does this automatically).

## Scalability comparison

Both benchmarks fit the same datasets: R draws them once per cell, then R and
Python read them. Each fit runs in its own worker process, and results are
checkpointed after every rep.

```bash
# 1. Draw the shared datasets for a cell
Rscript public/benchmark_r.R --gen additive_p10 100000

# 2. Run R and Python on that cell (can run in parallel)
Rscript public/benchmark_r.R additive_p10 100000
python  public/benchmark_python.py additive_p10 100000

# 3. Build the table, after all cells are done
python public/make_scalability_table.py
```

Scenarios are `additive_p{01,05,10,50,100}` and `logistic_p{01,05,10,50,100}` (the SLURM grid uses p >= 5), with
n = 1000 or 100000. Environment variables (see the script headers) restrict a run to a
rep range (`BENCH_REP_START/END`) or a subset of Python methods (`BENCH_METHODS`).

## SLURM (HPC cluster)

Set `ACCOUNT` (and adjust `--partition`, `module load` and the `wire/bin/activate`
line if your cluster differs), then submit from the project root:

```bash
# One-time setup: install R and Python packages
sbatch --account=$ACCOUNT public/slurm/run_install.sh

# Accuracy simulations (R and Python, Gaussian and logistic), sharded, with a
# merge job per script. GENERATE=1 also generates the datasets first.
GENERATE=1 ACCOUNT=$ACCOUNT bash public/slurm/submit_simulations.sh

# Scalability benchmark grid (scenario x n cells, sharded by rep range)
ACCOUNT=$ACCOUNT bash public/slurm/submit_benchmark_grid.sh

# Tables, after all jobs finish
python public/make_accuracy_table.py
python public/make_scalability_table.py
```

`run_generate.sh` and `run_logistic_generate.sh` are called by
`submit_simulations.sh` when `GENERATE=1`.
Log files are written to `logs/` in the project root.
