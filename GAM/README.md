# Generalized Additive Models

Code to reproduce the simulation and benchmark results from the paper for GAM section.

All scripts read/write relative to the **project root** (`data/`, `results/`).

## Requirements

**R packages** (mgcv, gamlss, gamlss.dist):
```bash
Rscript public/install_packages.R
```

**Python packages** (statsmodel, pyGAM)
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

## Scalability comparison

```bash
# Run R and Python benchmarks (can run in parallel; each checkpoints per rep)
Rscript public/benchmark_r.R
python  public/benchmark_python.py

# Build the table
python public/make_scalability_table.py
```

Each benchmark script accepts optional `<scenario> <n>` arguments to run a
single cell (useful for parallel job arrays):
```bash
Rscript public/benchmark_r.R additive_p10 100000
python  public/benchmark_python.py logistic_p50 1000
```

## SLURM (HPC cluster)

Edit `--account` and `--partition` in `slurm/*.sh` to match your cluster,
then submit from the project root:

```bash
# Full simulation pipeline (install → generate → simulate)
bash public/slurm/submit_all.sh

# Logistic pipeline only
bash public/slurm/submit_logistic.sh

# Benchmark grid (one job per scenario × n cell)
bash public/slurm/submit_benchmark_grid.sh
```

Log files are written to `logs/` in the project root.
The Python environment is activated from `wire/bin/activate`; adjust the
`source` line in the relevant scripts if you use a different virtual
environment or conda.
