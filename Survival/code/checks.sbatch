#!/bin/bash
#SBATCH --job-name=surv_checks
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module purge
module load "${R_MODULE:-Rtidyverse/4.5.1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE CUDA_VISIBLE_DEVICES="" RF_CORES=1
cd "$CODE_DIR"
mkdir -p "$EXPERIMENT_ROOT/audit"
Rscript native_checks.R "$EXPERIMENT_ROOT/audit"
"${SURV_PYTHON:?Set SURV_PYTHON}" native_checks.py "$EXPERIMENT_ROOT/audit"
Rscript -e 'capture.output(sessionInfo(),file=file.path(Sys.getenv("EXPERIMENT_ROOT"),"audit","R_sessionInfo.txt")); capture.output(lapply(c("survival","glmnet","randomForestSRC","gbm","survivalmodels","reticulate"),packageDescription),file=file.path(Sys.getenv("EXPERIMENT_ROOT"),"audit","R_package_descriptions.txt"))'
"${SURV_PYTHON:?Set SURV_PYTHON}" -m pip freeze > "$EXPERIMENT_ROOT/audit/python_main_freeze.txt"
"${DEEP_PYTHON:?Set DEEP_PYTHON}" -m pip freeze > "$EXPERIMENT_ROOT/audit/python_deep_freeze.txt"
lscpu > "$EXPERIMENT_ROOT/audit/lscpu.txt"
env | grep -E '^(SLURM_|OMP_|MKL_|OPENBLAS_|NUMEXPR_|BLIS_|VECLIB_|RF_CORES=|CUDA_VISIBLE_DEVICES=)' | sort > "$EXPERIMENT_ROOT/audit/job_environment.txt"
