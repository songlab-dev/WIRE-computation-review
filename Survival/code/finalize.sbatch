#!/bin/bash
#SBATCH --job-name=surv_final_audit
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${CODE_DIR:?CODE_DIR is required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$CODE_DIR"
PYTHON_BIN="${SURV_PYTHON:?Set SURV_PYTHON}"
"$PYTHON_BIN" summarize.py "$EXPERIMENT_ROOT" --require-complete
"$PYTHON_BIN" final_audit.py "$EXPERIMENT_ROOT"
"$PYTHON_BIN" collect_slurm.py "$EXPERIMENT_ROOT"
