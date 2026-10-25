#!/bin/bash
#SBATCH --job-name=surv_revision
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${CODE_DIR:?CODE_DIR is required}"
module purge
module load "${R_MODULE:-Rtidyverse/4.5.1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE CUDA_VISIBLE_DEVICES=""
cd "$CODE_DIR"
PYTHON_BIN="${SURV_PYTHON:?Set SURV_PYTHON}"
if [[ "${PHASE:-formal}" == prepare ]]; then
  "$PYTHON_BIN" prepare.py "$EXPERIMENT_ROOT"
else
  "$PYTHON_BIN" run_task.py "$EXPERIMENT_ROOT" "$SLURM_ARRAY_TASK_ID" --phase "${PHASE:-formal}" --timeout 3300
fi
