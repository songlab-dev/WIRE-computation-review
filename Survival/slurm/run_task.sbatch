#!/bin/bash
#SBATCH --job-name=surv_repro
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${TASK:?TASK is required}"
: "${EXPERIMENT_DIR:?EXPERIMENT_DIR is required}"

PYTHON_MODULE="${PYTHON_MODULE:-python3.11-anaconda/2024.02}"
R_MODULE="${R_MODULE:-Rtidyverse/4.5.1}"
PYTHON_ENV="${PYTHON_ENV:-surv_py}"
DEEPSURV_ENV="${DEEPSURV_ENV:-deepsurv_env}"

cd "$PROJECT_DIR"
mkdir -p "$EXPERIMENT_DIR/data" "$EXPERIMENT_DIR/results" logs

run_python() {
  module purge
  module load "$PYTHON_MODULE"
  conda run -n "$PYTHON_ENV" python "$@"
}

run_deepsurv_python() {
  module purge
  module load "$PYTHON_MODULE"
  conda run -n "$DEEPSURV_ENV" python "$@"
}

run_r() {
  module purge
  module load "$R_MODULE"
  Rscript "$@"
}

case "$TASK" in
  generate)
    run_python 01_generate_data.py
    ;;
  python_models)
    run_python 03_run_python_models.py
    ;;
  r_models)
    run_r 02_run_r_models.R
    ;;
  python_deepsurv)
    run_deepsurv_python 05_run_python_deepsurv.py
    ;;
  r_deepsurv)
    export RETICULATE_PYTHON="${RETICULATE_PYTHON:-$HOME/.conda/envs/$DEEPSURV_ENV/bin/python}"
    run_r 04_run_r_deepsurv.R
    ;;
  summarize)
    run_python 06_summarize_results.py
    ;;
  table)
    run_python 07_generate_paper_table.py
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac
