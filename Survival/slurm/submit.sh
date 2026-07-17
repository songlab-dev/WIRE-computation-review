#!/bin/bash
set -euo pipefail

MODE="${1:-main}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_ACCOUNT="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your cluster allocation}"

case "$MODE" in
  main)
    EXPERIMENT_DIR="outputs/main_n500"
    SAMPLE_SIZES="500"
    ;;
  sensitivity)
    EXPERIMENT_DIR="outputs/sensitivity_n1000_n1500"
    SAMPLE_SIZES="1000,1500"
    ;;
  *)
    echo "Usage: $0 [main|sensitivity]" >&2
    exit 2
    ;;
esac

export PROJECT_DIR EXPERIMENT_DIR SAMPLE_SIZES
export N_REPLICATIONS="${REPS:-50}"
export TARGET_CENSORING=0.30

cd "$PROJECT_DIR"
mkdir -p logs "$EXPERIMENT_DIR/results"

submit() {
  sbatch --parsable --account="$SLURM_ACCOUNT" "$@" slurm/run_task.sbatch
}

generate=$(submit --time=00:30:00 --export=ALL,TASK=generate)
python_models=$(submit --time=08:00:00 --dependency=afterok:"$generate" --export=ALL,TASK=python_models)
r_models=$(submit --time=10:00:00 --dependency=afterok:"$generate" --export=ALL,TASK=r_models)
python_deepsurv=$(submit --time=12:00:00 --dependency=afterok:"$generate" --export=ALL,TASK=python_deepsurv)
r_deepsurv=$(submit --time=12:00:00 --dependency=afterok:"$generate" --export=ALL,TASK=r_deepsurv)
summarize=$(submit --time=00:20:00 --dependency=afterok:"$python_models:$r_models:$python_deepsurv:$r_deepsurv" --export=ALL,TASK=summarize)
table=$(submit --time=00:10:00 --dependency=afterok:"$summarize" --export=ALL,TASK=table)

job_file="logs/${MODE}_jobs.txt"
printf '%s\n' \
  "generate=$generate" \
  "python_models=$python_models" \
  "r_models=$r_models" \
  "python_deepsurv=$python_deepsurv" \
  "r_deepsurv=$r_deepsurv" \
  "summarize=$summarize" \
  "table=$table" > "$job_file"

echo "Submitted $MODE experiment. Job IDs are in $job_file."
