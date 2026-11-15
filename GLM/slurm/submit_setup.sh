#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$GLM_PROJECT_ROOT"
command -v sbatch >/dev/null
mkdir -p "$GLM_PROJECT_ROOT/slurm/setup-logs"
sbatch --parsable --job-name=wire-glm-setup --account="$GLM_ACCOUNT" --partition="$GLM_PARTITION" \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=02:00:00 \
  --chdir="$GLM_PROJECT_ROOT" --export=ALL \
  --output="$GLM_PROJECT_ROOT/slurm/setup-logs/setup-%j.out" \
  --error="$GLM_PROJECT_ROOT/slurm/setup-logs/setup-%j.err" \
  "$GLM_PROJECT_ROOT/slurm/setup.sbatch"
