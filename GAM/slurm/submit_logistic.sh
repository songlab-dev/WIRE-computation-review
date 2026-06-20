#!/bin/bash
# Submit only the logistic pipeline:
#   logistic_generate → (logistic_simulate_r ∥ logistic_simulate_python)
#
# Run from the project root:
#   bash public/slurm/submit_logistic.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"

JID_LGEN=$(sbatch --parsable "$ROOT/public/slurm/run_logistic_generate.sh")
echo "logistic_generate:        $JID_LGEN"

JID_LR=$(sbatch --parsable --dependency=afterok:$JID_LGEN "$ROOT/public/slurm/run_logistic_simulate_r.sh")
echo "logistic_simulate_r:      $JID_LR"

JID_LPY=$(sbatch --parsable --dependency=afterok:$JID_LGEN "$ROOT/public/slurm/run_logistic_simulate_python.sh")
echo "logistic_simulate_python: $JID_LPY"
