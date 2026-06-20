#!/bin/bash
# Submit the full simulation pipeline in dependency order.
#
#   install → generate → (simulate_r ∥ simulate_python)
#   logistic_generate → (logistic_simulate_r ∥ logistic_simulate_python)
#
# Run from the project root:
#   bash public/slurm/submit_all.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"

JID_INSTALL=$(sbatch --parsable "$ROOT/public/slurm/run_install.sh")
echo "install:                  $JID_INSTALL"

JID_GEN=$(sbatch --parsable --dependency=afterok:$JID_INSTALL "$ROOT/public/slurm/run_generate.sh")
echo "generate_data:            $JID_GEN"

JID_R=$(sbatch --parsable --dependency=afterok:$JID_GEN "$ROOT/public/slurm/run_simulate_r.sh")
echo "simulate_r:               $JID_R"

JID_PY=$(sbatch --parsable --dependency=afterok:$JID_GEN "$ROOT/public/slurm/run_simulate_python.sh")
echo "simulate_python:          $JID_PY"

JID_LGEN=$(sbatch --parsable --dependency=afterok:$JID_INSTALL "$ROOT/public/slurm/run_logistic_generate.sh")
echo "logistic_generate:        $JID_LGEN"

JID_LR=$(sbatch --parsable --dependency=afterok:$JID_LGEN "$ROOT/public/slurm/run_logistic_simulate_r.sh")
echo "logistic_simulate_r:      $JID_LR"

JID_LPY=$(sbatch --parsable --dependency=afterok:$JID_LGEN "$ROOT/public/slurm/run_logistic_simulate_python.sh")
echo "logistic_simulate_python: $JID_LPY"
