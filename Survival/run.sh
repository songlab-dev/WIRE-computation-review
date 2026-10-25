#!/bin/bash
# Submit each stage from the Survival directory.
set -euo pipefail
cd "$(dirname "$0")"
[[ ! -f settings.local.sh ]] || source settings.local.sh

export SURV_PYTHON="${SURV_PYTHON:-$HOME/.conda/envs/surv_py/bin/python}"
export DEEP_PYTHON="${DEEP_PYTHON:-$HOME/.conda/envs/deepsurv_env/bin/python}"
export CODE_DIR="$PWD/code"
export CONCURRENCY="${CONCURRENCY:-16}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
command="${1:-help}"

if [[ "$command" == "help" ]]; then
  echo "Usage: ./run.sh {prepare|pilot|validate|run|retry|summary|verify-results}"
  exit 0
fi
if [[ "$command" == "verify-results" ]]; then
  "$SURV_PYTHON" code/verify_summary.py "$PWD/results"
  exit 0
fi
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to a new server directory}"
export EXPERIMENT_ROOT
mkdir -p "$CODE_DIR/logs" "$EXPERIMENT_ROOT/audit"
cd "$CODE_DIR"

submit_stage() {
  : "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to your allocation}"
  sbatch --parsable --export=ALL "$@"
}

case "$command" in
  prepare)
    export PHASE=prepare
    submit_stage job.sbatch
    ;;
  pilot)
    test -f "$EXPERIMENT_ROOT/READY.json"
    submit_stage checks.sbatch
    indices=$("$SURV_PYTHON" - "$EXPERIMENT_ROOT/tasks.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
selected = [
    r["task_id"] for r in rows
    if int(r["replication"]) == 1
    or (r["model"] == "DeepSurv" and int(r["n"]) == 1500
        and int(r["replication"]) <= 5)
]
print(",".join(selected))
PY
)
    "$SURV_PYTHON" submit.py "$EXPERIMENT_ROOT" --phase pilot \
      --indices "$indices" --concurrency "$CONCURRENCY"
    ;;
  validate)
    "$SURV_PYTHON" validate_pilot.py "$EXPERIMENT_ROOT" --concurrency "$CONCURRENCY"
    ;;
  run)
    : "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to your allocation}"
    "$SURV_PYTHON" submit.py "$EXPERIMENT_ROOT" --concurrency "$CONCURRENCY"
    ;;
  retry)
    : "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to your allocation}"
    "$SURV_PYTHON" summarize.py "$EXPERIMENT_ROOT"
    "$SURV_PYTHON" submit.py "$EXPERIMENT_ROOT" --retry --concurrency "$CONCURRENCY"
    ;;
  summary)
    submit_stage finalize.sbatch
    ;;
  *)
    echo "Unknown stage: $command" >&2
    exit 2
    ;;
esac
