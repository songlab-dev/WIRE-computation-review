#!/bin/bash -l
# One element = one scenario/seed; all competing methods run sequentially.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Use slurm/submit.sh}/slurm/common.sh"
glm_require_python
glm_header
if declare -F glm_check_node >/dev/null; then glm_check_node; fi
run_root="${1:?Expected the absolute planned run directory}"
array_index="${SLURM_ARRAY_TASK_ID:?Submit this script through slurm/submit.sh}"
glm_scratch="$(mktemp -d "${TMPDIR:-/tmp}/wire-glm.XXXXXX")"
glm_cleanup() {
  if [[ -d "$glm_scratch" && "${glm_scratch##*/}" == wire-glm.* ]]; then
    rm -rf -- "$glm_scratch"
  fi
}
trap glm_cleanup EXIT
cd "$GLM_PROJECT_ROOT"
srun --cpu-bind=cores "$GLM_PYTHON" slurm_cpu_relaxed.py run-task \
  --out "$run_root" --index "$array_index" --work-dir "$glm_scratch"
