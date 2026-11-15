#!/bin/bash -l
# afterany permits a diagnostic even when an array element is cancelled/OOM.
# CPU models may differ; the merger verifies completeness and other environments.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Use slurm/submit.sh}/slurm/common.sh"
glm_require_python
glm_header
run_root="${1:?Expected the absolute planned run directory}"
cd "$GLM_PROJECT_ROOT"
"$GLM_PYTHON" slurm_cpu_relaxed.py merge --out "$run_root"
export_args=(--results "$run_root/merged")
profile="$("$GLM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["profile"])' "$run_root/merged/run.json")"
if [[ "$profile" != full ]]; then export_args+=(--allow-validation); fi
"$GLM_PYTHON" export_tables.py "${export_args[@]}"
printf 'Public tables: %s\n' "$run_root/merged/public_tables"
