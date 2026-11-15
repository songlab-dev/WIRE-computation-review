#!/bin/bash -l
# Resources/account are supplied by submit_setup.sh, not hardcoded here.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Use slurm/submit_setup.sh}/slurm/common.sh"
cd "$GLM_PROJECT_ROOT"
glm_header
"$GLM_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Load Python 3.12 before setup"'
Rscript -e 'stopifnot(getRversion() >= "4.3.0")'
mkdir -p "$GLM_R_LIBS"
if [[ "$GLM_PYTHON" != "$GLM_PROJECT_ROOT/.venv/bin/python" ]]; then
  printf '%s\n' 'Setup creates project .venv; install dependencies yourself if GLM_PYTHON selects a custom environment.' >&2
  exit 2
fi
export PYTHON_FOR_VENV="$GLM_BOOTSTRAP_PYTHON"
bash install_dependencies.sh
"$GLM_PYTHON" run_benchmark.py --config configs/smoke.json --preflight
printf '%s\n' 'Setup passed. Next: bash slurm/submit.sh smoke'
