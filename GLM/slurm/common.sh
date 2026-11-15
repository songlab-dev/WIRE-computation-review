#!/usr/bin/env bash
set -euo pipefail
GLM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GLM_PROJECT_ROOT
GLM_MODULES=()
if [[ ! -f "$GLM_PROJECT_ROOT/slurm/site_config.sh" ]]; then
  printf '%s\n' 'Copy slurm/site_config.example.sh to slurm/site_config.sh; set your account and module names.' >&2
  exit 2
fi
source "$GLM_PROJECT_ROOT/slurm/site_config.sh"
: "${GLM_ACCOUNT:?Set GLM_ACCOUNT in slurm/site_config.sh}"
if [[ "$GLM_ACCOUNT" == CHANGE_ME ]]; then
  printf '%s\n' 'Set your authorized Great Lakes billing account; no default account is assumed.' >&2
  exit 2
fi
GLM_PARTITION="${GLM_PARTITION:-standard}"
GLM_BOOTSTRAP_PYTHON="${GLM_BOOTSTRAP_PYTHON:-python3}"
GLM_PYTHON="${GLM_PYTHON:-$GLM_PROJECT_ROOT/.venv/bin/python}"
GLM_R_LIBS="${GLM_R_LIBS:-$GLM_PROJECT_ROOT/.R-library}"
GLM_CPUS="${GLM_CPUS:-2}"
GLM_MEM="${GLM_MEM:-8G}"
GLM_TIME="${GLM_TIME:-24:00:00}"
GLM_MAX_PARALLEL="${GLM_MAX_PARALLEL:-10}"
GLM_CONSTRAINT="${GLM_CONSTRAINT:-}"
GLM_ALLOWED_NODES="${GLM_ALLOWED_NODES:-}"
GLM_EXPECTED_CPU="${GLM_EXPECTED_CPU:-}"
GLM_EXCLUSIVE_NODE="${GLM_EXCLUSIVE_NODE:-0}"
if [[ ! "$GLM_CPUS" =~ ^[0-9]+$ || "$GLM_CPUS" -lt 2 || ! "$GLM_MAX_PARALLEL" =~ ^[0-9]+$ || "$GLM_MAX_PARALLEL" -lt 1 ]]; then
  printf '%s\n' 'GLM_CPUS must be >=2; GLM_MAX_PARALLEL must be >=1.' >&2
  exit 2
fi
if [[ -n "${GLM_MODULES[*]:-}" ]]; then
  if ! type module &>/dev/null; then printf '%s\n' 'The module command is not initialized in this shell.' >&2; exit 2; fi
  module purge
  module load "${GLM_MODULES[@]}"
fi
export R_LIBS_USER="$GLM_R_LIBS"
export GLM_PIN_R="${GLM_PIN_R:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GLM_ALLOWED_NODES GLM_EXPECTED_CPU

glm_require_python() {
  if [[ ! -x "$GLM_PYTHON" ]]; then
    printf '%s\n' 'Python environment is missing; submit and complete slurm/submit_setup.sh first.' >&2
    exit 2
  fi
}

glm_header() {
  if command -v my_job_header &>/dev/null; then my_job_header; fi
}

glm_check_node() {
  if [[ -n "$GLM_ALLOWED_NODES" || -n "$GLM_EXPECTED_CPU" ]]; then
    "$GLM_PYTHON" "$GLM_PROJECT_ROOT/slurm/node_selection.py" check \
      --allow "$GLM_ALLOWED_NODES" --node "${SLURM_JOB_NODELIST:?Missing Slurm node allocation}" \
      --expected-cpu "$GLM_EXPECTED_CPU"
  fi
}
