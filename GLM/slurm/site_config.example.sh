# Copy to site_config.sh and edit before submitting. This is sourced by Bash.
# Never guess a billing account. Use the Great Lakes account you are authorized to use.
GLM_ACCOUNT="${GLM_ACCOUNT:-CHANGE_ME}"
GLM_PARTITION="${GLM_PARTITION:-standard}"

# Discover exact names on Great Lakes: module spider python; module spider R.
# Select Python 3.12 and R >= 4.3 with their required compiler modules.
# Example structure only: GLM_MODULES=("exact-python-module" "exact-R-module")
# Empty means retain the modules already loaded in your submission environment.
GLM_MODULES=()
GLM_BOOTSTRAP_PYTHON="${GLM_BOOTSTRAP_PYTHON:-python3}"
GLM_PYTHON="${GLM_PYTHON:-${GLM_PROJECT_ROOT}/.venv/bin/python}"
GLM_R_LIBS="${GLM_R_LIBS:-${GLM_PROJECT_ROOT}/.R-library}"
# Set 1 with R 4.4.0 to install the recorded direct R package versions.
GLM_PIN_R="${GLM_PIN_R:-0}"

# Each worker still fits using ONE thread. Reserve a second core for the RSS sampler.
GLM_CPUS="${GLM_CPUS:-2}"
GLM_MEM="${GLM_MEM:-8G}"
GLM_TIME="${GLM_TIME:-24:00:00}"
GLM_MAX_PARALLEL="${GLM_MAX_PARALLEL:-10}"

# CPU models need not match. Software, BLAS, modules and thread limits still match.
# Leave these three fields empty to permit any CPU in the selected partition.
# Optional scheduling restrictions can still be set explicitly if desired.
GLM_CONSTRAINT="${GLM_CONSTRAINT:-}"
# If CPU features are unavailable, supply comma-separated explicit node names
# verified from previous environment records. The submitter excludes all other
# nodes in the partition, and the array checks its node and CPU before fitting.
GLM_ALLOWED_NODES="${GLM_ALLOWED_NODES:-}"
GLM_EXPECTED_CPU="${GLM_EXPECTED_CPU:-}"
# Entire-node exclusivity may be expensive; enable only with an appropriate allocation.
GLM_EXCLUSIVE_NODE="${GLM_EXCLUSIVE_NODE:-0}"
