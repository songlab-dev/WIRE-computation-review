#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-".venv-glm"}
PYTHON=${PYTHON:-"python3"}
CRAN_REPO=${CRAN_REPO:-"https://cloud.r-project.org"}
SKIP_R=${SKIP_R:-"0"}
SKIP_PYTHON=${SKIP_PYTHON:-"0"}

R_PACKAGES=(
  glmnet
  Matrix
  MASS
  optparse
)

PYTHON_PACKAGES=(
  numpy
  pandas
  scikit-learn
  statsmodels
  glum
  skglm
  spglm
)

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

install_r_packages() {
  require_command Rscript

  local packages
  packages=$(printf '"%s",' "${R_PACKAGES[@]}")
  packages="c(${packages%,})"

  Rscript -e "
    repos <- c(CRAN = Sys.getenv('CRAN_REPO', '$CRAN_REPO'))
    pkgs <- $packages
    installed <- rownames(installed.packages())
    missing <- setdiff(pkgs, installed)
    if (length(missing)) {
      install.packages(missing, repos = repos)
    } else {
      message('R packages already installed.')
    }
  "
}

install_python_packages() {
  require_command "$PYTHON"

  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
  fi

  local python_bin="$VENV_DIR/bin/python"
  "$python_bin" -m pip install --upgrade pip setuptools wheel
  "$python_bin" -m pip install "${PYTHON_PACKAGES[@]}"
}

if [ "$SKIP_R" != "1" ]; then
  echo "[1/2] Installing R packages..."
  install_r_packages
else
  echo "[1/2] Skipping R packages."
fi

if [ "$SKIP_PYTHON" != "1" ]; then
  echo "[2/2] Installing Python packages into $VENV_DIR..."
  install_python_packages
else
  echo "[2/2] Skipping Python packages."
fi

echo "Done."
echo "Use this Python for the benchmark:"
echo "  PYTHON_BIN=$VENV_DIR/bin/python ./run_all.sh"
