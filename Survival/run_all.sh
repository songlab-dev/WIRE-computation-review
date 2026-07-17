#!/bin/bash
set -euo pipefail

MODE="${1:-main}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="${PYTHON_ENV:-surv_py}"
DEEPSURV_ENV="${DEEPSURV_ENV:-deepsurv_env}"
REPS="${REPS:-50}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
fi
if [ -n "${DEEPSURV_PYTHON_BIN:-}" ]; then
  DEEPSURV_PYTHON_BIN="$("$DEEPSURV_PYTHON_BIN" -c 'import sys; print(sys.executable)')"
fi

case "$MODE" in
  main)
    EXPERIMENT_DIR="outputs/main_n500"
    SAMPLE_SIZES="500"
    ;;
  sensitivity)
    EXPERIMENT_DIR="outputs/sensitivity_n1000_n1500"
    SAMPLE_SIZES="1000,1500"
    ;;
  *)
    echo "Usage: $0 [main|sensitivity]" >&2
    exit 2
    ;;
esac

run_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    "$PYTHON_BIN" "$@"
  else
    conda run -n "$PYTHON_ENV" python "$@"
  fi
}

run_deepsurv_python() {
  if [ -n "${DEEPSURV_PYTHON_BIN:-}" ]; then
    "$DEEPSURV_PYTHON_BIN" "$@"
  else
    conda run -n "$DEEPSURV_ENV" python "$@"
  fi
}

if [ -z "${RETICULATE_PYTHON:-}" ]; then
  if [ -n "${DEEPSURV_PYTHON_BIN:-}" ]; then
    RETICULATE_PYTHON="$DEEPSURV_PYTHON_BIN"
  else
    RETICULATE_PYTHON="$(conda run -n "$DEEPSURV_ENV" python -c 'import sys; print(sys.executable)')"
  fi
fi

export EXPERIMENT_DIR SAMPLE_SIZES RETICULATE_PYTHON
export N_REPLICATIONS="$REPS"
export TARGET_CENSORING="${TARGET_CENSORING:-0.30}"

cd "$PROJECT_DIR"
echo "[1/7] Generating shared datasets ($MODE, $REPS replications)..."
run_python 01_generate_data.py

echo "[2/7] Running R models..."
Rscript 02_run_r_models.R

echo "[3/7] Running Python models..."
run_python 03_run_python_models.py

echo "[4/7] Running R DeepSurv..."
Rscript 04_run_r_deepsurv.R

echo "[5/7] Running Python DeepSurv..."
run_deepsurv_python 05_run_python_deepsurv.py

echo "[6/7] Combining and summarizing results..."
run_python 06_summarize_results.py

echo "[7/7] Generating the LaTeX table..."
run_python 07_generate_paper_table.py

echo "Done. Results are in $EXPERIMENT_DIR/results/."
