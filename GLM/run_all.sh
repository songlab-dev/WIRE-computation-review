#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-".venv-glm/bin/python"}
REPS=${REPS:-5}
OUTDIR=${OUTDIR:-"data_glm"}

GEN_WORKERS=${GEN_WORKERS:-1}
R_WORKERS=${R_WORKERS:-1}
PY_WORKERS=${PY_WORKERS:-1}

echo "[1/6] Cleaning old data and results..."
rm -rf "$OUTDIR" results_R.csv results_python.csv results_all_raw.csv results_all_summary.csv paper_tables

echo "[2/6] Generating shared datasets with GEN_WORKERS=$GEN_WORKERS..."
Rscript 01_generate_data.R \
  --outdir "$OUTDIR" \
  --reps "$REPS" \
  --workers "$GEN_WORKERS"

echo "[3/6] Running R models with R_WORKERS=$R_WORKERS..."
Rscript 02_run_R_models.R \
  --manifest "$OUTDIR/manifest.csv" \
  --out results_R.csv \
  --workers "$R_WORKERS"

echo "[4/6] Running Python models with PY_WORKERS=$PY_WORKERS..."
"$PYTHON_BIN" 03_run_python_models.py \
  --manifest "$OUTDIR/manifest.csv" \
  --out results_python.csv \
  --workers "$PY_WORKERS"

echo "[5/6] Summarizing results..."
Rscript 04_summarize_results.R \
  --r-results results_R.csv \
  --py-results results_python.csv \
  --out-raw results_all_raw.csv \
  --out-summary results_all_summary.csv

echo "[6/6] Generating paper-ready tables..."
Rscript 05_generate_paper_tables.R \
  --summary results_all_summary.csv \
  --raw results_all_raw.csv \
  --out-dir paper_tables \
  --reps "$REPS"

echo "Done."
echo "Raw results: results_all_raw.csv"
echo "Summary:     results_all_summary.csv"
echo "Paper tables: paper_tables/ (LaTeX format, all_tables.tex is the master document)"
