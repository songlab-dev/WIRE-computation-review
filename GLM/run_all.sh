#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
profile="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in smoke|pilot|full|sensitivity) ;; *) echo "Usage: ./run_all.sh {smoke|pilot|full|sensitivity} [--resume] [--out DIR]" >&2; exit 2;; esac
py="${WIRE_PYTHON:-.venv/bin/python}"
if [[ ! -x "$py" ]]; then py="${WIRE_PYTHON:-python3}"; fi
out="results/$profile"
if [[ "$profile" == full ]]; then out="results/full100x5"; fi
# Read an optional --out so summarization uses the identical directory.
args=("$@")
for (( i=0; i<${#args[@]}; i++ )); do
  if [[ "${args[$i]}" == "--out" ]]; then out="${args[$((i+1))]:?--out requires a directory}"; fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$py" run_benchmark.py --config "configs/$profile.json" --out "$out" "${args[@]}"
"$py" check_results.py --out "$out"
"$py" summarize_results.py --out "$out"
export_args=(--results "$out")
if [[ "$profile" != full ]]; then export_args+=(--allow-validation); fi
"$py" export_tables.py "${export_args[@]}"
