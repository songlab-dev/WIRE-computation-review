#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
py="${PYTHON_FOR_VENV:-python3.12}"
"$py" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
r_args=()
if [[ "${GLM_PIN_R:-0}" == 1 ]]; then r_args+=(--reference); fi
Rscript install_R_dependencies.R "${r_args[@]}"
.venv/bin/python -m pip freeze > requirements-installed.txt
.venv/bin/python test_protocol.py
