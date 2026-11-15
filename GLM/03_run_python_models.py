#!/usr/bin/env python3
"""Same benchmark protocol, restricted to Python implementations."""
import subprocess, sys
from pathlib import Path
raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name("run_benchmark.py")),*sys.argv[1:],"--methods","Py_statsmodels","Py_sklearn","Py_glum","Py_spglm","Py_glum_lasso","Py_skglm"]))
