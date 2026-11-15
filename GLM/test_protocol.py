#!/usr/bin/env python3
"""Protocol invariants: reproducible files, sparse storage, folds, KKT."""
import importlib
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy import sparse
from benchmark_metrics import evaluate
gen=importlib.import_module("01_generate_data")
cfg=json.loads(Path(__file__).with_name("configs").joinpath("smoke.json").read_text())
with tempfile.TemporaryDirectory() as directory:
    _,a=gen.generate(cfg["scenarios"][1],2026091,1,cfg,Path(directory)/"a")
    _,b=gen.generate(cfg["scenarios"][1],2026091,1,cfg,Path(directory)/"b")
    assert a["sha256"]==b["sha256"],"Same seed must produce byte-identical data"
    x=gen.read_matrix(a["files"]["x"],a["n"],a["p"])
    assert sparse.isspmatrix_csc(x) and x.nnz < x.shape[0]*x.shape[1]
    folds=np.fromfile(a["files"]["foldid"],dtype="<i4")
    assert set(folds)==set(range(5)) and np.ptp(np.bincount(folds))<=1
    lam=np.fromfile(a["files"]["lambdas"],dtype="<f8")
    assert np.all(np.diff(lam)<0)
    y=np.fromfile(a["files"]["y"],dtype="<f8")
    # lambda_max must make the intercept-only fit satisfy the L1 KKT conditions.
    zero=np.zeros(a["p"])
    ev=evaluate(x,y,x,y,zero,zero,np.log(y.mean()),lam[0],cfg)
    assert ev["quality_ok"] and abs(ev["kkt_scaled"])<1e-12
print("Protocol invariants passed.")
