# Run Gaussian simulation (univariate and additive) using Python.
#
# For each (task, n, bootstrap replicate b), reads the pre-generated CSV from
# data/ and fits every Python method; writes one summary CSV per (task, n) to
# results/.
#
# Outputs (results/):
#   python_{task}_n{n}.csv  task in {global, local, additive_p01, ..., additive_p10}
#
# Prerequisites:
#   pip install -r public/requirements.txt
#   Rscript public/generate_data.R
#
# Run from the project root (or any directory — paths are resolved relative to
# this script's location):
#   python public/simulate_python.py

import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, s
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.gam.smooth_basis import CyclicCubicSplines

ROOT    = Path(__file__).parent.parent
DATA    = str(ROOT / "data")
RESULTS = str(ROOT / "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Metric helpers ────────────────────────────────────────────────────────────

def metrics(p, se, f0, y_test):
    return dict(
        mise     = float(np.mean((p - f0) ** 2)),
        test_mse = float(np.mean((y_test - p) ** 2)),
        coverage = float(np.mean(np.abs(p - f0) <= 1.96 * se)),
    )

def no_ci_metrics(p, f0, y_test):
    return dict(
        mise     = float(np.mean((p - f0) ** 2)),
        test_mse = float(np.mean((y_test - p) ** 2)),
        coverage = float(np.nan),
    )

def make_smoother(cls, *args, **kwargs):
    """Remove the 'transform' instance attribute that shadows the inherited
    transform() method needed by GLMGam.predict (statsmodels bug)."""
    sm = cls(*args, **kwargs)
    if hasattr(sm, "transform") and isinstance(sm.transform, str):
        del sm.transform
    return sm

def _sm_se(fit, smoother, x_train, x_pred):
    """SE of mean prediction from a fitted GLMGam via parameter covariance.

    GLMGam applies a centering constraint internally that reduces the raw
    smoother columns (e.g. 20) to a smaller constrained set (e.g. 19).
    smoother.transform() returns the unconstrained basis, so we recover the
    linear constraint matrix M via lstsq on the training data, then apply it
    to the prediction points.
    """
    X_tr_raw   = smoother.transform(x_train)                          # (n, d_raw)
    X_tr_con   = fit.model.exog                                        # (n, d_con)
    M          = np.linalg.lstsq(X_tr_raw, X_tr_con, rcond=None)[0]   # (d_raw, d_con)
    X_pred_con = smoother.transform(x_pred) @ M                        # (n_pred, d_con)
    cov        = fit.cov_params()
    var_pred   = np.einsum('ij,jk,ik->i', X_pred_con, cov, X_pred_con)
    return np.sqrt(np.clip(var_pred, 0, None))

# ── Per-replicate fitters ─────────────────────────────────────────────────────

def run_one_univariate(train, test, task):
    x      = train["x"].values[:, None]
    y      = train["y"].values
    x_test = test["x"].values[:, None]
    y_test = test["y"].values
    f0     = test["f0"].values
    x_pred = np.clip(x_test, x.min(), x.max())

    # P-splines, GCV (pyGAM)
    gam_fit = LinearGAM(s(0, n_splines=20, basis="ps")).gridsearch(x, y)
    p_pygam = gam_fit.predict(x_test)
    ci      = gam_fit.confidence_intervals(x_test, width=0.95)
    se_pygam = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)

    # B-splines, GCV (statsmodels)
    smoother_bs = BSplines(x, df=[20], degree=[3])
    fit_bs      = GLMGam(y, smoother=smoother_bs).fit()
    p_bs        = fit_bs.predict(exog_smooth=x_pred)
    se_bs       = _sm_se(fit_bs, smoother_bs, x, x_pred)

    res = dict(
        pygam_gcv          = metrics(p_pygam, se_pygam, f0, y_test),
        statsmodels_bs_gcv = metrics(p_bs, se_bs, f0, y_test),
    )

    # Cyclic cubic splines, GCV (statsmodels): global smooth only
    if task == "global":
        smoother_cc = make_smoother(CyclicCubicSplines, x, df=[20])
        fit_cc      = GLMGam(y, smoother=smoother_cc).fit()
        p_cc        = fit_cc.predict(exog_smooth=x_pred)
        se_cc       = _sm_se(fit_cc, smoother_cc, x, x_pred)
        res["statsmodels_cc_gcv"] = metrics(p_cc, se_cc, f0, y_test)

    return res

def run_one_additive(train, test):
    xcols  = [c for c in train.columns if c.startswith("x")]
    p      = len(xcols)
    X      = train[xcols].values
    y      = train["y"].values
    X_test = test[xcols].values
    y_test = test["y"].values
    f0     = test["f0"].values
    X_pred = np.clip(X_test, X.min(axis=0), X.max(axis=0))

    # P-splines, GCV (pyGAM)
    terms = s(0)
    for i in range(1, p):
        terms = terms + s(i)
    gam_fit  = LinearGAM(terms).gridsearch(X, y)
    p_pygam  = gam_fit.predict(X_test)
    ci       = gam_fit.confidence_intervals(X_test, width=0.95)
    se_pygam = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)

    # B-splines, GCV (statsmodels)
    smoother_bs = BSplines(X, df=[10] * p, degree=[3] * p)
    fit_bs      = GLMGam(y, smoother=smoother_bs).fit()
    p_bs        = fit_bs.predict(exog_smooth=X_pred)
    se_bs       = _sm_se(fit_bs, smoother_bs, X, X_pred)

    return dict(
        pygam_gcv          = metrics(p_pygam, se_pygam, f0, y_test),
        statsmodels_bs_gcv = metrics(p_bs, se_bs, f0, y_test),
    )

def run_one(path, task):
    d     = pd.read_csv(path)
    train = d[d["split"] == "train"]
    test  = d[d["split"] == "test"]
    if "x1" in d.columns:
        return run_one_additive(train, test)
    return run_one_univariate(train, test, task)

def run_mc(task, n):
    paths = sorted(glob.glob(f"{DATA}/{task}_n{n}_b*.csv"))
    reps  = [run_one(p, task) for p in paths]
    return {
        m: pd.DataFrame([r[m] for r in reps]).mean().to_dict()
        for m in reps[0]
    }

# ── Run ───────────────────────────────────────────────────────────────────────

for task in ("global", "local"):
    print(f"\n=== {task} ===")
    for n in (100, 500, 2000):
        out = f"{RESULTS}/python_{task}_n{n}.csv"
        if os.path.exists(out):
            print(f"  n={n}: skipping (already done)")
            continue
        print(f"\nn = {n}")
        res = run_mc(task, n)
        df  = pd.DataFrame(res).T.round(3)
        print(df)
        df.insert(0, "task", task)
        df.insert(1, "n", n)
        df.index.name = "method"
        df.reset_index().to_csv(out, index=False)

print("\n=== additive ===")
for p in (1, 10, 50, 100):
    subtask = f"additive_p{p:02d}"
    for n in (100, 500, 2000):
        out = f"{RESULTS}/python_{subtask}_n{n}.csv"
        if os.path.exists(out):
            print(f"  {subtask} n={n}: skipping (already done)")
            continue
        print(f"\n{subtask} n = {n}")
        res = run_mc(subtask, n)
        df  = pd.DataFrame(res).T.round(3)
        print(df)
        df.insert(0, "task", subtask)
        df.insert(1, "n", n)
        df.index.name = "method"
        df.reset_index().to_csv(out, index=False)
