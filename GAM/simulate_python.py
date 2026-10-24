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
import re
import sys
import time
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, s
from statsmodels.gam.api import BSplines
from statsmodels.gam.smooth_basis import CyclicCubicSplines
from gcv_grid import fit_statsmodels_gcv, fit_gamdist_gcv, predict_gamdist, pygam_lam_grid

ROOT    = Path(__file__).parent.parent
DATA    = str(ROOT / "data")
# GAM_RESULTS / GAM_MAX_REPS: pilot overrides (separate output dir, fewer reps).
RESULTS = str(ROOT / os.environ.get("GAM_RESULTS", "results"))
MAX_REPS = int(os.environ.get("GAM_MAX_REPS", "0"))   # 0 = all
os.makedirs(RESULTS, exist_ok=True)

# Per-fit wall-clock cap (s). After MAX_METHOD_TIMEOUTS timeouts in this job a
# method is skipped for the remaining reps and recorded as NA.
FIT_TIMEOUT_S       = 3600
MAX_METHOD_TIMEOUTS = 5
TIMEOUTS            = {}

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

def timed_fit(fn, name, timeout_s=FIT_TIMEOUT_S):
    """Run fn() (the fit for method `name`) with a hard wall-clock cap; NA
    metrics on timeout, error, or once the method has been skipped."""
    if TIMEOUTS.get(name, 0) >= MAX_METHOD_TIMEOUTS:
        return dict(NA_METRICS)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        return fn()
    except _Timeout:
        TIMEOUTS[name] = TIMEOUTS.get(name, 0) + 1
        print(f"    [{name} timeout > {timeout_s}s ({TIMEOUTS[name]}/{MAX_METHOD_TIMEOUTS})]")
    except Exception as e:
        print(f"    [{name} error: {e}]")
    finally:
        signal.alarm(0)
    return dict(NA_METRICS)

# ── Metric helpers ────────────────────────────────────────────────────────────

NA_METRICS = dict(mise=float("nan"), test_mse=float("nan"), coverage=float("nan"))

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

def gamdist_metrics(X, y, X_pred, f0, y_test):
    """Cubic-spline GAM via ADMM (gamdist), 5-point GCV; no CIs available."""
    model = fit_gamdist_gcv(X, y, family="normal")
    return no_ci_metrics(predict_gamdist(model, X_pred), f0, y_test)

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
    def _pygam():
        gam_fit  = LinearGAM(s(0, n_splines=20, basis="ps")).gridsearch(x, y, lam=pygam_lam_grid(len(y), "gauss"))
        p_pygam  = gam_fit.predict(x_test)
        ci       = gam_fit.confidence_intervals(x_test, width=0.95)
        se_pygam = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
        return metrics(p_pygam, se_pygam, f0, y_test)

    # B-splines, 5-point GCV (statsmodels)
    def _sm_bs():
        smoother_bs = BSplines(x, df=[20], degree=[3])
        fit_bs      = fit_statsmodels_gcv(y, smoother_bs)
        p_bs        = fit_bs.predict(exog_smooth=x_pred)
        se_bs       = _sm_se(fit_bs, smoother_bs, x, x_pred)
        return metrics(p_bs, se_bs, f0, y_test)

    res = dict(
        pygam_gcv          = timed_fit(_pygam, "pygam_gcv"),
        statsmodels_bs_gcv = timed_fit(_sm_bs, "statsmodels_bs_gcv"),
        gamdist_cr_gcv     = timed_fit(lambda: gamdist_metrics(x, y, x_pred, f0, y_test), "gamdist_cr_gcv"),
    )

    # Cyclic cubic splines, 5-point GCV (statsmodels): global smooth only
    if task == "global":
        def _sm_cc():
            smoother_cc = make_smoother(CyclicCubicSplines, x, df=[20])
            fit_cc      = fit_statsmodels_gcv(y, smoother_cc)
            p_cc        = fit_cc.predict(exog_smooth=x_pred)
            se_cc       = _sm_se(fit_cc, smoother_cc, x, x_pred)
            return metrics(p_cc, se_cc, f0, y_test)
        res["statsmodels_cc_gcv"] = timed_fit(_sm_cc, "statsmodels_cc_gcv")

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
    def _pygam():
        terms = s(0)
        for i in range(1, p):
            terms = terms + s(i)
        gam_fit  = LinearGAM(terms).gridsearch(X, y, lam=pygam_lam_grid(len(y), "gauss"))
        p_pygam  = gam_fit.predict(X_test)
        ci       = gam_fit.confidence_intervals(X_test, width=0.95)
        se_pygam = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
        return metrics(p_pygam, se_pygam, f0, y_test)

    # B-splines, 5-point GCV (statsmodels)
    def _sm_bs():
        smoother_bs = BSplines(X, df=[20] * p, degree=[3] * p)
        fit_bs      = fit_statsmodels_gcv(y, smoother_bs)
        p_bs        = fit_bs.predict(exog_smooth=X_pred)
        se_bs       = _sm_se(fit_bs, smoother_bs, X, X_pred)
        return metrics(p_bs, se_bs, f0, y_test)

    return dict(
        pygam_gcv          = timed_fit(_pygam, "pygam_gcv"),
        statsmodels_bs_gcv = timed_fit(_sm_bs, "statsmodels_bs_gcv"),
        gamdist_cr_gcv     = timed_fit(lambda: gamdist_metrics(X, y, X_pred, f0, y_test), "gamdist_cr_gcv"),
    )

def run_one(path, task):
    d     = pd.read_csv(path)
    train = d[d["split"] == "train"]
    test  = d[d["split"] == "test"]
    if "x1" in d.columns:
        return run_one_additive(train, test)
    return run_one_univariate(train, test, task)

CHUNK_DIR = str(Path(RESULTS) / "sim_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

def chunk_path(task, n, b):
    return f"{CHUNK_DIR}/python_{task}_n{n}_b{b:03d}.csv"

def run_cell(task, n):
    """Fit every rep of one (task, n) cell in [REP_START, REP_END] that has no
    chunk yet; one chunk file per rep. Returns the number of reps fitted."""
    paths = sorted(glob.glob(f"{DATA}/{task}_n{n}_b*.csv"))
    reps  = [(p, int(re.search(r"_b(\d+)\.csv$", p).group(1))) for p in paths]
    reps  = [(p, b) for p, b in reps if REP_START <= b <= REP_END]
    if MAX_REPS > 0:
        reps = reps[:MAX_REPS]
    k = 0
    for path, b in reps:
        chunk = chunk_path(task, n, b)
        if os.path.exists(chunk):
            continue
        rep_df = pd.DataFrame(run_one(path, task)).T
        rep_df.index.name = "method"
        rep_df.reset_index().to_csv(chunk, index=False)
        k += 1
    return k

def merge_cell(task, n, out):
    """Average all rep chunks of one cell into the summary CSV."""
    chunks = sorted(glob.glob(f"{CHUNK_DIR}/python_{task}_n{n}_b*.csv"))
    if not chunks:
        print(f"  {task} n={n}: no rep chunks")
        return
    reps = pd.concat(pd.read_csv(c) for c in chunks)
    df   = reps.groupby("method", sort=False).mean()
    print(f"  {task} n={n}: averaged {len(chunks)} rep chunks")
    print(df.round(3))
    df.insert(0, "task", task)
    df.insert(1, "n", n)
    df.reset_index().to_csv(out, index=False)

# ── Run ───────────────────────────────────────────────────────────────────────
# python <script> [--merge] [<task> <n>]
#   <task> <n>  run one cell (default: every cell)
#   --merge     no fitting; average existing rep chunks into the summary CSVs
# SIM_REP_START / SIM_REP_END restrict fitting to dataset files b<START>..b<END>
# (one shard of a cell). Sharded runs only write rep chunks; a later --merge
# run averages them.

args  = sys.argv[1:]
MERGE = bool(args) and args[0] == "--merge"
if MERGE:
    args = args[1:]
REP_START = int(os.environ.get("SIM_REP_START", "1"))
REP_END   = int(os.environ.get("SIM_REP_END", "999999"))
SHARDED   = "SIM_REP_START" in os.environ or "SIM_REP_END" in os.environ

cells = [(t, n) for t in ("global", "local", "additive_p01", "additive_p10", "additive_p50", "additive_p100") for n in (100, 500, 2000)]
if len(args) >= 2:
    cells = [(args[0], int(args[1]))]

for task, n in cells:
    out = f"{RESULTS}/python_{task}_n{n}.csv"
    if not MERGE:
        if not SHARDED and os.path.exists(out):
            print(f"  {task} n={n}: skipping (already done)")
            continue
        print(f"\n{task} n = {n}", flush=True)
        t0 = time.perf_counter()
        k  = run_cell(task, n)
        print(f"  [{task} n={n}: {time.perf_counter() - t0:.1f}s for {k} reps]", flush=True)
    if MERGE or not SHARDED:
        merge_cell(task, n, out)
