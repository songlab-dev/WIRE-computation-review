# Run logistic simulation (univariate and additive) using Python.
#
# For each (task, n, bootstrap replicate b), reads the pre-generated CSV from
# data/ and fits every Python method; writes one summary CSV per (task, n) to
# results/.
#
# Outputs (results/):
#   python_{task}_n{n}.csv
#   task in {logistic_global, logistic_local, logistic_p01, ..., logistic_p10}
#
# Prerequisites:
#   pip install -r public/requirements.txt
#   Rscript public/generate_logistic_data.R
#
# Run from the project root (or any directory — paths are resolved relative to
# this script's location):
#   python public/simulate_logistic_python.py

import glob
import os
import sys
import time
import re
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit, logit
from pygam import LogisticGAM, s
from statsmodels.gam.api import BSplines
from statsmodels.gam.smooth_basis import CyclicCubicSplines
import statsmodels.api as sm
from gcv_grid import fit_statsmodels_gcv, fit_gamdist_gcv, predict_gamdist, pygam_lam_grid

ROOT      = Path(__file__).parent.parent
DATA      = str(ROOT / "data")
# GAM_RESULTS / GAM_MAX_REPS: pilot overrides (separate output dir, fewer reps).
RESULTS   = str(ROOT / os.environ.get("GAM_RESULTS", "results"))
MAX_REPS  = int(os.environ.get("GAM_MAX_REPS", "0"))   # 0 = all
os.makedirs(RESULTS,   exist_ok=True)

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

# All metrics on the linear-predictor (log-odds) scale:
#   mise     — MSE between predicted LP and true LP
#   brier    — Brier score: mean((y - p_hat)^2)
#   coverage — nominal 95% CI coverage on the LP scale
NA_METRICS = dict(mise=float("nan"), brier=float("nan"), coverage=float("nan"))

def logistic_metrics(lp_hat, se_lp, f0_lp, y_test):
    p_hat = expit(lp_hat)
    return dict(
        mise     = float(np.mean((lp_hat - f0_lp) ** 2)),
        brier    = float(np.mean((y_test - p_hat) ** 2)),
        coverage = float(np.mean(np.abs(lp_hat - f0_lp) <= 1.96 * se_lp)),
    )

def _sm_se(fit, smoother, x_train, x_pred):
    """SE of mean prediction on the LP scale via parameter covariance."""
    X_tr_raw   = smoother.transform(x_train)
    X_tr_con   = fit.model.exog
    M          = np.linalg.lstsq(X_tr_raw, X_tr_con, rcond=None)[0]
    X_pred_con = smoother.transform(x_pred) @ M
    cov        = fit.cov_params()
    var_pred   = np.einsum('ij,jk,ik->i', X_pred_con, cov, X_pred_con)
    return np.sqrt(np.clip(var_pred, 0, None))

def _gamdist(X, y, X_pred, f0_lp, y_test, eps):
    """Cubic-spline GAM via ADMM (gamdist), 5-point GCV; no CIs, so coverage = NA."""
    model  = fit_gamdist_gcv(X, y, family="binomial")
    p_hat  = np.clip(predict_gamdist(model, X_pred), eps, 1 - eps)
    return dict(logistic_metrics(logit(p_hat), np.zeros(len(p_hat)), f0_lp, y_test),
                coverage=float("nan"))

def _make_smoother(cls, *args, **kwargs):
    """Remove the 'transform' attribute that shadows GLMGam.predict (statsmodels bug)."""
    sm_obj = cls(*args, **kwargs)
    if hasattr(sm_obj, "transform") and isinstance(sm_obj.transform, str):
        del sm_obj.transform
    return sm_obj

# ── Per-replicate fitters ─────────────────────────────────────────────────────

def run_one_logistic_univariate(train, test, task):
    x      = train["x"].values[:, None]
    y      = train["y"].values.astype(float)
    x_test = test["x"].values[:, None]
    y_test = test["y"].values.astype(float)
    f0_lp  = test["f0"].values
    x_pred = np.clip(x_test, x.min(), x.max())
    eps    = 1e-8

    def _pygam():
        gam_fit  = LogisticGAM(s(0)).gridsearch(x, y, lam=pygam_lam_grid(len(y), "logit"))
        p_hat    = np.clip(gam_fit.predict_proba(x_test), eps, 1 - eps)
        lp_hat   = logit(p_hat)
        ci       = np.clip(gam_fit.confidence_intervals(x_test, width=0.95), eps, 1 - eps)
        se_pygam = (logit(ci[:, 1]) - logit(ci[:, 0])) / (2 * 1.96)
        return logistic_metrics(lp_hat, se_pygam, f0_lp, y_test)

    def _sm_bs():
        smoother_bs = BSplines(x, df=[20], degree=[3])
        fit_bs      = fit_statsmodels_gcv(y, smoother_bs, family=sm.families.Binomial())
        p_bs        = np.clip(fit_bs.predict(exog_smooth=x_pred), eps, 1 - eps)
        lp_bs       = logit(p_bs)
        se_bs       = _sm_se(fit_bs, smoother_bs, x, x_pred)
        return logistic_metrics(lp_bs, se_bs, f0_lp, y_test)

    res = {
        "pygam_gcv":          timed_fit(_pygam, "pygam_gcv"),
        "statsmodels_bs_gcv": timed_fit(_sm_bs, "statsmodels_bs_gcv"),
        "gamdist_cr_gcv":     timed_fit(lambda: _gamdist(x, y, x_pred, f0_lp, y_test, eps), "gamdist_cr_gcv"),
    }

    if task == "logistic_global":
        def _sm_cc():
            smoother_cc = _make_smoother(CyclicCubicSplines, x, df=[20])
            fit_cc      = fit_statsmodels_gcv(y, smoother_cc, family=sm.families.Binomial())
            p_cc        = np.clip(fit_cc.predict(exog_smooth=x_pred), eps, 1 - eps)
            lp_cc       = logit(p_cc)
            se_cc       = _sm_se(fit_cc, smoother_cc, x, x_pred)
            return logistic_metrics(lp_cc, se_cc, f0_lp, y_test)
        res["statsmodels_cc_gcv"] = timed_fit(_sm_cc, "statsmodels_cc_gcv")

    return res

def run_one_logistic_additive(train, test):
    xcols  = [c for c in train.columns if c.startswith("x")]
    p      = len(xcols)
    X      = train[xcols].values
    y      = train["y"].values.astype(float)
    X_test = test[xcols].values
    y_test = test["y"].values.astype(float)
    f0_lp  = test["f0"].values
    X_pred = np.clip(X_test, X.min(axis=0), X.max(axis=0))
    eps    = 1e-8

    def _pygam():
        terms   = s(0)
        for i in range(1, p):
            terms = terms + s(i)
        gam_fit  = LogisticGAM(terms).gridsearch(X, y, lam=pygam_lam_grid(len(y), "logit"))
        p_hat    = np.clip(gam_fit.predict_proba(X_test), eps, 1 - eps)
        lp_hat   = logit(p_hat)
        ci       = np.clip(gam_fit.confidence_intervals(X_test, width=0.95), eps, 1 - eps)
        se_pygam = (logit(ci[:, 1]) - logit(ci[:, 0])) / (2 * 1.96)
        return logistic_metrics(lp_hat, se_pygam, f0_lp, y_test)

    def _sm_bs():
        smoother_bs = BSplines(X, df=[20] * p, degree=[3] * p)
        fit_bs      = fit_statsmodels_gcv(y, smoother_bs, family=sm.families.Binomial())
        p_bs        = np.clip(fit_bs.predict(exog_smooth=X_pred), eps, 1 - eps)
        lp_bs       = logit(p_bs)
        se_bs       = _sm_se(fit_bs, smoother_bs, X, X_pred)
        return logistic_metrics(lp_bs, se_bs, f0_lp, y_test)

    return {
        "pygam_gcv":          timed_fit(_pygam, "pygam_gcv"),
        "statsmodels_bs_gcv": timed_fit(_sm_bs, "statsmodels_bs_gcv"),
        "gamdist_cr_gcv":     timed_fit(lambda: _gamdist(X, y, X_pred, f0_lp, y_test, eps), "gamdist_cr_gcv"),
    }

def run_one(path, task):
    d     = pd.read_csv(path)
    train = d[d["split"] == "train"]
    test  = d[d["split"] == "test"]
    if "x1" in d.columns:
        return run_one_logistic_additive(train, test)
    return run_one_logistic_univariate(train, test, task)

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

cells = [(t, n) for t in ("logistic_global", "logistic_local", "logistic_p01", "logistic_p10", "logistic_p50", "logistic_p100") for n in (100, 500, 2000)]
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
