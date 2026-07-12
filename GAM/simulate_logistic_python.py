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
import re
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit, logit
from pygam import LogisticGAM, s
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.gam.smooth_basis import CyclicCubicSplines
import statsmodels.api as sm

ROOT      = Path(__file__).parent.parent
DATA      = str(ROOT / "data")
RESULTS   = str(ROOT / "results")
CHUNK_DIR = str(ROOT / "results" / "sim_chunks")
os.makedirs(RESULTS,   exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# Per-fit wall-clock cap; if > MAX_TIMEOUTS fits in a (tag, n) scenario time
# out, the remaining reps for that scenario are skipped.
FIT_TIMEOUT_S = 3600
MAX_TIMEOUTS  = 3

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

def timed_fit(fn, ctx, timeout_s=FIT_TIMEOUT_S):
    """Run fn() with a hard wall-clock cap.
    Returns None on timeout or error; increments ctx['timeout_count'] on timeout."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        result = fn()
        signal.alarm(0)
        return result
    except _Timeout:
        ctx["timeout_count"] += 1
        print(f"    [timeout > {timeout_s}s, count={ctx['timeout_count']}/{MAX_TIMEOUTS}]")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"    [error: {e}]")
        return None
    finally:
        signal.alarm(0)

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

def _make_smoother(cls, *args, **kwargs):
    """Remove the 'transform' attribute that shadows GLMGam.predict (statsmodels bug)."""
    sm_obj = cls(*args, **kwargs)
    if hasattr(sm_obj, "transform") and isinstance(sm_obj.transform, str):
        del sm_obj.transform
    return sm_obj

# ── Per-replicate fitters ─────────────────────────────────────────────────────

def run_one_logistic_univariate(train, test, task, ctx):
    x      = train["x"].values[:, None]
    y      = train["y"].values.astype(float)
    x_test = test["x"].values[:, None]
    y_test = test["y"].values.astype(float)
    f0_lp  = test["f0"].values
    x_pred = np.clip(x_test, x.min(), x.max())
    eps    = 1e-8

    def _pygam():
        gam_fit  = LogisticGAM(s(0)).gridsearch(x, y)
        p_hat    = np.clip(gam_fit.predict_proba(x_test), eps, 1 - eps)
        lp_hat   = logit(p_hat)
        ci       = np.clip(gam_fit.confidence_intervals(x_test, width=0.95), eps, 1 - eps)
        se_pygam = (logit(ci[:, 1]) - logit(ci[:, 0])) / (2 * 1.96)
        return logistic_metrics(lp_hat, se_pygam, f0_lp, y_test)

    def _sm_bs():
        smoother_bs = BSplines(x, df=[20], degree=[3])
        fit_bs      = GLMGam(y, smoother=smoother_bs, family=sm.families.Binomial()).fit()
        p_bs        = np.clip(fit_bs.predict(exog_smooth=x_pred), eps, 1 - eps)
        lp_bs       = logit(p_bs)
        se_bs       = _sm_se(fit_bs, smoother_bs, x, x_pred)
        return logistic_metrics(lp_bs, se_bs, f0_lp, y_test)

    res = {
        "pygam_gcv":          timed_fit(_pygam,  ctx) or NA_METRICS,
        "statsmodels_bs_gcv": timed_fit(_sm_bs,  ctx) or NA_METRICS,
    }

    if task == "logistic_global":
        def _sm_cc():
            smoother_cc = _make_smoother(CyclicCubicSplines, x, df=[20])
            fit_cc      = GLMGam(y, smoother=smoother_cc, family=sm.families.Binomial()).fit()
            p_cc        = np.clip(fit_cc.predict(exog_smooth=x_pred), eps, 1 - eps)
            lp_cc       = logit(p_cc)
            se_cc       = _sm_se(fit_cc, smoother_cc, x, x_pred)
            return logistic_metrics(lp_cc, se_cc, f0_lp, y_test)
        res["statsmodels_cc_gcv"] = timed_fit(_sm_cc, ctx) or NA_METRICS

    return res

def run_one_logistic_additive(train, test, ctx):
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
        gam_fit  = LogisticGAM(terms).gridsearch(X, y)
        p_hat    = np.clip(gam_fit.predict_proba(X_test), eps, 1 - eps)
        lp_hat   = logit(p_hat)
        ci       = np.clip(gam_fit.confidence_intervals(X_test, width=0.95), eps, 1 - eps)
        se_pygam = (logit(ci[:, 1]) - logit(ci[:, 0])) / (2 * 1.96)
        return logistic_metrics(lp_hat, se_pygam, f0_lp, y_test)

    def _sm_bs():
        smoother_bs = BSplines(X, df=[10] * p, degree=[3] * p)
        fit_bs      = GLMGam(y, smoother=smoother_bs, family=sm.families.Binomial()).fit()
        p_bs        = np.clip(fit_bs.predict(exog_smooth=X_pred), eps, 1 - eps)
        lp_bs       = logit(p_bs)
        se_bs       = _sm_se(fit_bs, smoother_bs, X, X_pred)
        return logistic_metrics(lp_bs, se_bs, f0_lp, y_test)

    return {
        "pygam_gcv":          timed_fit(_pygam, ctx) or NA_METRICS,
        "statsmodels_bs_gcv": timed_fit(_sm_bs, ctx) or NA_METRICS,
    }

def run_one(path, task, ctx):
    d     = pd.read_csv(path)
    train = d[d["split"] == "train"]
    test  = d[d["split"] == "test"]
    if "x1" in d.columns:
        return run_one_logistic_additive(train, test, ctx)
    return run_one_logistic_univariate(train, test, task, ctx)

def run_mc(tag, n):
    paths = sorted(glob.glob(f"{DATA}/{tag}_n{n}_b*.csv"))
    ctx   = {"timeout_count": 0}

    for path in paths:
        b     = re.search(r"_b(\d+)\.csv$", path).group(1)
        chunk = f"{CHUNK_DIR}/python_{tag}_n{n}_b{b}.csv"
        if os.path.exists(chunk):
            continue
        if ctx["timeout_count"] > MAX_TIMEOUTS:
            print(f"    >{MAX_TIMEOUTS} timeouts — skipping remaining reps")
            break
        res = run_one(path, tag, ctx)
        rep_df = pd.DataFrame(res).T
        rep_df.index.name = "method"
        rep_df.reset_index().to_csv(chunk, index=False)

    all_chunks = sorted(glob.glob(f"{CHUNK_DIR}/python_{tag}_n{n}_b*.csv"))
    if not all_chunks:
        return {}
    print(f"    averaging {len(all_chunks)} rep chunks")
    all_reps = [pd.read_csv(c).set_index("method") for c in all_chunks]
    combined = pd.concat(all_reps)
    return combined.groupby("method").mean().to_dict(orient="index")

# ── Run ───────────────────────────────────────────────────────────────────────

print("\n=== logistic ===")
for task in ("logistic_global", "logistic_local"):
    for n in (100, 500, 2000):
        out = f"{RESULTS}/python_{task}_n{n}.csv"
        if os.path.exists(out):
            print(f"  {task} n={n}: skipping (already done)")
            continue
        print(f"\n{task} n = {n}")
        res = run_mc(task, n)
        if not res:
            print("  no reps completed")
            continue
        df  = pd.DataFrame(res).T.round(3)
        print(df)
        df.insert(0, "task", task)
        df.insert(1, "n", n)
        df.index.name = "method"
        df.reset_index().to_csv(out, index=False)

for p in (1, 10, 50, 100):
    tag = f"logistic_p{p:02d}"
    for n in (100, 500, 2000):
        out = f"{RESULTS}/python_{tag}_n{n}.csv"
        if os.path.exists(out):
            print(f"  {tag} n={n}: skipping (already done)")
            continue
        print(f"\n{tag} n = {n}")
        res = run_mc(tag, n)
        if not res:
            print("  no reps completed")
            continue
        df  = pd.DataFrame(res).T.round(3)
        print(df)
        df.insert(0, "task", tag)
        df.insert(1, "n", n)
        df.index.name = "method"
        df.reset_index().to_csv(out, index=False)
