#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
    HAS_GLUM = True
except Exception:
    HAS_GLUM = False

try:
    from skglm import GeneralizedLinearEstimator, GeneralizedLinearEstimatorCV
    from skglm.datafits import Poisson as SkglmPoisson
    from skglm.penalties import L1
    from skglm.solvers import ProxNewton as SkglmProxNewton
    HAS_SKGLM = True
except Exception:
    HAS_SKGLM = False

try:
    from spglm.glm import GLM
    from spglm.family import Poisson as SpglmPoisson
    HAS_SPGLM = True
except Exception:
    HAS_SPGLM = False


def negloglik(y, mu):
    """Poisson negative log-likelihood per observation, dropping log(y!) term."""
    eps = 1e-8
    mu = np.maximum(np.asarray(mu).reshape(-1).astype(float), eps)
    y = np.asarray(y).reshape(-1).astype(float)
    return float(-np.mean(y * np.log(mu) - mu))


def mean_poisson_deviance(y, mu):
    """Mean Poisson deviance: 2 * mean( y*log(y/mu) - (y - mu) )."""
    eps = 1e-8
    y = np.asarray(y).reshape(-1).astype(float)
    mu = np.maximum(np.asarray(mu).reshape(-1).astype(float), eps)
    term = np.where(y > 0, y * np.log(np.maximum(y, eps) / mu), 0.0) - (y - mu)
    return float(np.mean(2.0 * term))


def read_dataset(row):
    X = pd.read_csv(row["x_csv"]).values
    y = pd.read_csv(row["y_csv"])["y"].values.astype(int)
    return X, y


def read_true_beta(row):
    beta_csv = row.get("beta_csv", None)
    if beta_csv is None or (isinstance(beta_csv, float) and np.isnan(beta_csv)):
        return None
    try:
        return pd.read_csv(beta_csv)["beta"].values.astype(float)
    except Exception:
        return None


def support_metrics(beta_hat, beta_true, tol_hat=1e-8, tol_true=1e-12):
    """Recall, precision and F1 for support recovery."""
    if beta_hat is None or beta_true is None:
        return np.nan, np.nan, np.nan
    beta_hat = np.asarray(beta_hat).reshape(-1)
    beta_true = np.asarray(beta_true).reshape(-1)
    m = min(len(beta_hat), len(beta_true))
    if m == 0:
        return np.nan, np.nan, np.nan
    beta_hat = beta_hat[:m]
    beta_true = beta_true[:m]
    S_hat = np.abs(beta_hat) > tol_hat
    S_true = np.abs(beta_true) > tol_true
    tp = int(np.sum(S_hat & S_true))
    fp = int(np.sum(S_hat & ~S_true))
    fn = int(np.sum(~S_hat & S_true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return recall, precision, f1


def fit_sklearn_unpenalized(X, y):
    # sklearn's PoissonRegressor always applies L2; alpha=0 disables shrinkage.
    model = PoissonRegressor(alpha=0.0, max_iter=1000, fit_intercept=True)
    model.fit(X, y)
    pred = model.predict(X)
    coef = np.asarray(model.coef_).reshape(-1)
    selected = int(np.sum(np.abs(coef) > 1e-8))
    return pred, selected, coef


def fit_statsmodels_glm(X, y):
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels not installed")
    Xd = sm.add_constant(X, has_constant="add")
    fit = sm.GLM(y, Xd, family=sm.families.Poisson()).fit(maxiter=50)
    pred = fit.predict(Xd)
    coef = np.asarray(fit.params[1:]).reshape(-1)
    selected = int(np.sum(np.abs(coef) > 1e-8))
    return pred, selected, coef


def fit_glum_unpenalized(X, y):
    if not HAS_GLUM:
        raise RuntimeError("glum not installed")
    model = GeneralizedLinearRegressor(family="poisson", alpha=0.0, fit_intercept=True, max_iter=1000)
    model.fit(X, y)
    pred = model.predict(X)
    coef = np.asarray(getattr(model, "coef_", np.array([]))).reshape(-1)
    selected = int(np.sum(np.abs(coef) > 1e-8))
    return pred, selected, coef


def fit_glum_lasso(X, y):
    if not HAS_GLUM:
        raise RuntimeError("glum not installed")
    # 5-fold CV over alpha (Lasso); l1_ratio = 1.0 fixes pure L1.
    model = GeneralizedLinearRegressorCV(
        family="poisson",
        l1_ratio=1.0,
        cv=5,
        fit_intercept=True,
        max_iter=1000,
    )
    model.fit(X, y)
    pred = model.predict(X)
    coef = np.asarray(getattr(model, "coef_", np.array([]))).reshape(-1)
    selected = int(np.sum(np.abs(coef) > 1e-8))
    return pred, selected, coef


def fit_skglm_default(X, y):
    if not HAS_SKGLM:
        raise RuntimeError("skglm not installed")
    # 5-fold CV over the Lasso regularization strength (alpha).
    # AndersonCD (the default) is not compatible with the Poisson datafit;
    # ProxNewton supports it.
    model = GeneralizedLinearEstimatorCV(
        datafit=SkglmPoisson(),
        penalty=L1(1.0),  # alpha placeholder; CV searches a path
        solver=SkglmProxNewton(),
        cv=5,
    )
    model.fit(X, y)
    coef = np.asarray(model.coef_).reshape(-1)
    intercept = np.asarray(getattr(model, "intercept_", 0.0)).reshape(-1)
    eta = np.asarray(X @ coef).reshape(-1) + (intercept[0] if intercept.size > 0 else 0.0)
    pred = np.exp(np.clip(eta, -20, 20))
    selected = int(np.sum(np.abs(coef) > 1e-8))
    return pred, selected, coef


def fit_spglm(X, y):
    if not HAS_SPGLM:
        raise RuntimeError("spglm not installed")
    y_col = np.asarray(y).reshape((-1, 1))
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    model = GLM(y_col, X_design, family=SpglmPoisson())
    res = model.fit()
    coef = np.asarray(getattr(res, "params", np.array([]))).reshape(-1)
    coef_no_intercept = coef[1:] if coef.size > 0 else coef
    for attr in ["mu", "predy", "predy_e"]:
        if hasattr(res, attr):
            return np.asarray(getattr(res, attr)).reshape(-1), np.nan, coef_no_intercept
    if coef.size > 0:
        eta = X_design @ coef
        pred = np.exp(np.clip(eta, -20, 20))
        return pred, np.nan, coef_no_intercept
    raise RuntimeError("Cannot extract predictions from spglm result object")


METHODS = {
    "PY_sklearn_unpenalized": fit_sklearn_unpenalized,
    "PY_statsmodels_glm": fit_statsmodels_glm,
    "PY_glum_unpenalized": fit_glum_unpenalized,
    "PY_glum_lasso": fit_glum_lasso,
    "PY_skglm_default": fit_skglm_default,
    "PY_spglm": fit_spglm,
}


def choose_methods(regime):
    methods = []
    if regime in ["dense", "sparse"]:
        methods += [
            ("PY_sklearn_unpenalized", "fast"),
            ("PY_glum_unpenalized", "fast"),
            ("PY_statsmodels_glm", "slow"),
            ("PY_spglm", "slow"),
        ]
    if regime in ["highdim", "sparse_highdim"]:
        methods += [
            ("PY_glum_lasso", "highdim"),
            ("PY_skglm_default", "highdim"),
        ]
    return methods


def run_one_task(task):
    row, method_name = task
    t0 = time.time()
    try:
        X, y = read_dataset(row)
        pred, selected, coef = METHODS[method_name](X, y)
        elapsed = time.time() - t0
        if row.get("regime") in ["highdim", "sparse_highdim"]:
            beta_true = read_true_beta(row)
            recall, precision, f1 = support_metrics(coef, beta_true)
        else:
            recall, precision, f1 = np.nan, np.nan, np.nan
        res = {
            "method": method_name,
            "time": elapsed,
            "loss": negloglik(y, pred),
            "mean_dev": mean_poisson_deviance(y, pred),
            "selected": selected,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "converged": True,
            "error": ""
        }
    except Exception as e:
        res = {
            "method": method_name,
            "time": np.nan,
            "loss": np.nan,
            "mean_dev": np.nan,
            "selected": np.nan,
            "recall": np.nan,
            "precision": np.nan,
            "f1": np.nan,
            "converged": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }

    n, p = int(row["n"]), int(row["p"])
    res.update({
        "id": row["id"],
        "regime": row["regime"],
        "n": n,
        "p": p,
        "rep": int(row["rep"]),
        "density": row.get("density", np.nan),
    })
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_glm/manifest.csv")
    parser.add_argument("--out", default="results_python.csv")
    parser.add_argument("--regime", default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.regime != "all":
        manifest = manifest[manifest["regime"] == args.regime]

    tasks = []
    for _, row in manifest.iterrows():
        methods = choose_methods(row["regime"])
        for method_name, _kind in methods:
            tasks.append((row.to_dict(), method_name))

    print(f"Python tasks: {len(tasks)}  Workers: {args.workers}", flush=True)

    rows = []
    if args.workers <= 1:
        for task in tasks:
            print(f"Running {task[1]} on {task[0]['id']}", flush=True)
            rows.append(run_one_task(task))
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one_task, task): task for task in tasks}
            for fut in cf.as_completed(futures):
                task = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    row, method_name = task
                    n, p = int(row["n"]), int(row["p"])
                    rows.append({
                        "id": row["id"], "regime": row["regime"], "n": n, "p": p,
                        "rep": int(row["rep"]), "density": row.get("density", np.nan),
                        "method": method_name, "time": np.nan, "loss": np.nan,
                        "mean_dev": np.nan, "selected": np.nan,
                        "recall": np.nan, "precision": np.nan, "f1": np.nan,
                        "converged": False,
                        "error": f"Parallel error: {type(e).__name__}: {str(e)}"
                    })
                print(f"Finished {task[1]} on {task[0]['id']}", flush=True)

    cols = ["id", "regime", "n", "p", "rep", "density",
            "method", "time", "loss", "mean_dev", "selected",
            "recall", "precision", "f1",
            "converged", "error"]
    out = pd.DataFrame(rows)
    out = out[cols] if len(out) else pd.DataFrame(columns=cols)
    out.to_csv(args.out, index=False)
    print(f"Saved Python results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
