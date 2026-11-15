#!/usr/bin/env python3
"""Fresh estimator for each repeat; warm starts only within a lambda path."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.special import xlogy

generate = importlib.import_module("01_generate_data")


def phase(path, value):
    # /proc exposes the host PID in some sandboxed runners while getpid()
    # exposes a namespace PID. Publish the PID that the RSS observer can read.
    pid=int(os.readlink("/proc/self")) if Path("/proc/self").exists() else os.getpid()
    tmp = Path(str(path)+".tmp"); tmp.write_text(json.dumps({"phase":value,"pid":pid}))
    tmp.replace(path)


def deviance(y, mu):
    return 2*np.mean(xlogy(y, y/mu)-y+mu)


def fit_unpen(method, x, y, conf):
    mi, tol = conf["max_iter"], conf["solver_tol"]
    if method == "Py_statsmodels":
        import statsmodels.api as sm
        xx = sm.add_constant(x.toarray() if sparse.issparse(x) else x, has_constant="add")
        fit = sm.GLM(y, xx, family=sm.families.Poisson()).fit(maxiter=mi, tol=tol, method="IRLS")
        return dict(coef=fit.params[1:].tolist(), intercept=float(fit.params[0]),
                    native_converged=bool(fit.converged), native_iterations=fit.fit_history.get("iteration"))
    if method == "Py_sklearn":
        from sklearn.linear_model import PoissonRegressor
        fit = PoissonRegressor(alpha=0, fit_intercept=True, solver="lbfgs", tol=tol, max_iter=mi).fit(x,y)
        return dict(coef=fit.coef_.tolist(),intercept=float(fit.intercept_),
                    native_converged=bool(fit.n_iter_ < mi),native_iterations=int(fit.n_iter_))
    if method == "Py_glum":
        from glum import GeneralizedLinearRegressor
        fit = GeneralizedLinearRegressor(family="poisson",alpha=0,l1_ratio=0,
                                        solver="irls-ls",fit_intercept=True,max_iter=mi,
                                        gradient_tol=tol,scale_predictors=False).fit(x,y)
        return dict(coef=fit.coef_.tolist(),intercept=float(fit.intercept_),
                    native_converged=bool(fit.n_iter_ < mi),native_iterations=int(fit.n_iter_))
    if method == "Py_spglm":
        from spglm.glm import GLM
        from spglm.family import Poisson
        # Explicit intercept; prevent the package from adding a second constant.
        xx = sparse.hstack([np.ones((len(y),1)),x],format="csr") if sparse.issparse(x) else np.column_stack([np.ones(len(y)),x])
        fit = GLM(y.reshape(-1,1),xx,family=Poisson(),constant=False).fit(tol=tol,max_iter=mi)
        b = np.asarray(fit.params).ravel(); ni = fit.model.fit_params["n_iter"]
        return dict(coef=b[1:].tolist(),intercept=float(b[0]),native_iterations=int(ni),
                    native_converged=bool(ni < mi))
    raise ValueError(method)


def path_fit(method, x, y, lambdas, conf):
    if method == "Py_glum_lasso":
        from glum import GeneralizedLinearRegressor
        fit = GeneralizedLinearRegressor(family="poisson",l1_ratio=1,
                    solver="irls-cd",alpha_search=True,alphas=lambdas,
                    scale_predictors=False,fit_intercept=True,selection="cyclic",
                    gradient_tol=conf["solver_tol"],max_iter=conf["max_iter"]).fit(x,y)
        return np.asarray(fit.coef_path_), np.asarray(fit.intercept_path_), int(fit.n_iter_)
    if method == "Py_skglm":
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import Poisson
        from skglm.penalties import L1
        from skglm.solvers import ProxNewton
        fit = GeneralizedLinearEstimator(datafit=Poisson(),penalty=L1(float(lambdas[0])),
                    solver=ProxNewton(fit_intercept=True,warm_start=True,
                                     tol=conf["solver_tol"],max_iter=conf["max_iter"]))
        coefs, intercepts, iterations = [], [], 0
        for lam in lambdas:
            fit.penalty.alpha = float(lam); fit.fit(x,y)
            coefs.append(fit.coef_.copy()); intercepts.append(float(fit.intercept_))
            iterations += int(fit.n_iter_)
        return np.asarray(coefs),np.asarray(intercepts),iterations
    raise ValueError(method)


def fit_pen(method, x, y, foldid, lambdas, conf):
    scores = np.empty((conf["cv_folds"], len(lambdas)))
    for k in range(conf["cv_folds"]):
        train, valid = foldid != k, foldid == k
        xx = x[train] if sparse.issparse(x) else np.asfortranarray(x[train])
        coefs, intercepts, _ = path_fit(method,xx,y[train],lambdas,conf)
        pred = np.exp(np.asarray(x[valid] @ coefs.T) + intercepts)
        scores[k] = np.sum(2*(xlogy(y[valid,None], y[valid,None]/pred)-y[valid,None]+pred),axis=0)
    cv = np.sum(scores,axis=0) / len(y)  # observation-weighted across folds
    if not np.all(np.isfinite(cv)):
        raise ValueError("Non-finite CV score; no silent removal of lambda candidates")
    idx = int(np.argmin(cv))  # first minimum: larger lambda if exactly tied
    coefs, intercepts, ni = path_fit(method,x,y,lambdas,conf)
    return dict(coef=coefs[idx].tolist(),intercept=float(intercepts[idx]),
                lambda_selected=float(lambdas[idx]),lambda_index=idx,
                cv_deviance=float(cv[idx]),cv_scores=cv.tolist(),
                native_converged=None,native_iterations=ni,
                lambda_boundary=bool(idx in {0,len(lambdas)-1}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--task",required=True);a=ap.parse_args()
    task=json.loads(Path(a.task).read_text());conf=task["config"]
    meta=json.loads(Path(task["dataset"]).read_text());files=meta["files"]
    x=generate.read_matrix(files["x"],meta["n"],meta["p"])
    y=np.fromfile(files["y"],dtype="<f8");foldid=np.fromfile(files["foldid"],dtype="<i4")
    lambdas=np.fromfile(files["lambdas"],dtype="<f8")
    fitfun=(lambda:fit_pen(task["method"],x,y,foldid,lambdas,conf)) if generate.is_penalized(meta["regime"]) else (lambda:fit_unpen(task["method"],x,y,conf))
    result={"status":"error","timings":[],"warnings":[]}
    try:
        phase(task["phase_file"],"warmup")
        # One untimed full fit includes lazy imports and JIT compilation.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); warm=fitfun()
        result["warmup_warnings"]=[str(v.message) for v in w];del warm;gc.collect()
        phase(task["phase_file"],"fit")
        last=None
        for _ in range(task["repeats"]):
            last=None;gc.collect()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always");start=time.perf_counter()
                last=fitfun();elapsed=time.perf_counter()-start
            result["timings"].append(elapsed);result["warnings"].extend(str(v.message) for v in w)
        phase(task["phase_file"],"evaluation")
        result.update(last);result["status"]="ok"
    except Exception as exc:
        result.update(error=repr(exc));phase(task["phase_file"],"error")
    result["warnings"]=list(dict.fromkeys(result["warnings"]))
    Path(task["result_file"]).write_text(json.dumps(result,allow_nan=False))
    phase(task["phase_file"],"done")


if __name__ == "__main__":
    main()
