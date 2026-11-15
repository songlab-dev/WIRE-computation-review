"""Language-independent diagnostics calculated by the orchestrator."""
import numpy as np
from scipy.special import xlogy


def deviance(y, mu):
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
        return float("nan")
    return float(2 * np.mean(xlogy(y, y / mu) - y + mu))


def evaluate(x, y, xt, yt, beta_true, coef, intercept, lam, conf):
    coef = np.asarray(coef, float); intercept = float(intercept)
    eta = np.asarray(x @ coef).ravel() + intercept
    mu = np.exp(eta); mut = np.exp(np.asarray(xt @ coef).ravel() + intercept)
    grad = np.asarray(x.T @ (mu - y)).ravel() / len(y)
    active = np.abs(coef) > conf["support_tol"]
    violation = np.where(active, np.abs(grad + lam * np.sign(coef)),
                         np.maximum(np.abs(grad) - lam, 0))
    raw_kkt = max(float(np.max(violation)), abs(float(np.mean(mu-y))))
    kkt = raw_kkt / max(1.0, float(np.mean(y)))
    true = np.abs(beta_true) > 1e-12
    tp = int(np.sum(active & true)); fp = int(np.sum(active & ~true)); fn = int(np.sum(~active & true))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {"train_deviance":deviance(y, mu), "test_deviance":deviance(yt, mut),
            "objective":float(np.mean(mu-y*eta) + lam*np.sum(np.abs(coef))),
            "kkt_raw":raw_kkt, "kkt_scaled":kkt,
            "quality_ok":bool(np.isfinite(kkt) and kkt <= conf["kkt_tol"]
                              and np.all(np.isfinite(mut)) and np.all(mut > 0)),
            "selected":int(active.sum()), "precision":precision, "recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.0}
