# Five-point grid GCV smoothing selection for statsmodels GLMGam and gamdist,
# neither of which selects smoothing by default (GLMGam uses alpha=0; gamdist
# fixes rel_dof), and the shared edf grid used for pyGAM as well. Keeps the fit minimising GCV = n * D / (n - edf)^2, with D the
# training deviance. All smooth terms share one grid value.

import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam
from gamdist import GAM as GamdistGAM

# Common search grid. pyGAM, statsmodels and gamdist all search the same 5 target
# effective degrees of freedom (edf) per smooth term, EDF_TARGETS, which is the
# only scale that does not depend on a package's basis or penalty definition.
# gamdist takes the targets directly (rel_dof). For pyGAM and statsmodels the
# penalty that gives each target was calibrated on pilot data (x ~ U(0,1),
# 20 basis functions per term); calibration and validation (achieved edf within
# about 0.5 of target for n >= 500, p <= 10) are in the project notes.
#   pygam_lam_ref : pyGAM lambda at n = 1000 (the same lambda for every term);
#                   lambda scales linearly with n, see pygam_lam_grid().
#   sm_r_uni/_add : statsmodels r, with alpha_j = r * tr(B_j'B_j) / tr(S_j), so r
#                   is scale-free; separate values for one smooth and additive fits.
EDF_TARGETS = (3.0, 5.0, 8.0, 12.0, 18.0)
GRID_TABLES = {
    "gauss": {
        "pygam_lam_ref": [4582.61, 257.343, 21.023, 2.02543, 0.0470901],
        "sm_r_uni": [150.533, 16.4118, 2.10423, 0.293672, 0.00673942],
        "sm_r_add": [425.06, 29.6744, 3.08226, 0.394863, 0.0127567],
    },
    "logit": {
        "pygam_lam_ref": [1083.11, 59.3334, 4.75733, 0.45574, 0.010649],
        "sm_r_uni": [35.7936, 3.75067, 0.475346, 0.0661201, 0.00153582],
        "sm_r_add": [80.9388, 5.33417, 0.529378, 0.0650944, 0.00185933],
    },
}


def pygam_lam_grid(n, family="gauss"):
    """pyGAM lambda grid (one value per grid point, shared by all terms)."""
    return np.array(GRID_TABLES[family]["pygam_lam_ref"]) * n / 1000.0


def sm_r_grid(family="gauss", additive=False):
    """statsmodels relative-penalty grid for the common edf targets."""
    return np.array(GRID_TABLES[family]["sm_r_add" if additive else "sm_r_uni"])


# gamdist: effective df per spline (must exceed 2, the penalty's null space).
GAMDIST_DOF_GRID = EDF_TARGETS


def _gcv(n, deviance, edf):
    denom = n - edf
    return n * deviance / denom**2 if denom > 0 else np.inf


def _sm_alpha_scale(smoother):
    """Per-term tr(B'B) / tr(S) for a statsmodels multivariate smoother."""
    scale = []
    for us in smoother.smoothers:
        tr_s = float(np.trace(us.cov_der2))
        scale.append(float(np.sum(us.basis ** 2)) / tr_s if tr_s > 0 else 1.0)
    return np.asarray(scale)


def fit_statsmodels_gcv(y, smoother, family=None, grid=None):
    """Fit GLMGam at each grid point; return the fit with the lowest GCV."""
    scale = _sm_alpha_scale(smoother)
    if grid is None:
        grid = sm_r_grid("logit" if family is not None else "gauss",
                         additive=len(smoother.smoothers) > 1)
    best, best_gcv = None, np.inf
    for r in grid:
        try:
            fit = GLMGam(y, smoother=smoother, alpha=list(r * scale),
                         family=family).fit()
            score = _gcv(fit.nobs, fit.deviance, fit.hat_matrix_trace)
        except Exception:
            continue
        if np.isfinite(score) and score < best_gcv:
            best, best_gcv = fit, score
    if best is None:
        raise RuntimeError("statsmodels GCV grid: no valid fit")
    return best


def fit_gamdist_gcv(X, y, family="normal", grid=GAMDIST_DOF_GRID):
    """Fit a gamdist spline GAM at each rel_dof; return the lowest-GCV model.

    X is a 2-D array; columns are named x0, x1, ... internally, so predict with
    predict_gamdist() rather than calling model.predict() directly.
    """
    Xdf = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    best, best_gcv = None, np.inf
    for dof in grid:
        try:
            mdl = GamdistGAM(family=family)
            for col in Xdf.columns:
                mdl.add_feature(name=col, type="spline", rel_dof=dof)
            mdl.fit(Xdf, y)
            # Same formula as mdl.gcv(), but guarded against n <= edf.
            score = _gcv(len(y), mdl.deviance(), mdl.dof())
        except Exception:
            continue
        if np.isfinite(score) and score < best_gcv:
            best, best_gcv = mdl, score
    if best is None:
        raise RuntimeError("gamdist GCV grid: no valid fit")
    return best


def predict_gamdist(model, X):
    """Mean response (inverse-link scale) from a fit_gamdist_gcv() model."""
    Xdf = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    return np.asarray(model.predict(Xdf), dtype=float)
