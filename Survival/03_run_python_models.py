"""Run the non-neural Python models used in the manuscript."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv


OUT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"


def make_surv_object(df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=df["event"].astype(bool).values,
        time=df["time"].astype(float).values,
    )


def test_support_mask(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    max_train_time = float(train["time"].max())
    mask = test["time"].astype(float).values < max_train_time
    if not np.any(mask):
        raise ValueError("No test observations inside training follow-up support.")
    return mask


def restrict_test_to_train_support(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return test.loc[test_support_mask(train, test)].copy()


def choose_time_grid(train: pd.DataFrame, test: pd.DataFrame, n_grid: int = 100) -> np.ndarray:
    lower = float(np.quantile(test["time"], 0.10))
    upper = float(np.quantile(test["time"], 0.90))
    train_max = float(train["time"].max())
    test_max = float(test["time"].max())
    train_events = train.loc[train["event"].astype(bool), "time"]
    train_event_max = float(train_events.max()) if not train_events.empty else train_max
    upper = min(upper, train_event_max * 0.99, train_max * 0.99, test_max * 0.99)
    if upper <= lower:
        lower = max(float(test["time"].min()), 1e-8)
        upper = min(train_event_max * 0.99, train_max * 0.99, test_max * 0.99)
    if upper <= lower:
        raise ValueError("Invalid time grid for IBS.")
    return np.linspace(lower, upper, n_grid)


def cindex(test: pd.DataFrame, risk_score: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            test["event"].astype(bool).values,
            test["time"].astype(float).values,
            np.asarray(risk_score, dtype=float),
        )[0]
    )


def step_function_matrix(surv_fns, time_grid: np.ndarray) -> np.ndarray:
    return np.asarray([fn(time_grid) for fn in surv_fns], dtype=float)


def cox_partial_loglik(time, event, eta) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    eta = np.asarray(eta, dtype=float)
    order = np.argsort(-time)
    event_ord = event[order]
    eta_ord = eta[order]
    max_eta = np.max(eta_ord)
    exp_eta = np.exp(eta_ord - max_eta)
    cum_risk = np.cumsum(exp_eta)
    event_idx = np.where(event_ord == 1)[0]
    if len(event_idx) == 0:
        return -np.inf
    return float(np.sum(eta_ord[event_idx] - (np.log(cum_risk[event_idx]) + max_eta)))


def choose_alpha_by_validation(model, x_val, y_val_time, y_val_event):
    coef_path = model.coef_
    if coef_path.ndim == 1:
        coef_path = coef_path.reshape(-1, 1)
    best_idx = 0
    best_pll = -np.inf
    for k in range(coef_path.shape[1]):
        pll = cox_partial_loglik(y_val_time, y_val_event, x_val @ coef_path[:, k])
        if pll > best_pll:
            best_pll = pll
            best_idx = k
    return best_idx, best_pll


def run_coxnet(setting: str, rep: int, p: int) -> dict[str, object]:
    train = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_train.csv")
    val = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_val.csv")
    test = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_test.csv")
    x_cols = [f"x{i + 1}" for i in range(p)]

    x_train = train[x_cols].values.astype(float)
    x_val = val[x_cols].values.astype(float)
    x_test = test[x_cols].values.astype(float)
    y_train = make_surv_object(train)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_train_std = (x_train - x_mean) / x_std
    x_val_std = (x_val - x_mean) / x_std
    x_test_std = (x_test - x_mean) / x_std

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path_model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alpha_min_ratio=0.01,
            n_alphas=100,
            fit_baseline_model=False,
        )
        path_model.fit(x_train_std, y_train)
    alpha_idx, _ = choose_alpha_by_validation(
        path_model,
        x_val_std,
        val["time"].values.astype(float),
        val["event"].values.astype(int),
    )
    selected_alpha = path_model.alphas_[alpha_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=[selected_alpha],
            fit_baseline_model=True,
        )
        model.fit(x_train_std, y_train)
    runtime = time.time() - start

    mask = test_support_mask(train, test)
    test_eval = test.loc[mask].copy()
    grid = choose_time_grid(train, test_eval)
    ibs = integrated_brier_score(
        y_train,
        make_surv_object(test_eval),
        step_function_matrix(model.predict_survival_function(x_test_std[mask]), grid),
        grid,
    )
    risk = model.predict(x_test_std)
    return {
        "method": "sksurv_CoxnetSurvivalAnalysis",
        "model_class": "Penalized Cox",
        "language": "Python",
        "ibs": float(ibs),
        "cindex_td": cindex(test, risk),
        "runtime_sec": runtime,
    }


def run_coxph(setting: str, rep: int, p: int) -> dict[str, object]:
    train = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_test.csv")
    x_cols = [f"x{i + 1}" for i in range(p)]
    train_fit = train[x_cols + ["time", "event"]].copy()
    test_fit = test[x_cols + ["time", "event"]].copy()
    model = CoxPHFitter()
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_fit, duration_col="time", event_col="event", show_progress=False)
    runtime = time.time() - start
    test_eval = restrict_test_to_train_support(train_fit, test_fit)
    grid = choose_time_grid(train_fit, test_eval)
    surv_df = model.predict_survival_function(test_eval[x_cols], times=grid)
    ibs = integrated_brier_score(
        make_surv_object(train_fit),
        make_surv_object(test_eval),
        surv_df.T.values,
        grid,
    )
    risk = model.predict_partial_hazard(test_fit[x_cols]).values.reshape(-1)
    return {
        "method": "lifelines_CoxPHFitter",
        "model_class": "CoxPH",
        "language": "Python",
        "ibs": float(ibs),
        "cindex_td": cindex(test_fit, risk),
        "runtime_sec": runtime,
    }


def run_rsf(setting: str, rep: int, p: int) -> dict[str, object]:
    train = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_test.csv")
    x_cols = [f"x{i + 1}" for i in range(p)]
    x_train = train[x_cols].values
    x_test = test[x_cols].values
    test_eval = restrict_test_to_train_support(train, test)
    grid = choose_time_grid(train, test_eval)
    model = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=100000 + rep,
    )
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, make_surv_object(train))
    runtime = time.time() - start
    ibs = integrated_brier_score(
        make_surv_object(train),
        make_surv_object(test_eval),
        step_function_matrix(model.predict_survival_function(test_eval[x_cols].values), grid),
        grid,
    )
    risk = model.predict(x_test)
    return {
        "method": "sksurv_RandomSurvivalForest",
        "model_class": "RSF",
        "language": "Python",
        "ibs": float(ibs),
        "cindex_td": cindex(test, risk),
        "runtime_sec": runtime,
    }


def run_boosting(setting: str, rep: int, p: int) -> dict[str, object]:
    train = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_test.csv")
    x_cols = [f"x{i + 1}" for i in range(p)]
    x_train = train[x_cols].values
    x_test = test[x_cols].values
    test_eval = restrict_test_to_train_support(train, test)
    grid = choose_time_grid(train, test_eval)
    model = GradientBoostingSurvivalAnalysis(
        loss="coxph",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=15,
        random_state=100000 + rep,
    )
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, make_surv_object(train))
    runtime = time.time() - start
    ibs = integrated_brier_score(
        make_surv_object(train),
        make_surv_object(test_eval),
        step_function_matrix(model.predict_survival_function(test_eval[x_cols].values), grid),
        grid,
    )
    risk = model.predict(x_test)
    return {
        "method": "sksurv_GradientBoostingSurvivalAnalysis",
        "model_class": "Boosting",
        "language": "Python",
        "ibs": float(ibs),
        "cindex_td": cindex(test, risk),
        "runtime_sec": runtime,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    config_columns = ["setting", "setting_type", "rep", "n", "p", "s", "rho"]
    cfgs = meta[config_columns].drop_duplicates()
    rows: list[dict[str, object]] = []
    for _, cfg in cfgs.iterrows():
        setting = str(cfg["setting"])
        setting_type = str(cfg["setting_type"])
        rep = int(cfg["rep"])
        p = int(cfg["p"])
        runners = (
            [run_coxnet, run_rsf, run_boosting]
            if setting_type == "highdim"
            else [run_coxph, run_rsf, run_boosting]
        )
        for runner in runners:
            base = cfg.to_dict()
            base.update({"language": "Python"})
            print(f"running Python {runner.__name__} {setting} rep={rep:03d}", flush=True)
            try:
                out = runner(setting, rep, p)
                rows.append({**base, **out, "status": "success", "error_message": ""})
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        **base,
                        "method": runner.__name__,
                        "model_class": "unknown",
                        "ibs": np.nan,
                        "cindex_td": np.nan,
                        "runtime_sec": np.nan,
                        "status": "failed",
                        "error_message": repr(exc),
                    }
                )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "python_models.csv", index=False)
    print("Wrote", RESULTS_DIR / "python_models.csv")


if __name__ == "__main__":
    main()
