"""Run the Python DeepSurv implementation used in the manuscript."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH
import torch


OUT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"


CONFIG = {
    "num_nodes": [32],
    "batch_norm": False,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 150,
    "patience": 20,
}


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def test_support_mask(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    max_train_time = float(train["time"].max())
    mask = test["time"].astype(float).values < max_train_time
    if not np.any(mask):
        raise ValueError("No test observations inside training follow-up support.")
    return mask


def cindex(test: pd.DataFrame, risk_score: np.ndarray) -> float:
    time = test["time"].astype(float).values
    event = test["event"].astype(bool).values
    risk = np.asarray(risk_score, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in range(len(time)):
        if not event[i]:
            continue
        mask = time[i] < time
        if not np.any(mask):
            continue
        diff = risk[i] - risk[mask]
        concordant += float(np.sum(diff > 0))
        concordant += 0.5 * float(np.sum(diff == 0))
        comparable += int(np.sum(mask))
    return concordant / comparable if comparable else np.nan


def censor_survival_values(train: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    y_time = train["time"].astype(float).values
    censor_event = 1 - train["event"].astype(int).values
    km_times: list[float] = []
    km_surv: list[float] = []
    surv = 1.0
    for t0 in np.unique(y_time):
        at_risk = int(np.sum(y_time >= t0))
        n_events = int(np.sum((y_time == t0) & (censor_event == 1)))
        if at_risk > 0 and n_events > 0:
            surv *= 1.0 - n_events / at_risk
            km_times.append(float(t0))
            km_surv.append(float(surv))
    if not km_times:
        return np.ones(len(np.asarray(times, dtype=float)), dtype=float)
    km_time = np.asarray(km_times, dtype=float)
    km_surv_arr = np.asarray(km_surv, dtype=float)
    idx = np.searchsorted(km_time, np.asarray(times, dtype=float), side="right") - 1
    out = np.ones(len(np.asarray(times, dtype=float)), dtype=float)
    valid = idx >= 0
    out[valid] = km_surv_arr[idx[valid]]
    return np.maximum(out, 1e-6)


def integrated_brier_score_manual(
    surv_mat: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_grid: np.ndarray,
) -> float:
    y_time = test["time"].astype(float).values
    y_event = test["event"].astype(int).values
    g_t = censor_survival_values(train, time_grid)
    g_y = censor_survival_values(train, y_time)
    bs_vec = []
    for k, t0 in enumerate(time_grid):
        s_hat = surv_mat[:, k]
        term1 = (y_time > t0).astype(float) * (1.0 - s_hat) ** 2 / g_t[k]
        term2 = ((y_time <= t0) & (y_event == 1)).astype(float) * s_hat**2 / g_y
        bs_vec.append(float(np.nanmean(term1 + term2)))
    return float(np.trapz(bs_vec, time_grid) / (time_grid[-1] - time_grid[0]))


def survival_df_to_matrix(surv: pd.DataFrame, time_grid: np.ndarray) -> np.ndarray:
    pred_times = surv.index.to_numpy(dtype=float)
    raw = surv.to_numpy(dtype=float).T
    idx = np.searchsorted(pred_times, time_grid, side="right") - 1
    idx = np.clip(idx, 0, len(pred_times) - 1)
    return np.clip(raw[:, idx], 0.0, 1.0)


def standardize_x(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = train[x_cols].values.astype("float32")
    x_val = val[x_cols].values.astype("float32")
    x_test = test[x_cols].values.astype("float32")
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (
        ((x_train - mean) / sd).astype("float32"),
        ((x_val - mean) / sd).astype("float32"),
        ((x_test - mean) / sd).astype("float32"),
    )


def run_one(cfg: pd.Series) -> dict[str, object]:
    setting = str(cfg["setting"])
    rep = int(cfg["rep"])
    p = int(cfg["p"])
    set_all_seeds(880000 + rep)
    x_cols = [f"x{i + 1}" for i in range(p)]
    train = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_train.csv")
    val = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_val.csv")
    test = pd.read_csv(DATA_DIR / f"{setting}_rep{rep:03d}_test.csv")

    x_train, x_val, x_test = standardize_x(train, val, test, x_cols)
    y_train = (
        train["time"].values.astype("float32"),
        train["event"].values.astype("float32"),
    )
    y_val = (
        val["time"].values.astype("float32"),
        val["event"].values.astype("float32"),
    )
    net = tt.practical.MLPVanilla(
        in_features=p,
        num_nodes=CONFIG["num_nodes"],
        out_features=1,
        batch_norm=CONFIG["batch_norm"],
        dropout=CONFIG["dropout"],
    )
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(CONFIG["learning_rate"])

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            x_train,
            y_train,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            callbacks=[tt.callbacks.EarlyStopping(patience=CONFIG["patience"])],
            val_data=(x_val, y_val),
            verbose=False,
        )
    runtime = time.time() - start

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.compute_baseline_hazards(input=x_train, target=y_train)
        surv = model.predict_surv_df(x_test)

    mask = test_support_mask(train, test)
    test_eval = test.loc[mask].copy()
    grid = choose_time_grid(train, test_eval)
    surv_mat = survival_df_to_matrix(surv, grid)
    risk = np.nanmean(1.0 - surv_mat, axis=1)
    return {
        **cfg.to_dict(),
        "method": "pycox_CoxPH_DeepSurv",
        "model_class": "DeepSurv",
        "language": "Python",
        "ibs": integrated_brier_score_manual(surv_mat[mask], train, test_eval, grid),
        "cindex_td": cindex(test, risk),
        "runtime_sec": float(runtime),
        "status": "success",
        "error_message": "",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    config_columns = ["setting", "setting_type", "rep", "n", "p", "s", "rho"]
    cfgs = meta[config_columns].drop_duplicates()
    rows: list[dict[str, object]] = []

    for _, cfg in cfgs.iterrows():
        setting = str(cfg["setting"])
        rep = int(cfg["rep"])
        print(f"running Python DeepSurv {setting} rep={rep:03d}", flush=True)
        try:
            rows.append(run_one(cfg))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    **cfg.to_dict(),
                    "method": "pycox_CoxPH_DeepSurv",
                    "model_class": "DeepSurv",
                    "language": "Python",
                    "ibs": np.nan,
                    "cindex_td": np.nan,
                    "runtime_sec": np.nan,
                    "status": "failed",
                    "error_message": repr(exc),
                }
            )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "python_deepsurv.csv", index=False)
    print("Wrote", RESULTS_DIR / "python_deepsurv.csv")


if __name__ == "__main__":
    main()
