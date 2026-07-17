"""Generate the two simulation settings reported in the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))
DATA_DIR = EXPERIMENT_DIR / "data"

N_REPLICATIONS = int(os.environ.get("N_REPLICATIONS", "50"))
TARGET_CENSORING = float(os.environ.get("TARGET_CENSORING", "0.30"))


def sample_sizes() -> list[int]:
    raw = os.environ.get("SAMPLE_SIZES", "500")
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("SAMPLE_SIZES must contain positive comma-separated integers.")
    return values


def standardize(values: np.ndarray) -> np.ndarray:
    sd = float(np.std(values))
    if sd == 0:
        raise ValueError("Cannot standardize a constant risk score.")
    return (values - float(np.mean(values))) / sd


_CHOL_CACHE: dict[tuple[int, float], np.ndarray] = {}


def generate_covariates(
    n: int,
    p: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    key = (p, rho)
    if key not in _CHOL_CACHE:
        index = np.arange(p)
        covariance = rho ** np.abs(np.subtract.outer(index, index))
        _CHOL_CACHE[key] = np.linalg.cholesky(covariance)
    return rng.normal(size=(n, p)) @ _CHOL_CACHE[key].T


def sparse_linear_score(x: np.ndarray, s: int = 50) -> np.ndarray:
    magnitudes = 0.93 ** np.arange(s)
    signs = np.where(np.arange(s) % 2 == 0, 1.0, -1.0)
    beta = np.zeros(x.shape[1])
    beta[:s] = signs * magnitudes
    return standardize(x @ beta)


def smooth_nonlinear_score(x: np.ndarray) -> np.ndarray:
    score = (
        0.4 * x[:, 0]
        + 0.8 * np.sin(x[:, 1])
        + 0.8 * (x[:, 2] ** 2 - 1.0)
        + 0.7 * np.tanh(x[:, 3] + x[:, 4])
        + 0.5 * np.exp(-(x[:, 5] ** 2))
    )
    return standardize(score)


def generate_censoring_time(
    event_time: np.ndarray,
    target: float,
    rng: np.random.Generator,
    max_iter: int = 100,
    tolerance: float = 0.003,
) -> np.ndarray:
    low = 1e-8
    high = float(np.quantile(event_time, 0.99) * 20.0)
    best_censoring: np.ndarray | None = None
    best_difference = np.inf

    for _ in range(max_iter):
        upper_bound = (low + high) / 2.0
        censoring = rng.uniform(0, upper_bound, size=len(event_time))
        censoring_rate = float(np.mean(event_time > censoring))
        difference = abs(censoring_rate - target)
        if difference < best_difference:
            best_difference = difference
            best_censoring = censoring.copy()
        if difference < tolerance:
            return censoring
        if censoring_rate > target:
            low = upper_bound
        else:
            high = upper_bound

    if best_censoring is None:
        raise RuntimeError("Censoring calibration failed.")
    return best_censoring


def add_outcome(
    frame: pd.DataFrame,
    risk_score: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    uniform = rng.uniform(size=frame.shape[0])
    event_time = (-np.log(uniform) / (0.1 * np.exp(risk_score))) ** (1.0 / 1.5)
    censoring_time = generate_censoring_time(event_time, TARGET_CENSORING, rng)
    frame["time"] = np.minimum(event_time, censoring_time)
    frame["event"] = (event_time <= censoring_time).astype(int)
    return frame, float(1.0 - frame["event"].mean())


def split_data(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    indices = rng.permutation(frame.shape[0])
    n_train = int(0.6 * frame.shape[0])
    n_validation = int(0.2 * frame.shape[0])
    frame["split"] = "test"
    frame.loc[indices[:n_train], "split"] = "train"
    frame.loc[indices[n_train : n_train + n_validation], "split"] = "val"
    return frame


def settings() -> list[dict[str, object]]:
    values = sample_sizes()
    high_dimensional = [
        {
            "setting": "setting1_highdim_s50" if n == 500 else f"setting1_highdim_s50_n{n}",
            "setting_type": "highdim",
            "n": n,
            "p": 500,
            "s": 50,
            "rho": 0.30,
        }
        for n in values
    ]
    nonlinear = [
        {
            "setting": (
                "setting2_smooth_nonlinear"
                if n == 500
                else f"setting2_smooth_nonlinear_n{n}"
            ),
            "setting_type": "nonlinear",
            "n": n,
            "p": 50,
            "s": "",
            "rho": 0.30,
        }
        for n in values
    ]
    return high_dimensional + nonlinear


def save_splits(frame: pd.DataFrame, prefix: Path) -> None:
    for split in ("train", "val", "test"):
        output = frame.loc[frame["split"] == split].drop(columns="split")
        output.to_csv(f"{prefix}_{split}.csv", index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []

    for setting_id, config in enumerate(settings()):
        setting = str(config["setting"])
        n = int(config["n"])
        p = int(config["p"])
        rho = float(config["rho"])

        for replication in range(1, N_REPLICATIONS + 1):
            seed = 910000 + setting_id * 1000 + replication
            rng = np.random.default_rng(seed)
            x = generate_covariates(n, p, rho, rng)
            if config["setting_type"] == "highdim":
                risk_score = sparse_linear_score(x, s=int(config["s"]))
            else:
                risk_score = smooth_nonlinear_score(x)

            frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(p)])
            frame, actual_censoring = add_outcome(frame, risk_score, rng)
            frame = split_data(frame, rng)
            save_splits(frame, DATA_DIR / f"{setting}_rep{replication:03d}")

            metadata.append(
                {
                    **config,
                    "rep": replication,
                    "seed": seed,
                    "target_censor_rate": TARGET_CENSORING,
                    "actual_censor_rate": actual_censoring,
                    "event_rate": float(frame["event"].mean()),
                    "n_train": int((frame["split"] == "train").sum()),
                    "n_val": int((frame["split"] == "val").sum()),
                    "n_test": int((frame["split"] == "test").sum()),
                }
            )
            print(f"generated {setting} rep={replication:03d} seed={seed}", flush=True)

    pd.DataFrame(metadata).to_csv(DATA_DIR / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
