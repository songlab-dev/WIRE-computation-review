"""Combine model outputs and create the manuscript summary table source."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))
RESULTS_DIR = EXPERIMENT_DIR / "results"

RESULT_FILES = (
    "python_models.csv",
    "r_models.csv",
    "python_deepsurv.csv",
    "r_deepsurv.csv",
)

RESULT_COLUMNS = [
    "setting",
    "setting_type",
    "rep",
    "n",
    "p",
    "s",
    "rho",
    "language",
    "method",
    "model_class",
    "ibs",
    "cindex_td",
    "runtime_sec",
    "status",
    "error_message",
]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def main() -> None:
    missing = [name for name in RESULT_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model outputs: {', '.join(missing)}")

    all_results = pd.concat(
        [pd.read_csv(RESULTS_DIR / name) for name in RESULT_FILES],
        ignore_index=True,
        sort=False,
    )
    failures = all_results.loc[all_results["status"] != "success"]
    if not failures.empty:
        failures.to_csv(RESULTS_DIR / "failed_runs.csv", index=False)
        raise RuntimeError(
            f"{len(failures)} model fits failed; see {RESULTS_DIR / 'failed_runs.csv'}"
        )

    all_results[RESULT_COLUMNS].to_csv(RESULTS_DIR / "all_results.csv", index=False)
    group_columns = [
        "setting",
        "setting_type",
        "n",
        "p",
        "s",
        "rho",
        "model_class",
        "language",
    ]
    summary = (
        all_results.groupby(group_columns, dropna=False)[
            ["ibs", "cindex_td", "runtime_sec"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    flatten_columns(summary).to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'all_results.csv'}")
    print(f"Wrote {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
