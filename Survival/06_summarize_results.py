"""Combine model outputs and create the manuscript summary table source."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))
DATA_DIR = EXPERIMENT_DIR / "data"
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

KEY_COLUMNS = ["setting", "rep", "model_class", "language"]
METRIC_COLUMNS = ["ibs", "cindex_td", "runtime_sec"]
MODEL_CLASSES = {
    "highdim": ["Penalized Cox", "RSF", "Boosting", "DeepSurv"],
    "nonlinear": ["CoxPH", "RSF", "Boosting", "DeepSurv"],
}


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def validate_results(all_results: pd.DataFrame) -> None:
    missing_columns = sorted(set(RESULT_COLUMNS) - set(all_results.columns))
    if missing_columns:
        raise ValueError(f"Missing result columns: {', '.join(missing_columns)}")

    failures = all_results.loc[all_results["status"] != "success"]
    if not failures.empty:
        failures.to_csv(RESULTS_DIR / "failed_runs.csv", index=False)
        raise RuntimeError(
            f"{len(failures)} model fits failed; "
            f"see {RESULTS_DIR / 'failed_runs.csv'}"
        )

    duplicates = all_results.loc[
        all_results.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS
    ]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate model results found: "
            f"{duplicates.drop_duplicates().head().to_dict('records')}"
        )

    metrics = all_results[METRIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_metrics = ~np.isfinite(metrics.to_numpy())
    if invalid_metrics.any():
        bad_rows = all_results.loc[invalid_metrics.any(axis=1), KEY_COLUMNS]
        raise ValueError(
            "Non-finite evaluation metrics found: "
            f"{bad_rows.head().to_dict('records')}"
        )
    if not all_results["cindex_td"].between(0, 1).all():
        raise ValueError("C-index values must be between 0 and 1.")
    if (all_results["runtime_sec"] < 0).any():
        raise ValueError("Runtime values must be nonnegative.")

    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing simulation metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    configs = metadata[["setting", "setting_type", "rep"]].drop_duplicates()
    expected = {
        (str(row.setting), int(row.rep), model_class, language)
        for row in configs.itertuples(index=False)
        for model_class in MODEL_CLASSES[str(row.setting_type)]
        for language in ("R", "Python")
    }
    observed = {
        (str(row.setting), int(row.rep), str(row.model_class), str(row.language))
        for row in all_results[KEY_COLUMNS].itertuples(index=False)
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "Incomplete result coverage. "
            f"Missing examples: {missing[:5]}; "
            f"unexpected examples: {unexpected[:5]}"
        )


def main() -> None:
    missing = [name for name in RESULT_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model outputs: {', '.join(missing)}")

    all_results = pd.concat(
        [pd.read_csv(RESULTS_DIR / name) for name in RESULT_FILES],
        ignore_index=True,
        sort=False,
    )
    validate_results(all_results)

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
        all_results.groupby(group_columns, dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    flatten_columns(summary).to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'all_results.csv'}")
    print(f"Wrote {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
