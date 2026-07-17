"""Generate a manuscript-ready LaTeX table from summary.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(os.environ.get("EXPERIMENT_DIR", "outputs/main_n500"))

MODEL_ORDER = {
    "highdim": ["Penalized Cox", "RSF", "Boosting", "DeepSurv"],
    "nonlinear": ["CoxPH", "RSF", "Boosting", "DeepSurv"],
}

PACKAGE_NAMES = {
    ("Penalized Cox", "R"): r"R (\texttt{glmnet})",
    ("Penalized Cox", "Python"): r"Python (\texttt{scikit-survival})",
    ("CoxPH", "R"): r"R (\texttt{survival})",
    ("CoxPH", "Python"): r"Python (\texttt{lifelines})",
    ("RSF", "R"): r"R (\texttt{randomForestSRC})",
    ("RSF", "Python"): r"Python (\texttt{scikit-survival})",
    ("Boosting", "R"): r"R (\texttt{gbm})",
    ("Boosting", "Python"): r"Python (\texttt{scikit-survival})",
    ("DeepSurv", "R"): r"R (\texttt{survivalmodels})",
    ("DeepSurv", "Python"): r"Python (\texttt{pycox})",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "manuscript_table.tex",
    )
    return parser.parse_args()


def setting_label(setting: str, setting_type: str, n: int) -> str:
    if setting_type == "highdim":
        base = "High-dimensional sparse Cox"
    else:
        base = "Smooth nonlinear survival"
    return base if n == 500 else f"{base} ($n={n}$)"


def metric(mean: float, sd: float, bold: bool) -> str:
    value = f"{mean:.3f} ({sd:.3f})"
    return rf"\textbf{{{value}}}" if bold else value


def make_rows(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    setting_order = summary[["setting", "setting_type", "n"]].drop_duplicates()

    for setting_index, (_, setting_row) in enumerate(setting_order.iterrows(), start=1):
        setting = str(setting_row["setting"])
        setting_type = str(setting_row["setting_type"])
        n = int(setting_row["n"])
        block = summary.loc[summary["setting"] == setting].copy()
        best_ibs = f"{block['ibs_mean'].min():.3f}"
        best_cindex = f"{block['cindex_td_mean'].max():.3f}"
        label = f"Setting {setting_index}: {setting_label(setting, setting_type, n)}"
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label}}}}}\\")

        for model_class in MODEL_ORDER[setting_type]:
            for language_index, language in enumerate(("R", "Python")):
                row = block.loc[
                    (block["model_class"] == model_class)
                    & (block["language"] == language)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected one row for {setting}, {model_class}, {language}; "
                        f"found {len(row)}."
                    )
                values = row.iloc[0]
                model_label = model_class if language_index == 0 else ""
                package = PACKAGE_NAMES[(model_class, language)]
                ibs = metric(
                    float(values["ibs_mean"]),
                    float(values["ibs_std"]),
                    f"{float(values['ibs_mean']):.3f}" == best_ibs,
                )
                cindex = metric(
                    float(values["cindex_td_mean"]),
                    float(values["cindex_td_std"]),
                    f"{float(values['cindex_td_mean']):.3f}" == best_cindex,
                )
                runtime = metric(
                    float(values["runtime_sec_mean"]),
                    float(values["runtime_sec_std"]),
                    False,
                )
                lines.append(
                    f"{model_label} & {package} & {ibs} & {cindex} & {runtime} \\\\"
                )
        lines.append(r"\hline")

    return lines


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    repetitions = int(os.environ.get("N_REPLICATIONS", "50"))
    replication_label = "replication" if repetitions == 1 else "replications"
    censoring = 100 * float(os.environ.get("TARGET_CENSORING", "0.30"))
    rows = "\n".join(make_rows(summary))
    table = rf"""\begin{{table}}[htbp]
\centering
\small
\caption{{Simulation results comparing representative R and Python survival-analysis implementations under {censoring:.0f}\% censoring. Values are mean (SD) over {repetitions} {replication_label}.}}
\label{{tab:survival_simulation_results}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{lllcc}}
\hline
Model & Implementation & IBS & C-index & Time (s) \\
\hline
{rows}
\end{{tabular}}
\vspace{{4pt}}
\begin{{minipage}}{{\linewidth}}
\footnotesize
\textit{{Notes:}}
IBS denotes the IPCW-integrated Brier score, where smaller values indicate better prediction.
C-index measures discrimination, where larger values indicate better risk ranking.
Time denotes mean model-fitting runtime in seconds per replication, excluding data generation and evaluation.
Runtime differences reflect package-specific implementations and tuning procedures and should not be interpreted as language-level benchmarks.
Bold values indicate the best predictive performance within each setting for IBS and C-index.
\end{{minipage}}
\end{{table}}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
