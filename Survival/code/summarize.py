"""Audit all expected fits and generate four-metric tables from formal runs only."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from run_task import code_hash

PACKAGES = {
    ("CoxPH", "R"): "survival",
    ("CoxPH", "Python"): "lifelines",
    ("Coxnet", "R"): "glmnet",
    ("Coxnet", "Python"): "scikit-survival",
    ("RSF", "R"): "randomForestSRC",
    ("RSF", "Python"): "scikit-survival",
    ("Boosting", "R"): "gbm",
    ("Boosting", "Python"): "scikit-survival",
    ("DeepSurv", "R"): "survivalmodels",
    ("DeepSurv", "Python"): "pycox",
}
METRICS = ["ibs", "cindex", "fitting_time_s", "peak_rss_mib"]
GROUP = ["setting", "n", "model", "language"]


def latex_table(summary, n):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Survival simulation results at $n="
        + str(n)
        + r"$. Values are mean (SD) over 100 independent datasets; memory is peak resident set size (RSS).}",
        r"\label{tab:survival_simulation_results"
        + ("" if n == 500 else "_n" + str(n))
        + r"}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Implementation & IBS & C-index & Time (s) & Peak RSS (MiB) \\",
        r"\midrule",
    ]
    for setting, title in [
        ("highdim", "High-dimensional sparse Cox"),
        ("nonlinear", "Smooth nonlinear survival"),
    ]:
        lines.append(r"\multicolumn{6}{l}{\textit{" + title + r"}}\\")
        for model in (["Coxnet"] if setting == "highdim" else ["CoxPH"]) + [
            "RSF",
            "Boosting",
            "DeepSurv",
        ]:
            for lang in ["R", "Python"]:
                z = summary[
                    (summary.setting == setting)
                    & (summary.n == n)
                    & (summary.model == model)
                    & (summary.language == lang)
                ].iloc[0]
                label = (
                    {"Coxnet": "Penalized Cox", "CoxPH": "Cox PH"}.get(model, model)
                    if lang == "R"
                    else ""
                )
                values = []
                for metric in METRICS:
                    digits = (
                        3
                        if metric in ["ibs", "cindex"]
                        else 2
                        if metric == "fitting_time_s"
                        else 1
                    )
                    values.append(
                        f"{z[metric+'_mean']:.{digits}f} ({z[metric+'_sd']:.{digits}f})"
                    )
                lines.append(
                    label
                    + " & "
                    + PACKAGES[(model, lang)]
                    + " ("
                    + lang
                    + ") & "
                    + " & ".join(values)
                    + r" \\"
                )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / "summary"
    out.mkdir(exist_ok=True)
    manifest = pd.read_csv(root / "tasks.csv")
    if (root / "RUN_SCOPE.json").exists():
        manifest = manifest[
            manifest.model.isin(
                json.loads((root / "RUN_SCOPE.json").read_text())["models"]
            )
        ]
    expected = set(manifest.task_id)
    records = {}
    attempts = []
    registry = (
        json.loads((root / "release_registry.json").read_text())["models"]
        if (root / "release_registry.json").exists()
        else {
            model: {"result_root": str(root), "code_sha256": code_hash()}
            for model in manifest.model.unique()
        }
    )
    for result_root in sorted({r["result_root"] for r in registry.values()}):
        for p in (Path(result_root) / "formal").glob("*/result.json"):
            r = json.loads(p.read_text())
            if (
                r["model"] not in registry
                or registry[r["model"]]["result_root"] != result_root
            ):
                continue
            assert r["task_id"] in expected and r["task_id"] not in records
            assert (
                r["code_sha256"] == registry[r["model"]]["code_sha256"]
            ), "Unexpected model-specific release"
            records[r["task_id"]] = r
            for attempt in p.parent.glob("attempt_*/result.json"):
                attempts.append(json.loads(attempt.read_text()))
    rows = []
    for row in manifest.to_dict("records"):
        result = records.get(row["task_id"], {})
        flat = {**row, "status": result.get("status", "missing")}
        for name in METRICS + [
            "attempt",
            "error",
            "hostname",
            "started_utc",
            "finished_utc",
            "worker_total_wall_s",
            "n_test_evaluated",
            "code_sha256",
            "config_sha256",
            "data_config_sha256",
        ]:
            flat[name] = result.get(name)
        if flat["status"] == "success":
            assert all(np.isfinite(flat[m]) for m in METRICS)
            assert flat["n_test_evaluated"] == int(row["n"] * 0.2)
        rows.append(flat)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "all_runs.csv", index=False)
    failure = frame[frame.status != "success"]
    failure.to_csv(out / "failed_or_missing.csv", index=False)
    (out / "retry_indices.txt").write_text(
        ",".join(str(int(i)) for i in failure.task_id) + "\n"
    )
    failed_attempts = [
        {
            k: r.get(k)
            for k in (
                "task_id",
                "dataset_id",
                "model",
                "language",
                "attempt",
                "status",
                "error",
            )
        }
        for r in attempts
        if r["status"] != "success"
    ]
    (out / "failed_attempts.json").write_text(json.dumps(failed_attempts, indent=2))
    good = frame[frame.status == "success"]
    groups = []
    for key, expected_rows in frame.groupby(GROUP, sort=False):
        valid = expected_rows[expected_rows.status == "success"]
        item = dict(zip(GROUP, key))
        item.update(
            expected=len(expected_rows),
            success=len(valid),
            failed_or_missing=len(expected_rows) - len(valid),
        )
        for metric in METRICS:
            item[metric + "_mean"] = valid[metric].mean()
            item[metric + "_sd"] = valid[metric].std(ddof=1)
        groups.append(item)
    summary = pd.DataFrame(groups)
    summary.to_csv(out / "summary.csv", index=False)
    data_meta = [
        json.loads(p.read_text()) for p in (root / "data").glob("*/metadata.json")
    ]
    rates = (
        pd.DataFrame(data_meta)
        .groupby(["setting", "n"])
        .censoring_rate.agg(["mean", "std", "min", "max"])
    )
    rates.to_csv(out / "censoring_rates.csv")
    counts = {
        "expected": len(manifest),
        "success": len(good),
        "failed_or_missing": len(failure),
        "failed_attempts": len(failed_attempts),
        "tasks_with_failed_attempts": len(set(r["task_id"] for r in failed_attempts)),
        "code_releases": {model: r["code_sha256"] for model, r in registry.items()},
    }
    (out / "completion.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))
    if a.require_complete and len(failure):
        raise SystemExit(
            "Formal results are incomplete; no final LaTeX tables produced"
        )
    if not len(failure):
        assert (
            len(summary) == len(manifest[GROUP].drop_duplicates())
            and (summary.success == 100).all()
        )
        if len(summary) == 48:
            for n in (500, 1000, 1500):
                (out / f"results_n{n}.tex").write_text(latex_table(summary, n))
        pair = good.pivot(
            index=["setting", "n", "model", "replication"],
            columns="language",
            values=METRICS,
        )
        differences = pd.DataFrame(
            {metric: pair[metric]["Python"] - pair[metric]["R"] for metric in METRICS}
        )
        differences.to_csv(out / "paired_python_minus_R.csv")
    with (out / "raw_records.jsonl").open("w") as stream:
        for task_id in sorted(records):
            stream.write(json.dumps(records[task_id]) + "\n")


if __name__ == "__main__":
    main()
