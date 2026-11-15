#!/usr/bin/env python3
"""Export published GLM tables without changing experimental records.

Full exports require the complete canonical 100-dataset x 5-fit experiment.
Smoke, pilot and sensitivity exports require --allow-validation and are labeled.
Numerical checks run on temporary copies; CV differences remain informational.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from check_results import verify_results
from run_benchmark import ROOT, make_seeds, run_fingerprint, source_hashes
from summarize_results import LABELS, tex_escape

PENALIZED = {"highdim", "sparse_highdim"}
IDENTITY = ["scenario", "seed", "method"]
METRICS = {"fit_seconds": "median_fit_seconds", "peak_rss_mib": "peak_rss_mib",
           "test_deviance": "test_deviance", "f1": "f1"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def booleans(series, name):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid boolean in " + name)
    return values.isin(["true", "1"])


def integers(series, name):
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values).all() or (values != np.floor(values)).any():
        raise ValueError("Noninteger value in " + name)
    return values.astype("int64")


def validate_records(root, allow_validation=False):
    root = Path(root).resolve()
    if not (root / "run.json").is_file() and (root / "merged/run.json").is_file():
        root = root / "merged"
    run = json.loads((root / "run.json").read_text())
    conf = run["config"]
    if conf["profile"] == "full":
        canonical = json.loads((ROOT / "configs/full.json").read_text())
        if conf != canonical:
            raise ValueError("Full tables require the canonical configs/full.json (100 datasets x 5 fits).")
    elif not allow_validation:
        raise ValueError("This is not a full run; use --allow-validation for labeled exploratory tables.")
    seeds = make_seeds(conf)
    if run.get("selected_seeds") != seeds:
        raise ValueError("The run must include every configured independent dataset.")
    hashes = source_hashes()
    if run.get("source_sha256") != hashes:
        raise ValueError("Recorded benchmark source differs from this release; use its original source.")
    if run.get("fingerprint") != run_fingerprint(conf, hashes, seeds):
        raise ValueError("Invalid run fingerprint")
    if len(seeds) != conf["statistical_replicates"] or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid independent-dataset seed specification")

    frame = pd.read_csv(root / "raw_results.csv")
    required = set(IDENTITY + ["regime", "n", "p", "density", "timing_dataset", "success",
                               "quality_ok", "status", "requested_repeats", "completed_repeats",
                               "median_fit_seconds", "total_fit_seconds", "peak_rss_mib", "rss_samples",
                               "kkt_scaled", "objective", "test_deviance", "f1", "lambda_selected", "cv_deviance"])
    if required - set(frame.columns):
        raise ValueError("Missing raw columns: " + ", ".join(sorted(required - set(frame.columns))))
    for column in ["seed", "n", "p", "requested_repeats", "completed_repeats", "rss_samples"]:
        frame[column] = integers(frame[column], column)
    for column in ["success", "quality_ok", "timing_dataset"]:
        frame[column] = booleans(frame[column], column)
    if frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate dataset-method record")
    expected = {}
    for scenario in conf["scenarios"]:
        methods = conf["penalized_methods"] if scenario["regime"] in PENALIZED else conf["unpenalized_methods"]
        for seed in seeds:
            for method in methods:
                expected[(scenario["id"], seed, method)] = scenario
    actual = set(frame[IDENTITY].itertuples(index=False, name=None))
    if actual != set(expected):
        raise ValueError(f"Incomplete/unexpected dataset-method grid: {len(actual)} of {len(expected)} expected")
    for row in frame.itertuples(index=False):
        scenario = expected[(row.scenario, row.seed, row.method)]
        if (row.regime, row.n, row.p, row.density) != (scenario["regime"], scenario["n"], scenario["p"], scenario.get("density", 1.0)):
            raise ValueError("Recorded design does not match configuration")
        timing = row.seed in conf["timing_seeds"]
        repeats = conf["timing_repeats"] if timing else 1
        if row.timing_dataset != timing or row.requested_repeats != repeats:
            raise ValueError("Recorded timing protocol does not match configuration")
        if not 0 <= row.completed_repeats <= repeats or row.rss_samples < 0:
            raise ValueError("Invalid recorded fit/sample count")
        if row.success:
            if not row.quality_ok or row.status != "ok" or row.completed_repeats != repeats:
                raise ValueError("A successful task did not complete its quality/fit requirements")
            if not np.isfinite(row.kkt_scaled) or not 0 <= row.kkt_scaled <= conf["kkt_tol"]:
                raise ValueError("A successful task failed the recorded KKT criterion")
            if not np.isfinite(row.median_fit_seconds) or row.median_fit_seconds <= 0:
                raise ValueError("Invalid successful-task timing")
            if not np.isfinite(row.test_deviance) or row.test_deviance < 0:
                raise ValueError("Invalid successful-task test deviance")
            if row.regime in PENALIZED and (not np.isfinite(row.f1) or not 0 <= row.f1 <= 1):
                raise ValueError("Invalid successful-task support F1")
        if row.rss_samples == 0 and not pd.isna(row.peak_rss_mib):
            raise ValueError("An unsampled RSS must be missing, not zero")
        if row.rss_samples > 0 and (not np.isfinite(row.peak_rss_mib) or row.peak_rss_mib <= 0):
            raise ValueError("Invalid sampled peak RSS")

    trials = pd.read_csv(root / "fit_trials.csv")
    for column in ["seed", "trial"]:
        trials[column] = integers(trials[column], column)
    if trials.duplicated(IDENTITY + ["trial"]).any():
        raise ValueError("Duplicate measured trial")
    seconds = pd.to_numeric(trials.seconds, errors="raise")
    if not np.isfinite(seconds).all() or (seconds <= 0).any():
        raise ValueError("Invalid measured trial time")
    if not set(trials[IDENTITY].itertuples(index=False, name=None)).issubset(actual):
        raise ValueError("Unexpected measured-trial identity")
    by_task = {key: group for key, group in trials.groupby(IDENTITY, sort=False)}
    for row in frame.itertuples(index=False):
        trial = by_task.get((row.scenario, row.seed, row.method))
        count = 0 if trial is None else len(trial)
        if count != row.completed_repeats:
            raise ValueError("Measured trial count differs from the task checkpoint")
        if count:
            if set(trial.trial) != set(range(1, count + 1)):
                raise ValueError("Measured trial indices are not consecutive")
            if not np.isclose(trial.seconds.median(), row.median_fit_seconds, rtol=1e-10, atol=1e-12):
                raise ValueError("Dataset timing is not the median of its measured fits")
            if not np.isclose(trial.seconds.sum(), row.total_fit_seconds, rtol=1e-10, atol=1e-12):
                raise ValueError("Dataset total timing differs from its measured fits")
            if "timing_dataset" in trial and not booleans(trial.timing_dataset, "trial timing_dataset").eq(row.timing_dataset).all():
                raise ValueError("Trial timing flag differs from its dataset")

    # The checker writes verification.json; isolate that side effect from inputs.
    with tempfile.TemporaryDirectory(prefix="glm-export-check-") as private:
        for name in ["raw_results.csv", "run.json"]:
            shutil.copy2(root / name, Path(private) / name)
        with contextlib.redirect_stdout(io.StringIO()):
            verification = verify_results(private)
    return root, conf, frame, trials, verification


def summary(frame, conf):
    rows = []
    for scenario in conf["scenarios"]:
        methods = conf["penalized_methods"] if scenario["regime"] in PENALIZED else conf["unpenalized_methods"]
        for method in methods:
            group = frame[frame.scenario.eq(scenario["id"]) & frame.method.eq(method)]
            okay = group[group.success & group.timing_dataset]
            row = {"scenario": scenario["id"], "regime": scenario["regime"], "n": scenario["n"],
                   "p": scenario["p"], "density": scenario.get("density", 1.0), "method": method,
                   "implementation": LABELS[method], "attempted_datasets": len(group),
                   "successful_datasets": int(group.success.sum()), "failed_datasets": int((~group.success).sum()),
                   "attempted_timing_datasets": int(group.timing_dataset.sum()),
                   "successful_timing_datasets": len(okay),
                   "requested_measured_fits": int(group.requested_repeats.sum()),
                   "completed_measured_fits": int(group.completed_repeats.sum())}
            for metric, column in METRICS.items():
                source = okay if metric in {"fit_seconds", "peak_rss_mib"} else group[group.success]
                if metric == "f1" and scenario["regime"] not in PENALIZED:
                    source = source.iloc[:0]
                values = pd.to_numeric(source[column], errors="raise").dropna()
                n = len(values)
                sd = float(values.std(ddof=1)) if n > 1 else np.nan
                row.update({metric + "_mean": float(values.mean()) if n else np.nan,
                            metric + "_sd": sd, metric + "_mcse": sd / math.sqrt(n) if n > 1 else np.nan,
                            metric + "_n": n})
            rows.append(row)
    return pd.DataFrame(rows)


def tex_performance_table(frame, conf, penalized):
    scenarios = [s["id"] for s in conf["scenarios"] if (s["regime"] in PENALIZED) == penalized]
    methods = conf["penalized_methods"] if penalized else conf["unpenalized_methods"]
    if not scenarios:
        return "% No scenarios in this category.\n"
    caption = (r"$\ell_1$-penalized" if penalized else "Unpenalized") + " Poisson regression: mean fitting time and peak process RSS."
    if conf["profile"] != "full":
        caption = "EXPLORATORY / " + tex_escape(conf["profile"]) + ": " + caption
    label = "tab:glm-pen-performance" if penalized else "tab:glm-unpen-performance"
    lines = [r"\begin{table}[!tbp]", r"\centering", r"\caption{" + caption + r"}\label{" + label + "}",
             r"\scriptsize\setlength{\tabcolsep}{3.7pt}\renewcommand{\arraystretch}{1.13}",
             r"\begin{tabular}{@{}l" + "r" * len(scenarios) + r"@{}}", r"\toprule"]
    for index, (metric, title) in enumerate([("fit_seconds", "Fitting time (seconds)"), ("peak_rss_mib", "Peak memory (MiB)")]):
        if index:
            lines.append(r"\midrule")
        lines.extend([r"\multicolumn{" + str(len(scenarios) + 1) + r"}{@{}l}{\textit{" + title + r"}}\\",
                      "Implementation & " + " & ".join(map(tex_escape, scenarios)) + r"\\", r"\midrule"])
        for method in methods:
            language, package = LABELS[method].split(": ", 1)
            cells = [language + r": \texttt{" + tex_escape(package) + "}"]
            for scenario in scenarios:
                block = frame[frame.scenario.eq(scenario)]
                values = block[metric + "_mean"]
                value = block.loc[block.method.eq(method), metric + "_mean"].iloc[0]
                cell = "--" if not np.isfinite(value) else f"{value:.3g}"
                if np.isfinite(value) and math.isclose(value, values.min(), rel_tol=1e-12, abs_tol=1e-15):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(" & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                  r"\begin{minipage}{0.98\linewidth}\scriptsize\textit{Note.} "
                  + f"Each scenario has {conf['statistical_replicates']} independent datasets. "
                  + f"Time averages the within-dataset median of {conf['timing_repeats']} measured fits; one untimed warm-up precedes each task. "
                  + "RSS averages sampled per-dataset peaks over the fitting block, including the runtime and resident inputs. "
                  + "Means exclude unsuccessful tasks. Missing RSS samples remain missing. "
                  + "The companion CSV reports SD, MCSE and effective dataset counts for each metric. Boldface marks the smallest mean per scenario and metric. "])
    failures = frame[frame.failed_datasets.gt(0)]
    if len(failures):
        notes = [tex_escape(r.scenario + " / " + LABELS[r.method]) + f": {r.successful_datasets}/{r.attempted_datasets} successful" for r in failures.itertuples()]
        lines.append("; ".join(notes) + ". Other entries use all configured datasets. ")
    else:
        lines.append("All entries use every configured dataset. ")
    missing = frame[frame.peak_rss_mib_n.lt(frame.successful_timing_datasets)]
    if len(missing):
        notes = [tex_escape(r.scenario + " / " + LABELS[r.method]) + f": RSS available for {r.peak_rss_mib_n} datasets" for r in missing.itertuples()]
        lines.append("; ".join(notes) + ". ")
    lines.extend([r"\end{minipage}", r"\end{table}"])
    return "\n".join(line.rstrip() for line in lines) + "\n"


def main_cell(mean, sd, metric):
    if not np.isfinite(mean):
        return "--"
    digits = (2 if mean >= 1 else 3 if mean >= 0.1 else 4) if metric == "fit_seconds" else (1 if metric == "peak_rss_mib" else 3)
    value = f"{mean:.{digits}f}"
    return value + (f" ({sd:.{digits}f})" if np.isfinite(sd) else "")


def tex_table(frame, conf, penalized):
    """Match the final manuscript endpoints; mean (between-dataset SD)."""
    scenarios = [s for s in conf["scenarios"] if (s["regime"] in PENALIZED) == penalized]
    if not scenarios:
        return "% No scenarios in this category.\n"
    panels = [scenarios] if penalized else [[s for s in scenarios if s["regime"] == regime] for regime in ["dense", "sparse"]]
    panels = [panel for panel in panels if panel]
    methods = conf["penalized_methods"] if penalized else conf["unpenalized_methods"]
    columns = ["fit_seconds", "peak_rss_mib", "test_deviance"] + (["f1"] if penalized else [])
    headers = ["Implementation", "Time (s)", "Peak memory (MiB)", "Test deviance"] + ([r"Support $F_1$"] if penalized else [])
    lines = []
    for index, panel in enumerate(panels):
        caption = "Computational and statistical performance for " + (r"$\ell_1$-penalized" if penalized else "unpenalized") + " Poisson regression. "
        if not penalized:
            caption += "Panel " + ("A" if panel[0]["regime"] == "dense" else "B") + ": " + panel[0]["regime"] + " designs. "
        caption += "Entries are mean (SD) across independent datasets."
        if conf["profile"] != "full":
            caption = "EXPLORATORY / " + tex_escape(conf["profile"]) + ": " + caption
        lines.extend([r"\begin{table}[!p]", r"\centering"])
        if index:
            lines.append(r"\ContinuedFloat")
        lines.append(r"\caption{" + caption + "}")
        if index == 0:
            lines.append(r"\label{" + ("tab:glm-pen-summary" if penalized else "tab:glm-unpen-summary") + "}")
        lines.extend([r"\scriptsize\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.13}",
                      r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "r" * len(columns) + r"@{}}",
                      r"\toprule", " & ".join(headers) + r"\\", r"\midrule"])
        for ordinal, scenario in enumerate(panel):
            if ordinal:
                lines.append(r"\midrule")
            heading = tex_escape(scenario["id"]) + f": $n={scenario['n']:,}$, $p={scenario['p']:,}$, " + r"$\delta=" + str(scenario.get("density", 1.0)) + "$"
            lines.append(r"\multicolumn{" + str(len(headers)) + r"}{@{}l}{\textit{" + heading + r"}}\\")
            group = frame[frame.scenario.eq(scenario["id"])]
            for method in methods:
                row = group[group.method.eq(method)].iloc[0]
                language, package = LABELS[method].split(": ", 1)
                cells = [language + r": \texttt{" + tex_escape(package) + "}"]
                for metric in columns:
                    mean, sd = row[metric + "_mean"], row[metric + "_sd"]
                    cell = main_cell(mean, sd, metric)
                    if metric in {"fit_seconds", "peak_rss_mib"} and np.isfinite(mean) and math.isclose(mean, group[metric + "_mean"].min(), rel_tol=1e-12, abs_tol=1e-15):
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
                lines.append(" & ".join(cells) + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular*}", r"\par\smallskip",
                      r"\begin{minipage}{0.98\linewidth}\scriptsize\textit{Note.} "
                      + f"{conf['statistical_replicates']} independent datasets per scenario. Time uses the within-dataset median of {conf['timing_repeats']} measured fits. "
                      + "RSS is the sampled peak over each fitting block, including runtime and inputs. "
                      + "SD and MCSE use independent datasets. Boldface marks the smallest mean time/RSS. "])
        selected = frame[frame.scenario.isin([s["id"] for s in panel])]
        failed = selected[selected.failed_datasets.gt(0)]
        if len(failed):
            lines.append("; ".join(tex_escape(r.scenario + " / " + LABELS[r.method]) + f": {r.successful_datasets}/{r.attempted_datasets} successful" for r in failed.itertuples()) + ". Other entries use all configured datasets. ")
        else:
            lines.append("All tasks in this panel succeeded. ")
        missing = selected[selected.peak_rss_mib_n.lt(selected.successful_timing_datasets)]
        if len(missing):
            lines.append("Missing RSS samples have smaller effective counts, reported in the companion CSV. ")
        lines.extend([r"\end{minipage}", r"\end{table}"])
    return "\n".join(line.rstrip() for line in lines) + "\n"


def export(results, destination=None, allow_validation=False):
    root, conf, frame, trials, verification = validate_records(results, allow_validation)
    destination = Path(destination).resolve() if destination else root / "public_tables"
    if destination == root:
        raise ValueError("Choose a separate table directory; experimental inputs are read-only.")
    data = summary(frame, conf)
    cpu_models = sorted(frame.cpu_model.dropna().unique().tolist()) if "cpu_model" in frame else []
    metadata = {
        "profile": conf["profile"], "canonical_full_run": conf["profile"] == "full",
        "independent_datasets_per_scenario": conf["statistical_replicates"],
        "timing_datasets_per_scenario": len(conf["timing_seeds"]), "timed_fits_per_dataset_method": conf["timing_repeats"],
        "untimed_warmups_per_dataset_method": 1, "dataset_method_tasks": len(frame),
        "successful_tasks": int(frame.success.sum()), "failed_tasks": int((~frame.success).sum()),
        "requested_measured_fits": int(frame.requested_repeats.sum()), "recorded_measured_fits": len(trials),
        "successful_task_measured_fits": int(frame.loc[frame.success, "completed_repeats"].sum()),
        "missing_successful_rss_records": int((frame.success & frame.timing_dataset & frame.peak_rss_mib.isna()).sum()),
        "cpu_models": cpu_models, "cv_difference_policy": verification["cv_difference_policy"],
        "cv_groups_outside_previous_limit": verification["cv_groups_outside_previous_limit"],
        "numerical_merge_checks_passed": verification["merge_checks_passed"],
        "input_sha256": {name: sha256(root / name) for name in ["run.json", "raw_results.csv", "fit_trials.csv"]},
        "exporter_sha256": sha256(__file__),
        "metrics": {
            "fit_seconds": "Mean of per-dataset median measured fitting times, seconds.",
            "peak_rss_mib": "Mean sampled peak worker RSS per fitting block, MiB; runtime and inputs included.",
            "test_deviance": "Mean holdout Poisson deviance on each successful independent dataset.",
            "f1": "Support recovery F1 for successful penalized datasets only; support threshold is recorded in the configuration.",
            "sd": "Between-independent-dataset sample SD.", "mcse": "SD / sqrt(actual nonmissing successful dataset count)."},
    }
    for name in ["worker_hostname", "cpu_model"]:
        if name in frame:
            metadata["same_" + name + "_within_dataset"] = bool(frame.groupby(["scenario", "seed"])[name].nunique().eq(1).all())
    destination.mkdir(parents=True, exist_ok=True)
    for penalized, number, category in [(False, 1, "unpenalized"), (True, 2, "penalized")]:
        subset = data[data.regime.isin(PENALIZED).eq(penalized)]
        if not penalized:
            subset = subset.drop(columns=[c for c in subset.columns if c.startswith("f1_")])
        subset.to_csv(destination / f"table{number}_{category}.csv", index=False)
        (destination / f"table{number}_main.tex").write_text(tex_table(subset, conf, penalized))
        (destination / f"table{number}_time_memory.tex").write_text(tex_performance_table(subset, conf, penalized))
        ids = [s["id"] for s in conf["scenarios"] if (s["regime"] in PENALIZED) == penalized]
        methods = conf["penalized_methods"] if penalized else conf["unpenalized_methods"]
        for metric, stem in [("fit_seconds", "time"), ("peak_rss_mib", "memory")]:
            wide = subset.pivot(index="method", columns="scenario", values=metric + "_mean").reindex(index=methods, columns=ids)
            wide.index = pd.Index([LABELS[m] for m in wide.index], name="Implementation")
            wide.to_csv(destination / f"{stem}_{category}.csv")
    columns = [c for c in ["scenario", "seed", "method", "status", "error", "requested_repeats", "completed_repeats", "quality_ok", "kkt_scaled", "cpu_model"] if c in frame]
    frame.loc[~frame.success, columns].to_csv(destination / "failures.csv", index=False)
    pd.DataFrame([{**s, "density": s.get("density", 1.0)} for s in conf["scenarios"]]).to_csv(destination / "scenarios.csv", index=False)
    (destination / "run_summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (destination / "README.md").write_text(
        "# Poisson GLM performance tables\n\n"
        + ("Canonical full run.\n\n" if conf["profile"] == "full" else "EXPLORATORY / VALIDATION RUN: " + conf["profile"] + ".\n\n")
        + f"{len(frame)} dataset-method tasks; {metadata['successful_tasks']} successful; {metadata['failed_tasks']} unsuccessful. "
        + f"{len(trials)} measured fits recorded, including fits from unsuccessful quality checks.\n\n"
        + "The main TeX tables report time, peak RSS and test deviance; penalized rows also report support F1, matching the final manuscript endpoints. "
        + "The two table CSVs contain means, SD, MCSE and denominators for those endpoints. "
        + "The four wide CSVs and *_time_memory.tex contain the time/memory-only panels. Main TeX requires booktabs and caption. "
        + "SD and MCSE use independent datasets. Failed tasks remain in attempted counts, while successful nonmissing observations form each metric's denominator.\n\n"
        + "CPU differences are recorded and permitted; software/BLAS checks remain part of the runner and merger. "
        + f"{len(verification['cv_groups_outside_previous_limit'])} dataset groups exceeded the previous CV-score tolerance; "
        + "these differences remain recorded and do not block export. Inspect failures.csv and run_summary.json with the tables.\n")
    print(f"Exported {metadata['successful_tasks']}/{len(frame)} successful tasks to {destination}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Completed local run or Slurm merged directory")
    parser.add_argument("--out", help="Table directory (default: RESULTS/public_tables)")
    parser.add_argument("--allow-validation", action="store_true", help="Allow clearly labeled non-full profiles")
    args = parser.parse_args()
    export(args.results, args.out, args.allow_validation)


if __name__ == "__main__":
    main()
