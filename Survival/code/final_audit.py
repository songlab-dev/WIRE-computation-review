"""Check every active result and every paired DeepSurv input/log after completion."""
import argparse, collections, csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from run_task import code_hash

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
a = ap.parse_args()
root = a.root.resolve()
manifest = pd.read_csv(root / "tasks.csv").set_index("task_id")
registry = (
    json.loads((root / "release_registry.json").read_text())["models"]
    if (root / "release_registry.json").exists()
    else {
        model: {
            "result_root": str(root),
            "code_dir": str(Path(__file__).resolve().parent),
            "code_sha256": code_hash(),
        }
        for model in manifest.model.unique()
    }
)
complete = json.loads((root / "summary/completion.json").read_text())
assert complete["success"] == 4800 and complete["failed_or_missing"] == 0
records = {}
locations = {}
for result_root in sorted({release["result_root"] for release in registry.values()}):
    for p in (Path(result_root) / "formal").glob("*/result.json"):
        r = json.loads(p.read_text())
        model = r["model"]
        release = registry[model]
        if release["result_root"] != result_root:
            continue
        assert r["status"] == "success" and r["code_sha256"] == release["code_sha256"]
        task = int(r["task_id"])
        assert task not in records
        for key in (
            "dataset_id",
            "model",
            "language",
            "n",
            "replication",
            "data_seed",
            "training_seed",
        ):
            assert r[key] == manifest.loc[task, key]
        assert "Gold 6154" in r["cpu_model"]
        assert r["requested_cpu_cores"] == "4"
        assert int(r["requested_memory_mb"]) == 32768
        assert not any(v["during_fit"] for v in r["observed_children"].values())
        if r["language"] == "Python":
            assert all(x["num_threads"] == 1 for x in r["threadpools"])
        if model == "RSF" and r["language"] == "R":
            assert r["details"]["rf_cores"] == 1
        if model == "RSF":
            assert r["details"]["trees"] == 200
            assert r["details"]["mtry"] == int(np.sqrt(r["p"]))
            assert r["details"]["bootstrap_draws"] == int(r["n"] * 0.6)
        if model == "Boosting":
            assert r["details"]["trees"] == 300
        if model == "Coxnet":
            assert r["details"]["penalty_candidates"] == 100
        if model == "DeepSurv":
            assert r["details"]["batch_size"] == 150
            assert (
                r["details"]["torch_threads"] == 1
                and r["details"]["torch_interop_threads"] == 1
            )
            if r["language"] == "R":
                assert r["details"]["backend_pid"] == r["postfit_process"]["pid"]
        records[task] = r
        locations[task] = p.parent / r["attempt"]
assert set(records) == set(manifest.index)
warnings = collections.Counter(
    (r["model"], r["language"], r.get("warnings", "").strip())
    for r in records.values()
    if r.get("warnings", "").strip()
)
(root / "audit/worker_warnings.json").write_text(
    json.dumps(
        [
            {"model": key[0], "language": key[1], "text": key[2], "count": value}
            for key, value in warnings.items()
        ],
        indent=2,
    )
)
paired = []
for dataset, group in manifest[manifest.model == "DeepSurv"].groupby(
    "dataset_id", sort=False
):
    ids = {row.language: int(i) for i, row in group.iterrows()}
    paths = {lang: locations[i] for lang, i in ids.items()}
    arrays = {
        lang: np.load(path / "deep_training_audit.npz") for lang, path in paths.items()
    }
    for key in ("x", "xv", "time", "event", "val_time", "val_event"):
        assert np.array_equal(arrays["R"][key], arrays["Python"][key]), (dataset, key)
    logs = {
        lang: pd.read_csv(path / "training_log.csv") for lang, path in paths.items()
    }
    assert all(np.isfinite(df.to_numpy()).all() for df in logs.values()), dataset
    assert logs["R"].shape == logs["Python"].shape
    predictions = {
        lang: np.loadtxt(path / "survival.csv", delimiter=",")
        for lang, path in paths.items()
    }
    difference = float(abs(predictions["R"] - predictions["Python"]).max())
    paired.append(
        {
            "dataset_id": dataset,
            "epochs": len(logs["R"]),
            "max_abs_prediction_difference": difference,
            "max_abs_log_difference": float(
                abs(logs["R"].to_numpy() - logs["Python"].to_numpy()).max()
            ),
        }
    )
    for arr in arrays.values():
        arr.close()
pd.DataFrame(paired).to_csv(root / "audit/deep_pair_checks.csv", index=False)
superseded = []
for p in (root / "formal").glob("*/result.json"):
    r = json.loads(p.read_text())
    if r["model"] == "DeepSurv" and registry["DeepSurv"]["result_root"] != str(root):
        superseded.append(r)
with (root / "audit/superseded_deep_batch128_records.jsonl").open("w") as f:
    for r in superseded:
        f.write(json.dumps(r) + "\n")
report = {
    "active_fits": len(records),
    "model_language_cells": 48,
    "replications_per_cell": 100,
    "active_failure_count": 0,
    "superseded_deep_batch128_fits": len(superseded),
    "fits_with_worker_warnings": sum(warnings.values()),
    "hardware_models": sorted({r["cpu_model"] for r in records.values()}),
    "nodes": sorted({r["hostname"] for r in records.values()}),
    "os_thread_counts": sorted(
        {r["postfit_process"]["Threads"] for r in records.values()}
    ),
    "computational_threads": 1,
    "max_peak_rss_mib": max(r["peak_rss_mib"] for r in records.values()),
    "max_fitting_time_s": max(r["fitting_time_s"] for r in records.values()),
    "all_deep_training_logs_finite": True,
    "deep_paired_datasets": len(paired),
    "deep_max_abs_prediction_difference": max(
        r["max_abs_prediction_difference"] for r in paired
    ),
    "deep_max_abs_log_difference": max(r["max_abs_log_difference"] for r in paired),
    "selected_releases": registry,
}
(root / "audit/final_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
