"""Gate formal fitting on source-specific native and complete-cell pilot checks."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from run_task import code_hash, atomic_json

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--concurrency", type=int, default=32)
a = ap.parse_args()
root = a.root.resolve()
fingerprint = code_hash()
manifest = pd.read_csv(root / "tasks.csv")
pilot = manifest[
    (manifest.replication == 1)
    | (
        (manifest.model == "DeepSurv")
        & (manifest.n == 1500)
        & (manifest.replication <= 5)
    )
]
if (root / "RUN_SCOPE.json").exists():
    scope = json.loads((root / "RUN_SCOPE.json").read_text())["models"]
    pilot = manifest[
        manifest.model.isin(scope)
        & (
            (manifest.replication == 1)
            | ((manifest.n == 1500) & (manifest.replication <= 5))
        )
    ]
datasets = pd.read_csv(root / "datasets.csv")
assert len(datasets) == 600 and datasets.data_seed.nunique() == 600
assert sorted(datasets.data_seed) == list(range(20226091, 20226691))
assert len(manifest) == 4800 and manifest.task_id.nunique() == 4800
for _, cell in datasets.groupby(["setting", "n"]):
    assert len(cell) == 100 and set(cell.replication) == set(range(1, 101))
    assert cell.training_seed.value_counts().to_dict() == {
        s: 20 for s in range(2026091, 2026096)
    }
for row in datasets.to_dict("records"):
    split = pd.read_csv(root / "data" / row["dataset_id"] / "splits.csv")
    assert len(split) == row["n"] and split.row_id.nunique() == row["n"]
    assert set(split.row_id) == set(range(row["n"]))
    assert split.split.value_counts().to_dict() == {
        "train": int(0.6 * row["n"]),
        "val": int(0.2 * row["n"]),
        "test": int(0.2 * row["n"]),
    }
records = {}
for p in (root / "pilot").glob("*/result.json"):
    r = json.loads(p.read_text())
    r["_path"] = str(p)
    if r["code_sha256"] == fingerprint:
        records[r["task_id"]] = r
missing = [
    int(i)
    for i in pilot.task_id
    if i not in records or records[i]["status"] != "success"
]
if missing:
    raise SystemExit(
        "Pilot missing/failed/stale task IDs: " + ",".join(map(str, missing))
    )
checks = {
    name: json.loads((root / "audit" / name).read_text())
    for name in [
        "rsf_native_R.json",
        "rsf_native_Python.json",
        "native_deep_split.json",
        "metric_and_memory_checks.json",
    ]
}
gbm_checks = json.loads((root / "audit/gbm_native_tree_constraints.json").read_text())
assert all(
    x["nonempty_leaves"] <= 4
    and x["splits"] <= 3
    and x["min_nonempty_leaf_weight"] >= 15
    for x in gbm_checks
)
assert checks["rsf_native_R.json"]["weighted_30_draws_15_unique_leaves"] > 1
assert checks["rsf_native_Python.json"]["weighted_30_draws_15_unique_leaves"] == 1
assert checks["native_deep_split.json"]["exact_ordered_validation"]
assert (
    abs(
        checks["metric_and_memory_checks.json"]["ibs_reference"]
        - checks["metric_and_memory_checks.json"]["ibs_common"]
    )
    < 1e-12
)
aligned = []
for dataset in pilot.dataset_id.unique():
    rr = [r for r in records.values() if r["dataset_id"] == dataset]
    assert len(rr) == len(pilot[pilot.dataset_id == dataset])
    ds = root / "data" / dataset
    metadata = json.loads((ds / "metadata.json").read_text())
    assert all(r["n_test_evaluated"] == metadata["n_test"] for r in rr)
    assert (
        len({r["grid_lower"] for r in rr}) == 1
        and len({r["grid_upper"] for r in rr}) == 1
    )
    deep = {r["language"]: r for r in rr if r["model"] == "DeepSurv"}
    arrays = {
        lang: np.load(
            Path(r["_path"]).parent / r["attempt"] / "deep_training_audit.npz"
        )
        for lang, r in deep.items()
    }
    for key in ("x", "xv", "time", "event", "val_time", "val_event"):
        assert np.array_equal(arrays["R"][key], arrays["Python"][key]), (
            dataset,
            key,
            "DeepSurv actual input mismatch",
        )
    r = deep["R"]
    assert r["details"]["backend_pid"] == r["postfit_process"]["pid"]
    for r in rr:
        assert not any(v["during_fit"] for v in r["observed_children"].values())
        assert "Gold 6154" in r["cpu_model"]
        assert r["requested_cpu_cores"] == "4"
        if r["model"] == "DeepSurv":
            assert (
                r["details"]["torch_threads"] == 1
                and r["details"]["torch_interop_threads"] == 1
            )
            assert r["details"]["batch_size"] == 150
            log = pd.read_csv(
                Path(r["_path"]).parent / r["attempt"] / "training_log.csv"
            )
            assert np.isfinite(log.to_numpy()).all()
        if r["model"] == "RSF" and r["language"] == "R":
            assert r["details"]["rf_cores"] == 1 and r["details"]["trees"] == 200
            tr = pd.read_csv(ds / "train.csv", usecols=["time", "event"])
            assert (
                r["details"]["prediction_times"]
                == tr.loc[tr.event == 1, "time"].nunique()
            )
        if r["language"] == "Python":
            assert all(p["num_threads"] == 1 for p in r["threadpools"])
        if r["model"] == "Coxnet":
            assert r["details"]["penalty_candidates"] == 100
        if r["model"] == "Boosting" and r["language"] == "Python":
            assert r["details"]["max_leaves"] <= 4
    sr = np.loadtxt(
        Path(deep["R"]["_path"]).parent / deep["R"]["attempt"] / "survival.csv",
        delimiter=",",
    )
    sp = np.loadtxt(
        Path(deep["Python"]["_path"]).parent
        / deep["Python"]["attempt"]
        / "survival.csv",
        delimiter=",",
    )
    aligned.append(
        {
            "dataset": dataset,
            "ordered_train_val_test_match": True,
            "deep_max_abs_prediction_difference": float(abs(sr - sp).max()),
        }
    )
records = [records[int(i)] for i in pilot.task_id]
main_pilot = [r for r in records if r["replication"] == 1]
total_worker = sum(r["worker_total_wall_s"] for r in main_pilot) * 100
total_fit = sum(r["fitting_time_s"] for r in main_pilot) * 100
info = {
    "code_sha256": fingerprint,
    "pilot_fits": len(records),
    "concurrency": a.concurrency,
    "predicted_total_worker_hours": total_worker / 3600,
    "predicted_total_fit_hours": total_fit / 3600,
    "ideal_wall_hours_at_concurrency": total_worker / a.concurrency / 3600,
    "eta_limitation": "One representative replication per cell; estimate excludes queue waits and retries.",
    "max_pilot_peak_rss_mib": max(r["peak_rss_mib"] for r in records),
    "max_pilot_worker_wall_s": max(r["worker_total_wall_s"] for r in records),
    "dataset_checks": aligned,
}
atomic_json(root / "PILOT_VALIDATED.json", info)
pd.DataFrame(
    [
        {
            k: r.get(k)
            for k in (
                "task_id",
                "setting",
                "n",
                "model",
                "language",
                "fitting_time_s",
                "worker_total_wall_s",
                "peak_rss_mib",
                "ibs",
                "cindex",
            )
        }
        for r in records
    ]
).to_csv(root / "audit" / "pilot_results.csv", index=False)
print(json.dumps(info, indent=2))
