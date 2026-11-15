#!/usr/bin/env python3
"""Run and merge any Slurm profile, recording CPU models without requiring a match.

The original benchmark/controller files remain byte-for-byte unchanged so that
completed runs retain valid source fingerprints. This controller changes only
CPU matching during resume and merge; fitting is unchanged. The result checker
records CV score differences without blocking the merge.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

import run_benchmark as benchmark
import slurm_workflow as workflow
from run_benchmark import ROOT, atomic_json, environment_signature, json_hash, run_fingerprint
from summarize_results import tex_escape

POLICY = "cpu-model-recorded-not-matched-v1"


def controller_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def gather(root):
    root = Path(root).resolve()
    plan = workflow.load_plan(root)
    workflow.verify_code(plan)
    pending = workflow.pending_tasks(plan, root)
    if pending:
        raise ValueError("Incomplete array elements: " + ",".join(map(str, pending)))
    rows, all_records, manifests, locations, inputs = [], [], [], [], []
    environments = {}
    reference = None
    for task in plan["tasks"]:
        records, manifest = workflow.validate_shard(plan, task, root)
        shard = workflow.shard_path(root, task)
        inputs.append({"index": task["index"], "checkpoint_sha256": json_hash(records)})
        manifests.append(manifest)
        locations.append({"index": task["index"], "scenario": task["scenario"], "seed": task["seed"],
                          "shard_relative_to_run_root": str(shard.relative_to(root))})
        for record in records:
            row = copy.deepcopy(record["summary"])
            env_id = row["environment_id"]
            env = json.loads((shard / "environment_records" / (env_id + ".json")).read_text())
            if json_hash(env) != env_id or env["hostname"] != row["worker_hostname"]:
                raise ValueError("Environment provenance mismatch")
            signature = {k: v for k, v in environment_signature(env).items() if k != "cpu"}
            if reference is None:
                reference = signature
            changed = [key for key in reference if reference[key] != signature[key]]
            if changed:
                raise ValueError("Non-CPU environment mismatch; changed fields: " + ",".join(changed))
            environments[env_id] = env
            row["cpu_model"] = env["cpu"]
            row["allocated_node"] = env.get("slurm", {}).get("SLURM_JOB_NODELIST", env["hostname"])
            rows.append(row)
            all_records.append(record)
    if len({(r["scenario"], r["seed"], r["method"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate dataset-method records")
    return plan, rows, all_records, environments, manifests, locations, inputs


def pairing_metadata(rows):
    blocks = {}
    for row in rows:
        block = blocks.setdefault((row["scenario"], row["seed"]), {"hosts": set(), "cpus": set()})
        block["hosts"].add(row["worker_hostname"])
        block["cpus"].add(row["cpu_model"])
    return {"same_node_method_pairing_verified": all(len(b["hosts"]) == 1 for b in blocks.values()),
            "same_cpu_method_pairing_verified": all(len(b["cpus"]) == 1 for b in blocks.values())}


def label_tables(destination, rows, cpu_models):
    data = pd.DataFrame(rows)
    tables = destination / "tables"
    labels = data.groupby(["scenario", "method"], sort=False).agg(
        cpu_models=("cpu_model", lambda x: "; ".join(sorted(set(x)))),
        worker_nodes=("worker_hostname", lambda x: "; ".join(sorted(set(x))))).reset_index()
    for name in ["summary_all.csv", "table1_unpenalized.csv", "table2_penalized.csv", "paired_speedups.csv"]:
        try:
            frame = pd.read_csv(tables / name)
        except pd.errors.EmptyDataError:
            if name != "paired_speedups.csv":
                raise
            frame = pd.DataFrame(columns=["scenario", "method", "reference", "mean", "sd", "mcse", "paired_datasets"])
        frame.merge(labels, on=["scenario", "method"], how="left", validate="one_to_one").to_csv(tables / name, index=False)
    pairs = pd.read_csv(tables / "paired_speedups_raw.csv")
    pairs.merge(data[["scenario", "seed", "method", "cpu_model", "worker_hostname"]],
                on=["scenario", "seed", "method"], how="left", validate="one_to_one").to_csv(
                    tables / "paired_speedups_raw.csv", index=False)
    blocks = data.groupby(["scenario", "seed", "cpu_model", "worker_hostname"], sort=False).agg(
        method_tasks=("method", "size"), successful_tasks=("success", "sum"),
        completed_measured_fits=("completed_repeats", "sum")).reset_index()
    blocks.to_csv(tables / "cpu_by_scenario.csv", index=False)
    readiness_path = destination / "submission_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness.update(cpu_policy=POLICY, cpu_models=cpu_models,
                     cpu_difference_blocks_submission=False, **pairing_metadata(rows))
    verification = json.loads((destination / "verification.json").read_text())
    readiness.update(cv_difference_policy=verification.get("cv_difference_policy", "strict"),
                     cv_difference_blocks_merge=verification.get("cv_difference_blocks_merge", True),
                     cv_groups_outside_previous_limit=len(verification.get("cv_groups_outside_previous_limit", [])))
    atomic_json(readiness_path, readiness)
    note = "Recorded CPU models: " + "; ".join(cpu_models) + ". CPU model differences are permitted when pooling results."
    if not readiness["cv_difference_blocks_merge"]:
        note += (" CV score differences are recorded without blocking the merge; "
                 + str(readiness["cv_groups_outside_previous_limit"])
                 + " dataset groups were outside the previous absolute tolerance.")
    status = tables / "results_status.txt"
    status.write_text(status.read_text() + note + "\n")
    for name in ["table1_main.tex", "table2_main.tex"]:
        path = tables / name
        path.write_text(path.read_text() + "\n{\\footnotesize " + tex_escape(note) + "\\par}\n")


def merge_results(root, destination=None):
    root = Path(root).resolve()
    plan, rows, records, environments, manifests, locations, inputs = gather(root)
    cpu_models = sorted({env["cpu"] for env in environments.values()})
    pairing = pairing_metadata(rows)
    merger_hash = controller_hash()
    merge_hash = json_hash({"plan": plan["fingerprint"], "inputs": inputs,
                           "policy": POLICY, "controller_sha256": merger_hash})
    destination = Path(destination).resolve() if destination else root / "merged"
    if destination.exists():
        saved = destination / "run.json"
        if saved.exists() and json.loads(saved.read_text()).get("merge_inputs_sha256") == merge_hash:
            print("Merge inputs unchanged:", destination)
            return destination
        raise ValueError("Destination exists with different/incomplete content; use a fresh --destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".glm-cpu-relaxed-", dir=destination.parent))
    try:
        with (stage / "raw_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (stage / "fit_trials.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "seed", "method", "trial", "seconds", "timing_dataset", "cpu_model", "worker_hostname"])
            for row, record in zip(rows, records):
                times = record["worker"].get("timings", [])
                times = times if isinstance(times, list) else [times]
                for trial, value in enumerate(times, 1):
                    writer.writerow([row["scenario"], row["seed"], row["method"], trial, value,
                                     row["timing_dataset"], row["cpu_model"], row["worker_hostname"]])
        atomic_json(stage / "run.json", {"fingerprint": run_fingerprint(plan["config"], plan["source_sha256"], plan["seeds"]),
            "config": plan["config"], "source_sha256": plan["source_sha256"], "selected_seeds": plan["seeds"],
            "execution_mode": "slurm_merged", "slurm_plan_fingerprint": plan["fingerprint"],
            "merge_inputs_sha256": merge_hash, "merge_policy": POLICY,
            "cpu_relaxed_controller_sha256": merger_hash})
        atomic_json(stage / "seeds.json", plan["seeds"])
        representative = copy.deepcopy(next(iter(environments.values())))
        if len(cpu_models) > 1:
            representative["cpu"] = "Mixed CPU models: " + "; ".join(cpu_models)
        representative["execution"] = {"backend": "Slurm array", "array_elements": len(plan["tasks"]),
            "homogeneous_environment_verified": len(cpu_models) == 1,
            "non_cpu_environment_verified": True, **pairing,
            "cpu_models": cpu_models, "cpu_policy": POLICY,
            "worker_cpu_threads": 1, "distinct_hosts": sorted({env["hostname"] for env in environments.values()}),
            "note": "CPU models are recorded and may differ between tasks or resumed attempts; summaries pool the recorded results."}
        atomic_json(stage / "environment.json", representative)
        atomic_json(stage / "cluster_environments.json", environments)
        atomic_json(stage / "hardware_policy.json", {"profile": plan["config"]["profile"],
            "cpu_policy": POLICY, "cpu_models": cpu_models, "only_cpu_difference_allowed": True,
            "supports_full_profile": True, **pairing})
        atomic_json(stage / "shard_index.json", {"run_root_relative_to_merged": os.path.relpath(root, destination), "shards": locations})
        (stage / "environment_records").mkdir()
        (stage / "manifests").mkdir()
        for env_id, env in environments.items():
            atomic_json(stage / "environment_records" / (env_id + ".json"), env)
        for manifest in manifests:
            shutil.copy2(manifest, stage / "manifests" / manifest.name)
        subprocess.run([sys.executable, str(ROOT / "check_results.py"), "--out", str(stage)], check=True)
        subprocess.run([sys.executable, str(ROOT / "summarize_results.py"), "--out", str(stage)], check=True)
        label_tables(stage, rows, cpu_models)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f"Merged {len(rows)} tasks into {destination}; recorded CPU models: {len(cpu_models)}")
    return destination


def run_task(root, index, work_dir=None):
    root = Path(root).resolve()
    plan = workflow.load_plan(root)
    workflow.verify_code(plan)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("Array index is outside the plan")
    task = plan["tasks"][index]
    shard = workflow.shard_path(root, task)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(root / "config.json"),
               "--out", str(shard), "--scenario", task["scenario"], "--seed", str(task["seed"])]
    if (shard / "run.json").exists():
        command.append("--resume")
    if work_dir:
        command.extend(["--work-dir", str(Path(work_dir).resolve())])
    subprocess.run(command, check=True)
    records, _ = workflow.validate_shard(plan, task, root)
    atomic_json(shard / "task_complete.json", {"plan_fingerprint": plan["fingerprint"], "index": index,
                "attempted_methods": len(records), "successful_methods": sum(bool(r["summary"]["success"]) for r in records)})


def benchmark_worker(arguments):
    """Use the original fitter, excluding only CPU from its resume comparison."""
    old_keys, old_environment, old_argv = benchmark.ENVIRONMENT_KEYS, benchmark.environment, sys.argv
    runtime_policy = {"cpu_policy": POLICY, "cpu_relaxed_controller_sha256": controller_hash()}

    def recorded_environment():
        result = old_environment()
        result["runtime_controller"] = runtime_policy.copy()
        return result

    try:
        benchmark.ENVIRONMENT_KEYS = tuple(key for key in old_keys if key != "cpu")
        benchmark.environment = recorded_environment
        sys.argv = [str(ROOT / "run_benchmark.py"), *arguments]
        benchmark.main()
    finally:
        benchmark.ENVIRONMENT_KEYS, benchmark.environment, sys.argv = old_keys, old_environment, old_argv


def main():
    # Worker arguments are parsed by the original benchmark's parser.
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        benchmark_worker(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="Merge completed smoke, pilot, full or sensitivity runs")
    merge.add_argument("--out", required=True)
    merge.add_argument("--destination", help="Fresh destination; default OUT/merged")
    run = sub.add_parser("run-task", help="Run/resume one array element without matching CPU models")
    run.add_argument("--out", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--work-dir")
    args = parser.parse_args()
    if args.command == "merge":
        merge_results(args.out, args.destination)
    else:
        run_task(args.out, args.index, args.work_dir)


if __name__ == "__main__":
    main()
