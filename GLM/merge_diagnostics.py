#!/usr/bin/env python3
"""Merge completed smoke/pilot runs, retaining CPU labels and paired blocks."""
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

import slurm_workflow as workflow
from run_benchmark import ROOT, atomic_json, environment_signature, json_hash, run_fingerprint


def gather(root):
    root = Path(root).resolve()
    plan = workflow.load_plan(root)
    workflow.verify_code(plan)
    if plan["config"]["profile"] not in {"smoke", "pilot"}:
        raise ValueError("This CPU-aware merger is for smoke/pilot diagnostics only; use the formal workflow for full runs")
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
        pair_hosts, pair_cpus = set(), set()
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
                raise ValueError("Only CPU may differ in diagnostics; changed fields: " + ",".join(changed))
            pair_hosts.add(env["hostname"])
            pair_cpus.add(env["cpu"])
            environments[env_id] = env
            row["cpu_model"] = env["cpu"]
            row["allocated_node"] = env.get("slurm", {}).get("SLURM_JOB_NODELIST", env["hostname"])
            rows.append(row)
            all_records.append(record)
        if len(pair_hosts) != 1 or len(pair_cpus) != 1:
            raise ValueError("Methods in one scenario/seed must share a node and CPU: shard " + str(task["index"]))
    if len({(r["scenario"], r["seed"], r["method"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate dataset-method records")
    return plan, rows, all_records, environments, manifests, locations, inputs


def label_tables(destination, rows):
    data = pd.DataFrame(rows)
    tables = destination / "tables"
    labels = data.groupby(["scenario", "method"], sort=False).agg(
        cpu_models=("cpu_model", lambda x: "; ".join(sorted(set(x)))),
        worker_nodes=("worker_hostname", lambda x: "; ".join(sorted(set(x))))).reset_index()
    for name in ["summary_all.csv", "table1_unpenalized.csv", "table2_penalized.csv", "paired_speedups.csv"]:
        frame = pd.read_csv(tables / name)
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
    readiness.update(diagnostic_only=True, allowed_for_submission=False,
                     cpu_policy="CPU models retained; same-node method pairing verified")
    atomic_json(readiness_path, readiness)
    status = tables / "results_status.txt"
    status.write_text("CPU-AWARE DIAGNOSTICS: retain CPU labels; do not treat all rows as one machine.\n" + status.read_text())


def merge_diagnostics(root, destination=None):
    root = Path(root).resolve()
    plan, rows, records, environments, manifests, locations, inputs = gather(root)
    cpu_models = sorted({env["cpu"] for env in environments.values()})
    merger_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    merge_hash = json_hash({"plan": plan["fingerprint"], "inputs": inputs,
                           "policy": "cpu-aware-diagnostics-v1", "merger_sha256": merger_hash})
    destination = Path(destination).resolve() if destination else root / "diagnostic_merged"
    if destination.exists():
        saved = destination / "run.json"
        if saved.exists() and json.loads(saved.read_text()).get("merge_inputs_sha256") == merge_hash:
            print("Diagnostic inputs unchanged:", destination)
            return destination
        raise ValueError("Destination exists with different/incomplete content; use a fresh --destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".glm-diagnostic-", dir=destination.parent))
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
            "merge_inputs_sha256": merge_hash, "merge_policy": "cpu-aware-diagnostics-v1",
            "diagnostic_merge_sha256": merger_hash, "diagnostic_only": True})
        atomic_json(stage / "seeds.json", plan["seeds"])
        representative = copy.deepcopy(next(iter(environments.values())))
        if len(cpu_models) > 1:
            representative["cpu"] = "MIXED CPU (diagnostic): " + "; ".join(cpu_models)
        representative["execution"] = {"backend": "Slurm array", "array_elements": len(plan["tasks"]),
            "homogeneous_environment_verified": len(cpu_models) == 1,
            "non_cpu_environment_verified": True, "same_node_method_pairing_verified": True,
            "cpu_models": cpu_models, "diagnostic_only": True,
            "worker_cpu_threads": 1, "distinct_hosts": sorted({env["hostname"] for env in environments.values()}),
            "note": "CPU models vary across dataset blocks; original per-attempt environments retained. Pilot timing is for resource planning."}
        atomic_json(stage / "environment.json", representative)
        atomic_json(stage / "cluster_environments.json", environments)
        atomic_json(stage / "hardware_policy.json", {"profile": plan["config"]["profile"], "diagnostic_only": True,
            "allowed_for_submission": False, "cpu_models": cpu_models,
            "only_cpu_difference_allowed": True, "same_node_method_pairing_verified": True})
        atomic_json(stage / "shard_index.json", {"run_root_relative_to_merged": os.path.relpath(root, destination), "shards": locations})
        (stage / "environment_records").mkdir()
        (stage / "manifests").mkdir()
        for env_id, env in environments.items():
            atomic_json(stage / "environment_records" / (env_id + ".json"), env)
        for manifest in manifests:
            shutil.copy2(manifest, stage / "manifests" / manifest.name)
        subprocess.run([sys.executable, str(ROOT / "check_results.py"), "--out", str(stage)], check=True)
        subprocess.run([sys.executable, str(ROOT / "summarize_results.py"), "--out", str(stage)], check=True)
        label_tables(stage, rows)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f"Merged {len(rows)} diagnostic tasks with CPU labels into {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing completed Slurm smoke/pilot run directory")
    parser.add_argument("--destination", help="Fresh result destination; default OUT/diagnostic_merged")
    args = parser.parse_args()
    merge_diagnostics(args.out, args.destination)


if __name__ == "__main__":
    main()
