#!/usr/bin/env python3
"""One Slurm element per scenario/seed; validate before pooling any results."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from run_benchmark import (ROOT, atomic_json, checkpoint_records,
                           environment_signature, json_hash, make_seeds,
                           run_fingerprint, source_hashes)


def methods_for(conf, scenario):
    return conf["penalized_methods"] if scenario["regime"] in {"highdim","sparse_highdim"} else conf["unpenalized_methods"]


def task_grid(conf, seeds):
    return [{"index":len(seeds)*ordinal+j,"scenario":scenario["id"],
             "scenario_index":ordinal,"seed":seed,
             "methods":methods_for(conf,scenario),
             "timing_dataset":seed in conf["timing_seeds"]}
            for ordinal,scenario in enumerate(conf["scenarios"])
            for j,seed in enumerate(seeds)]


def plan_payload(conf, hashes):
    seeds=make_seeds(conf)
    return {"config":conf,"source_sha256":hashes,"seeds":seeds,
            "tasks":task_grid(conf,seeds)}


def create_plan(config, out, resume=False):
    conf=json.loads(Path(config).read_text())
    # Local temporary files are removed; this changes storage, not the experiment.
    conf["keep_data"]=False
    ids=[s["id"] for s in conf["scenarios"]]
    if not ids or len(set(ids))!=len(ids) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",s) for s in ids):
        raise ValueError("Scenario IDs must be unique, nonempty, safe directory names")
    if len(set(conf["timing_seeds"]))!=len(conf["timing_seeds"]) or conf["statistical_replicates"]<len(conf["timing_seeds"]):
        raise ValueError("Invalid seed/replication specification")
    if conf["statistical_replicates"]<1 or conf["timing_repeats"]<1:
        raise ValueError("Positive replication counts are required")
    for scenario in conf["scenarios"]:
        methods=methods_for(conf,scenario)
        if not methods or len(set(methods))!=len(methods):raise ValueError("Empty or duplicate method list")
    payload=plan_payload(conf,source_hashes());fingerprint=json_hash(payload)
    out=Path(out).resolve();path=out/"plan.json"
    if path.exists():
        old=load_plan(out)
        if not resume:raise ValueError("Plan exists; use --resume or a new --out")
        if old["fingerprint"]!=fingerprint:raise ValueError("Configuration/source changed; use a new output directory")
        return old
    if out.exists() and any(out.iterdir()):raise ValueError("Nonempty output without a valid plan; use a fresh --out")
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"config.json",conf)
    atomic_json(path,{**payload,"fingerprint":fingerprint,"execution_mode":"slurm_array"})
    (out/"logs").mkdir(exist_ok=True)
    return load_plan(out)


def load_plan(out):
    root=Path(out).resolve();plan=json.loads((root/"plan.json").read_text())
    payload={key:plan[key] for key in ["config","source_sha256","seeds","tasks"]}
    if json_hash(payload)!=plan["fingerprint"]:raise ValueError("Plan fingerprint is invalid")
    if json.loads((root/"config.json").read_text())!=plan["config"]:raise ValueError("Saved configuration was changed")
    if plan["seeds"]!=make_seeds(plan["config"]) or plan["tasks"]!=task_grid(plan["config"],plan["seeds"]):
        raise ValueError("Plan task grid does not match the saved protocol")
    return plan


def verify_code(plan):
    if source_hashes()!=plan["source_sha256"]:
        raise ValueError("Benchmark source changed after planning; restore the original code or use a fresh output directory")


def shard_path(root, task):
    return Path(root)/"shards"/f'task-{task["index"]:04d}'


def validate_shard(plan, task, root):
    shard=shard_path(root,task);run=json.loads((shard/"run.json").read_text())
    conf=copy.deepcopy(plan["config"])
    scenario=conf["scenarios"][task["scenario_index"]]
    conf["scenarios"]=[scenario]
    if run["config"]!=conf or run["source_sha256"]!=plan["source_sha256"]:
        raise ValueError("Shard configuration/source mismatch: "+str(task["index"]))
    if run["fingerprint"]!=run_fingerprint(conf,plan["source_sha256"],[task["seed"]]):
        raise ValueError("Shard fingerprint mismatch")
    if run.get("selected_seeds")!=[task["seed"]] or json.loads((shard/"seeds.json").read_text())!=[task["seed"]]:
        raise ValueError("Shard seed selection mismatch")
    records=checkpoint_records(shard)
    keys={(r["summary"]["scenario"],int(r["summary"]["seed"]),r["summary"]["method"]) for r in records}
    expected={(task["scenario"],task["seed"],m) for m in task["methods"]}
    if keys!=expected:raise ValueError("Incomplete/unexpected method checkpoints in shard "+str(task["index"]))
    order={method:i for i,method in enumerate(task["methods"])}
    records.sort(key=lambda r:order[r["summary"]["method"]])
    repeats=conf["timing_repeats"] if task["timing_dataset"] else 1
    for record in records:
        row=record["summary"]
        if (row["n"],row["p"],row["regime"],row["density"])!=(scenario["n"],scenario["p"],scenario["regime"],scenario.get("density",1.0)):
            raise ValueError("Shard design mismatch")
        if row["timing_dataset"]!=task["timing_dataset"] or row["requested_repeats"]!=repeats:
            raise ValueError("Timing/statistical replication mismatch")
        if not 0<=row["completed_repeats"]<=repeats:raise ValueError("Invalid completed repetitions")
        if row["success"] and (row["completed_repeats"]!=repeats or not row["quality_ok"] or row["kkt_scaled"]>conf["kkt_tol"]):
            raise ValueError("Successful checkpoint failed its recorded quality criteria")
        if row["status"]!=record["worker"]["status"]:raise ValueError("Worker/status mismatch")
    manifest=shard/"manifests"/(task["scenario"]+"_"+str(task["seed"])+".json")
    meta=json.loads(manifest.read_text())
    if (meta["id"],meta["seed"],meta["scenario_index"],meta["n"],meta["p"],meta["test_n"])!=(task["scenario"],task["seed"],task["scenario_index"],scenario["n"],scenario["p"],conf["test_n"]):
        raise ValueError("Dataset manifest mismatch")
    if not meta.get("sha256"):raise ValueError("Dataset checksums are missing")
    return records,manifest


def pending_tasks(plan, root):
    pending=[]
    for task in plan["tasks"]:
        try:validate_shard(plan,task,root)
        except (OSError,ValueError,KeyError,TypeError):pending.append(task["index"])
    return pending


def run_task(out, index, work_dir=None):
    root=Path(out).resolve();plan=load_plan(root);verify_code(plan)
    if not 0<=index<len(plan["tasks"]):raise ValueError("Array index is outside the plan")
    task=plan["tasks"][index];shard=shard_path(root,task)
    command=[sys.executable,str(ROOT/"run_benchmark.py"),"--config",str(root/"config.json"),
             "--out",str(shard),"--scenario",task["scenario"],"--seed",str(task["seed"])]
    if (shard/"run.json").exists():command.append("--resume")
    if work_dir:command.extend(["--work-dir",str(Path(work_dir).resolve())])
    subprocess.run(command,check=True)
    records,_=validate_shard(plan,task,root)
    atomic_json(shard/"task_complete.json",{"plan_fingerprint":plan["fingerprint"],"index":index,
                "attempted_methods":len(records),"successful_methods":sum(bool(r["summary"]["success"]) for r in records)})


def merge_results(out, destination=None):
    root=Path(out).resolve();plan=load_plan(root);verify_code(plan)
    pending=pending_tasks(plan,root)
    if pending:raise ValueError(f"Cannot merge: {len(pending)} array elements are incomplete. Pending IDs: "+",".join(map(str,pending)))
    rows=[];all_records=[];environments={};manifests=[];locations=[];inputs=[]
    signature=None;reference_cpu=None
    for task in plan["tasks"]:
        records,manifest=validate_shard(plan,task,root);shard=shard_path(root,task)
        inputs.append({"index":task["index"],"checkpoint_sha256":json_hash(records)})
        manifests.append(manifest);all_records.extend(records)
        locations.append({"index":task["index"],"scenario":task["scenario"],"seed":task["seed"],
                          "shard_relative_to_run_root":str(shard.relative_to(root))})
        for record in records:
            row=record["summary"];env_id=row["environment_id"]
            env=json.loads((shard/"environment_records"/(env_id+".json")).read_text())
            if json_hash(env)!=env_id or env["hostname"]!=row["worker_hostname"]:
                raise ValueError("Environment provenance mismatch")
            this_signature=environment_signature(env)
            if signature is None:signature=this_signature;reference_cpu=env["cpu"]
            if this_signature!=signature:
                changed=[key for key in signature if signature[key]!=this_signature[key]]
                raise ValueError(f"Heterogeneous environment in shard {task['index']}: {changed}; CPU {env['cpu']} vs {reference_cpu}. Do not pool; select a homogeneous node constraint and start a new run.")
            environments[env_id]=env;rows.append(row)
    if len({(r["scenario"],r["seed"],r["method"]) for r in rows})!=len(rows):raise ValueError("Duplicate merged task keys")
    merge_hash=json_hash({"plan":plan["fingerprint"],"inputs":inputs})
    destination=Path(destination).resolve() if destination else root/"merged"
    if destination.exists():
        saved=destination/"run.json"
        if saved.exists() and json.loads(saved.read_text()).get("merge_inputs_sha256")==merge_hash:
            print("Merged inputs unchanged:",destination);return destination
        raise ValueError("Merged destination exists with different/incomplete content; use a fresh --destination")
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".glm-merge-",dir=destination.parent))
    try:
        with (stage/"raw_results.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (stage/"fit_trials.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["scenario","seed","method","trial","seconds","timing_dataset"])
            for record in all_records:
                row=record["summary"];times=record["worker"].get("timings",[])
                times=times if isinstance(times,list) else [times]
                for i,t in enumerate(times):writer.writerow([row["scenario"],row["seed"],row["method"],i+1,t,row["timing_dataset"]])
        atomic_json(stage/"run.json",{"fingerprint":run_fingerprint(plan["config"],plan["source_sha256"],plan["seeds"]),
                    "config":plan["config"],"source_sha256":plan["source_sha256"],"selected_seeds":plan["seeds"],
                    "execution_mode":"slurm_merged","slurm_plan_fingerprint":plan["fingerprint"],
                    "merge_inputs_sha256":merge_hash})
        atomic_json(stage/"seeds.json",plan["seeds"])
        representative=copy.deepcopy(next(iter(environments.values())))
        representative["execution"]={"backend":"Slurm array","array_elements":len(plan["tasks"]),
                  "homogeneous_environment_verified":True,"worker_cpu_threads":1,
                  "distinct_hosts":sorted({env["hostname"] for env in environments.values()}),
                  "environment_records":len(environments),"note":"System RAM/CPU count describe the representative node; Slurm allocations are recorded per attempt."}
        atomic_json(stage/"environment.json",representative)
        atomic_json(stage/"cluster_environments.json",environments)
        atomic_json(stage/"shard_index.json",{"run_root_relative_to_merged":os.path.relpath(root,destination),"shards":locations})
        (stage/"environment_records").mkdir();(stage/"manifests").mkdir()
        for env_id,env in environments.items():atomic_json(stage/"environment_records"/(env_id+".json"),env)
        for manifest in manifests:shutil.copy2(manifest,stage/"manifests"/manifest.name)
        stage.rename(destination)
    except BaseException:
        # Only this newly created private staging directory is removed.
        shutil.rmtree(stage);raise
    print(f"Merged {len(plan['tasks'])} complete array elements; {len(rows)} dataset-method attempts into {destination}")
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    plan=sub.add_parser("plan");plan.add_argument("--config",required=True);plan.add_argument("--out",required=True);plan.add_argument("--resume",action="store_true")
    for name in ["count","pending","status","run-task","merge"]:
        p=sub.add_parser(name);p.add_argument("--out",required=True)
        if name=="run-task":p.add_argument("--index",required=True,type=int);p.add_argument("--work-dir")
        if name=="merge":p.add_argument("--destination")
    args=parser.parse_args()
    if args.command=="plan":
        p=create_plan(args.config,args.out,args.resume);print(f"Plan: {len(p['tasks'])} array elements; {sum(len(t['methods']) for t in p['tasks'])} dataset-method attempts; {args.out}")
    elif args.command=="run-task":run_task(args.out,args.index,args.work_dir)
    elif args.command=="merge":merge_results(args.out,args.destination)
    else:
        p=load_plan(args.out);verify_code(p)
        if args.command=="count":print(len(p["tasks"]))
        elif args.command=="pending":print(",".join(map(str,pending_tasks(p,args.out))))
        else:
            pending=pending_tasks(p,args.out)
            print(json.dumps({"profile":p["config"]["profile"],"planned_array_elements":len(p["tasks"]),
                  "complete_array_elements":len(p["tasks"])-len(pending),"pending_array_ids":pending},indent=2))


if __name__=="__main__":main()
