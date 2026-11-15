#!/usr/bin/env python3
"""Sequential paired benchmark, restartable after each method/dataset task."""
import os
for _v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"]:
    os.environ[_v]="1"
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_info
from benchmark_metrics import evaluate
gen=importlib.import_module("01_generate_data")
ROOT=Path(__file__).resolve().parent


ENVIRONMENT_KEYS=("platform","architecture","cpu","python","python_packages","R","python_BLAS","thread_limits","loaded_modules")


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def source_hashes():
    names=["01_generate_data.py","python_worker.py","r_worker.R","benchmark_metrics.py","run_benchmark.py"]
    if (ROOT/"slurm_workflow.py").exists():names.append("slurm_workflow.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def make_seeds(conf):
    seeds=list(conf["timing_seeds"]);rng=np.random.default_rng(conf["accuracy_seed"])
    while len(seeds)<conf["statistical_replicates"]:
        s=int(rng.integers(1,2**31-1))
        if s not in seeds:seeds.append(s)
    return seeds


def run_fingerprint(conf,hashes,seeds):
    return json_hash({"config":conf,"source":hashes,"selected_seeds":seeds})


def environment_signature(env):
    return {key:env[key] for key in ENVIRONMENT_KEYS}


def checkpoint_records(out):
    records=[];seen=set()
    for path in sorted(Path(out).glob("tasks/*/*/diagnostics.json")):
        record=json.loads(path.read_text());row=record["summary"]
        key=(row["scenario"],int(row["seed"]),row["method"])
        if key in seen:raise ValueError("Duplicate task checkpoint: "+str(key))
        times=record["worker"].get("timings",[])
        times=times if isinstance(times,list) else [times]
        if len(times)!=int(row["completed_repeats"]):raise ValueError("Corrupt timing checkpoint: "+str(key))
        seen.add(key);records.append(record)
    return records


def restore_checkpoints(out):
    """Atomic diagnostics are authoritative after a preemption or partial CSV write."""
    records=checkpoint_records(out)
    if not records:return []
    rows=[r["summary"] for r in records]
    for name in ["raw_results.csv","fit_trials.csv"]:
        tmp=Path(out)/(name+".tmp")
        with tmp.open("w",newline="") as f:
            if name=="raw_results.csv":
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            else:
                w=csv.writer(f);w.writerow(["scenario","seed","method","trial","seconds","timing_dataset"])
                for record in records:
                    row=record["summary"];times=record["worker"].get("timings",[])
                    times=times if isinstance(times,list) else [times]
                    for i,t in enumerate(times):w.writerow([row["scenario"],row["seed"],row["method"],i+1,t,row["timing_dataset"]])
        tmp.replace(Path(out)/name)
    return rows


def environment():
    packages=["numpy","scipy","pandas","statsmodels","scikit-learn","glum","skglm","spglm","psutil","threadpoolctl","numba"]
    versions={p:importlib.metadata.version(p) for p in packages}
    rcode='p<-c("Matrix","jsonlite","glmnet","fastglm","speedglm","biglm");cat(jsonlite::toJSON(list(R=R.version.string,packages=setNames(lapply(p,function(z)as.character(packageVersion(z))),p),BLAS=extSoftVersion()["BLAS"]),auto_unbox=TRUE))'
    rr=subprocess.run(["Rscript","-e",rcode],capture_output=True,text=True)
    rinfo=json.loads(rr.stdout) if rr.returncode==0 else {"error":rr.stderr[-2000:]}
    cpu=platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu=line.split(":",1)[1].strip();break
    return {"recorded_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "platform":platform.platform(),"architecture":platform.machine(),
            "cpu":cpu,"logical_cpus":psutil.cpu_count(),"ram_gib":psutil.virtual_memory().total/2**30,
            "hostname":socket.gethostname(),"cpu_affinity":psutil.Process().cpu_affinity() if hasattr(psutil.Process(),"cpu_affinity") else None,
            "slurm":{key:os.environ[key] for key in ["SLURM_JOB_ID","SLURM_ARRAY_JOB_ID","SLURM_ARRAY_TASK_ID","SLURM_JOB_PARTITION","SLURM_CPUS_PER_TASK","SLURM_MEM_PER_NODE","SLURM_JOB_NODELIST"] if key in os.environ},
            "loaded_modules":os.environ.get("LOADEDMODULES",""),
            "python":sys.version,"python_packages":versions,"R":rinfo,
            "python_BLAS":threadpool_info(),"thread_limits":{v:os.environ[v] for v in os.environ if v.endswith("NUM_THREADS")},
            "rss_definition":"Sampled worker resident set size during the fitting block; includes runtime and resident input data; excludes parent process. Warmup and imports precede sampling."}


def preflight(conf):
    if not shutil.which("Rscript"): raise RuntimeError("Rscript is required; install R and run install_R_dependencies.R")
    if len(set(conf["timing_seeds"])) != len(conf["timing_seeds"]): raise ValueError("Timing seeds must be unique")
    if conf["statistical_replicates"] < len(conf["timing_seeds"]): raise ValueError("statistical_replicates must include all timing datasets")
    if conf["timing_repeats"] < 1 or conf["cv_folds"] < 2 or conf["lambda_count"] < 2: raise ValueError("Invalid repeat/CV/path size")
    # Fail before a long experiment if any selected package is missing.
    rpkgs={"R_stats":"stats","R_fastglm":"fastglm","R_speedglm":"speedglm","R_biglm":"biglm","R_glmnet":"glmnet"}
    pp={"Py_statsmodels":"statsmodels","Py_sklearn":"sklearn","Py_glum":"glum","Py_spglm":"spglm","Py_glum_lasso":"glum","Py_skglm":"skglm"}
    for m in conf["unpenalized_methods"]+conf["penalized_methods"]:
        if m in pp: importlib.import_module(pp[m])
        elif m not in rpkgs: raise ValueError("Unknown method: "+m)
    pkgs=sorted({rpkgs[m] for m in conf["unpenalized_methods"]+conf["penalized_methods"] if m in rpkgs}|{"jsonlite","Matrix"})
    code='p<-c('+','.join('"'+p+'"' for p in pkgs)+');bad<-p[!vapply(p,requireNamespace,logical(1),quietly=TRUE)];if(length(bad))stop(paste(bad,collapse=", "))'
    subprocess.run(["Rscript","-e",code],check=True)
    if "R_fastglm" in conf["unpenalized_methods"]:
        subprocess.run(["Rscript","-e",'stopifnot(packageVersion("fastglm") >= "0.1.2")'],check=True)


def execute(task, directory):
    directory.mkdir(parents=True,exist_ok=True)
    taskpath=directory/"task.json";taskpath.write_text(json.dumps(task,indent=2))
    cmd=["Rscript",str(ROOT/"r_worker.R"),"--task",str(taskpath)] if task["method"].startswith("R_") else [sys.executable,str(ROOT/"python_worker.py"),"--task",str(taskpath)]
    peak=0;baseline=None;samples=0;start=time.monotonic();timed_out=False
    for p in [task["phase_file"],task["result_file"]]:
        Path(p).unlink(missing_ok=True)
    with (directory/"worker.log").open("w") as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=log,env=os.environ.copy())
        while proc.poll() is None:
            try:
                state=json.loads(Path(task["phase_file"]).read_text())
                if state.get("phase")=="fit":
                    ps=psutil.Process(int(state["pid"]))
                    rss=ps.memory_info().rss;peak=max(peak,rss);samples+=1
                    if baseline is None: baseline=rss
            except (FileNotFoundError,json.JSONDecodeError,psutil.Error): pass
            if time.monotonic()-start > task["config"]["task_timeout_seconds"]:
                proc.kill();proc.wait();timed_out=True;break
            time.sleep(task["config"]["memory_poll_seconds"])
    resultpath=Path(task["result_file"])
    if timed_out: result={"status":"timeout","error":"Task wall-time limit includes loading and warmup", "timings":[]}
    elif resultpath.exists(): result=json.loads(resultpath.read_text())
    else: result={"status":"error","error":"Worker exited without result; see worker.log", "timings":[]}
    result.update(peak_rss_mib=peak/2**20 if samples else None,
                  rss_baseline_mib=baseline/2**20 if baseline else None,
                  rss_samples=samples,worker_exit_code=proc.returncode)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default=str(ROOT/"configs/full.json"))
    ap.add_argument("--out",default="results/full");ap.add_argument("--resume",action="store_true")
    ap.add_argument("--scenario",action="append",help="Repeat to select scenario IDs")
    ap.add_argument("--seed",type=int,action="append",help="Select a saved protocol seed; repeat to select several")
    ap.add_argument("--work-dir",help="Private node-local directory for input matrices and worker communication")
    ap.add_argument("--methods",nargs="+",help="Restrict to named methods; subsets are recorded")
    ap.add_argument("--preflight",action="store_true");a=ap.parse_args()
    conf=json.loads(Path(a.config).read_text())
    if a.scenario: conf["scenarios"]=[s for s in conf["scenarios"] if s["id"] in a.scenario]
    if a.methods:
        for key in ["unpenalized_methods","penalized_methods"]:
            conf[key]=[m for m in conf[key] if m in a.methods]
    if not conf["scenarios"]: raise ValueError("No scenarios selected")
    protocol_seeds=make_seeds(conf)
    if a.seed and (len(set(a.seed))!=len(a.seed) or not set(a.seed)<=set(protocol_seeds)):
        raise ValueError("Selected seeds must be unique members of the protocol seed list")
    seeds=[seed for seed in protocol_seeds if not a.seed or seed in a.seed]
    preflight(conf)
    if a.preflight: print(json.dumps(environment(),indent=2));return
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    # One writer per output directory, even if an array element is accidentally resubmitted.
    import fcntl
    lock=(out/"run.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("Another process is writing this output directory")
    hashes=source_hashes();fingerprint=run_fingerprint(conf,hashes,seeds)
    runfile=out/"run.json";raw=out/"raw_results.csv"
    current_env=environment()
    if runfile.exists():
        old=json.loads(runfile.read_text())
        if old.get("execution_mode")=="slurm_merged":raise ValueError("Resume the Slurm shards, not a merged result directory")
        if not a.resume: raise ValueError("Output exists; use --resume or a fresh --out")
        if old["fingerprint"] != fingerprint: raise ValueError("Config/source changed; use a fresh output directory")
        # Hardware, package and interpreter changes invalidate pooled timings.
        envpath=out/"environment.json";oldenv=json.loads(envpath.read_text())
        for key in ENVIRONMENT_KEYS:
            if oldenv[key] != current_env[key]: raise ValueError("Environment changed: "+key+"; use a fresh output directory")
    else:
        atomic_json(runfile,{"fingerprint":fingerprint,"config":conf,"source_sha256":hashes,"selected_seeds":seeds})
        atomic_json(out/"environment.json",current_env)
    env_id=json_hash(current_env);(out/"environment_records").mkdir(exist_ok=True)
    atomic_json(out/"environment_records"/(env_id+".json"),current_env)
    existing=restore_checkpoints(out)
    if raw.exists() and not existing:raise ValueError("Raw CSV exists without recoverable task checkpoints; inspect the output")
    done={(r["scenario"],int(r["seed"]),r["method"]) for r in existing}
    atomic_json(out/"seeds.json",seeds)
    work=Path(a.work_dir).resolve() if a.work_dir else out
    if a.work_dir and conf.get("keep_data",False):
        raise ValueError("Node-local work requires keep_data=false; the Slurm planner sets this storage-only option")
    for si,scenario in enumerate(conf["scenarios"]):
        methods=conf["penalized_methods"] if gen.is_penalized(scenario["regime"]) else conf["unpenalized_methods"]
        for seed in seeds:
            pending=[m for m in methods if (scenario["id"],seed,m) not in done]
            if not pending:continue
            datadir=work/"data"/(scenario["id"]+"_"+str(seed))
            # Scenario index is frozen in the unfiltered full/smoke configuration.
            original=json.loads(Path(a.config).read_text())["scenarios"]
            ordinal=next(i for i,s in enumerate(original) if s["id"]==scenario["id"])
            dataset,meta=gen.generate(scenario,seed,ordinal,conf,datadir)
            (out/"manifests").mkdir(exist_ok=True)
            manifest=out/"manifests"/(datadir.name+".json")
            if manifest.exists() and json.loads(manifest.read_text())["sha256"]!=meta["sha256"]:
                raise ValueError("Regenerated dataset differs from the checkpointed checksums")
            atomic_json(manifest,meta)
            x=gen.read_matrix(meta["files"]["x"],meta["n"],meta["p"])
            xt=gen.read_matrix(meta["files"]["xt"],meta["test_n"],meta["p"])
            y=np.fromfile(meta["files"]["y"],dtype="<f8");yt=np.fromfile(meta["files"]["yt"],dtype="<f8")
            beta=np.fromfile(meta["files"]["beta"],dtype="<f8")
            for method in pending:
                is_timing=seed in conf["timing_seeds"];repeats=conf["timing_repeats"] if is_timing else 1
                td=out/"tasks"/datadir.name/method;td.mkdir(parents=True,exist_ok=True)
                runtime=work/"workers"/datadir.name/method if a.work_dir else td
                runtime.mkdir(parents=True,exist_ok=True)
                task={"config":conf,"dataset":str(dataset),"method":method,"repeats":repeats,
                      "phase_file":str(runtime/"phase.json"),"result_file":str(runtime/"worker_result.json")}
                print(f'{scenario["id"]} seed={seed} {method} repeats={repeats}',flush=True)
                result=execute(task,runtime)
                if runtime!=td:
                    for name in ["task.json","worker.log","worker_result.json","phase.json"]:
                        if (runtime/name).exists():shutil.copy2(runtime/name,td/name)
                times=result.get("timings",[]);times=times if isinstance(times,list) else [times]
                warns=result.get("warnings",[]);warns=warns if isinstance(warns,list) else [warns]
                row={"scenario":scenario["id"],"regime":scenario["regime"],"n":meta["n"],"p":meta["p"],
                     "density":scenario.get("density",1.0),"realized_density":meta["realized_density"],
                     "seed":seed,"method":method,"timing_dataset":is_timing,"requested_repeats":repeats,
                     "environment_id":env_id,"worker_hostname":current_env["hostname"],
                     "completed_repeats":len(times),"total_fit_seconds":sum(times) if times else None,
                     "median_fit_seconds":float(np.median(times)) if times else None,
                     "fits_per_second":len(times)/sum(times) if times and sum(times)>0 else None,
                     "peak_rss_mib":result.get("peak_rss_mib"),"rss_baseline_mib":result.get("rss_baseline_mib"),
                     "rss_samples":result["rss_samples"],"native_converged":result.get("native_converged"),
                     "native_iterations":result.get("native_iterations"),"status":result["status"],
                     "lambda_selected":result.get("lambda_selected",0.0),"lambda_index":result.get("lambda_index"),
                     "lambda_boundary":result.get("lambda_boundary",False),"cv_deviance":result.get("cv_deviance"),
                     "warning_count":len(warns),"error":result.get("error",""),
                     "zero_true_columns":meta["zero_true_columns"]}
                diagnostics={k:None for k in ["train_deviance","test_deviance","objective","kkt_raw","kkt_scaled","selected","precision","recall","f1"]}
                quality=False
                if result["status"]=="ok":
                    try:
                        ev=evaluate(x,y,xt,yt,beta,result["coef"],result["intercept"],row["lambda_selected"],conf)
                        quality=ev.pop("quality_ok");diagnostics.update(ev)
                    except Exception as exc:row["error"]="Evaluation failed: "+repr(exc)
                row.update(diagnostics,quality_ok=quality,
                           success=bool(result["status"]=="ok" and quality and result.get("native_converged") is not False and len(times)==repeats))
                row={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
                atomic_json(td/"diagnostics.json",{"summary":row,"worker":result})
                with raw.open("a",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=list(row));
                    if f.tell()==0:w.writeheader()
                    w.writerow(row)
                with (out/"fit_trials.csv").open("a",newline="") as f:
                    w=csv.writer(f)
                    if f.tell()==0:w.writerow(["scenario","seed","method","trial","seconds","timing_dataset"])
                    for i,t in enumerate(times):w.writerow([scenario["id"],seed,method,i+1,t,is_timing])
                done.add((scenario["id"],seed,method))
                print(f'  status={row["status"]} quality={quality} median={row["median_fit_seconds"]}',flush=True)
            del x,xt,y,yt,beta
            if not conf.get("keep_data",False):shutil.rmtree(datadir)
    print("Completed. Summarize with: python summarize_results.py --out "+str(out),flush=True)


if __name__=="__main__":main()
