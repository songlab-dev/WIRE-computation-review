#!/usr/bin/env python3
"""Slurm control tests. Synthetic fixtures are NOT experimental results."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import slurm_workflow as workflow
from run_benchmark import (ROOT, atomic_json, checkpoint_records, json_hash,
                           restore_checkpoints, run_fingerprint)


def fixture_environment(host="test-node", cpu="TEST CPU"):
    return {"platform":"TEST","architecture":"TEST","cpu":cpu,"python":"TEST",
            "python_packages":{"fixture":"1"},"R":{"R":"TEST","packages":{}},
            "python_BLAS":[],"thread_limits":{"OMP_NUM_THREADS":"1"},
            "loaded_modules":"TEST","hostname":host,"slurm":{"SLURM_JOB_ID":"TEST"}}


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.root=self.base/"run"
        self.conf=json.loads((ROOT/"configs/smoke.json").read_text())
        self.conf["scenarios"]=self.conf["scenarios"][:2]
        self.conf["statistical_replicates"]=2
        self.conf["unpenalized_methods"]=["R_stats","Py_statsmodels"]
        self.config=self.base/"config.json";atomic_json(self.config,self.conf)
        self.plan=workflow.create_plan(self.config,self.root)

    def tearDown(self):self.tmp.cleanup()

    def write_shard(self,index,failed_method=None,env=None):
        task=self.plan["tasks"][index];shard=workflow.shard_path(self.root,task)
        shard.mkdir(parents=True,exist_ok=True)
        conf=copy.deepcopy(self.plan["config"]);scenario=conf["scenarios"][task["scenario_index"]]
        conf["scenarios"]=[scenario]
        atomic_json(shard/"run.json",{"config":conf,"source_sha256":self.plan["source_sha256"],
                    "selected_seeds":[task["seed"]],"fingerprint":run_fingerprint(conf,self.plan["source_sha256"],[task["seed"]])})
        atomic_json(shard/"seeds.json",[task["seed"]])
        env=env or fixture_environment();env_id=json_hash(env)
        (shard/"environment_records").mkdir(exist_ok=True)
        atomic_json(shard/"environment_records"/(env_id+".json"),env)
        (shard/"manifests").mkdir(exist_ok=True)
        atomic_json(shard/"manifests"/(task["scenario"]+"_"+str(task["seed"])+".json"),
                    {"id":task["scenario"],"seed":task["seed"],"scenario_index":task["scenario_index"],
                     "n":scenario["n"],"p":scenario["p"],"test_n":conf["test_n"],"sha256":{"fixture":"TEST NOT DATA"}})
        repeats=conf["timing_repeats"] if task["timing_dataset"] else 1
        for method in task["methods"]:
            okay=method!=failed_method;times=[0.1]*repeats if okay else []
            row={"scenario":task["scenario"],"seed":task["seed"],"method":method,
                 "n":scenario["n"],"p":scenario["p"],"regime":scenario["regime"],
                 "density":scenario.get("density",1.0),"timing_dataset":task["timing_dataset"],
                 "requested_repeats":repeats,"completed_repeats":len(times),
                 "quality_ok":okay,"kkt_scaled":0 if okay else None,"success":okay,
                 "status":"ok" if okay else "error","environment_id":env_id,"worker_hostname":env["hostname"]}
            directory=shard/"tasks"/(task["scenario"]+"_"+str(task["seed"]))/method
            directory.mkdir(parents=True,exist_ok=True)
            atomic_json(directory/"diagnostics.json",{"summary":row,"worker":{"timings":times,"status":row["status"]}})
        return shard

    def complete(self):
        for i in range(len(self.plan["tasks"])):self.write_shard(i)

    def test_full_grid_counts(self):
        p=workflow.create_plan(ROOT/"configs/full.json",self.base/"full")
        self.assertEqual(len(p["tasks"]),1100)
        self.assertEqual(sum(len(t["methods"]) for t in p["tasks"]),6800)
        self.assertEqual(sum(t["timing_dataset"] for t in p["tasks"]),1100)
        self.assertEqual(sum(len(t["methods"])*p["config"]["timing_repeats"] for t in p["tasks"] if t["timing_dataset"]),34000)
        self.assertEqual(len(p["seeds"]),100)
        self.assertEqual(len(set(p["seeds"])),100)
        self.assertEqual(p["config"]["timing_repeats"],5)
        self.assertTrue(all(t["timing_dataset"] for t in p["tasks"]))

    def test_resume_plan_and_source_guard(self):
        with self.assertRaises(ValueError):workflow.create_plan(self.config,self.root)
        self.assertEqual(workflow.create_plan(self.config,self.root,True),self.plan)
        changed=copy.deepcopy(self.conf);changed["timing_repeats"]+=1;atomic_json(self.config,changed)
        with self.assertRaises(ValueError):workflow.create_plan(self.config,self.root,True)

    def test_saved_config_tampering(self):
        atomic_json(self.root/"config.json",{})
        with self.assertRaises(ValueError):workflow.load_plan(self.root)

    def test_pending_and_incomplete_merge(self):
        self.assertEqual(workflow.pending_tasks(self.plan,self.root),[0,1,2,3])
        self.write_shard(0)
        self.assertEqual(workflow.pending_tasks(self.plan,self.root),[1,2,3])
        with self.assertRaisesRegex(ValueError,"incomplete"):workflow.merge_results(self.root)
        self.assertFalse((self.root/"merged").exists())

    def test_merge_preserves_failure_and_is_idempotent(self):
        self.complete();self.write_shard(0,failed_method="R_stats")
        self.assertEqual(workflow.pending_tasks(self.plan,self.root),[])
        merged=workflow.merge_results(self.root)
        with (merged/"raw_results.csv").open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),8);self.assertEqual(sum(r["success"]=="False" for r in rows),1)
        self.assertEqual(json.loads((merged/"run.json").read_text())["config"]["statistical_replicates"],2)
        before=(merged/"raw_results.csv").read_bytes()
        self.assertEqual(workflow.merge_results(self.root),merged)
        self.assertEqual((merged/"raw_results.csv").read_bytes(),before)

    def test_different_hosts_same_cpu_are_allowed(self):
        self.complete();self.write_shard(3,env=fixture_environment(host="test-node-2"))
        merged=workflow.merge_results(self.root)
        env=json.loads((merged/"environment.json").read_text())
        self.assertEqual(env["execution"]["distinct_hosts"],["test-node","test-node-2"])

    def test_mixed_cpu_rejected(self):
        self.complete();self.write_shard(2,env=fixture_environment(cpu="DIFFERENT TEST CPU"))
        with self.assertRaisesRegex(ValueError,"Heterogeneous"):workflow.merge_results(self.root)
        self.assertFalse((self.root/"merged").exists())

    def test_mixed_blas_rejected(self):
        self.complete();env=fixture_environment();env["python_BLAS"]=[{"version":"OTHER TEST"}]
        self.write_shard(2,env=env)
        with self.assertRaisesRegex(ValueError,"python_BLAS"):workflow.merge_results(self.root)

    def test_environment_provenance_rejected(self):
        self.complete();shard=workflow.shard_path(self.root,self.plan["tasks"][0])
        path=next((shard/"environment_records").glob("*.json"));env=json.loads(path.read_text())
        env["cpu"]="TAMPERED";atomic_json(path,env)
        with self.assertRaisesRegex(ValueError,"provenance"):workflow.merge_results(self.root)

    def test_partial_csv_recovered_from_atomic_checkpoints(self):
        shard=self.write_shard(0)
        (shard/"raw_results.csv").write_text("partial,csv\nBROKEN")
        (shard/"fit_trials.csv").write_text("partial,csv\nBROKEN")
        rows=restore_checkpoints(shard);self.assertEqual(len(rows),2)
        with (shard/"fit_trials.csv").open() as f:self.assertEqual(len(list(csv.DictReader(f))),4)
        self.assertEqual(len(checkpoint_records(shard)),2)

    def test_duplicate_checkpoint_rejected(self):
        shard=self.write_shard(0);path=next(shard.glob("tasks/*/*/diagnostics.json"))
        duplicate=path.parent.parent/"duplicate";duplicate.mkdir()
        shutil.copy2(path,duplicate/"diagnostics.json")
        with self.assertRaisesRegex(ValueError,"Duplicate"):checkpoint_records(shard)

    def test_unsupported_seed_rejected_before_preflight(self):
        result=subprocess.run([sys.executable,str(ROOT/"run_benchmark.py"),"--config",str(ROOT/"configs/smoke.json"),
                              "--seed","-1","--out",str(self.base/"badseed")],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn("Selected seeds",result.stderr)
        self.assertFalse((self.base/"badseed").exists())


class SubmissionTests(unittest.TestCase):
    def test_mock_submission_and_spooled_array(self):
        """Mocks sbatch/srun locally: no cluster access and no chargeable jobs."""
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);project=base/"GLM"
            shutil.copytree(ROOT,project,ignore=shutil.ignore_patterns(".venv",".R-library","__pycache__","results","site_config.sh","setup-logs"))
            shutil.copy2(project/"slurm/site_config.example.sh",project/"slurm/site_config.sh")
            binary=base/"bin";binary.mkdir();journal=base/"mock-sbatch.jsonl";srunlog=base/"mock-srun.json"
            mock=binary/"sbatch"
            mock.write_text("#!"+sys.executable+"\nimport os,json,sys\nfrom pathlib import Path\np=Path(os.environ['MOCK_SBATCH_JOURNAL'])\nn=len(p.read_text().splitlines()) if p.exists() else 0\nf=p.open('a');f.write(json.dumps(sys.argv[1:])+'\\n');f.close();print(9001+n)\n")
            mock.chmod(0o755)
            mockrun=binary/"srun"
            mockrun.write_text("#!"+sys.executable+"\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ['MOCK_SRUN_LOG']).write_text(json.dumps(sys.argv[1:]))\n")
            mockrun.chmod(0o755)
            env={**os.environ,"PATH":str(binary)+os.pathsep+os.environ["PATH"],"GLM_ACCOUNT":"TEST_NO_CHARGE", "GLM_PYTHON":sys.executable,
                 "MOCK_SBATCH_JOURNAL":str(journal),"MOCK_SRUN_LOG":str(srunlog)}
            runroot=base/"run"
            result=subprocess.run(["bash",str(project/"slurm/submit.sh"),"full","--out",str(runroot)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(len(calls),2)
            self.assertIn("--array=0-1099%10",calls[0]);self.assertIn("--dependency=afterany:9001",calls[1])
            self.assertIn("--account=TEST_NO_CHARGE",calls[0]);self.assertIn("--cpus-per-task=2",calls[0])
            self.assertEqual(calls[0][-1],str(runroot))
            # Slurm executes a spooled script, so dirname($0) is NOT the project directory.
            spool=base/"spool";spool.mkdir();script=spool/"slurm_script"
            shutil.copy2(project/"slurm/array.sbatch",script)
            env.update(SLURM_SUBMIT_DIR=str(project),SLURM_ARRAY_TASK_ID="0",TMPDIR=str(base))
            result=subprocess.run(["bash",str(script),str(runroot)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            arguments=json.loads(srunlog.read_text())
            self.assertIn("--cpu-bind=cores",arguments);self.assertIn("run-task",arguments)
            work=Path(arguments[arguments.index("--work-dir")+1]);self.assertFalse(work.exists())
            self.assertFalse(list(base.glob("wire-glm.*")))


if __name__=="__main__":unittest.main(verbosity=2)
