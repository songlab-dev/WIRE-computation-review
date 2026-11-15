#!/usr/bin/env python3
"""CPU policy regression tests; synthetic fixtures are not benchmark results."""
import copy
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_benchmark as benchmark
import slurm_cpu_relaxed as relaxed
import test_slurm as controls
from run_benchmark import atomic_json, json_hash, source_hashes


class CpuPolicyTests(unittest.TestCase):
    def setUp(self):
        self.control = controls.ControlTests()
        self.control.setUp()
        self.control.complete()

    def tearDown(self):
        self.control.tearDown()

    def test_mixed_cpus_are_accepted_and_recorded(self):
        self.control.write_shard(2, env=controls.fixture_environment(cpu="OTHER CPU"))
        _, rows, _, environments, *_ = relaxed.gather(self.control.root)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["cpu_model"] for row in rows}, {"TEST CPU", "OTHER CPU"})
        self.assertEqual(len(environments), 2)

    def test_full_profile_is_accepted(self):
        # Small fixtures exercise profile routing, not the publication gate.
        conf = copy.deepcopy(self.control.conf)
        conf["profile"] = "full"
        config = self.control.base / "full-fixture.json"
        atomic_json(config, conf)
        self.control.root = self.control.base / "full-fixture"
        self.control.plan = relaxed.workflow.create_plan(config, self.control.root)
        self.control.complete()
        self.control.write_shard(2, env=controls.fixture_environment(cpu="OTHER CPU"))
        plan, rows, *_ = relaxed.gather(self.control.root)
        self.assertEqual(plan["config"]["profile"], "full")
        self.assertEqual(len(rows), 8)

    def test_resume_provenance_may_span_nodes_and_cpus(self):
        shard = self.control.root / "shards/task-0000"
        env = controls.fixture_environment(host="NEW NODE", cpu="NEW CPU")
        env_id = json_hash(env)
        atomic_json(shard / "environment_records" / (env_id + ".json"), env)
        path = next(shard.glob("tasks/*/*/diagnostics.json"))
        checkpoint = json.loads(path.read_text())
        checkpoint["summary"].update(environment_id=env_id, worker_hostname=env["hostname"])
        atomic_json(path, checkpoint)
        _, rows, *_ = relaxed.gather(self.control.root)
        self.assertEqual(len(rows), 8)
        self.assertFalse(relaxed.pairing_metadata(rows)["same_node_method_pairing_verified"])
        self.assertFalse(relaxed.pairing_metadata(rows)["same_cpu_method_pairing_verified"])

    def test_changed_software_is_rejected(self):
        env = controls.fixture_environment(cpu="OTHER CPU")
        env["python"] = "OTHER PYTHON"
        self.control.write_shard(2, env=env)
        with self.assertRaisesRegex(ValueError, "changed fields: python"):
            relaxed.gather(self.control.root)

    def test_changed_blas_is_rejected(self):
        env = controls.fixture_environment(cpu="OTHER CPU")
        env["python_BLAS"] = [{"version": "OTHER BLAS"}]
        self.control.write_shard(2, env=env)
        with self.assertRaisesRegex(ValueError, "python_BLAS"):
            relaxed.gather(self.control.root)

    def test_incomplete_shards_are_rejected(self):
        shard = self.control.root / "shards/task-0000"
        next(shard.glob("tasks/*/*/diagnostics.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            relaxed.gather(self.control.root)

    def test_environment_tampering_is_rejected(self):
        shard = self.control.root / "shards/task-0000"
        path = next((shard / "environment_records").glob("*.json"))
        env = json.loads(path.read_text())
        env["cpu"] = "TAMPERED CPU"
        atomic_json(path, env)
        with self.assertRaisesRegex(ValueError, "provenance"):
            relaxed.gather(self.control.root)

    def test_source_guard_is_preserved(self):
        changed = {**source_hashes(), "python_worker.py": "CHANGED SOURCE"}
        with patch.object(relaxed.workflow, "source_hashes", return_value=changed):
            with self.assertRaisesRegex(ValueError, "source changed"):
                relaxed.gather(self.control.root)

    def test_merge_preserves_failures_and_is_idempotent(self):
        self.control.write_shard(0, failed_method="R_stats")
        self.control.write_shard(2, env=controls.fixture_environment(cpu="OTHER CPU"))
        with patch.object(relaxed.subprocess, "run"), patch.object(relaxed, "label_tables"):
            merged = relaxed.merge_results(self.control.root)
        before = (merged / "raw_results.csv").read_bytes()
        with (merged / "raw_results.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["success"] == "False" for row in rows), 1)
        self.assertFalse(json.loads((merged / "environment.json").read_text())["execution"]["homogeneous_environment_verified"])
        self.assertEqual(relaxed.merge_results(self.control.root), merged)
        self.assertEqual((merged / "raw_results.csv").read_bytes(), before)

    def test_existing_checkpoint_resumes_on_new_cpu_without_refitting(self):
        task = self.control.plan["tasks"][0]
        shard = relaxed.workflow.shard_path(self.control.root, task)
        atomic_json(shard / "environment.json", controls.fixture_environment())
        before = {p: p.read_bytes() for p in shard.glob("tasks/*/*/diagnostics.json")}
        hashes = source_hashes()
        keys = benchmark.ENVIRONMENT_KEYS
        env = controls.fixture_environment(cpu="NEW CPU", host="NEW NODE")
        with patch.object(benchmark, "preflight"), patch.object(benchmark, "environment", return_value=env), \
                patch.object(benchmark, "execute") as execute:
            relaxed.benchmark_worker(["--config", str(self.control.root / "config.json"),
                "--out", str(shard), "--scenario", task["scenario"], "--seed", str(task["seed"]), "--resume"])
            execute.assert_not_called()
        self.assertEqual(source_hashes(), hashes)
        self.assertEqual(benchmark.ENVIRONMENT_KEYS, keys)
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)
        self.assertEqual(json.loads((shard / "environment.json").read_text())["cpu"], "TEST CPU")
        environments = [json.loads(p.read_text()) for p in (shard / "environment_records").glob("*.json")]
        self.assertEqual({e["cpu"] for e in environments}, {"TEST CPU", "NEW CPU"})
        current = next(e for e in environments if e["cpu"] == "NEW CPU")
        self.assertEqual(current["runtime_controller"]["cpu_policy"], relaxed.POLICY)

    def test_resume_still_rejects_changed_software(self):
        task = self.control.plan["tasks"][0]
        shard = relaxed.workflow.shard_path(self.control.root, task)
        atomic_json(shard / "environment.json", controls.fixture_environment())
        keys = benchmark.ENVIRONMENT_KEYS
        env = controls.fixture_environment(cpu="NEW CPU")
        env["python"] = "NEW PYTHON"
        with patch.object(benchmark, "preflight"), patch.object(benchmark, "environment", return_value=env):
            with self.assertRaisesRegex(ValueError, "Environment changed: python"):
                relaxed.benchmark_worker(["--config", str(self.control.root / "config.json"),
                    "--out", str(shard), "--scenario", task["scenario"], "--seed", str(task["seed"]), "--resume"])
        self.assertEqual(benchmark.ENVIRONMENT_KEYS, keys)

    def test_array_task_uses_relaxed_worker(self):
        with patch.object(relaxed.subprocess, "run") as run:
            relaxed.run_task(self.control.root, 0, work_dir=self.control.base / "scratch")
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "slurm_cpu_relaxed.py")
        self.assertEqual(command[2], "worker")
        self.assertIn("--resume", command)
        self.assertIn("--work-dir", command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
