#!/usr/bin/env python3
"""CPU-aware diagnostic merge policy tests; fixtures are not benchmark results."""
import copy
import json
import unittest

import merge_diagnostics as diagnostic
from run_benchmark import atomic_json, json_hash
import test_slurm as controls


class DiagnosticPolicyTests(unittest.TestCase):
    def setUp(self):
        self.control = controls.ControlTests()
        self.control.setUp()
        self.control.complete()

    def tearDown(self): self.control.tearDown()

    def test_different_cpus_in_separate_dataset_blocks_are_allowed(self):
        self.control.write_shard(2, env=controls.fixture_environment(cpu="OTHER CPU"))
        plan, rows, records, environments, *_ = diagnostic.gather(self.control.root)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["cpu_model"] for row in rows}, {"TEST CPU", "OTHER CPU"})
        self.assertEqual(len(environments), 2)

    def test_different_software_remains_rejected(self):
        env = controls.fixture_environment(cpu="OTHER CPU")
        env["python"] = "DIFFERENT PYTHON"
        self.control.write_shard(2, env=env)
        with self.assertRaisesRegex(ValueError, "changed fields: python"):
            diagnostic.gather(self.control.root)

    def test_incomplete_shards_remain_rejected(self):
        shard = self.control.root / "shards/task-0000"
        next(shard.glob("tasks/*/*/diagnostics.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            diagnostic.gather(self.control.root)

    def test_full_profile_is_rejected(self):
        conf = copy.deepcopy(self.control.conf)
        conf["profile"] = "full"
        config = self.control.base / "formal.json"
        atomic_json(config, conf)
        root = self.control.base / "formal"
        diagnostic.workflow.create_plan(config, root)
        with self.assertRaisesRegex(ValueError, "diagnostics only"):
            diagnostic.gather(root)

    def test_methods_on_different_hosts_in_one_block_are_rejected(self):
        shard = self.control.root / "shards/task-0000"
        env = controls.fixture_environment(host="OTHER HOST")
        env_id = json_hash(env)
        atomic_json(shard / "environment_records" / (env_id + ".json"), env)
        path = next(shard.glob("tasks/*/*/diagnostics.json"))
        checkpoint = json.loads(path.read_text())
        checkpoint["summary"].update(environment_id=env_id, worker_hostname=env["hostname"])
        atomic_json(path, checkpoint)
        with self.assertRaisesRegex(ValueError, "share a node"):
            diagnostic.gather(self.control.root)


if __name__ == "__main__": unittest.main(verbosity=2)
