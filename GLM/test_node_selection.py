#!/usr/bin/env python3
"""Node-policy checks and mocked Slurm submissions; no experimental fits."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from run_benchmark import ROOT
from slurm.node_selection import check_node, cpu_model, explicit_nodes, select_nodes


class NodePolicyTests(unittest.TestCase):
    def test_candidate_pool_excludes_every_other_partition_node(self):
        selection = select_nodes(["node-b", "node-a"], ["node-a", "node-b", "node-c"], "TEST")
        self.assertEqual(selection["allowed_nodes"], ["node-a", "node-b"])
        self.assertEqual(selection["excluded_nodes"], ["node-c"])

    def test_missing_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            select_nodes(["missing"], ["node-a"], "TEST")

    def test_empty_pool_or_partition_is_rejected(self):
        with self.assertRaises(ValueError): select_nodes([], ["node-a"], "TEST")
        with self.assertRaises(ValueError): select_nodes(["node-a"], [], "TEST")

    def test_ambiguous_names_are_rejected(self):
        for names in ["node[1-2]", "node-a, node-b", "node-a,node-a", "node-a,"]:
            with self.subTest(names=names), self.assertRaises(ValueError): explicit_nodes(names)

    def test_allocation_and_cpu_are_both_checked(self):
        self.assertTrue(check_node("node-a", ["node-a"], "TEST CPU", "TEST CPU")["node_policy_passed"])
        with self.assertRaisesRegex(ValueError, "outside"):
            check_node("new-node", ["node-a"], "TEST CPU", "TEST CPU")
        with self.assertRaisesRegex(ValueError, "CPU differs"):
            check_node("node-a", ["node-a"], "OTHER CPU", "TEST CPU")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            check_node("node-a,node-b", ["node-a", "node-b"], "TEST CPU", "TEST CPU")


class MockedNodeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.project = self.base / "GLM"
        shutil.copytree(ROOT, self.project, ignore=shutil.ignore_patterns(
            ".venv", ".R-library", "__pycache__", "results", "site_config.sh", "setup-logs"))
        shutil.copy2(self.project / "slurm/site_config.example.sh", self.project / "slurm/site_config.sh")
        binary = self.base / "bin"
        binary.mkdir()
        self.journal = self.base / "sbatch.jsonl"
        self.srunlog = self.base / "srun.json"
        scripts = {
            "sinfo": "print('node-a\\nnode-b\\nnode-c')\n",
            "scontrol": "print('\\n'.join(sys.argv[-1].split(',')))\n",
            "sbatch": "p=Path(os.environ['MOCK_JOURNAL']); n=len(p.read_text().splitlines()) if p.exists() else 0; f=p.open('a'); f.write(json.dumps(sys.argv[1:])+'\\n'); f.close(); print(9001+n)\n",
            "srun": "Path(os.environ['MOCK_SRUN']).write_text(json.dumps(sys.argv[1:]))\n",
        }
        for name, body in scripts.items():
            p = binary / name
            p.write_text("#!" + sys.executable + "\nimport os,sys,json\nfrom pathlib import Path\n" + body)
            p.chmod(0o755)
        self.env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                    "GLM_ACCOUNT": "TEST_NO_CHARGE", "GLM_PYTHON": sys.executable,
                    "GLM_ALLOWED_NODES": "node-a,node-b", "GLM_EXPECTED_CPU": cpu_model(),
                    "GLM_MAX_PARALLEL": "10", "GLM_CONSTRAINT": "", "GLM_EXCLUSIVE_NODE": "0",
                    "MOCK_JOURNAL": str(self.journal), "MOCK_SRUN": str(self.srunlog)}
        self.runroot = self.base / "run"

    def tearDown(self): self.tmp.cleanup()

    def submit(self, *options):
        return subprocess.run(["bash", str(self.project / "slurm/submit.sh"), "full",
                               "--out", str(self.runroot), *options], env=self.env,
                              capture_output=True, text=True)

    def array(self, node="node-a", cpu=None):
        spool = self.base / "spool"
        spool.mkdir(exist_ok=True)
        script = spool / "slurm_script"
        shutil.copy2(self.project / "slurm/array.sbatch", script)
        env = {**self.env, "SLURM_SUBMIT_DIR": str(self.project), "SLURM_ARRAY_TASK_ID": "0",
               "SLURM_JOB_NODELIST": node, "TMPDIR": str(self.base)}
        if cpu is not None: env["GLM_EXPECTED_CPU"] = cpu
        return subprocess.run(["bash", str(script), str(self.runroot)], env=env,
                              capture_output=True, text=True)

    def test_pool_is_applied_without_changing_array_or_dependency(self):
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.journal.read_text().splitlines()]
        self.assertIn("--exclude=node-c", calls[0])
        self.assertFalse(any(a.startswith("--nodelist=") for a in calls[0]))
        self.assertIn("--nodes=1", calls[0])
        self.assertIn("--array=0-1099%10", calls[0])
        self.assertIn("--dependency=afterany:9001", calls[1])
        self.assertFalse(any(a.startswith("--exclude=") for a in calls[1]))
        record = json.loads((self.runroot / "submissions.jsonl").read_text())
        self.assertEqual(record["node_selection"]["allowed_nodes"], ["node-a", "node-b"])
        self.assertEqual(record["expected_cpu"], cpu_model())
        self.assertEqual(self.array().returncode, 0)
        self.assertTrue(self.srunlog.exists())
        self.assertFalse(list(self.base.glob("wire-glm.*")))

    def test_unknown_pool_stops_before_planning_or_submission(self):
        self.env["GLM_ALLOWED_NODES"] = "missing"
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("absent", result.stderr)
        self.assertFalse(self.journal.exists())
        self.assertFalse(self.runroot.exists())

    def test_newly_added_node_stops_before_worker_launch(self):
        result = self.array(node="node-c")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside", result.stderr)
        self.assertFalse(self.srunlog.exists())

    def test_changed_cpu_stops_before_worker_launch(self):
        result = self.array(cpu="WRONG TEST CPU")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU differs", result.stderr)
        self.assertFalse(self.srunlog.exists())

    def test_dry_run_submits_nothing(self):
        result = self.submit("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exclude=node-c", result.stdout)
        self.assertFalse(self.journal.exists())
        self.assertTrue((self.runroot / "plan.json").exists())


if __name__ == "__main__": unittest.main(verbosity=2)
