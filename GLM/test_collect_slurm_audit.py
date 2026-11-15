#!/usr/bin/env python3
"""Diagnostic-export controls; synthetic fixtures are not experimental results."""
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import unittest

import collect_slurm_audit as audit
import test_slurm as controls


def row(method, cv=2.0, success=True, regime="highdim"):
    return {"scenario": "H1", "seed": 123, "method": method, "regime": regime,
            "success": success, "status": "ok" if success else "error",
            "objective": 1.0 if success else None, "test_deviance": 10.0 if success else None,
            "kkt_scaled": 1e-8 if success else None, "lambda_selected": 0.2 if success else None,
            "lambda_index": 1 if success else None, "cv_deviance": cv if success else None}


class PairAuditTests(unittest.TestCase):
    def test_absolute_cv_boundary_is_not_relaxed_by_small_relative_difference(self):
        rows = [row("A", 0.0), row("B", audit.DEVIANCE_LIMIT)]
        result = audit.paired_audit(rows, 1e-5)[0]
        self.assertIn("cv_range_at_or_above_limit", result["issues"])
        self.assertEqual(result["audit_status"], "issues")
        rows = [row("A", 1e6), row("B", 1e6 + 0.01)]
        result = audit.paired_audit(rows, 1e-5)[0]
        self.assertLess(result["cv_relative_range_diagnostic_only"], 1e-6)
        self.assertIn("cv_range_at_or_above_limit", result["issues"])

    def test_failed_methods_do_not_disappear_from_denominators(self):
        rows = [row("A"), row("B", 2.0 + 1e-5), row("C", success=False)]
        result = audit.paired_audit(rows, 1e-5)[0]
        self.assertEqual((result["recorded_methods"], result["successful_methods"], result["failed_methods"]), (3, 2, 1))
        self.assertEqual(result["failed_method_names"], "C")
        self.assertEqual(result["audit_status"], "passed")

    def test_lambda_missing_scores_and_objective_are_all_reported(self):
        rows = [row("A"), row("B")]
        rows[1].update(lambda_selected=0.1, cv_deviance=math.nan, objective=1.01)
        result = audit.paired_audit(rows, 1e-5)[0]
        for reason in ["selected_lambda_differs_or_invalid", "cv_missing_or_nonfinite",
                       "objective_range_at_or_above_limit"]:
            self.assertIn(reason, result["issues"])

    def test_unpenalized_checks_holdout_and_kkt(self):
        rows = [row("A", regime="dense"), row("B", regime="dense")]
        rows[1].update(test_deviance=10.1, kkt_scaled=1e-3)
        result = audit.paired_audit(rows, 1e-5)[0]
        self.assertIn("holdout_range_at_or_above_limit", result["issues"])
        self.assertIn("successful_task_kkt_invalid", result["issues"])
        self.assertIsNone(result["selected_cv_range"])

    def test_single_success_and_all_failure_are_not_called_paired_passes(self):
        for rows in [[row("A"), row("B", success=False)], [row("A", success=False)]]:
            self.assertEqual(audit.paired_audit(rows, 1e-5)[0]["audit_status"], "not_applicable")
        self.assertFalse(audit.successful({"success": "False"}))


class CvAuditTests(unittest.TestCase):
    def test_complete_curve_and_zero_based_selected_index_are_preserved(self):
        item = row("A")
        worker = {"cv_scores": [3.0, 2.0, 4.0], "lambda_index": 1,
                  "lambda_selected": 0.2, "cv_deviance": 2.0}
        curves, issues = audit.cv_audit([item], [{"worker": worker}], 3)
        self.assertEqual([curve["cv_deviance"] for curve in curves], [3.0, 2.0, 4.0])
        self.assertEqual([curve["selected"] for curve in curves], [False, True, False])
        self.assertEqual(issues, [])

    def test_first_minimum_and_scalar_consistency_are_checked(self):
        item = row("A")
        worker = {"cv_scores": [1.0, 1.0, 3.0], "lambda_index": 1,
                  "lambda_selected": 0.2, "cv_deviance": 2.0}
        _, issues = audit.cv_audit([item], [{"worker": worker}], 3)
        reasons = {issue["issue"] for issue in issues}
        self.assertIn("selected_index_not_first_cv_minimum", reasons)
        self.assertIn("summary_cv_differs_from_selected_path_score", reasons)

    def test_missing_nonfinite_and_wrong_length_successful_paths_are_flagged(self):
        for scores in [None, [1.0], [1.0, math.inf, 3.0]]:
            _, issues = audit.cv_audit([row("A")], [{"worker": {"cv_scores": scores}}], 3)
            self.assertTrue(issues)
        _, issues = audit.cv_audit([row("A", success=False)], [{"worker": {"status": "error"}}], 3)
        self.assertEqual(issues, [])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.control = controls.ControlTests()
        self.control.setUp()
        self.control.complete()
        # Valid failure checkpoints are completed tasks; export must retain them.
        self.control.write_shard(0, failed_method="R_stats")

    def tearDown(self):
        self.control.tearDown()

    def snapshot(self):
        root = self.control.root
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    def test_archive_preserves_original_checkpoint_values_failures_and_source_files(self):
        before = self.snapshot()
        destination = self.control.base / "audit.tar.gz"
        with contextlib.redirect_stdout(io.StringIO()):
            audit.export_audit(self.control.root, destination)
        self.assertEqual(self.snapshot(), before)
        with tarfile.open(destination) as packed:
            prefix = "GLM_smoke_audit/"
            summary = json.load(packed.extractfile(prefix + "summary.json"))
            self.assertEqual(summary["recorded_method_tasks"], 8)
            self.assertEqual(summary["failed_method_tasks"], 1)
            self.assertFalse(summary["publication_verified"])
            self.assertFalse(summary["relative_cv_difference_changes_acceptance"])
            records = [json.loads(line) for line in packed.extractfile(prefix + "worker_results.jsonl")]
            self.assertEqual(len(records), 8)
            failed = [record for record in records if not record["summary"]["success"]]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["worker"]["status"], "error")
            for filename, digest in self.control.plan["source_sha256"].items():
                self.assertEqual(hashlib.sha256(packed.extractfile(prefix + "source/" + filename).read()).hexdigest(), digest)
            raw = list(csv.DictReader(io.StringIO(packed.extractfile(prefix + "raw_results.csv").read().decode())))
            self.assertEqual(len(raw), 8)

    def test_refuses_to_replace_existing_archive(self):
        destination = self.control.base / "existing.tar.gz"
        destination.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            audit.export_audit(self.control.root, destination)
        self.assertEqual(destination.read_bytes(), b"original")

    def test_refuses_to_write_inside_original_run(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "outside"):
            audit.export_audit(self.control.root, self.control.root / "archive.tar.gz")
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
