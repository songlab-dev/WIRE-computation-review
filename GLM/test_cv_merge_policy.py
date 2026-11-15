"""Synthetic controls for the user's nonblocking CV merge policy."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from check_results import verify_results


class CvMergePolicyTests(unittest.TestCase):
    def setUp(self):
        self.private = tempfile.TemporaryDirectory()
        self.root = Path(self.private.name)
        (self.root / "run.json").write_text(json.dumps({"config": {"kkt_tol": 1e-5}}))
        self.rows = [{"scenario": "H1", "seed": 123, "regime": "highdim",
                      "method": method, "success": True, "status": "ok", "error": "",
                      "requested_repeats": 1, "completed_repeats": 1,
                      "kkt_scaled": 1e-8, "objective": 1.0, "test_deviance": 2.0,
                      "lambda_selected": 0.2, "cv_deviance": score}
                     for method, score in [("R_glmnet", 3.0), ("Py_glum_lasso", 3.01)]]

    def tearDown(self):
        self.private.cleanup()

    def verify(self, require_all_success=False):
        pd.DataFrame(self.rows).to_csv(self.root / "raw_results.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            return verify_results(self.root, require_all_success)

    def test_cv_difference_allows_merge_and_remains_reported(self):
        report = self.verify()
        self.assertTrue(report["merge_checks_passed"])
        self.assertFalse(report["paired_checks_passed"])
        self.assertFalse(report["cv_difference_blocks_merge"])
        self.assertEqual(len(report["cv_groups_outside_previous_limit"]), 1)
        self.assertAlmostEqual(report["cv_groups_outside_previous_limit"][0]["selected_cv_score_range"], 0.01)
        self.assertEqual(report["cv_groups_outside_previous_limit"][0]["seed"], 123)
        self.assertEqual(pd.read_csv(self.root / "raw_results.csv").cv_deviance.tolist(), [3.0, 3.01])
        self.assertEqual(json.loads((self.root / "verification.json").read_text()), report)

    def test_close_cv_scores_are_recorded_as_equivalent(self):
        self.rows[1]["cv_deviance"] = 3.00001
        self.assertTrue(self.verify()["paired_checks_passed"])

    def test_lambda_and_objective_checks_are_preserved(self):
        for key, value, message in [("lambda_selected", 0.3, "Different selected lambda"),
                                    ("objective", 1.01, "Penalized objectives differ")]:
            old = self.rows[1][key]
            self.rows[1][key] = value
            with self.assertRaisesRegex(AssertionError, message):
                self.verify()
            self.rows[1][key] = old

    def test_success_kkt_check_is_preserved(self):
        self.rows[1]["kkt_scaled"] = 1e-3
        with self.assertRaises(AssertionError):
            self.verify()

    def test_failure_denominator_and_require_all_success_are_preserved(self):
        self.rows[1].update(success=False, status="error", completed_repeats=0, kkt_scaled=None)
        report = self.verify()
        self.assertEqual(report["tasks"], 2)
        self.assertFalse(report["all_success"])
        with self.assertRaises(AssertionError):
            self.verify(require_all_success=True)

    def test_duplicate_and_invalid_repeat_counts_are_still_rejected(self):
        self.rows[1]["method"] = self.rows[0]["method"]
        with self.assertRaisesRegex(AssertionError, "Duplicate"):
            self.verify()
        self.rows[1]["method"] = "Py_glum_lasso"
        self.rows[1]["completed_repeats"] = 2
        with self.assertRaises(AssertionError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
