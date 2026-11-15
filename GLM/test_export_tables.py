"""Table guards and statistical denominators; controls are not new GLM fits."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import export_tables as tables
from run_benchmark import ROOT, run_fingerprint, source_hashes


class ExportControlTests(unittest.TestCase):
    def setUp(self):
        self.private = tempfile.TemporaryDirectory()
        self.root = Path(self.private.name) / "SYNTHETIC_CONTROL_NOT_EXPERIMENT"
        self.root.mkdir()
        self.conf = json.loads((ROOT / "configs/smoke.json").read_text())
        self.conf.update(profile="smoke", statistical_replicates=2, timing_seeds=[101, 102], timing_repeats=5)
        self.conf["scenarios"] = [self.conf["scenarios"][0], self.conf["scenarios"][2]]
        self.conf["unpenalized_methods"] = ["R_stats", "R_biglm"]
        self.conf["penalized_methods"] = ["R_glmnet", "Py_glum_lasso"]
        self.rows, self.trials = [], []
        for scenario in self.conf["scenarios"]:
            methods = self.conf["penalized_methods"] if scenario["regime"] in tables.PENALIZED else self.conf["unpenalized_methods"]
            for j, seed in enumerate(self.conf["timing_seeds"]):
                for k, method in enumerate(methods):
                    times = [(i + 1) * 0.1 * (j + 1) * (k + 1) for i in range(5)]
                    self.rows.append({"scenario": scenario["id"], "regime": scenario["regime"],
                        "n": scenario["n"], "p": scenario["p"], "density": scenario.get("density", 1.0),
                        "seed": seed, "method": method, "timing_dataset": True, "success": True,
                        "quality_ok": True, "status": "ok", "error": "", "requested_repeats": 5,
                        "completed_repeats": 5, "median_fit_seconds": float(np.median(times)),
                        "total_fit_seconds": sum(times), "peak_rss_mib": 100 + 20 * j + k, "rss_samples": 3,
                        "kkt_scaled": 1e-9, "objective": 1.0, "test_deviance": 2.0,
                        "f1": (0.2 + j * 0.2) if scenario["regime"] in tables.PENALIZED else np.nan,
                        "lambda_selected": 0.2 if scenario["regime"] in tables.PENALIZED else np.nan,
                        "cv_deviance": 1.0 if scenario["regime"] in tables.PENALIZED else np.nan,
                        "cpu_model": "SYNTHETIC TEST CPU", "worker_hostname": "SYNTHETIC TEST NODE"})
                    self.trials.extend({"scenario": scenario["id"], "seed": seed, "method": method,
                                        "trial": i + 1, "seconds": t, "timing_dataset": True}
                                       for i, t in enumerate(times))
        self.write()
        (self.root / "verification.json").write_text('{"original_input":"DO NOT MODIFY"}\n')

    def tearDown(self):
        self.private.cleanup()

    def write(self):
        hashes = source_hashes()
        seeds = self.conf["timing_seeds"]
        (self.root / "run.json").write_text(json.dumps({"config": self.conf, "source_sha256": hashes,
            "selected_seeds": seeds, "fingerprint": run_fingerprint(self.conf, hashes, seeds)}))
        pd.DataFrame(self.rows).to_csv(self.root / "raw_results.csv", index=False)
        pd.DataFrame(self.trials).to_csv(self.root / "fit_trials.csv", index=False)

    def export(self, allow=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return tables.export(self.root, self.root / "export", allow)

    def test_mcse_uses_two_independent_datasets_not_ten_measured_fits(self):
        self.export()
        frame = pd.read_csv(self.root / "export/table1_unpenalized.csv")
        row = frame[frame.method.eq("R_stats")].iloc[0]
        self.assertAlmostEqual(row.fit_seconds_mean, 0.45)
        self.assertAlmostEqual(row.fit_seconds_sd, 0.3 / np.sqrt(2))
        self.assertAlmostEqual(row.fit_seconds_mcse, 0.15)
        self.assertEqual(row.fit_seconds_n, 2)
        self.assertFalse(any(c.startswith("f1_") or "train_deviance" in c or "throughput" in c or "precision" in c or "recall" in c for c in frame.columns))
        self.assertIn("test_deviance_mean", frame.columns)

    def test_final_manuscript_endpoints_and_f1_denominator(self):
        self.export()
        frame = pd.read_csv(self.root / "export/table2_penalized.csv")
        row = frame[frame.method.eq("R_glmnet")].iloc[0]
        self.assertAlmostEqual(row.f1_mean, 0.3)
        self.assertAlmostEqual(row.f1_mcse, 0.1)
        self.assertEqual(row.f1_n, 2)
        self.assertIn("Support $F_1$", (self.root / "export/table2_main.tex").read_text())

    def test_failure_stays_in_attempted_denominator(self):
        row = self.rows[1]
        row.update(success=False, quality_ok=False, status="error", completed_repeats=0,
                   median_fit_seconds=np.nan, total_fit_seconds=np.nan, peak_rss_mib=np.nan, rss_samples=0)
        self.trials = [r for r in self.trials if not (r["scenario"] == row["scenario"] and r["seed"] == row["seed"] and r["method"] == row["method"])]
        self.write()
        metadata = self.export()
        frame = pd.read_csv(self.root / "export/table1_unpenalized.csv")
        biglm = frame[frame.method.eq("R_biglm")].iloc[0]
        self.assertEqual((biglm.attempted_datasets, biglm.successful_datasets, biglm.fit_seconds_n), (2, 1, 1))
        self.assertTrue(pd.isna(biglm.fit_seconds_mcse))
        self.assertEqual(metadata["failed_tasks"], 1)
        self.assertIn("1/2 successful", (self.root / "export/table1_main.tex").read_text())

    def test_missing_rss_is_missing_with_its_own_denominator(self):
        self.rows[0].update(peak_rss_mib=np.nan, rss_samples=0)
        self.write()
        self.export()
        row = pd.read_csv(self.root / "export/table1_unpenalized.csv").iloc[0]
        self.assertEqual((row.fit_seconds_n, row.peak_rss_mib_n), (2, 1))
        self.assertEqual(row.peak_rss_mib_mean, 120)
        self.assertTrue(pd.isna(row.peak_rss_mib_mcse))

    def test_inputs_are_unchanged_and_outputs_are_deterministic(self):
        names = ["run.json", "raw_results.csv", "fit_trials.csv", "verification.json"]
        before = {n: tables.sha256(self.root / n) for n in names}
        self.export()
        files = {p.name: p.read_bytes() for p in (self.root / "export").iterdir()}
        self.export()
        self.assertEqual(before, {n: tables.sha256(self.root / n) for n in names})
        self.assertEqual(files, {p.name: p.read_bytes() for p in (self.root / "export").iterdir()})

    def test_cv_differences_are_retained_without_blocking(self):
        self.rows[-1]["cv_deviance"] = 1.01
        self.write()
        metadata = self.export()
        self.assertEqual(metadata["cv_difference_policy"], "record_only")
        self.assertEqual(len(metadata["cv_groups_outside_previous_limit"]), 1)

    def test_nonfull_run_requires_explicit_labeled_export(self):
        with self.assertRaisesRegex(ValueError, "allow-validation"):
            self.export(allow=False)
        self.export()
        self.assertIn("EXPLORATORY", (self.root / "export/table1_main.tex").read_text())

    def test_old_protocol_cannot_be_relabeled_as_full(self):
        self.conf.update(profile="full", statistical_replicates=50)
        self.write()
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.export()

    def test_missing_or_duplicate_task_cannot_be_published(self):
        row = self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "grid"):
            self.export()
        self.rows.extend([row, row.copy()])
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.export()

    def test_missing_trial_and_incorrect_median_are_rejected(self):
        trial = self.trials.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "trial count"):
            self.export()
        self.trials.append(trial)
        self.rows[0]["median_fit_seconds"] += 0.01
        self.write()
        with self.assertRaisesRegex(ValueError, "median"):
            self.export()

    def test_source_and_design_tampering_are_rejected(self):
        run = json.loads((self.root / "run.json").read_text())
        run["source_sha256"]["python_worker.py"] = "CHANGED"
        (self.root / "run.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.export()
        self.rows[0]["n"] += 1
        self.write()
        with self.assertRaisesRegex(ValueError, "design"):
            self.export()

    def test_objective_and_kkt_guards_remain_active(self):
        self.rows[0]["objective"] += 0.1
        self.write()
        with self.assertRaisesRegex(AssertionError, "objectives differ"):
            self.export()
        self.rows[0].update(objective=1.0, kkt_scaled=1e-3)
        self.write()
        with self.assertRaisesRegex(ValueError, "KKT"):
            self.export()



if __name__ == "__main__":
    unittest.main(verbosity=2)
