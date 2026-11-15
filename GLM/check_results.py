#!/usr/bin/env python3
"""Check recorded results; CV score differences are recorded without blocking."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def verify_results(root, require_all_success=False):
    root = Path(root)
    df = pd.read_csv(root / "raw_results.csv")
    okay = df.success.astype(str).str.lower().eq("true")
    assert not df.duplicated(["scenario", "seed", "method"]).any(), "Duplicate task rows"
    assert ((df.completed_repeats <= df.requested_repeats) & (df.completed_repeats >= 0)).all()
    conf = json.loads((root / "run.json").read_text())["config"]
    assert df.loc[okay, "kkt_scaled"].le(conf["kkt_tol"]).all()
    checks, cv_differences = [], []
    for (scenario, seed), group in df[okay].groupby(["scenario", "seed"]):
        unpen = group.regime.iloc[0] in {"dense", "sparse"}
        cv_range, cv_equal = None, None
        if unpen and len(group) > 1:
            assert np.ptp(group.objective) < 1e-6, (scenario, int(seed), "Unpenalized objectives differ")
            assert np.ptp(group.test_deviance) < 1e-4, (scenario, int(seed), "Holdout predictions differ")
        if not unpen and len(group) > 1:
            assert group.lambda_selected.nunique() == 1, (scenario, int(seed), "Different selected lambda; inspect CV scores")
            values = pd.to_numeric(group.cv_deviance, errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).all():
                cv_range = float(np.ptp(values))
            cv_equal = bool(cv_range is not None and cv_range < 1e-4)
            # User policy: this comparison is informational only. Preserve
            # original values, fit status, and every success/failure denominator.
            if not cv_equal:
                cv_differences.append({
                    "scenario": scenario, "seed": int(seed), "selected_cv_score_range": cv_range,
                    "previous_limit": 1e-4,
                    "cv_by_method": {method: float(value) if np.isfinite(value) else None
                                     for method, value in zip(group.method, values)},
                })
            assert np.ptp(group.objective) < 1e-6, (scenario, int(seed), "Penalized objectives differ")
        checks.append({
            "scenario": scenario, "seed": int(seed), "successful_methods": len(group),
            "merge_checks_passed": True, "paired_checks_passed": cv_equal is not False,
            "selected_cv_score_range": cv_range, "cv_scores_within_previous_limit": cv_equal,
        })
    if require_all_success:
        assert okay.all(), df.loc[~okay, ["scenario", "method", "status", "error", "kkt_scaled"]].to_string(index=False)
    verification = {
        "all_success": bool(okay.all()), "tasks": len(df), "checks": checks,
        "merge_checks_passed": True,
        "paired_checks_passed": all(check["paired_checks_passed"] for check in checks),
        "cv_difference_policy": "record_only", "cv_difference_blocks_merge": False,
        "cv_groups_outside_previous_limit": cv_differences,
    }
    (root / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    print(f"Checked {len(df)} recorded tasks; {int(okay.sum())} successful. "
          f"CV differences are recorded and do not block merging "
          f"({len(cv_differences)} groups outside the previous limit).")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-all-success", action="store_true")
    args = parser.parse_args()
    verify_results(args.out, args.require_all_success)


if __name__ == "__main__":
    main()
