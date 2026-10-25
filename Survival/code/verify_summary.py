"""Independently recompute every published mean and sample SD from per-fit CSV."""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
root = parser.parse_args().root.resolve()
rows = list(csv.DictReader((root / "summary/all_runs.csv").open()))
summary = list(csv.DictReader((root / "summary/summary.csv").open()))
assert len(rows) == 4800 and len({r["task_id"] for r in rows}) == 4800
assert all(r["status"] == "success" for r in rows)
groups = collections.defaultdict(list)
for r in rows:
    groups[tuple(r[k] for k in ("setting", "n", "model", "language"))].append(r)
assert len(groups) == len(summary) == 48
verified = 0
for cell in summary:
    key = tuple(cell[k] for k in ("setting", "n", "model", "language"))
    observations = groups[key]
    assert len(observations) == 100
    assert {int(r["replication"]) for r in observations} == set(range(1, 101))
    for metric in ("ibs", "cindex", "fitting_time_s", "peak_rss_mib"):
        values = [float(r[metric]) for r in observations]
        assert all(math.isfinite(x) for x in values)
        for label, expected in (
            ("mean", statistics.mean(values)),
            ("sd", statistics.stdev(values)),
        ):
            actual = float(cell[metric + "_" + label])
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
                key,
                metric,
                label,
            )
            verified += 1
report = {
    "valid_fits": len(rows),
    "cells": len(groups),
    "replications_per_cell": 100,
    "independently_verified_mean_and_sample_sd_values": verified,
}
(root / "audit/summary_integrity.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
