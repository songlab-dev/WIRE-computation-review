import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sksurv.tree import SurvivalTree
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score
from common import evaluate, process_status, partial_loglik, lambda_max

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
x = np.arange(1, 43, dtype=float)[:, None]
t = np.r_[1.0, np.repeat(100.0, 41)]
e = np.r_[True, np.repeat(False, 41)]
fit = SurvivalTree(min_samples_split=30, min_samples_leaf=1, random_state=2026091).fit(
    x, Surv.from_arrays(e, t)
)
sizes = fit.tree_.n_node_samples[fit.tree_.children_left < 0]
assert sorted(sizes.tolist()) == [1, 41]
old = SurvivalTree(min_samples_split=10, min_samples_leaf=15, random_state=2026091).fit(
    x, Surv.from_arrays(e, t)
)
weighted = SurvivalTree(
    min_samples_split=30, min_samples_leaf=1, random_state=2026091
).fit(
    x[:20],
    Surv.from_arrays(e[:20], t[:20]),
    sample_weight=np.r_[np.repeat(2.0, 15), np.zeros(5)],
)
assert weighted.tree_.n_leaves == 1
(a.out / "rsf_native_Python.json").write_text(
    json.dumps(
        {
            "new_child_sizes": sizes.tolist(),
            "old_child_sizes": old.tree_.n_node_samples[
                old.tree_.children_left < 0
            ].tolist(),
            "weighted_30_draws_15_unique_leaves": int(weighted.tree_.n_leaves),
            "conclusion": "Python counts unique positive-weight observations for min_samples_split; it is not identical to R bootstrap draw-count stopping.",
        },
        indent=2,
    )
)
# Exact ties in event/censoring times are deliberately included.
train = pd.DataFrame(
    {
        "time": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
        "event": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    }
)
test = pd.DataFrame(
    {"time": [1.5, 2, 3, 4, 5, 6, 7, 8, 9], "event": [1, 0, 1, 1, 0, 1, 0, 1, 1]}
)
grid = np.linspace(2, 7, 21)
pred = np.exp(-np.arange(1, 10)[:, None] * grid[None, :] / 40)
got = evaluate(train, test, pred, grid)
ref = integrated_brier_score(
    Surv.from_arrays(train.event.astype(bool), train.time),
    Surv.from_arrays(test.event.astype(bool), test.time),
    pred,
    grid,
)
assert abs(got["ibs"] - ref) < 1e-12
# Late observations stay in the denominator and are known event-free through
# the horizon; their unavailable future censoring weights are never requested.
late = test.copy()
late.loc[8, "time"] = 1000
later = evaluate(train, late, pred, grid)
assert later["n_test_evaluated"] == 9
# Verify grouped Breslow risk sets against an independent direct calculation,
# and verify lambda_max using a numerical likelihood gradient.
rng = np.random.default_rng(5432)
xx = rng.normal(size=(len(train), 4))
eta = xx @ np.array([0.2, -0.1, 0.3, 0.1])
direct = sum(
    eta[i] - np.log(np.exp(eta[train.time.to_numpy() >= train.time.iloc[i]]).sum())
    for i in range(len(train))
    if train.event.iloc[i]
)
assert (
    abs(partial_loglik(train.time.to_numpy(), train.event.to_numpy(), eta) - direct)
    < 1e-12
)
eps = 1e-6
gradient = np.array(
    [
        (
            partial_loglik(
                train.time.to_numpy(), train.event.to_numpy(), eps * xx[:, j]
            )
            - partial_loglik(
                train.time.to_numpy(), train.event.to_numpy(), -eps * xx[:, j]
            )
        )
        / (2 * eps * len(train))
        for j in range(4)
    ]
)
assert (
    abs(
        lambda_max(xx, train.time.to_numpy(), train.event.to_numpy())
        - abs(gradient).max()
    )
    < 1e-8
)
# VmHWM retains a touched native allocation after it is freed.
before = process_status()["VmHWM"]
block = np.ones(64 * 1024 * 1024, dtype="uint8")
touched = int(block.sum())
peak = process_status()["VmHWM"]
del block
after = process_status()["VmHWM"]
assert touched > 0 and peak >= before + 50 * 1024 and after >= peak
(a.out / "metric_and_memory_checks.json").write_text(
    json.dumps(
        {
            "ibs_reference": float(ref),
            "ibs_common": got["ibs"],
            "late_test_retained": later["n_test_evaluated"],
            "rss_before_kib": before,
            "rss_peak_kib": peak,
            "rss_after_release_kib": after,
        },
        indent=2,
    )
)
print("Python native checks passed")
