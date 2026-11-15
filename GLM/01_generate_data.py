#!/usr/bin/env python3
"""One generator for both languages; sparse files remain sparse."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread


def is_penalized(regime):
    return regime in {"highdim", "sparse_highdim"}


def matrix(rng, n, p, regime, density):
    if regime.startswith("sparse"):
        # Independent Bernoulli positions, not a fixed number of nonzeros.
        rows, cols, values = [], [], []
        for j in range(p):
            idx = np.flatnonzero(rng.random(n) < density)
            rows.extend(idx); cols.extend([j] * len(idx))
            values.extend(rng.normal(size=len(idx)))
        return sparse.csc_matrix((values, (rows, cols)), shape=(n, p))
    x = np.asfortranarray(rng.normal(size=(n, p)))
    if regime == "dense":
        for j in range(1, p):
            x[:, j] = 0.3 * x[:, j - 1] + np.sqrt(1 - 0.3**2) * x[:, j]
    return x


def save_matrix(x, path):
    if sparse.issparse(x):
        mmwrite(str(path.with_suffix(".mtx")), x, precision=17)
        return str(path.with_suffix(".mtx"))
    path = path.with_suffix(".bin")
    np.asarray(x, dtype="<f8").ravel(order="F").tofile(path)
    return str(path)


def read_matrix(path, n, p):
    if str(path).endswith(".mtx"):
        return sparse.csc_matrix(mmread(path), dtype=np.float64)
    return np.fromfile(path, dtype="<f8").reshape((n, p), order="F")


def generate(scenario, seed, index, config, out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    n, p, regime = scenario["n"], scenario["p"], scenario["regime"]
    # Scenario ordinal, rather than Python's unstable hash(), controls the stream.
    streams = np.random.SeedSequence([seed, index]).spawn(5)
    rx, rb, ry, rt, rf = [np.random.default_rng(s) for s in streams]
    density = scenario.get("density", 1.0)
    x = matrix(rx, n, p, regime, density)
    beta = np.zeros(p); s0 = min(20 if is_penalized(regime) else 10, p)
    beta[:s0] = np.linspace(0.5, 0.15 if is_penalized(regime) else 0.1, s0)
    if is_penalized(regime):
        beta[:s0] *= rb.choice([-1.0, 1.0], size=s0)
    eta = np.asarray(x @ beta).ravel()
    # No clipping: clipping would change the specified data-generating model.
    if not np.all(np.isfinite(eta)) or np.max(eta) > 25:
        raise ValueError("Unsafe Poisson means: revise the design explicitly instead of clipping")
    y = ry.poisson(np.exp(eta)).astype(float)
    if y.mean() <= 0:
        raise ValueError("All-zero response has no finite intercept MLE")
    xt = matrix(rt, config["test_n"], p, regime, density)
    yt = rt.poisson(np.exp(np.asarray(xt @ beta).ravel())).astype(float)
    foldid = np.empty(n, dtype="<i4")
    foldid[rf.permutation(n)] = np.arange(n) % config["cv_folds"]
    lambda_max = float(np.max(np.abs(np.asarray(x.T @ (y - y.mean())).ravel())) / n)
    lambdas = np.geomspace(max(lambda_max, 1e-12),
                          max(lambda_max, 1e-12) * config["lambda_min_ratio"],
                          config["lambda_count"])
    files = {"x":save_matrix(x, out / "X"), "xt":save_matrix(xt, out / "X_test")}
    for name, arr, dtype in [("y", y, "<f8"), ("yt", yt, "<f8"),
                             ("beta", beta, "<f8"), ("foldid", foldid, "<i4"),
                             ("lambdas", lambdas, "<f8")]:
        files[name] = str(out / (name + ".bin")); np.asarray(arr, dtype=dtype).tofile(files[name])
    hashes = {k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in files.items()}
    meta = dict(scenario, seed=int(seed), scenario_index=index, test_n=config["test_n"],
                s0=s0, files=files, sha256=hashes, realized_density=(x.nnz/(n*p) if sparse.issparse(x) else 1.0),
                eta_min=float(eta.min()), eta_max=float(eta.max()),
                zero_true_columns=int(np.sum(np.asarray((x != 0).sum(axis=0)).ravel()[:s0] == 0)))
    path = out / "dataset.json"; path.write_text(json.dumps(meta, indent=2))
    return path, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--out", default="data_glm"); ap.add_argument("--seed", type=int, default=2026091)
    args = ap.parse_args(); conf = json.loads(Path(args.config).read_text())
    manifest = []
    for i, sc in enumerate(conf["scenarios"]):
        _, meta = generate(sc, args.seed, i, conf, Path(args.out)/sc["id"]); manifest.append(meta)
    Path(args.out, "manifest.json").write_text(json.dumps(manifest, indent=2))
