"""Generate one dataset per call and save to data/{regime}_n{n}_p{p}_s{seed}."""
import argparse, csv, os
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.special import expit


def ar1_cholesky(p, rho=0.5):
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(Sigma)


def gen_dense(n, p, seed, out_dir):
    rng = np.random.default_rng(seed)
    L = ar1_cholesky(p)
    X = (rng.standard_normal((n, p)) @ L.T).astype(np.float32)
    k = max(10, p // 5)
    beta = np.zeros(p, dtype=np.float32)
    beta[:k] = rng.choice([-1.0, 1.0], size=k)
    y = (rng.random(n) < expit(X @ beta)).astype(np.int32)
    df = pd.DataFrame(X, columns=[f"num_{j}" for j in range(p)])
    df["y"] = y
    path = os.path.join(out_dir, f"dense_n{n}_p{p}_s{seed}.csv.gz")
    df.to_csv(path, index=False, compression="gzip")
    print(f"  wrote {path}  shape={df.shape}  y_mean={y.mean():.3f}")
    return path


def gen_mixed(n, p, seed, out_dir):
    rng = np.random.default_rng(seed)
    n_cat = 10
    n_num = p - n_cat
    L = ar1_cholesky(n_num)
    X_num = (rng.standard_normal((n, n_num)) @ L.T).astype(np.float32)
    df = pd.DataFrame(X_num, columns=[f"num_{j}" for j in range(n_num)])
    for j in range(n_cat):
        L_j = int(rng.integers(5, 51))
        df[f"cat_{j}"] = rng.integers(0, L_j, size=n).astype(str)
    lin = (0.8 * X_num[:, 0] - 0.6 * X_num[:, 1] + 0.3 * X_num[:, 2])
    for j in range(3):
        cat_codes = df[f"cat_{j}"].astype(int).to_numpy()
        lin += 0.5 * ((cat_codes % 3) == 0).astype(np.float32)
    y = (rng.random(n) < expit(lin)).astype(np.int32)
    df["y"] = y
    path = os.path.join(out_dir, f"mixed_n{n}_p{p}_s{seed}.csv.gz")
    df.to_csv(path, index=False, compression="gzip")
    print(f"  wrote {path}  shape={df.shape}  y_mean={y.mean():.3f}")
    return path


def gen_sparse(n, p, density, seed, out_dir):
    rng = np.random.default_rng(seed)
    X = sp.random(n, p, density=density, format="csr", random_state=int(seed),
                  data_rvs=lambda k: rng.normal(size=k)).astype(np.float32)
    beta_idx = rng.choice(p, size=min(200, max(50, p // 1000)), replace=False)
    beta = rng.normal(size=len(beta_idx))
    logits = np.asarray(X[:, beta_idx].dot(beta)).ravel()
    y = (rng.random(n) < expit(logits)).astype(np.int32)
    stem = os.path.join(out_dir, f"sparse_n{n}_p{p}_s{seed}")
    np.savez_compressed(stem + ".npz",
                        data=X.data, indices=X.indices,
                        indptr=X.indptr, shape=np.array(X.shape))
    np.save(stem + "_y.npy", y)
    print(f"  wrote {stem}.npz  nnz={X.nnz}  y_mean={y.mean():.3f}")
    return stem + ".npz"


def load_task(task_list, task_id):
    with open(task_list, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"task_id {task_id} not found in {task_list}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime",    choices=["dense", "mixed", "sparse"])
    ap.add_argument("--n",         type=int)
    ap.add_argument("--p",         type=int)
    ap.add_argument("--density",   type=float, default=0.01)
    ap.add_argument("--seed",      type=int, default=1)
    ap.add_argument("--out-dir",   default="data")
    ap.add_argument("--task-id",   type=int)
    ap.add_argument("--task-list", default="tasks_data.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.task_id is not None:
        task = load_task(args.task_list, args.task_id)
        regime  = task["regime"]
        n       = int(task["n"])
        p       = int(task["p"])
        density = float(task["density"]) if task.get("density") else 0.01
        seed    = int(task["seed"])
    else:
        regime  = args.regime
        n       = args.n
        p       = args.p
        density = args.density
        seed    = args.seed

    print(f"[gen_data] regime={regime} n={n} p={p} seed={seed}")
    if regime == "dense":
        gen_dense(n, p, seed, args.out_dir)
    elif regime == "mixed":
        gen_mixed(n, p, seed, args.out_dir)
    elif regime == "sparse":
        gen_sparse(n, p, density, seed, args.out_dir)
    else:
        raise ValueError(f"Unknown regime: {regime}")


if __name__ == "__main__":
    main()
