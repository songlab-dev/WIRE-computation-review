"""Train one classifier on one dataset and write a one-row result file."""
import argparse, csv, importlib.metadata, json, os, platform, random, time, resource, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, psutil
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, recall_score, f1_score, confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def peak_mb():
    x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return x / 1024.0 if os.name != "darwin" else x / (1024.0 * 1024.0)


def gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    except Exception:
        pass
    return None


def gen_dense(n, p, seed):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=max(10, p // 5),
        n_redundant=max(5, p // 10), weights=[0.6, 0.4],
        class_sep=1.0, random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int32), ""


def gen_sparse(n, p, density, seed):
    rng = np.random.default_rng(seed)
    X = sp.random(n, p, density=density, format="csr", random_state=seed,
                  data_rvs=lambda k: rng.normal(size=k)).astype(np.float32)
    beta_idx = rng.choice(p, size=min(200, max(50, p // 1000)), replace=False)
    beta = rng.normal(size=len(beta_idx))
    logits = np.asarray(X[:, beta_idx].dot(beta)).ravel()
    pr = expit(logits)
    y = (rng.random(n) < pr).astype(np.int32)
    return X, y, ""


def gen_mixed(n, p, seed):
    rng = np.random.default_rng(seed)
    n_cat = 10
    n_num = p - n_cat
    df = pd.DataFrame({f"num_{j}": rng.normal(size=n).astype(np.float32) for j in range(n_num)})
    cat_cols, levels = [], []
    for j in range(n_cat):
        L = int(rng.integers(5, 51))
        levels.append(L)
        col = f"cat_{j}"
        cat_cols.append(col)
        df[col] = pd.Categorical(rng.integers(0, L, size=n).astype(str))
    lin = (0.8 * df["num_0"].to_numpy()
           - 0.6 * df["num_1"].to_numpy()
           + 0.3 * df["num_2"].to_numpy())
    for j in range(3):
        lin += 0.5 * ((df[f"cat_{j}"].cat.codes.to_numpy() % 3) == 0)
    pr = expit(lin)
    y = (rng.random(n) < pr).astype(np.int32)
    return df, y, ",".join(map(str, levels))


def preprocess_mixed(Xtr, Xte):
    num_cols = [c for c in Xtr.columns if c.startswith("num_")]
    cat_cols = [c for c in Xtr.columns if c.startswith("cat_")]
    ct = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])
    return ct.fit_transform(Xtr), ct.transform(Xte)


def one_hot(Xtr, Xte):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return enc.fit_transform(Xtr), enc.transform(Xte)


def prepare_X(Xtr, Xte, toarray=False):
    if isinstance(Xtr, pd.DataFrame):
        Xtr2, Xte2 = preprocess_mixed(Xtr, Xte)
    else:
        Xtr2, Xte2 = Xtr, Xte
    if toarray:
        if sp.issparse(Xtr2):
            Xtr2, Xte2 = Xtr2.toarray(), Xte2.toarray()
        if sp.issparse(Xte2):
            Xte2 = Xte2.toarray()
    return Xtr2, Xte2


def split_data(X, y, seed):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    if isinstance(X, pd.DataFrame):
        return X.iloc[tr].copy(), X.iloc[te].copy(), y[tr], y[te]
    return X[tr], X[te], y[tr], y[te]


def metrics_from_output(y_true, pred, proba=None, score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy":    float(accuracy_score(y_true, pred)),
        "recall":      float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "f1":          float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss":    np.nan,
        "auroc":       np.nan,
    }
    if proba is not None:
        p1 = proba[:, 1] if (hasattr(proba, "ndim") and proba.ndim == 2) else np.asarray(proba).ravel()
        out["log_loss"] = float(log_loss(y_true, np.clip(p1, 1e-6, 1 - 1e-6)))
        out["auroc"] = float(roc_auc_score(y_true, p1))
    elif score is not None:
        out["auroc"] = float(roc_auc_score(y_true, score))
    return out


CSV_HEADER = ["run_id", "regime", "method", "package", "implementation", "solver",
              "hardware", "n", "p", "sparsity", "categorical_levels", "seed",
              "train_time_s", "predict_time_s", "peak_memory_mb", "gpu_peak_memory_mb",
              "accuracy", "recall", "specificity", "f1", "log_loss", "auroc", "notes"]


def append_row(path, row):
    rows_dir = (path[:-4] if path.endswith(".csv") else path) + "_rows"
    os.makedirs(rows_dir, exist_ok=True)
    rid = str(row["run_id"]).replace("/", "_")
    hw = (str(row.get("hardware") or "cpu").strip().lower()
          .replace("/", "_") or "cpu")
    final = os.path.join(rows_dir, f"{rid}__{hw}.csv")
    tmp = f"{final}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, final)


def write_metadata(run_id, out_dir, use_gpu):
    meta = {
        "run_id":           run_id,
        "slurm_job_id":     os.getenv("SLURM_JOB_ID"),
        "slurm_nodelist":   os.getenv("SLURM_JOB_NODELIST"),
        "slurm_cpus":       os.getenv("SLURM_CPUS_PER_TASK"),
        "platform":         platform.processor(),
        "ram_gb":           round(psutil.virtual_memory().total / 1e9, 1),
        "python":           platform.python_version(),
    }
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                meta["gpu_name"] = torch.cuda.get_device_name(0)
                meta["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            pass
    pkgs = ["scikit-learn", "xgboost", "lightgbm", "catboost", "torch",
            "skglm", "hummingbird-ml", "tensorflow", "tensorflow-probability",
            "cuml-cu12"]
    meta["pkg_versions"] = {}
    for pkg in pkgs:
        try:
            meta["pkg_versions"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"metadata_{run_id}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def check_convergence(model, note):
    if hasattr(model, "n_iter_") and np.any(np.asarray(model.n_iter_) >= model.max_iter):
        note += " WARN:not_converged"
    return note


def load_data(path, regime):
    if regime in ("dense", "mixed"):
        df = pd.read_csv(path)
        y  = df.pop("y").to_numpy(dtype=np.int32)
        if regime == "dense":
            return df.to_numpy(dtype=np.float32), y, ""
        else:
            cat_levels = ",".join(
                str(df[c].nunique()) for c in df.columns if c.startswith("cat_")
            )
            for col in df.columns:
                if col.startswith("cat_"):
                    df[col] = pd.Categorical(df[col].astype(str))
            return df, y, cat_levels
    else:
        data = np.load(path)
        X = sp.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"]),
        )
        y = np.load(path.replace(".npz", "_y.npy"))
        return X, y, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime",    required=True, choices=["dense", "sparse", "mixed"])
    ap.add_argument("--model",     required=True)
    ap.add_argument("--n",         type=int, required=True)
    ap.add_argument("--p",         type=int, required=True)
    ap.add_argument("--density",   type=float, default=0.01)
    ap.add_argument("--seed",      type=int, default=1)
    ap.add_argument("--gpu",       action="store_true")
    ap.add_argument("--out",       required=True)
    ap.add_argument("--data-file", default=None,
                    help="Pre-generated csv.gz/npz dataset; --n/--p/--seed become metadata.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    if args.data_file:
        X, y, cat_levels = load_data(args.data_file, args.regime)
    elif args.regime == "dense":
        X, y, cat_levels = gen_dense(args.n, args.p, args.seed)
    elif args.regime == "sparse":
        X, y, cat_levels = gen_sparse(args.n, args.p, args.density, args.seed)
    else:
        X, y, cat_levels = gen_mixed(args.n, args.p, args.seed)

    Xtr, Xte, ytr, yte = split_data(X, y, args.seed)

    note = ""
    method = package = implementation = solver = ""
    train_t = pred_t = 0.0

    if args.model == "sklearn_logit_lbfgs":
        from sklearn.linear_model import LogisticRegression
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=args.seed)
        package, method, implementation, solver = "scikit-learn", "logistic", "cpu", "lbfgs"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        note = check_convergence(model, note)
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "sklearn_logit_newton":
        from sklearn.linear_model import LogisticRegression
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        Xtr2 = Xtr2.astype(np.float64); Xte2 = Xte2.astype(np.float64)
        model = LogisticRegression(solver="newton-cholesky", max_iter=200, random_state=args.seed)
        package, method, implementation, solver = "scikit-learn", "logistic", "cpu", "newton-cholesky"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        note = check_convergence(model, note)
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "sklearn_logit_saga":
        from sklearn.linear_model import LogisticRegression
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        model = LogisticRegression(solver="saga", penalty="l2", max_iter=500, random_state=args.seed)
        package, method, implementation, solver = "scikit-learn", "logistic", "cpu", "saga"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        note = check_convergence(model, note)
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "sklearn_linearsvc":
        from sklearn.svm import LinearSVC
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        model = LinearSVC(dual="auto", random_state=args.seed)
        package, method, implementation, solver = "scikit-learn", "linear_svm", "cpu", "liblinear/ovr"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = model.predict(Xte2); score = model.decision_function(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, score=score)

    elif args.model == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        model = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=args.seed)
        package, method, implementation, solver = "scikit-learn", "random_forest", "cpu", "bagged_trees"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "skglm_logit":
        from skglm import SparseLogisticRegression
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        model = SparseLogisticRegression(alpha=1e-4, l1_ratio=0.5, max_epochs=200)
        package, method, implementation, solver = "skglm", "logistic", "cpu_sparse", "prox/sparse"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "xgb":
        from xgboost import XGBClassifier
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", device="cuda" if args.gpu else "cpu",
            n_jobs=1, random_state=args.seed,
        )
        package, method, implementation, solver = "xgboost", "gbdt", "gpu" if args.gpu else "cpu", "hist"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "lgbm":
        from lightgbm import LGBMClassifier
        Xtr2, Xte2 = prepare_X(Xtr, Xte)
        kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=args.seed, n_jobs=1, verbose=-1)
        if args.gpu:
            kw["device_type"] = "cuda"
        model = LGBMClassifier(**kw)
        package, method, implementation, solver = "lightgbm", "gbdt", "gpu" if args.gpu else "cpu", "histogram"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = model.predict(Xte2); proba = model.predict_proba(Xte2); pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "catboost":
        from catboost import CatBoostClassifier
        if isinstance(Xtr, pd.DataFrame):
            cat_features = [c for c in Xtr.columns if c.startswith("cat_")]
            Xtr2, Xte2 = Xtr, Xte
        else:
            Xtr2, Xte2 = Xtr, Xte
            cat_features = None
        model = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=8, loss_function="Logloss",
            eval_metric="AUC", verbose=False, random_seed=args.seed,
            task_type="GPU" if args.gpu else "CPU",
        )
        package, method, implementation, solver = "catboost", "gbdt", "gpu" if args.gpu else "cpu", "ordered_boosting"
        t1 = time.perf_counter(); model.fit(Xtr2, ytr, cat_features=cat_features); train_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        pred = model.predict(Xte2).astype(int).ravel()
        proba = model.predict_proba(Xte2)
        pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "torch_mlp":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = nn.Sequential(nn.Linear(Xtr2.shape[1], 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.tensor(Xtr2), torch.tensor(ytr).float().view(-1, 1))
        dl = DataLoader(ds, batch_size=512, shuffle=True)
        package, method, implementation, solver = "torch", "mlp", device, "adam"
        t1 = time.perf_counter()
        model.train()
        for _ in range(10):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        train_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            pr = torch.sigmoid(model(torch.tensor(Xte2).to(device))).cpu().numpy().ravel()
        pred = (pr >= 0.5).astype(int)
        pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=pr)

    elif args.model == "tfp_glm":
        import tensorflow as tf
        import tensorflow_probability as tfp
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        Xtr2 = np.c_[np.ones(len(Xtr2), dtype=np.float32), Xtr2.astype(np.float32)]
        Xte2 = np.c_[np.ones(len(Xte2), dtype=np.float32), Xte2.astype(np.float32)]
        package, method, implementation, solver = "tensorflow_probability", "logistic", "cpu/gpu", "fisher_scoring"
        l2 = float(args.n) / 1000.0
        t1 = time.perf_counter()
        coeffs, _, _, _ = tfp.glm.fit(
            model_matrix=tf.constant(Xtr2, tf.float32),
            response=tf.constant(ytr, tf.float32),
            model=tfp.glm.Bernoulli(),
            l2_regularizer=l2,
            maximum_iterations=100,
        )
        train_t = time.perf_counter() - t1
        coeffs_np = coeffs.numpy()
        if np.any(np.isnan(coeffs_np)):
            note += " WARN:NaN_coeffs"
            coeffs_np = np.nan_to_num(coeffs_np, nan=0.0)
        t2 = time.perf_counter()
        logits = tf.linalg.matvec(tf.constant(Xte2, tf.float32), tf.constant(coeffs_np, tf.float32)).numpy()
        pr = expit(logits)
        pred = (pr >= 0.5).astype(int)
        pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=pr)

    elif args.model == "tfdf_rf":
        import tensorflow_decision_forests as tfdf
        if not isinstance(Xtr, pd.DataFrame):
            row = _make_row(args, "tfdf_rf", "random_forest", "tfdf", "cpu", "forest",
                            cat_levels, 0, 0, np.nan, np.nan, np.nan,
                            "skipped: requires mixed regime")
            append_row(args.out, row)
            print(row)
            return
        trdf, tedf = Xtr.copy(), Xte.copy()
        trdf["label"] = ytr
        tedf["label"] = yte
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(trdf, label="label")
        test_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(tedf, label="label")
        model = tfdf.keras.RandomForestModel(num_trees=300, random_seed=args.seed)
        package, method, implementation, solver = "tfdf", "random_forest", "cpu", "forest"
        t1 = time.perf_counter(); model.fit(train_ds, verbose=0); train_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        pr = model.predict(test_ds, verbose=0).ravel()
        pred = (pr >= 0.5).astype(int)
        pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=pr)

    elif args.model == "cuml_logit":
        import cupy as cp
        import pandas as _pd
        import pandas.api.types as _pat
        if not hasattr(_pat, "is_interval"):
            _pat.is_interval = lambda obj: isinstance(obj, _pd.Interval)
        from cuml.linear_model import LogisticRegression as CuLogit
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        model = CuLogit(max_iter=500)
        package, method, implementation, solver = "cuml", "logistic", "gpu", "quasi_newton"
        t1 = time.perf_counter()
        model.fit(cp.asarray(Xtr2), cp.asarray(ytr))
        train_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        pred  = cp.asnumpy(model.predict(cp.asarray(Xte2)))
        proba = cp.asnumpy(model.predict_proba(cp.asarray(Xte2)))[:, 1]
        pred_t = time.perf_counter() - t2
        mets = metrics_from_output(yte, pred, proba=proba)

    elif args.model == "hummingbird_rf":
        from sklearn.ensemble import RandomForestClassifier
        from hummingbird.ml import convert
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        Xtr2 = Xtr2.astype(np.float32)
        Xte2 = Xte2.astype(np.float32)
        base = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=args.seed).fit(Xtr2, ytr)
        package, method, implementation, solver = "hummingbird", "random_forest_inference", "torch_backend", "convert"
        t1 = time.perf_counter(); hb = convert(base, "torch", Xtr2[:256]); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = hb.predict(Xte2); pred_t = time.perf_counter() - t2
        note = "train_time is conversion only; compare predict_time with sklearn_rf"
        mets = metrics_from_output(yte, pred)

    elif args.model == "hummingbird_xgb":
        from xgboost import XGBClassifier
        from hummingbird.ml import convert
        Xtr2, Xte2 = prepare_X(Xtr, Xte, toarray=True)
        Xtr2 = Xtr2.astype(np.float32)
        Xte2 = Xte2.astype(np.float32)
        base = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=1, random_state=args.seed,
        )
        base.fit(Xtr2, ytr)
        package, method, implementation, solver = "hummingbird", "xgb_inference", "torch_backend", "convert"
        t1 = time.perf_counter(); hb = convert(base, "torch", Xtr2[:256]); train_t = time.perf_counter() - t1
        t2 = time.perf_counter(); pred = hb.predict(Xte2); pred_t = time.perf_counter() - t2
        note = "train_time is conversion only; compare predict_time with xgb"
        mets = metrics_from_output(yte, pred)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    run_id = f"{args.model}_{args.regime}_n{args.n}_p{args.p}_s{args.seed}"
    write_metadata(run_id, os.path.dirname(args.out) or ".", args.gpu)

    row = {
        "run_id":               run_id,
        "regime":               args.regime,
        "method":               method,
        "package":              package,
        "implementation":       implementation,
        "solver":               solver,
        "hardware":             "gpu" if args.gpu else "cpu",
        "n":                    args.n,
        "p":                    args.p,
        "sparsity":             args.density if args.regime == "sparse" else "",
        "categorical_levels":   cat_levels if args.regime == "mixed" else "",
        "seed":                 args.seed,
        "train_time_s":         round(train_t, 6),
        "predict_time_s":       round(pred_t, 6),
        "peak_memory_mb":       round(peak_mb(), 2),
        "gpu_peak_memory_mb":   gpu_peak_mb() or "",
        "accuracy":             round(float(mets["accuracy"]), 6),
        "recall":               round(float(mets["recall"]), 6),
        "specificity":          "" if np.isnan(mets["specificity"]) else round(float(mets["specificity"]), 6),
        "f1":                   round(float(mets["f1"]), 6),
        "log_loss":             "" if np.isnan(mets["log_loss"]) else round(float(mets["log_loss"]), 6),
        "auroc":                "" if np.isnan(mets["auroc"])    else round(float(mets["auroc"]), 6),
        "notes":                note,
    }
    append_row(args.out, row)
    print(row)


def _make_row(args, model, method, package, impl, solver,
              cat_levels, train_t, pred_t, acc, ll, auc, note):
    return {
        "run_id":             f"{model}_{args.regime}_n{args.n}_p{args.p}_s{args.seed}",
        "regime":             args.regime, "method": method, "package": package,
        "implementation":     impl, "solver": solver,
        "hardware":           "gpu" if args.gpu else "cpu",
        "n": args.n, "p": args.p,
        "sparsity":           args.density if args.regime == "sparse" else "",
        "categorical_levels": cat_levels if args.regime == "mixed" else "",
        "seed":               args.seed,
        "train_time_s":       train_t, "predict_time_s": pred_t,
        "peak_memory_mb":     round(peak_mb(), 2), "gpu_peak_memory_mb": "",
        "accuracy":           acc, "recall": "", "specificity": "", "f1": "",
        "log_loss": ll, "auroc": auc,
        "notes":              note,
    }


if __name__ == "__main__":
    main()
