"""Merge python_results.csv + r_results.csv, compute summaries and the
R-vs-Python comparison tables.

Usage:
    python analyze_results.py --py results/run_v2/python_results.csv \
        --r results/run_v2/r_results.csv --save --crosslang
"""
import argparse, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


NUMERIC = ["train_time_s", "predict_time_s", "peak_memory_mb",
           "accuracy", "recall", "specificity", "f1", "log_loss", "auroc"]

GROUP_COLS = ["regime", "method", "package", "implementation", "solver",
              "hardware", "n", "p"]


def load(py_path, r_path):
    frames = []
    for path, lang in [(py_path, "python"), (r_path, "r")]:
        if path and os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["language"] = lang
            frames.append(df)
        else:
            print(f"  [skip] {path} not found")
    if not frames:
        raise FileNotFoundError("No result files found.")
    df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summary(df):
    present_group = [c for c in GROUP_COLS if c in df.columns]
    present_num   = [c for c in NUMERIC if c in df.columns]
    agg = (df.groupby(present_group, dropna=False)[present_num]
             .agg(["mean", "std", "count"])
             .round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def pivot_table(df, metric="auroc", time_col="train_time_s"):
    df2 = df.copy()
    df2["pkg_solver"] = df2["package"] + "/" + df2["solver"].fillna("")
    grp = df2.groupby(["regime", "n", "p", "pkg_solver"], dropna=False)
    tbl = grp[[metric, time_col]].mean().unstack("pkg_solver").round(4)
    return tbl


def convergence_warnings(df):
    mask = df["notes"].astype(str).str.contains("WARN:not_converged", na=False)
    if mask.any():
        return df.loc[mask, ["run_id", "package", "solver", "n", "p", "seed"]]
    return pd.DataFrame(columns=["run_id"])


def _py(d):
    return d["language"] == "python"

def _r(d):
    return d["language"] == "r"

MATCHED_FAMILIES = [
    dict(label="Logistic (unpenalized)",
         py=lambda d: _py(d) & (d["method"] == "logistic")
                      & (d["package"] == "scikit-learn")
                      & d["solver"].isin(["lbfgs", "newton-cholesky"]),
         r =lambda d: _r(d) & d["package"].isin(["base_R", "speedglm"])),
    dict(label="Ridge logistic",
         py=lambda d: _py(d) & (d["method"] == "logistic")
                      & d["solver"].fillna("").str.contains("saga"),
         r =lambda d: _r(d) & (d["package"] == "glmnet")
                      & d["run_id"].str.contains("ridge")),
    dict(label="Lasso logistic",
         py=lambda d: pd.Series(False, index=d.index),
         r =lambda d: _r(d) & (d["package"] == "glmnet")
                      & d["run_id"].str.contains("lasso")),
    dict(label="Linear SVM",
         py=lambda d: _py(d) & (d["method"] == "linear_svm"),
         r =lambda d: _r(d) & (d["package"] == "LiblineaR")),
    dict(label="Random forest",
         py=lambda d: _py(d) & (d["method"] == "random_forest")
                      & (d["package"] == "scikit-learn"),
         r =lambda d: _r(d) & d["package"].isin(["randomForest", "ranger"])),
    dict(label="XGBoost",
         py=lambda d: _py(d) & (d["package"] == "xgboost"),
         r =lambda d: _r(d) & (d["package"] == "xgboost")),
    dict(label="LightGBM",
         py=lambda d: _py(d) & (d["package"] == "lightgbm"),
         r =lambda d: _r(d) & (d["package"] == "lightgbm")),
    dict(label="CatBoost",
         py=lambda d: _py(d) & (d["package"] == "catboost"),
         r =lambda d: _r(d) & (d["package"] == "catboost")),
]

GPU_PACKAGES = ["xgboost", "catboost", "torch", "cuml"]


def _mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def crosslang_tables(df, out_dir):
    dense = df[df["regime"] == "dense"].copy()
    dense_pycpu = dense[~((dense["language"] == "python")
                          & (dense["hardware"] == "gpu"))]
    written = []

    def emit(name, rows, cols):
        path = os.path.join(out_dir, name)
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        written.append(path)

    # Scaling vs n (p=100) and vs p (n=50000), dense regime.
    for axis, fixed_col, fixed_val, vary in [
            ("n", "p", 100, sorted(dense["n"].dropna().unique())),
            ("p", "n", 50000, sorted(dense["p"].dropna().unique()))]:
        rows = []
        sub = dense_pycpu[dense_pycpu[fixed_col] == fixed_val]
        for fam in MATCHED_FAMILIES:
            for v in vary:
                at = sub[sub[axis] == v]
                pat, rat = fam["py"](at), fam["r"](at)
                pt = _mean(at.loc[pat, "train_time_s"])
                rt = _mean(at.loc[rat, "train_time_s"])
                rows.append(dict(
                    family=fam["label"], **{axis: int(v)},
                    py_train_s=round(pt, 4), r_train_s=round(rt, 4),
                    r_over_py=(round(rt / pt, 2)
                               if pt and not np.isnan(pt) and not np.isnan(rt)
                               else np.nan),
                    py_auroc=round(_mean(at.loc[pat, "auroc"]), 4),
                    r_auroc=round(_mean(at.loc[rat, "auroc"]), 4)))
        emit(f"xlang_scaling_{axis}.csv", rows,
             ["family", axis, "py_train_s", "r_train_s", "r_over_py",
              "py_auroc", "r_auroc"])

    # Scaling vs n (p=100), mixed regime.
    mixed = df[df["regime"] == "mixed"].copy()
    mixed_pycpu = mixed[~((mixed["language"] == "python")
                          & (mixed["hardware"] == "gpu"))]
    sub = mixed_pycpu[mixed_pycpu["p"] == 100]
    rows = []
    for fam in MATCHED_FAMILIES:
        for v in sorted(mixed["n"].dropna().unique()):
            at = sub[sub["n"] == v]
            pat, rat = fam["py"](at), fam["r"](at)
            pt = _mean(at.loc[pat, "train_time_s"])
            rt = _mean(at.loc[rat, "train_time_s"])
            rows.append(dict(
                family=fam["label"], n=int(v),
                py_train_s=round(pt, 4), r_train_s=round(rt, 4),
                r_over_py=(round(rt / pt, 2)
                           if pt and not np.isnan(pt) and not np.isnan(rt)
                           else np.nan),
                py_auroc=round(_mean(at.loc[pat, "auroc"]), 4),
                r_auroc=round(_mean(at.loc[rat, "auroc"]), 4)))
    emit("xlang_scaling_n_mixed.csv", rows,
         ["family", "n", "py_train_s", "r_train_s", "r_over_py",
          "py_auroc", "r_auroc"])

    # Inference, memory, calibration — dense grand means per family.
    inf, mem, cal = [], [], []
    for fam in MATCHED_FAMILIES:
        P, Rr = dense_pycpu[fam["py"](dense_pycpu)], dense_pycpu[fam["r"](dense_pycpu)]
        inf.append(dict(family=fam["label"],
                        py_predict_s=round(_mean(P["predict_time_s"]), 4),
                        r_predict_s=round(_mean(Rr["predict_time_s"]), 4)))
        r_mem = _mean(Rr["peak_memory_mb"])
        mem.append(dict(family=fam["label"],
                        py_peak_mb=round(_mean(P["peak_memory_mb"]), 1),
                        r_peak_mb=round(r_mem, 1),
                        r_mem_reliable=bool(r_mem >= 50)
                        if not np.isnan(r_mem) else False))
        cal.append(dict(family=fam["label"],
                        py_log_loss=round(_mean(P["log_loss"]), 4),
                        r_log_loss=round(_mean(Rr["log_loss"]), 4)))
    emit("xlang_inference.csv", inf,
         ["family", "py_predict_s", "r_predict_s"])
    emit("xlang_memory.csv", mem,
         ["family", "py_peak_mb", "r_peak_mb", "r_mem_reliable"])
    emit("xlang_calibration.csv", cal,
         ["family", "py_log_loss", "r_log_loss"])

    # GPU vs CPU (Python only), dense grand means per package.
    gpu = []
    pyd = dense[dense["language"] == "python"]
    for pkg in GPU_PACKAGES:
        sub = pyd[pyd["package"] == pkg]
        cpu, gp = sub[sub["hardware"] == "cpu"], sub[sub["hardware"] == "gpu"]
        ct, gt = _mean(cpu["train_time_s"]), _mean(gp["train_time_s"])
        ca, ga = _mean(cpu["auroc"]), _mean(gp["auroc"])
        gpu.append(dict(
            package=pkg,
            cpu_train_s=round(ct, 4), gpu_train_s=round(gt, 4),
            speedup=(round(ct / gt, 2)
                     if gt and not np.isnan(gt) and not np.isnan(ct)
                     else np.nan),
            cpu_auroc=round(ca, 4), gpu_auroc=round(ga, 4),
            auroc_delta=(round(ga - ca, 4)
                         if not np.isnan(ca) and not np.isnan(ga)
                         else np.nan)))
    emit("gpu_vs_cpu.csv", gpu,
         ["package", "cpu_train_s", "gpu_train_s", "speedup",
          "cpu_auroc", "gpu_auroc", "auroc_delta"])

    print("\nCross-language tables written:")
    for p in written:
        print(f"  {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py",   default="results/python_results.csv")
    ap.add_argument("--r",    default="results/r_results.csv")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--crosslang", action="store_true")
    args = ap.parse_args()

    print("Loading results...")
    df = load(args.py, args.r)
    print(f"  {len(df)} rows loaded, {df['seed'].nunique()} unique seeds, "
          f"{df['run_id'].nunique()} unique run IDs\n")

    warns = convergence_warnings(df)
    if len(warns):
        print(f"[!] Convergence warnings ({len(warns)} runs):")
        print(warns.to_string(index=False))
        print()

    smry = summary(df)
    print("=== Summary (mean ± SD across seeds) ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 30,
                            "display.width", 160):
        print(smry.to_string(index=False))

    for regime in df["regime"].dropna().unique():
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        print(f"\n=== AUROC pivot — {regime} regime ===")
        try:
            tbl = pivot_table(sub, metric="auroc", time_col="train_time_s")
            print(tbl.to_string())
        except Exception as e:
            print(f"  (pivot failed: {e})")

    hb = df[df["package"] == "hummingbird"].copy()
    if not hb.empty:
        print("\n=== Hummingbird inference speedup (predict_time_s) ===")
        baselines = df[df["package"].isin(["scikit-learn", "xgboost"])]
        for _, row in hb.iterrows():
            base_model = "random_forest" if "rf" in row["method"] else "gbdt"
            base_pkg   = "scikit-learn"  if "rf" in row["method"] else "xgboost"
            baseline   = baselines[
                (baselines["method"] == base_model) &
                (baselines["package"] == base_pkg) &
                (baselines["n"] == row["n"]) &
                (baselines["p"] == row["p"]) &
                (baselines["regime"] == row["regime"])
            ]["predict_time_s"].mean()
            if not np.isnan(baseline) and baseline > 0:
                speedup = baseline / row["predict_time_s"]
                print(f"  {row['run_id']}: {speedup:.2f}x faster than {base_pkg}/{base_model}")

    if args.save:
        out_dir = os.path.dirname(args.py) or "results"
        os.makedirs(out_dir, exist_ok=True)
        s_path = os.path.join(out_dir, "summary_by_group.csv")
        m_path = os.path.join(out_dir, "all_results_merged.csv")
        smry.to_csv(s_path, index=False)
        df.to_csv(m_path, index=False)
        print(f"\nSaved: {s_path}, {m_path}")

    if args.crosslang:
        out_dir = os.path.dirname(args.py) or "results"
        os.makedirs(out_dir, exist_ok=True)
        crosslang_tables(df, out_dir)


if __name__ == "__main__":
    main()
