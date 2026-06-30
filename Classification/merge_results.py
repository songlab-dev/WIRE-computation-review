"""Merge per-task one-row result files into a single results CSV.

Each array task writes <out>_rows/<run_id>__<hw>.csv (atomic, no locks); this
concatenates them, de-duplicates by (run_id, hardware) keeping the newest, and
writes the merged CSV.

Usage:
  python merge_results.py results/run_v2/python_results.csv
  python merge_results.py results/run_v2/python_results.csv results/run_v2/r_results.csv
"""
import csv, glob, os, sys

HEADER = ["run_id", "regime", "method", "package", "implementation", "solver",
          "hardware", "n", "p", "sparsity", "categorical_levels", "seed",
          "train_time_s", "predict_time_s", "peak_memory_mb", "gpu_peak_memory_mb",
          "accuracy", "recall", "specificity", "f1", "log_loss", "auroc", "notes"]


def merge(out_path):
    rows_dir = (out_path[:-4] if out_path.endswith(".csv") else out_path) + "_rows"
    if not os.path.isdir(rows_dir):
        print(f"  no rows dir: {rows_dir} — skipping")
        return
    files = sorted(glob.glob(os.path.join(rows_dir, "*.csv")),
                   key=os.path.getmtime)
    by_key = {}
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["run_id"], (r.get("hardware") or "").strip().lower())
                by_key[key] = r
    merged = sorted(by_key.values(),
                    key=lambda r: (r.get("regime", ""), r.get("method", ""),
                                   r.get("run_id", ""), r.get("hardware", "")))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow(r)
    print(f"  merged {len(files)} row files -> {len(merged)} unique rows "
          f"-> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python merge_results.py <out.csv> [<out2.csv> ...]")
    for p in sys.argv[1:]:
        merge(p)
