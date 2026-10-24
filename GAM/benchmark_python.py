# Scalability benchmark — Python methods.
#
# For each (scenario, n) cell, runs REPS replicates of fit + predict for all
# Python methods and records elapsed time, itr/sec, total time, peak memory
# (max RSS), and MISE. Results are checkpointed after every rep so a wall-clock
# kill never loses completed work.
#
# Outputs (results/benchmark_chunks/):
#   python_{scenario}_n{n}.csv   per-cell chunk files
# Outputs (results/):
#   python_benchmark.csv          combined file (all chunks)
#
# CLI usage (optional — selects one cell):
#   python public/benchmark_python.py <scenario> <n>
#   e.g.  python public/benchmark_python.py additive_p10 100000
#
# Without arguments, runs all (scenario, n) cells in the default grid.
#
# SLURM job-budget integration: set BENCH_JOB_BUDGET_S to the job's wall-clock
# limit in seconds. The script stops launching new fits within JOB_MARGIN_S of
# the budget and checkpoints cleanly rather than being SIGKILLed mid-run.
#
# Each fit runs in its own worker process under `timeout` and `/usr/bin/time`,
# as in benchmark_r.R, so max_rss_mb is measured the same way in both languages.
# The "baseline" method loads the data without fitting (interpreter + data
# floor). Timings follow bench::mark: one untimed fit, then BENCH_ITERS
# (default 1) timed fits. Set BENCH_TIMEOUT_BIN / BENCH_TIME_BIN to override
# the tool paths.
#
# Run from the project root (or any directory — paths are resolved relative to
# this script's location):
#   python public/benchmark_python.py

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, LogisticGAM, s
from statsmodels.gam.api import BSplines
import statsmodels.api as sm
from scipy.special import logit
from gcv_grid import fit_statsmodels_gcv, fit_gamdist_gcv, predict_gamdist, pygam_lam_grid

ROOT    = Path(__file__).parent.parent
# GAM_RESULTS / GAM_MAX_REPS: pilot overrides (separate output dir, fewer reps).
RESULTS  = str(ROOT / os.environ.get("GAM_RESULTS", "results"))
MAX_REPS = int(os.environ.get("GAM_MAX_REPS", "0"))   # 0 = all
os.makedirs(RESULTS, exist_ok=True)

REPS    = MAX_REPS or 100
N_TEST  = 2000          # held-out points for the MISE estimate
EPS     = 1e-6

BENCH_TIMEOUT_S  = 1800   # per-fit wall-clock cap (s)
MAX_ALLFAIL_REPS = 1000   # effectively disabled (REPS=100)
# BENCH_REP_START / BENCH_REP_END: run only this rep range (one shard of a cell).
REP_START = int(os.environ.get("BENCH_REP_START", "1"))
REP_END   = int(os.environ.get("BENCH_REP_END", str(REPS)))
SHARDED   = (REP_START, REP_END) != (1, REPS)
# BENCH_METHODS: comma list; run only these Python methods (default all). With it
# set the shared "baseline" worker is skipped. BENCH_TAG: suffix for the chunk
# file name, so a method-subset pass writes its own files (resume is per file).
BENCH_METHODS = [m for m in os.environ.get("BENCH_METHODS", "").split(",") if m]
BENCH_TAG     = os.environ.get("BENCH_TAG", "")
# Rows from the common-edf-grid code carry grid_ver = 2. pyGAM and statsmodels rows
# without it used earlier, non-comparable grids and are ignored by the table scripts.
GRID_VER      = 2
# After this many timeouts in a cell (per job), skip that method for the
# remaining reps; skipped reps are recorded as not completed (NaN).
MAX_METHOD_TIMEOUTS = 5
BENCH_ITERS      = int(os.environ.get("BENCH_ITERS", "1"))
STARTUP_SLACK_S  = 300    # worker start-up allowance (s)
BASELINE         = "baseline"

# Global job wall-clock budget (set by SLURM scripts via BENCH_JOB_BUDGET_S).
# Defaults to 0 (disabled). When set, the script stops before being SIGKILLed.
JOB_START    = time.perf_counter()
JOB_BUDGET_S = int(float(os.environ.get("BENCH_JOB_BUDGET_S", "0")))
JOB_MARGIN_S = int(float(os.environ.get("BENCH_JOB_MARGIN_S",
                                        str(BENCH_TIMEOUT_S + 300))))


def _past_deadline():
    return (JOB_BUDGET_S > 0
            and time.perf_counter() - JOB_START > JOB_BUDGET_S - JOB_MARGIN_S)


NA_RESULT = dict(elapsed_s=float("nan"), itr_per_sec=float("nan"),
                 total_time_s=float("nan"), max_rss_mb=float("nan"),
                 mise=float("nan"))


def bench_mark(fn, *args, iterations=BENCH_ITERS):
    """Mirror of bench::mark(fn(), iterations=, check=FALSE, filter_gc=FALSE).
    Returns (timings, model from the untimed first fit)."""
    model = fn(*args)
    times = []
    for _ in range(iterations):
        t = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t)
    total = float(np.sum(times))
    return dict(elapsed_s=float(np.median(times)),
                itr_per_sec=len(times) / total,
                total_time_s=total), model




def _find_tools():
    """Return (timeout_bin, time_bin, is_gnu_time)."""
    to = (os.environ.get("BENCH_TIMEOUT_BIN")
          or shutil.which("timeout") or shutil.which("gtimeout"))
    tb = os.environ.get("BENCH_TIME_BIN", "/usr/bin/time")
    if not to or not os.path.exists(tb):
        sys.exit("benchmark needs coreutils `timeout` and /usr/bin/time "
                 "(set BENCH_TIMEOUT_BIN / BENCH_TIME_BIN)")
    gnu = subprocess.run([tb, "-v", "-o", os.devnull, "true"],
                         capture_output=True).returncode == 0
    return to, tb, gnu


def _parse_max_rss_mb(path, gnu):
    """Max RSS (MB) from /usr/bin/time output: GNU reports kB, BSD bytes."""
    try:
        txt = Path(path).read_text()
    except OSError:
        return float("nan")
    if gnu:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)
        return int(m.group(1)) / 1024 if m else float("nan")
    m = re.search(r"(\d+)\s+maximum resident set size", txt)
    return int(m.group(1)) / 1024**2 if m else float("nan")


def run_isolated(scenario, method, data_path, tmpdir):
    """Run one method in a worker process; all-NaN result on failure."""
    out_path = os.path.join(tmpdir, f"{method}.json")
    rss_path = os.path.join(tmpdir, f"{method}.time")
    cap = BENCH_TIMEOUT_S * (1 + BENCH_ITERS) + STARTUP_SLACK_S
    cmd = [TIMEOUT_BIN, "-k", "30", str(cap),
           TIME_BIN, "-v" if GNU_TIME else "-l", "-o", rss_path,
           sys.executable, os.path.abspath(__file__), "--worker",
           scenario, method, data_path, out_path]
    rc = subprocess.run(cmd).returncode
    if rc in (124, 137):
        print(f"    [{method} timeout > {cap}s]")
        return dict(NA_RESULT, timed_out=True)
    if rc != 0 or not os.path.exists(out_path):
        print(f"    [{method} worker failed, exit status {rc}]")
        return dict(NA_RESULT)
    with open(out_path) as f:
        res = json.load(f)
    fit_ok = res.pop("fit_ok")
    res["max_rss_mb"] = _parse_max_rss_mb(rss_path, GNU_TIME) if fit_ok else float("nan")
    return res


def worker_main(scenario, method, data_path, out_path):
    """Fit + predict one method on one replicate; write results as JSON."""
    methods = SCENARIOS[scenario]
    d = np.load(data_path)
    X, y, Xt, f0 = d["X"], d["y"], d["Xt"], d["f0"]
    res = {k: v for k, v in NA_RESULT.items() if k != "max_rss_mb"}
    res["fit_ok"] = method == BASELINE
    # Cell given up (marker written by monitor/giveup.py after >= 10 failed fits in
    # this cell): record a failed fit without running it. Workers are fresh processes,
    # so this also applies to jobs that were started before the marker existed.
    flag = Path(RESULTS) / "benchmark_giveup" / f"{scenario}_n{len(y)}_{method}.flag"
    if method != BASELINE and flag.exists():
        print(f"    [{method}: cell given up, skipping]")
        with open(out_path, "w") as f:
            json.dump(res, f)
        return
    if method != BASELINE:
        fit_fn, pred_fn = {nm: (f, p) for nm, f, p in methods}[method]
        try:
            timing, model = bench_mark(fit_fn, X, y)
            res.update(timing, fit_ok=True)
        except Exception as e:
            print(f"    [{method} error: {e}]")
            model = None
        if model is not None:
            try:
                pred = pred_fn(model, X, Xt)
                res["mise"] = float(np.mean((np.asarray(pred) - f0) ** 2))
            except Exception as e:
                print(f"    [{method} predict failed: {e}]")
    with open(out_path, "w") as f:
        json.dump(res, f)


def _pygam_terms(p):
    terms = s(0)
    for i in range(1, p):
        terms = terms + s(i)
    return terms


def _bs_smoother(X, p):
    return BSplines(X, df=[20]*p, degree=[3]*p)   # 20 per term, like pyGAM, so the edf targets are reachable


# ── Shared datasets ───────────────────────────────────────────────────────────
# Drawn once by `Rscript benchmark_r.R --gen <scenario> <n>` and read here, so
# every method in both languages fits the same data. One raw float64 file per
# rep, column-major: X (n x p), y (n), Xt (N_TEST x p), f0 (N_TEST).
DATA_DIR = os.environ.get("BENCH_DATA_DIR", f"{RESULTS}/benchmark_data")


def read_rep(scenario, n, rep):
    path = f"{DATA_DIR}/{scenario}_n{n}_rep{rep:03d}.bin"
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run `Rscript benchmark_r.R --gen {scenario} {n}` first")
    p = int(scenario.split("_p")[-1])
    v = np.fromfile(path, dtype="<f8")
    o = n * p
    X  = np.ascontiguousarray(v[:o].reshape(p, n).T)
    y  = v[o:o + n]
    Xt = np.ascontiguousarray(v[o + n:o + n + N_TEST * p].reshape(p, N_TEST).T)
    f0 = v[o + n + N_TEST * p:]
    return X, y, Xt, f0


# ── Scenario factories ────────────────────────────────────────────────────────

def make_additive_p(p):
    def fit_pygam(X, y):
        return LinearGAM(_pygam_terms(p)).gridsearch(X, y, lam=pygam_lam_grid(len(y), "gauss"))

    def pred_pygam(model, X, Xt):
        return model.predict(Xt)

    def fit_sm(X, y):
        return fit_statsmodels_gcv(y, _bs_smoother(X, p))

    def pred_sm(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        return model.predict(exog_smooth=Xp)

    def fit_gd(X, y):
        return fit_gamdist_gcv(X, y, family="normal")

    def pred_gd(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        return predict_gamdist(model, Xp)

    return [("pygam", fit_pygam, pred_pygam),
            ("statsmodels_bs", fit_sm, pred_sm),
            ("gamdist_cr", fit_gd, pred_gd)]


def make_logistic_p(p):
    def fit_pygam(X, y):
        return LogisticGAM(_pygam_terms(p)).gridsearch(X, y, lam=pygam_lam_grid(len(y), "logit"))

    def pred_pygam(model, X, Xt):
        return logit(np.clip(model.predict_proba(Xt), EPS, 1 - EPS))

    def fit_sm(X, y):
        return fit_statsmodels_gcv(y, _bs_smoother(X, p),
                                   family=sm.families.Binomial())

    def pred_sm(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        p_hat = np.clip(model.predict(exog_smooth=Xp), EPS, 1 - EPS)
        return logit(p_hat)

    def fit_gd(X, y):
        return fit_gamdist_gcv(X, y, family="binomial")

    def pred_gd(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        return logit(np.clip(predict_gamdist(model, Xp), EPS, 1 - EPS))

    return [("pygam", fit_pygam, pred_pygam),
            ("statsmodels_bs", fit_sm, pred_sm),
            ("gamdist_cr", fit_gd, pred_gd)]


P_VALUES  = (1, 5, 10, 50, 100)
SCENARIOS = {f"additive_p{p:02d}": make_additive_p(p) for p in P_VALUES}
SCENARIOS.update({f"logistic_p{p:02d}": make_logistic_p(p) for p in P_VALUES})

# Worker mode: python benchmark_python.py --worker <scenario> <method> <data> <out>
if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
    worker_main(*sys.argv[2:6])
    sys.exit(0)

# ── Run ───────────────────────────────────────────────────────────────────────
TIMEOUT_BIN, TIME_BIN, GNU_TIME = _find_tools()
CHUNKS_DIR = f"{RESULTS}/benchmark_chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Optional CLI args select one cell: python benchmark_python.py <scenario> <n>
if len(sys.argv) >= 3:
    SEL_SCENARIOS, SEL_NS = [sys.argv[1]], [int(sys.argv[2])]
else:
    SEL_SCENARIOS, SEL_NS = list(SCENARIOS), [1_000, 100_000]


def process_cell(scenario, n):
    """Run/resume one (scenario, n) cell. Returns True if deferred."""
    methods = [m for m in SCENARIOS[scenario] if not BENCH_METHODS or m[0] in BENCH_METHODS]
    shard = f"_r{REP_START}-{REP_END}" if SHARDED else ""
    chunk = f"{CHUNKS_DIR}/python_{scenario}_n{n}{shard}{BENCH_TAG}.csv"
    prev_rows, done = [], REP_START - 1
    if os.path.exists(chunk):
        prev = pd.read_csv(chunk)
        prev_rows = prev.to_dict("records")
        done = int(prev["rep"].max()) if not prev.empty else done
        if done >= REP_END:
            print(f"  {scenario} n={n}: complete (reps {REP_START}-{REP_END}) — skipping")
            return False
        print(f"  {scenario} n={n}: resuming at rep {done+1} (reps {REP_START}-{REP_END})")
    else:
        print(f"  {scenario} n={n}")
    rows = list(prev_rows)

    def _save():
        """Checkpoint completed reps so a wall-clock kill never loses work."""
        pd.DataFrame(rows).to_csv(chunk, index=False)

    allfail_reps, deferred, stopped = 0, False, False
    timeouts = {}
    for rep in range(done + 1, REP_END + 1):
        X, y, Xt, f0 = read_rep(scenario, n, rep)
        if _past_deadline():
            print(f"  [job budget reached before rep {rep}: stopping cleanly "
                  f"({done}/{REP_END} done)]")
            stopped = True
            break
        rep_rows, rep_all_failed = [], True
        with tempfile.TemporaryDirectory(prefix="bench_") as tmpdir:
            data_path = os.path.join(tmpdir, "data.npz")
            np.savez(data_path, X=X, y=y, Xt=Xt, f0=f0)
            for name in ([] if BENCH_METHODS else [BASELINE]) + [nm for nm, *_ in methods]:
                if _past_deadline():
                    print(f"  [job budget reached mid-rep {rep}: discarding partial "
                          f"rep, stopping cleanly ({done}/{REP_END} done)]")
                    stopped = True
                    break
                if timeouts.get(name, 0) >= MAX_METHOD_TIMEOUTS:
                    res = dict(NA_RESULT)   # skipped: too many timeouts
                else:
                    res = run_isolated(scenario, name, data_path, tmpdir)
                    if res.pop("timed_out", False):
                        timeouts[name] = timeouts.get(name, 0) + 1
                        if timeouts[name] == MAX_METHOD_TIMEOUTS:
                            print(f"    [{name}: {MAX_METHOD_TIMEOUTS} timeouts — "
                                  f"skipping it for the remaining reps]")
                if name != BASELINE and not np.isnan(res["elapsed_s"]):
                    rep_all_failed = False
                rep_rows.append({
                    "scenario": scenario, "method": name,
                    "n": n, "rep": rep, **res, "grid_ver": GRID_VER,
                })
        if stopped:
            break
        rows.extend(rep_rows)
        done = rep
        _save()
        if rep_all_failed:
            allfail_reps += 1
            print(f"    rep {rep}: all methods failed "
                  f"({allfail_reps}/{MAX_ALLFAIL_REPS})")
            if allfail_reps >= MAX_ALLFAIL_REPS:
                print(f"    deferring {scenario} n={n} after "
                      f"{MAX_ALLFAIL_REPS} all-fail reps")
                deferred = True
                break
    if len(rows) > len(prev_rows):
        df_chunk = pd.DataFrame(rows)
        for name in ([] if BENCH_METHODS else [BASELINE]) + [nm for nm, *_ in methods]:
            sub = df_chunk[df_chunk["method"] == name]
            if not sub.empty and "max_rss_mb" in sub:
                print(f"    {name}: {sub['elapsed_s'].median():.3f}s  "
                      f"{sub['itr_per_sec'].median():.3g} itr/s  "
                      f"max_rss={sub['max_rss_mb'].median():.1f}MB  "
                      f"mise={sub['mise'].median():.4g}  "
                      f"({sub['rep'].nunique()} reps)")
        _save()
        print(f"  saved {chunk} ({done}/{REP_END} reps)")
    else:
        print(f"  {scenario} n={n}: no new reps added")
    return deferred


# Process all viable cells first; revisit deferred (all-fail) cells at the end.
deferred_cells = []
for scenario in SEL_SCENARIOS:
    for n in SEL_NS:
        if process_cell(scenario, n):
            deferred_cells.append((scenario, n))
if deferred_cells:
    print(f"\n=== revisiting {len(deferred_cells)} deferred cell(s) ===")
    for scenario, n in deferred_cells:
        process_cell(scenario, n)

# Combine all chunks into the final CSV.
import glob as _glob
chunks = sorted(_glob.glob(f"{CHUNKS_DIR}/python_*.csv"))
if chunks:
    combined = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    out = f"{RESULTS}/python_benchmark.csv"
    tmp = f"{out}.{os.getpid()}.tmp"   # shards finish concurrently: write atomically
    combined.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"\ncombined {len(chunks)} chunks -> python_benchmark.csv")
else:
    print("No chunks found — nothing written.")
