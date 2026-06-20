# Scalability benchmark — Python methods.
#
# For each (scenario, n) cell, runs REPS replicates of fit + predict for all
# Python methods and records elapsed time, peak memory, and MISE. Results are
# checkpointed after every rep so a wall-clock kill never loses completed work.
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
# Run from the project root (or any directory — paths are resolved relative to
# this script's location):
#   python public/benchmark_python.py

import os
import sys
import signal
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, LogisticGAM, s
from statsmodels.gam.api import GLMGam, BSplines
import statsmodels.api as sm
from scipy.special import expit, logit

ROOT    = Path(__file__).parent.parent
RESULTS = str(ROOT / "results")
os.makedirs(RESULTS, exist_ok=True)

REPS    = 100
N_TEST  = 2000          # held-out points for the MISE estimate
EPS     = 1e-6
rng     = np.random.default_rng(2024)

BENCH_TIMEOUT_S  = 1800   # per-fit wall-clock cap (s)
MAX_ALLFAIL_REPS = 1000   # effectively disabled (REPS=100)

# Global job wall-clock budget (set by SLURM scripts via BENCH_JOB_BUDGET_S).
# Defaults to 0 (disabled). When set, the script stops before being SIGKILLed.
JOB_START    = time.perf_counter()
JOB_BUDGET_S = int(float(os.environ.get("BENCH_JOB_BUDGET_S", "0")))
JOB_MARGIN_S = int(float(os.environ.get("BENCH_JOB_MARGIN_S",
                                        str(BENCH_TIMEOUT_S + 300))))


def _past_deadline():
    return (JOB_BUDGET_S > 0
            and time.perf_counter() - JOB_START > JOB_BUDGET_S - JOB_MARGIN_S)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def run_timed(fn, *args, timeout_s=BENCH_TIMEOUT_S):
    """Time fn(*args) with a hard wall-clock cap.
    Returns (elapsed_s, peak_mb, model); (nan, nan, None) on timeout/error."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        model = fn(*args)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        return elapsed, peak_bytes / 1024**2, model
    except _Timeout:
        print(f"    [timeout > {timeout_s}s]")
        return float("nan"), float("nan"), None
    except Exception as e:
        print(f"    [error after {time.perf_counter() - t0:.1f}s: {e}]")
        return float("nan"), float("nan"), None
    finally:
        signal.alarm(0)
        if tracemalloc.is_tracing():
            tracemalloc.stop()


# ── 10-component signal library ───────────────────────────────────────────────
def wood_bump(x):
    return 0.2*x**11*(10*(1-x))**6 + 10*(10*x)**3*(1-x)**10

COMPONENT_FNS = [
    lambda x: np.sin(2*np.pi*x),
    lambda x: wood_bump(x),
    lambda x: np.exp(2*x),
    lambda x: (2*x - 1)**3,
    lambda x: np.sin(4*np.pi*x),
    lambda x: 40*x**2*(1-x)**2,
    lambda x: np.abs(2*x - 1),
    lambda x: np.log(1 + 5*x),
    lambda x: 20*(x - 0.5)**2,
    lambda x: x,
]


def _components(p):
    return [COMPONENT_FNS[j % len(COMPONENT_FNS)] for j in range(p)]


def _signal(fns_p, X):
    return sum(f(X[:, j]) for j, f in enumerate(fns_p))


def _pygam_terms(p):
    terms = s(0)
    for i in range(1, p):
        terms = terms + s(i)
    return terms


def _bs_smoother(X, p):
    return BSplines(X, df=[10]*p, degree=[3]*p)


# ── Scenario factories ────────────────────────────────────────────────────────

def make_additive_p(p):
    fns_p = _components(p)

    def gen(n):
        X = rng.uniform(0, 1, size=(n, p))
        y = _signal(fns_p, X) + rng.standard_normal(n)
        Xt = rng.uniform(0, 1, size=(N_TEST, p))
        f0 = _signal(fns_p, Xt)
        return X, y, Xt, f0

    def fit_pygam(X, y):
        return LinearGAM(_pygam_terms(p)).gridsearch(X, y)

    def pred_pygam(model, X, Xt):
        return model.predict(Xt)

    def fit_sm(X, y):
        return GLMGam(y, smoother=_bs_smoother(X, p)).fit()

    def pred_sm(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        return model.predict(exog_smooth=Xp)

    return gen, [("pygam", fit_pygam, pred_pygam),
                 ("statsmodels_bs", fit_sm, pred_sm)]


def make_logistic_p(p):
    fns_p = _components(p)
    # Calibrate LP scale using a separate RNG so the main stream is unaffected.
    cal = np.random.default_rng(1234 + p)
    raw = _signal(fns_p, cal.uniform(0, 1, size=(20_000, p)))
    mu, sd = float(raw.mean()), float(raw.std())

    def lp_of(X):
        return 1.5 * (_signal(fns_p, X) - mu) / sd

    def gen(n):
        X = rng.uniform(0, 1, size=(n, p))
        y = rng.binomial(1, expit(lp_of(X))).astype(float)
        Xt = rng.uniform(0, 1, size=(N_TEST, p))
        f0 = lp_of(Xt)
        return X, y, Xt, f0

    def fit_pygam(X, y):
        return LogisticGAM(_pygam_terms(p)).gridsearch(X, y)

    def pred_pygam(model, X, Xt):
        return logit(np.clip(model.predict_proba(Xt), EPS, 1 - EPS))

    def fit_sm(X, y):
        return GLMGam(y, smoother=_bs_smoother(X, p),
                      family=sm.families.Binomial()).fit()

    def pred_sm(model, X, Xt):
        Xp = np.clip(Xt, X.min(axis=0), X.max(axis=0))
        p_hat = np.clip(model.predict(exog_smooth=Xp), EPS, 1 - EPS)
        return logit(p_hat)

    return gen, [("pygam", fit_pygam, pred_pygam),
                 ("statsmodels_bs", fit_sm, pred_sm)]


P_VALUES  = (1, 5, 10, 50, 100)
SCENARIOS = {f"additive_p{p:02d}": make_additive_p(p) for p in P_VALUES}
SCENARIOS.update({f"logistic_p{p:02d}": make_logistic_p(p) for p in P_VALUES})

# ── Run ───────────────────────────────────────────────────────────────────────
CHUNKS_DIR = f"{RESULTS}/benchmark_chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Optional CLI args select one cell: python benchmark_python.py <scenario> <n>
if len(sys.argv) >= 3:
    SEL_SCENARIOS, SEL_NS = [sys.argv[1]], [int(sys.argv[2])]
else:
    SEL_SCENARIOS, SEL_NS = list(SCENARIOS), [1_000, 100_000]


def process_cell(scenario, n):
    """Run/resume one (scenario, n) cell. Returns True if deferred."""
    gen_fn, methods = SCENARIOS[scenario]
    chunk = f"{CHUNKS_DIR}/python_{scenario}_n{n}.csv"
    prev_rows, done = [], 0
    if os.path.exists(chunk):
        prev = pd.read_csv(chunk)
        prev_rows = prev.to_dict("records")
        done = int(prev["rep"].max()) if not prev.empty else 0
        if done >= REPS:
            print(f"  {scenario} n={n}: complete ({done} reps) — skipping")
            return False
        print(f"  {scenario} n={n}: resuming at rep {done+1} (have {done}/{REPS})")
    else:
        print(f"  {scenario} n={n}")
    rows = list(prev_rows)

    def _save():
        """Checkpoint completed reps so a wall-clock kill never loses work."""
        pd.DataFrame(rows).to_csv(chunk, index=False)

    allfail_reps, deferred, stopped = 0, False, False
    for rep in range(1, REPS + 1):
        X, y, Xt, f0 = gen_fn(n)   # always draw to keep the RNG stream aligned
        if rep <= done:
            continue
        if _past_deadline():
            print(f"  [job budget reached before rep {rep}: stopping cleanly "
                  f"({done}/{REPS} done)]")
            stopped = True
            break
        rep_rows, rep_all_failed = [], True
        for name, fit_fn, pred_fn in methods:
            if _past_deadline():
                print(f"  [job budget reached mid-rep {rep}: discarding partial "
                      f"rep, stopping cleanly ({done}/{REPS} done)]")
                stopped = True
                break
            elapsed, mem_mb, model = run_timed(fit_fn, X, y)
            mise = float("nan")
            if model is not None:
                rep_all_failed = False
                try:
                    pred = pred_fn(model, X, Xt)
                    mise = float(np.mean((np.asarray(pred) - f0) ** 2))
                except Exception as e:
                    print(f"    [{name} predict failed: {e}]")
            rep_rows.append({
                "scenario": scenario, "method": name,
                "n": n, "rep": rep,
                "elapsed_s": elapsed, "mem_mb": mem_mb, "mise": mise,
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
        for name, *_ in methods:
            sub = df_chunk[df_chunk["method"] == name]
            if not sub.empty:
                print(f"    {name}: {sub['elapsed_s'].median():.3f}s  "
                      f"{sub['mem_mb'].median():.1f}MB  "
                      f"mise={sub['mise'].median():.4g}  "
                      f"({sub['rep'].nunique()} reps)")
        _save()
        print(f"  saved {chunk} ({done}/{REPS} reps)")
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
    combined.to_csv(f"{RESULTS}/python_benchmark.csv", index=False)
    print(f"\ncombined {len(chunks)} chunks -> python_benchmark.csv")
else:
    print("No chunks found — nothing written.")
