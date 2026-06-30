"""Run the whole benchmark end-to-end on the local machine.

Pipeline: generate task lists -> generate datasets -> Python CPU models
-> (optional) Python GPU models -> R models -> merge -> analyze.

On a cluster, submit each stage as a SLURM array instead (see README); this
driver is the single-machine reference path.

Examples:
    python run_all.py                 # full run (CPU + R), then merge + analyze
    python run_all.py --gpu           # also run the Python GPU stage
    python run_all.py --max-tasks 3   # smoke test: first 3 tasks per stage
    python run_all.py --steps data,cpu,merge,analyze
"""
import argparse, csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PY_OUT = os.path.join(RESULTS, "python_results.csv")
R_OUT = os.path.join(RESULTS, "r_results.csv")


def n_tasks(task_list):
    with open(os.path.join(HERE, task_list), newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def loop_tasks(label, runner_cmd, task_list, max_tasks):
    total = n_tasks(task_list)
    upper = total if max_tasks is None else min(max_tasks, total)
    print(f"\n[{label}] {upper}/{total} tasks from {task_list}")
    for tid in range(upper):
        run(runner_cmd(tid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="comma list of: gentasks,data,cpu,gpu,r,merge,analyze,all")
    ap.add_argument("--gpu", action="store_true",
                    help="include the Python GPU stage in 'all'")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap tasks per stage (smoke test)")
    args = ap.parse_args()

    steps = args.steps.split(",")
    if "all" in steps:
        steps = ["gentasks", "data", "cpu", "r", "merge", "analyze"]
        if args.gpu:
            steps.insert(3, "gpu")

    os.makedirs(RESULTS, exist_ok=True)
    py = sys.executable

    if "gentasks" in steps:
        print("\n[gentasks] building task lists")
        run([py, "generate_tasks.py"])

    if "data" in steps:
        loop_tasks("data",
                   lambda t: [py, "gen_data.py", "--task-id", str(t),
                              "--task-list", "tasks_data.csv"],
                   "tasks_data.csv", args.max_tasks)

    if "cpu" in steps:
        loop_tasks("cpu",
                   lambda t: [py, "run_task.py", "--task-id", str(t),
                              "--task-list", "tasks_cpu.csv", "--out", PY_OUT],
                   "tasks_cpu.csv", args.max_tasks)

    if "gpu" in steps:
        loop_tasks("gpu",
                   lambda t: [py, "run_task.py", "--task-id", str(t),
                              "--task-list", "tasks_gpu.csv", "--out", PY_OUT, "--gpu"],
                   "tasks_gpu.csv", args.max_tasks)

    if "r" in steps:
        loop_tasks("r",
                   lambda t: ["Rscript", "run_r_task.R", "--task-id", str(t),
                              "--task-list", "tasks_r.csv", "--out", R_OUT],
                   "tasks_r.csv", args.max_tasks)

    if "merge" in steps:
        print("\n[merge]")
        run([py, "merge_results.py", PY_OUT, R_OUT])

    if "analyze" in steps:
        print("\n[analyze]")
        run([py, "analyze_results.py", "--py", PY_OUT, "--r", R_OUT,
             "--save", "--crosslang"])

    print("\nDone.")


if __name__ == "__main__":
    main()
