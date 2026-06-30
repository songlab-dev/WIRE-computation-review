"""Read one row from a task CSV by --task-id and call bench_classification.py."""
import argparse, csv, subprocess, sys, os


def load_task(task_list, task_id):
    with open(task_list, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"task_id {task_id} not found in {task_list}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id",   type=int, required=True)
    ap.add_argument("--task-list", required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--gpu",       action="store_true")
    args = ap.parse_args()

    task = load_task(args.task_list, args.task_id)

    cmd = [
        sys.executable, "bench_classification.py",
        "--regime",  task["regime"],
        "--model",   task["model"],
        "--n",       task["n"],
        "--p",       task["p"],
        "--seed",    task["seed"],
        "--out",     args.out,
    ]
    if task.get("density"):
        cmd += ["--density", task["density"]]
    if task.get("data_file"):
        cmd += ["--data-file", task["data_file"]]
    if args.gpu:
        cmd += ["--gpu"]

    print(f"[task {args.task_id}] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
