"""Preserve scheduler records, including superseded attempts, from all releases."""
import argparse
import csv
import io
import json
import subprocess
from pathlib import Path
from run_task import code_hash

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
args = ap.parse_args()
root = args.root.resolve()
tasks = {
    int(row["task_id"]): row for row in csv.DictReader((root / "tasks.csv").open())
}
registry = (
    json.loads((root / "release_registry.json").read_text())["models"]
    if (root / "release_registry.json").exists()
    else {
        model: {"result_root": str(root), "code_sha256": code_hash()}
        for model in {row["model"] for row in tasks.values()}
    }
)
roots = {root} | {Path(x["result_root"]) for x in registry.values()}
history = {}
for result_root in sorted(roots):
    for row in json.loads((result_root / "submission_history.json").read_text()):
        history[str(row["job_id"])] = dict(row, result_root=str(result_root))
cols = [
    "JobID",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "TotalCPU",
    "AllocCPUS",
    "MaxRSS",
    "NodeList",
    "Reason",
]
output = subprocess.check_output(
    ["sacct", "-j", ",".join(history), "--parsable2", "--format=" + ",".join(cols)],
    text=True,
)
audit = root / "audit"
audit.mkdir(exist_ok=True)
(audit / "slurm_accounting.psv").write_text(output)
rows = list(csv.DictReader(io.StringIO(output), delimiter="|"))
terminal, superseded, cpu = [], [], []


def seconds(value):
    day, _, rest = value.partition("-")
    if not rest:
        rest, day = day, "0"
    values = list(map(float, rest.split(":")))
    return float(day) * 86400 + sum(v * 60**i for i, v in enumerate(values[::-1]))


for row in rows:
    job = row["JobID"]
    if "." in job:
        continue
    array, _, index = job.partition("_")
    submitted = history.get(array, {})
    scope = submitted.get("phase", "unknown")
    is_superseded = False
    if scope == "formal" and index.isdigit():
        model = tasks[int(index)]["model"]
        selected = registry[model]
        is_superseded = (
            submitted["result_root"] != selected["result_root"]
            or submitted["code_sha256"] != selected["code_sha256"]
        )
    row.update(phase=scope, superseded=is_superseded)
    if is_superseded:
        superseded.append(row)
    if row["State"] not in ["COMPLETED", "RUNNING", "PENDING", "COMPLETING"]:
        terminal.append(row)
    if row["State"] == "COMPLETED" and float(row["ElapsedRaw"]) > 0:
        cpu.append(
            {
                "job_id": job,
                "phase": scope,
                "superseded": is_superseded,
                "allocated_cpus": int(row["AllocCPUS"]),
                "wall_s": float(row["ElapsedRaw"]),
                "cpu_s": seconds(row["TotalCPU"]),
                "average_cpu_cores_used": seconds(row["TotalCPU"])
                / float(row["ElapsedRaw"]),
            }
        )
(audit / "slurm_noncompletion_records.json").write_text(json.dumps(terminal, indent=2))
(audit / "slurm_superseded_records.json").write_text(json.dumps(superseded, indent=2))
(audit / "slurm_cpu_usage.json").write_text(json.dumps(cpu, indent=2))
active_terminal = [
    r for r in terminal if r["phase"] == "formal" and not r["superseded"]
]
active_cpu = [r for r in cpu if r["phase"] == "formal" and not r["superseded"]]
report = {
    "scheduler_rows": len(rows),
    "all_noncompletion_records": len(terminal),
    "superseded_scheduler_records": len(superseded),
    "active_formal_noncompletion_records": len(active_terminal),
    "active_formal_completed_jobs": len(active_cpu),
    "max_average_cpu_cores_used": max(
        (r["average_cpu_cores_used"] for r in active_cpu), default=None
    ),
}
(audit / "slurm_summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
