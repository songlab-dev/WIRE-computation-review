"""Submit a bounded array on verified CPUs; retry only unsuccessful task IDs."""
import argparse, datetime, json, os, subprocess
from pathlib import Path
from run_task import code_hash, RUNTIME_FILES

HERE = Path(__file__).resolve().parent


def compact(values):
    values = sorted(set(values))
    parts = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    contiguous = ",".join(parts)
    stride_parts = []
    for residue in range(8):
        group = [v for v in values if v % 8 == residue]
        if not group:
            continue
        start = previous = group[0]
        for value in group[1:] + [None]:
            if value is not None and value == previous + 8:
                previous = value
                continue
            stride_parts.append(
                str(start) if start == previous else f"{start}-{previous}:8"
            )
            start = previous = value
    strided = ",".join(stride_parts)
    return min((contiguous, strided), key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--phase", choices=["pilot", "formal"], default="formal")
    ap.add_argument("--indices", help="Comma-separated integer IDs or inclusive ranges")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--code-dir",
        type=Path,
        default=HERE,
        help="Validated immutable worker-code directory",
    )
    a = ap.parse_args()
    root = a.root.resolve()
    assert 1 <= a.concurrency <= 100
    code_dir = a.code_dir.resolve()
    assert all(
        (code_dir / name).read_bytes() == (HERE / name).read_bytes()
        for name in RUNTIME_FILES
    )
    if a.phase == "formal":
        assert (root / "READY.json").exists()
        assert (
            json.loads((root / "PILOT_VALIDATED.json").read_text())["code_sha256"]
            == code_hash()
        )
    if a.retry:
        assert a.phase == "formal" and not a.indices
        a.indices = (root / "summary/retry_indices.txt").read_text().strip()
    if a.indices is None:
        a.indices = "0-4799"
    if not a.indices:
        print("No failed or missing fits to retry")
        return
    ids = []
    for chunk in a.indices.split(","):
        value, _, step = chunk.partition(":")
        fields = value.split("-")
        ids.extend(range(int(fields[0]), int(fields[-1]) + 1, int(step) if step else 1))
    assert len(set(ids)) == len(ids) and min(ids) >= 0 and max(ids) < 4800
    if (root / "RUN_SCOPE.json").exists():
        import csv

        allowed = json.loads((root / "RUN_SCOPE.json").read_text())["models"]
        with (root / "tasks.csv").open() as stream:
            task_rows = list(csv.DictReader(stream))
        ids = [i for i in ids if task_rows[i]["model"] in allowed]
    pending = []
    for task_id in ids:
        results = list((root / a.phase).glob(f"{task_id:04d}_*/result.json"))
        if results:
            r = json.loads(results[0].read_text())
            if r["status"] == "success" and r["code_sha256"] == code_hash():
                continue
            if a.phase == "formal" and r["status"] == "success":
                raise RuntimeError(
                    "Existing successful formal fit has a different source fingerprint"
                )
        pending.append(task_id)
    if not pending:
        print("All requested fits already succeeded or are outside this run scope")
        return
    logpath = root / "submission_history.json"
    history = json.loads(logpath.read_text()) if logpath.exists() else []
    # Avoid duplicate submission while an earlier formal array is active.
    if a.phase == "formal":
        for item in history:
            if item["phase"] == "formal":
                q = subprocess.run(
                    ["squeue", "-h", "-j", item["job_id"]],
                    text=True,
                    capture_output=True,
                )
                if q.stdout.strip():
                    raise RuntimeError("Formal array still active: " + item["job_id"])
    env = os.environ.copy()
    env.update(CODE_DIR=str(code_dir), EXPERIMENT_ROOT=str(root), PHASE=a.phase)
    exclusion_path = root / "audit/exclude_nodes.txt"
    exclusion = exclusion_path.read_text().strip() if exclusion_path.exists() else ""
    cmd = [
        "sbatch",
        "--parsable",
        "--array=" + compact(pending) + "%" + str(a.concurrency),
        "--export=ALL,CODE_DIR,EXPERIMENT_ROOT,PHASE",
        str(code_dir / "job.sbatch"),
    ]
    if exclusion:
        cmd.insert(2, "--exclude=" + exclusion)
    (code_dir / "logs").mkdir(exist_ok=True)
    job = (
        subprocess.check_output(cmd, cwd=code_dir, env=env, text=True)
        .strip()
        .split(";")[0]
    )
    record = {
        "job_id": job,
        "phase": a.phase,
        "tasks": len(pending),
        "indices": compact(pending),
        "concurrency": a.concurrency,
        "code_sha256": code_hash(),
        "code_dir": str(code_dir),
        "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    history.append(record)
    logpath.write_text(json.dumps(history, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
