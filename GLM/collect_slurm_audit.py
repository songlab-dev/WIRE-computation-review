#!/usr/bin/env python3
"""Export existing Slurm checkpoints for diagnosis, without refitting or merging.

Run from GLM: .venv/bin/python collect_slurm_audit.py --out results/slurm/full
The numerical limits below match check_results.py; relative differences are
informational only. No benchmark source, plan, checkpoint, or limit is changed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


OBJECTIVE_LIMIT = 1e-6
DEVIANCE_LIMIT = 1e-4
UNPENALIZED = {"dense", "sparse"}


def successful(row):
    return str(row.get("success", "")).strip().lower() == "true"


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def numeric_range(rows, key):
    values = [finite_number(row.get(key)) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return max(values) - min(values)


def by_method(rows, key):
    return json.dumps({row["method"]: row.get(key) for row in rows}, ensure_ascii=False)


def paired_audit(rows, kkt_tol):
    """Inspect every block, retaining failures and every numerical issue."""
    blocks = defaultdict(list)
    for row in rows:
        blocks[(row["scenario"], int(row["seed"]))].append(row)
    output = []
    for (scenario, seed), block in sorted(blocks.items()):
        okay = [row for row in block if successful(row)]
        penalized = block[0]["regime"] not in UNPENALIZED
        issues = []
        kkts = [finite_number(row.get("kkt_scaled")) for row in okay]
        if any(value is None or value > kkt_tol for value in kkts):
            issues.append("successful_task_kkt_invalid")
        objective_range = numeric_range(okay, "objective")
        test_range = numeric_range(okay, "test_deviance")
        cv_range = numeric_range(okay, "cv_deviance") if penalized else None
        lambdas = [finite_number(row.get("lambda_selected")) for row in okay] if penalized else []
        lambda_count = len(set(value for value in lambdas if value is not None))
        if len(okay) > 1:
            if objective_range is None:
                issues.append("objective_missing_or_nonfinite")
            elif objective_range >= OBJECTIVE_LIMIT:
                issues.append("objective_range_at_or_above_limit")
            if penalized:
                if any(value is None for value in lambdas) or lambda_count != 1:
                    issues.append("selected_lambda_differs_or_invalid")
                if cv_range is None:
                    issues.append("cv_missing_or_nonfinite")
                elif cv_range >= DEVIANCE_LIMIT:
                    issues.append("cv_range_at_or_above_limit")
            elif test_range is None:
                issues.append("holdout_missing_or_nonfinite")
            elif test_range >= DEVIANCE_LIMIT:
                issues.append("holdout_range_at_or_above_limit")
        cv_values = [finite_number(row.get("cv_deviance")) for row in okay] if penalized else []
        relative = None
        if cv_range is not None:
            relative = cv_range / max(1.0, max(abs(value) for value in cv_values))
        output.append({
            "scenario": scenario, "seed": seed, "regime": block[0]["regime"],
            "recorded_methods": len(block), "successful_methods": len(okay),
            "failed_methods": len(block) - len(okay),
            "successful_method_names": ";".join(row["method"] for row in okay),
            "failed_method_names": ";".join(row["method"] for row in block if not successful(row)),
            "objective_range": objective_range, "objective_limit_strict": OBJECTIVE_LIMIT,
            "holdout_deviance_range": test_range,
            "selected_lambda_count": lambda_count if penalized else None,
            "selected_cv_range": cv_range,
            "cv_limit_strict": DEVIANCE_LIMIT if penalized else None,
            "cv_relative_range_diagnostic_only": relative,
            "max_scaled_kkt": max((value for value in kkts if value is not None), default=None),
            "kkt_limit_inclusive": kkt_tol,
            "lambda_by_method": by_method(okay, "lambda_selected") if penalized else "",
            "cv_by_method": by_method(okay, "cv_deviance") if penalized else "",
            "objective_by_method": by_method(okay, "objective"),
            "audit_status": "issues" if issues else ("passed" if len(okay) > 1 else "not_applicable"),
            "issues": ";".join(issues),
        })
    return output


CV_FIELDS = ["scenario", "seed", "method", "success", "status", "lambda_index_zero_based",
             "selected", "lambda_selected", "cv_deviance", "cpu_model", "worker_hostname"]
WORKER_ISSUE_FIELDS = ["scenario", "seed", "method", "issue"]


def cv_audit(rows, records, lambda_count):
    """Preserve full recorded paths and identify malformed successful paths."""
    curves, issues = [], []
    for row, record in zip(rows, records):
        if row["regime"] in UNPENALIZED:
            continue
        worker = record["worker"]
        scores = worker.get("cv_scores")
        valid_list = isinstance(scores, list)
        selected = worker.get("lambda_index")

        def report(reason):
            issues.append({"scenario": row["scenario"], "seed": row["seed"],
                           "method": row["method"], "issue": reason})

        if successful(row):
            if not valid_list or len(scores) != lambda_count:
                report("successful_cv_path_missing_or_wrong_length")
            elif any(finite_number(value) is None for value in scores):
                report("successful_cv_path_nonfinite")
            else:
                if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < len(scores):
                    report("selected_cv_index_invalid")
                else:
                    if selected != min(range(len(scores)), key=lambda index: scores[index]):
                        report("selected_index_not_first_cv_minimum")
                    if finite_number(row.get("cv_deviance")) != finite_number(scores[selected]):
                        report("summary_cv_differs_from_selected_path_score")
            for key in ["lambda_index", "lambda_selected", "cv_deviance"]:
                if row.get(key) != worker.get(key):
                    report("summary_worker_mismatch_" + key)
        if valid_list:
            for index, score in enumerate(scores):
                curves.append({"scenario": row["scenario"], "seed": row["seed"],
                               "method": row["method"], "success": successful(row),
                               "status": row["status"], "lambda_index_zero_based": index,
                               "selected": index == selected, "lambda_selected": row.get("lambda_selected"),
                               "cv_deviance": score, "cpu_model": row.get("cpu_model"),
                               "worker_hostname": row.get("worker_hostname")})
    return curves, issues


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_audit(root, destination=None):
    # This validated, CPU-relaxed reader never calls a fitter or writes shards.
    import slurm_cpu_relaxed as relaxed

    root = Path(root).resolve()
    plan, rows, records, environments, manifests, locations, inputs = relaxed.gather(root)
    profile = plan["config"]["profile"]
    safe_profile = "".join(char if char.isalnum() or char in "-_" else "_" for char in profile)
    name = "GLM_" + safe_profile + "_audit"
    destination = Path(destination or (name + ".tar.gz")).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Write the diagnostic archive outside the original results directory")
    if destination.exists():
        raise FileExistsError("Archive already exists; choose a new --archive filename: " + str(destination))
    blocks = paired_audit(rows, plan["config"]["kkt_tol"])
    curves, worker_issues = cv_audit(rows, records, plan["config"]["lambda_count"])
    failures = [row for row in rows if not successful(row)]
    summary = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnosis_only_no_refitting_no_publication_certification",
        "profile": profile, "plan_fingerprint": plan["fingerprint"],
        "collector_sha256": sha256(Path(__file__)),
        "checkpoint_coverage_verified": True, "non_cpu_environment_verified": True,
        "cpu_policy": relaxed.POLICY, "cpu_models": sorted({env["cpu"] for env in environments.values()}),
        "array_elements": len(plan["tasks"]), "recorded_method_tasks": len(rows),
        "successful_method_tasks": len(rows) - len(failures), "failed_method_tasks": len(failures),
        "failure_count_by_method": dict(Counter(row["method"] for row in failures)),
        "paired_blocks": len(blocks),
        "blocks_with_numerical_issues": sum(block["audit_status"] == "issues" for block in blocks),
        "blocks_without_successful_method_pairs": sum(block["successful_methods"] < 2 for block in blocks),
        "worker_cv_issues": len(worker_issues), "recorded_cv_path_values": len(curves),
        "limits": {"objective_range_strictly_below": OBJECTIVE_LIMIT,
                   "cv_and_unpenalized_holdout_range_strictly_below": DEVIANCE_LIMIT,
                   "kkt_at_or_below": plan["config"]["kkt_tol"]},
        "relative_cv_difference_changes_acceptance": False,
        "publication_verified": False, **relaxed.pairing_metadata(rows),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".glm-audit-", dir=destination.parent) as private:
        stage = Path(private) / name
        stage.mkdir()
        write_csv(stage / "raw_results.csv", rows)
        write_csv(stage / "paired_audit.csv", blocks)
        write_csv(stage / "failed_tasks.csv", failures, list(rows[0]))
        write_csv(stage / "cv_scores.csv", curves, CV_FIELDS)
        write_csv(stage / "worker_cv_issues.csv", worker_issues, WORKER_ISSUE_FIELDS)
        with (stage / "worker_results.jsonl").open("w", encoding="utf-8") as stream:
            for row, record in zip(rows, records):
                # Original summaries and worker dictionaries are retained verbatim
                # as values; CPU/node annotations are a separate field.
                stream.write(json.dumps({**record, "recorded_hardware": {
                    "cpu_model": row["cpu_model"], "allocated_node": row["allocated_node"]}},
                    ensure_ascii=False) + "\n")
        shutil.copyfile(root / "plan.json", stage / "plan.json")
        shutil.copyfile(root / "config.json", stage / "config.json")
        write_json(stage / "summary.json", summary)
        write_json(stage / "checkpoint_inputs.json", inputs)
        write_json(stage / "shard_index.json", locations)
        for folder in ["environment_records", "manifests", "source", "worker_logs"]:
            (stage / folder).mkdir()
        for env_id, environment in environments.items():
            write_json(stage / "environment_records" / (env_id + ".json"), environment)
        for manifest in manifests:
            shutil.copyfile(manifest, stage / "manifests" / manifest.name)
        source = Path(__file__).resolve().parent
        for filename in dict.fromkeys([*plan["source_sha256"], "slurm_cpu_relaxed.py",
                                       "check_results.py", "summarize_results.py", Path(__file__).name]):
            shutil.copyfile(source / filename, stage / "source" / filename)
        for task in plan["tasks"]:
            shard = relaxed.workflow.shard_path(root, task)
            block_rows = [row for row in rows if row["scenario"] == task["scenario"] and row["seed"] == task["seed"]]
            for row in block_rows:
                if row["regime"] in UNPENALIZED and successful(row):
                    continue
                log = shard / "tasks" / (task["scenario"] + "_" + str(task["seed"])) / row["method"] / "worker.log"
                if log.is_file():
                    folder = stage / "worker_logs" / ("task-" + str(task["index"]).zfill(4))
                    folder.mkdir(exist_ok=True)
                    shutil.copyfile(log, folder / (row["method"] + ".log"))
        (stage / "README.txt").write_text(
            "Diagnostic export of existing checkpoints; no models were fitted.\n"
            "This archive does not certify the full run or bypass check_results.py.\n"
            "paired_audit.csv uses the original absolute numerical limits.\n"
            "Relative CV differences are informational and do not change acceptance.\n"
            "cv_scores.csv contains all recorded path values with zero-based indices.\n"
            "lambda_selected is the chosen lambda, not the lambda at every path index.\n"
            "Exact full lambda grids and per-fold coefficients were not saved by the workers.\n"
            "worker_results.jsonl retains original summaries, final coefficients, native\n"
            "diagnostics, warnings, warmup warnings, and timings, including failures.\n"
            "Dataset manifests/checksums are included; matrix files are omitted.\n",
            encoding="utf-8")
        checksums = {str(path.relative_to(stage)): sha256(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "archive_file_sha256.json", checksums)
        packed = Path(private) / "audit.tar.gz"
        with tarfile.open(packed, "w:gz") as archive:
            archive.add(stage, arcname=name)
        # Both files are on the same filesystem: publish atomically, without
        # replacing an existing archive or altering anything under root.
        os.link(packed, destination)
    print(f"Profile={profile}; array elements={len(plan['tasks'])}; recorded tasks={len(rows)}; "
          f"successful={len(rows) - len(failures)}; failed={len(failures)}")
    print(f"Blocks with numerical issues={summary['blocks_with_numerical_issues']}; "
          f"worker CV issues={len(worker_issues)}")
    h1 = [block for block in blocks if block["scenario"] == "H1" and block["selected_cv_range"] is not None]
    for block in sorted(h1, key=lambda value: value["selected_cv_range"], reverse=True)[:10]:
        print(f"H1 seed={block['seed']}; selected CV range={block['selected_cv_range']:.10g}; "
              f"limit=<{DEVIANCE_LIMIT:g}; status={block['audit_status']}")
        print("  CV by method: " + block["cv_by_method"])
        print("  Lambda by method: " + block["lambda_by_method"])
    print("Diagnostic archive: " + str(destination))
    print("Original results unchanged; no refitting; numerical acceptance limits unchanged.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing Slurm run root, e.g. results/slurm/full")
    parser.add_argument("--archive", help="New archive path outside the run root; default GLM_PROFILE_audit.tar.gz")
    args = parser.parse_args()
    export_audit(args.out, args.archive)


if __name__ == "__main__":
    main()
