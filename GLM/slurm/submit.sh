#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
glm_require_python
cd "$GLM_PROJECT_ROOT"
profile="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in smoke|pilot|full|sensitivity) ;; *) printf '%s\n' 'Profile must be smoke, pilot, full or sensitivity.' >&2; exit 2;; esac
run_root="$GLM_PROJECT_ROOT/results/slurm/$profile"
if [[ "$profile" == full ]]; then run_root="$GLM_PROJECT_ROOT/results/slurm/full100x5"; fi
resume=0;dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) resume=1; shift;;
    --dry-run) dry_run=1; shift;;
    --out) run_root="${2:?--out requires a directory}"; shift 2;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
run_root="$("$GLM_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$run_root")"
node_selection='{}'
excluded_nodes=""
if [[ -n "$GLM_ALLOWED_NODES" ]]; then
  node_selection="$("$GLM_PYTHON" "$GLM_PROJECT_ROOT/slurm/node_selection.py" select \
    --partition "$GLM_PARTITION" --allow "$GLM_ALLOWED_NODES")"
  excluded_nodes="$("$GLM_PYTHON" -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["excluded_nodes"]))' "$node_selection")"
  GLM_ALLOWED_NODES="$("$GLM_PYTHON" -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["allowed_nodes"]))' "$node_selection")"
  export GLM_ALLOWED_NODES
  printf 'Allowed worker nodes: %s\n' "$GLM_ALLOWED_NODES"
fi
if [[ -n "$GLM_EXPECTED_CPU" ]]; then printf 'Required worker CPU: %s\n' "$GLM_EXPECTED_CPU"; fi
plan_args=(--config "$GLM_PROJECT_ROOT/configs/$profile.json" --out "$run_root")
if [[ "$resume" -eq 1 ]]; then plan_args+=(--resume); fi
"$GLM_PYTHON" slurm_workflow.py plan "${plan_args[@]}"
count="$("$GLM_PYTHON" slurm_workflow.py count --out "$run_root")"
if [[ "$resume" -eq 1 ]]; then
  array_spec="$("$GLM_PYTHON" slurm_workflow.py pending --out "$run_root")"
else
  array_spec="0-$((count-1))"
fi
base=(--parsable --account="$GLM_ACCOUNT" --partition="$GLM_PARTITION" --nodes=1 --ntasks=1 \
      --chdir="$GLM_PROJECT_ROOT" --export=ALL)
array_args=("${base[@]}" --job-name="wire-glm-$profile" --cpus-per-task="$GLM_CPUS" \
      --mem="$GLM_MEM" --time="$GLM_TIME" --array="${array_spec}%${GLM_MAX_PARALLEL}" \
      --output="$run_root/logs/array-%A_%a.out" --error="$run_root/logs/array-%A_%a.err")
if [[ -n "$GLM_CONSTRAINT" ]]; then array_args+=(--constraint="$GLM_CONSTRAINT"); fi
if [[ -n "$excluded_nodes" ]]; then array_args+=(--exclude="$excluded_nodes"); fi
if [[ "$GLM_EXCLUSIVE_NODE" == 1 ]]; then array_args+=(--exclusive); fi
merge_args=("${base[@]}" --job-name="wire-glm-merge-$profile" --cpus-per-task=2 --mem=4G --time=01:00:00 \
      --output="$run_root/logs/merge-%j.out" --error="$run_root/logs/merge-%j.err")
if [[ "$dry_run" -eq 1 ]]; then
  if [[ -n "$array_spec" ]]; then
    printf 'Array command: '; printf '%q ' sbatch "${array_args[@]}" "$GLM_PROJECT_ROOT/slurm/array.sbatch" "$run_root"; printf '\n'
  else
    printf '%s\n' 'All task checkpoints are complete; only merging is required.'
  fi
  printf 'Merge command: '; printf '%q ' sbatch "${merge_args[@]}" '--dependency=afterany:ARRAY_JOB_ID' "$GLM_PROJECT_ROOT/slurm/merge.sbatch" "$run_root"; printf '\n'
  printf '%s\n' 'No job submitted. The plan is saved; use --resume when submitting this plan.'
  exit 0
fi
command -v sbatch >/dev/null
array_id=""
if [[ -n "$array_spec" ]]; then
  array_id="$(sbatch "${array_args[@]}" "$GLM_PROJECT_ROOT/slurm/array.sbatch" "$run_root")"
  array_id="${array_id%%;*}"
  if [[ ! "$array_id" =~ ^[0-9]+$ ]]; then printf 'Unexpected sbatch result: %s\n' "$array_id" >&2; exit 2; fi
  printf 'Array job: %s\n' "$array_id"
  merge_args+=(--dependency="afterany:$array_id")
fi
merge_id="$(sbatch "${merge_args[@]}" "$GLM_PROJECT_ROOT/slurm/merge.sbatch" "$run_root")"
merge_id="${merge_id%%;*}"
if [[ ! "$merge_id" =~ ^[0-9]+$ ]]; then printf 'Unexpected sbatch result: %s\n' "$merge_id" >&2; exit 2; fi
"$GLM_PYTHON" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1])/"submissions.jsonl"; record={"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"array_job_id":sys.argv[2],"merge_job_id":sys.argv[3],"array_spec":sys.argv[4],"node_selection":json.loads(sys.argv[5]),"expected_cpu":sys.argv[6]}; f=p.open("a"); f.write(json.dumps(record)+"\n"); f.close()' "$run_root" "$array_id" "$merge_id" "$array_spec" "$node_selection" "$GLM_EXPECTED_CPU"
printf 'Merge job: %s\nPublic tables: %s/merged/public_tables\n' "$merge_id" "$run_root"
