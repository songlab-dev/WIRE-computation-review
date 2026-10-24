#!/bin/bash
# Submit the benchmark grid: (scenario, n) cells, split into rep-range shards so
# no job exceeds TARGET_H hours. Each job runs R and Python in parallel on its
# shard and writes its own chunk files, so shards never share an output file.
#
#   families : additive, logistic      p : 5, 10, 50, 100  (p = 1 dropped)      n : 1000, 100000
#
# Per-cell budgets below come from the pilot (1 rep per cell): HREP is wall
# hours per rep (slower of R / Python, both fits per method, worker start-up);
# logistic p50/p100 at n=1000 did not finish a rep in the pilot and use the
# per-fit cap worst case (5 R methods x 2 fits x 1800 s + start-up ~ 5.5 h).
# HSKIP is hours per rep once a method that keeps timing out is skipped (the
# benchmark scripts drop a method after 5 timeouts in a job), from the pilot
# fits that completed. MEM is GB per job: whole-job peak RSS from sacct x 1.3,
# at least 14.
# Billing on the standard partition is max(CPUs x 7 GB, requested memory), so
# MEM <= 14 GB costs the same as 2 CPUs alone; --time is billed only as used.
#
# Run from the project root:
#   ACCOUNT=<your_slurm_account> bash public/slurm/submit_benchmark_grid.sh

set -e
: "${ACCOUNT:?set ACCOUNT=<your_slurm_account>}"
REPS=${REPS:-100}
# Pass selection (defaults: everything, R and Python, job prefix "bm"):
#   PY_METHODS  run only these Python methods (e.g. gamdist_cr, or pygam,statsmodels_bs)
#   TAG         chunk-file tag for the Python half (e.g. _fix), so a subset pass resumes
#               from its own files; also used as the job-name prefix suffix
#   SKIP_R=1    do not run the R half (it still generates the shared datasets)
PY_METHODS=${PY_METHODS:-}
TAG=${TAG:-}
SKIP_R=${SKIP_R:-0}
PREFIX="bm${TAG}"
TARGET_H=${TARGET_H:-48}     # max hours per shard job
SAFETY=${SAFETY:-1.25}       # multiplier on pilot time

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/benchmark_chunks"
JOBDIR="$ROOT/logs/benchmark_jobs"
mkdir -p "$JOBDIR"

#      scenario       n       HREP  HSKIP  MEM
CELLS="additive_p05   1000    0.01  0.01   14
additive_p05   100000  0.23  0.23   14
additive_p10   1000    0.02  0.02   14
additive_p10   100000  0.51  0.51   14
additive_p50   1000    0.24  0.24   14
additive_p50   100000  1.90  0.45   28
additive_p100  1000    3.00  3.00   14
additive_p100  100000  3.30  3.20   46
logistic_p05   1000    0.01  0.01   14
logistic_p05   100000  0.47  0.47   14
logistic_p10   1000    0.03  0.03   14
logistic_p10   100000  1.82  0.80   14
logistic_p50   1000    5.50  5.50   14
logistic_p50   100000  4.30  1.05   26
logistic_p100  1000    5.50  5.50   14
logistic_p100  100000  4.95  0.60   54"

echo "Submitting benchmark grid (REPS=$REPS, TARGET_H=$TARGET_H, SAFETY=$SAFETY)..."
job_count=0
while read -r scenario n hrep hskip mem; do
    # Largest shard (reps per job) that fits in TARGET_H: the first 5 reps cost
    # HREP each (timeouts not yet skipped), later reps HSKIP; and its wall time.
    read -r per hours <<< "$(awk -v t="$TARGET_H" -v h="$hrep" -v h2="$hskip" \
        -v s="$SAFETY" -v r="$REPS" 'function est(k) {
            return (k < 5 ? k : 5) * h + (k > 5 ? k - 5 : 0) * h2 }
        BEGIN { k = 1; while (k < r && est(k + 1) * s <= t) k++
                m = int((r + k - 1) / k); k = int((r + m - 1) / m)   # balanced shards
                x = est(k) * s; hrs = int(x) + (x > int(x)) + 1; if (hrs < 2) hrs = 2
                print k, hrs }')"

    for ((start = 1; start <= REPS; start += per)); do
        end=$((start + per - 1)); [ "$end" -gt "$REPS" ] && end=$REPS
        if [ "$per" -lt "$REPS" ]; then tag="${scenario}_n${n}_r${start}-${end}"; else tag="${scenario}_n${n}"; fi
        script="$JOBDIR/${PREFIX}_${tag}.sh"

        cat > "$script" << EOF
#!/bin/bash
#SBATCH --job-name=${PREFIX}_${tag}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=${mem}G
#SBATCH --time=${hours}:00:00
#SBATCH --output=${ROOT}/logs/${PREFIX}_${tag}_%j.out
#SBATCH --error=${ROOT}/logs/${PREFIX}_${tag}_%j.err

cd "${ROOT}"
module load R/4.3.1
module load python/3.11.5   # wire/ was built with this Python
source "${ROOT}/wire/bin/activate"

# Stop launching new fits before the --time limit and checkpoint cleanly.
export BENCH_JOB_BUDGET_S=$((hours * 3600))
export BENCH_REP_START=${start} BENCH_REP_END=${end}
export BENCH_METHODS=${PY_METHODS} BENCH_TAG=${TAG}

echo "Cell: ${scenario} n=${n} reps ${start}-${end} (mem=${mem}GB, time=${hours}h)"

# Draw this shard's datasets once (node-local); R and Python both read them.
export BENCH_DATA_DIR="\${TMPDIR:-/tmp}/bench_data_\${SLURM_JOB_ID}"
trap 'rm -rf "\$BENCH_DATA_DIR"' EXIT
Rscript "${ROOT}/public/benchmark_r.R" --gen ${scenario} ${n} || exit 1
$( [ "${SKIP_R}" = "1" ] || echo "Rscript \"${ROOT}/public/benchmark_r.R\" ${scenario} ${n} > \"${ROOT}/logs/${PREFIX}_${tag}_r.log\" 2>&1 &" )
python "${ROOT}/public/benchmark_python.py" ${scenario} ${n} \\
    > "${ROOT}/logs/${PREFIX}_${tag}_py.log" 2>&1 &
wait
EOF
        jid=$(sbatch --parsable "$script")
        job_count=$((job_count + 1))
        printf "  %-36s reps %3d-%-3d  %3sh  %2sG  job %s\n" "$tag" "$start" "$end" "$hours" "$mem" "$jid"
    done
done <<< "$CELLS"

echo "All $job_count jobs submitted. Monitor with: squeue -u \$USER"
