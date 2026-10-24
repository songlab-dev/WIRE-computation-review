#!/bin/bash
# Submit the accuracy simulations as cell / rep-range shards, then one merge job
# per script that averages the rep chunks into results/{r,python}_<task>_n<n>.csv.
#
#   scripts : simulate_r.R, simulate_python.py, simulate_logistic_r.R,
#             simulate_logistic_python.py
#   cells   : univariate (global, local) and additive p = 1, 10, 50, 100;
#             n = 100, 500, 2000
#
# Reps per cell: REPS_LIGHT (default 500) for univariate and p <= 10,
# REPS_HEAVY (default 100) for p >= 50. Every method fits the same dataset
# files b001.. in both languages.
#
# Cells whose whole run is under 1 h are bundled into one job per script;
# heavier cells are split into rep ranges of at most TARGET_H hours. HREP below
# is hours per rep from the pilot (3 reps per cell); the R logistic p50/p100
# cells did not finish in the pilot and use a 1 h/rep placeholder. Rep chunks
# are checkpointed, so a shard that hits its time limit loses only the rep in
# progress: resubmitting skips finished reps and the merge job re-averages.
#
# Each job uses 1 CPU and 7 GB (pilot peaks <= 3 GB; 7 GB per CPU is billed
# the same as the CPU alone), with BLAS/OpenMP pinned to one thread.
#
# Run from the project root:
#   ACCOUNT=<your_slurm_account> bash public/slurm/submit_simulations.sh
#   GENERATE=1 ... also (re)generates data/ first; move any old data/ aside first.

set -e
: "${ACCOUNT:?set ACCOUNT=<your_slurm_account>}"
REPS_LIGHT=${REPS_LIGHT:-500}
REPS_HEAVY=${REPS_HEAVY:-100}
TARGET_H=${TARGET_H:-48}
SAFETY=${SAFETY:-1.25}
GENERATE=${GENERATE:-0}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs results/sim_chunks
JOBDIR="$ROOT/logs/sim_jobs"
mkdir -p "$JOBDIR"

# ── Data generation (optional) ────────────────────────────────────────────────
DEP_G=""; DEP_L=""
if [ "$GENERATE" = "1" ]; then
    if ls data/*_b*.csv > /dev/null 2>&1; then
        echo "data/ already has datasets; move it aside (e.g. mv data data_old) or unset GENERATE"
        exit 1
    fi
    JID_G=$(sbatch --parsable --account="$ACCOUNT" public/slurm/run_generate.sh)
    JID_L=$(sbatch --parsable --account="$ACCOUNT" public/slurm/run_logistic_generate.sh)
    DEP_G="--dependency=afterok:$JID_G"; DEP_L="--dependency=afterok:$JID_L"
    echo "generate: $JID_G   logistic_generate: $JID_L"
fi

#      script                       task             n     HREP
CELLS="simulate_r.R                 global           100   0.0002
simulate_r.R                 global           500   0.0001
simulate_r.R                 global           2000  0.0001
simulate_r.R                 local            100   0.0001
simulate_r.R                 local            500   0.0001
simulate_r.R                 local            2000  0.0001
simulate_r.R                 additive_p01     100   0.0001
simulate_r.R                 additive_p01     500   0.0001
simulate_r.R                 additive_p01     2000  0.0001
simulate_r.R                 additive_p10     100   0.0007
simulate_r.R                 additive_p10     500   0.0006
simulate_r.R                 additive_p10     2000  0.0007
simulate_r.R                 additive_p50     100   0.0017
simulate_r.R                 additive_p50     500   0.0898
simulate_r.R                 additive_p50     2000  0.0920
simulate_r.R                 additive_p100    100   0.0003
simulate_r.R                 additive_p100    500   0.0071
simulate_r.R                 additive_p100    2000  1.1612
simulate_python.py           global           100   0.0002
simulate_python.py           global           500   0.0003
simulate_python.py           global           2000  0.0006
simulate_python.py           local            100   0.0001
simulate_python.py           local            500   0.0004
simulate_python.py           local            2000  0.0006
simulate_python.py           additive_p01     100   0.0001
simulate_python.py           additive_p01     500   0.0003
simulate_python.py           additive_p01     2000  0.0006
simulate_python.py           additive_p10     100   0.0013
simulate_python.py           additive_p10     500   0.0035
simulate_python.py           additive_p10     2000  0.0062
simulate_python.py           additive_p50     100   0.0078
simulate_python.py           additive_p50     500   0.0193
simulate_python.py           additive_p50     2000  0.0359
simulate_python.py           additive_p100    100   0.0265
simulate_python.py           additive_p100    500   0.0532
simulate_python.py           additive_p100    2000  0.0913
simulate_logistic_r.R        logistic_global  100   0.0004
simulate_logistic_r.R        logistic_global  500   0.0002
simulate_logistic_r.R        logistic_global  2000  0.0001
simulate_logistic_r.R        logistic_local   100   0.0002
simulate_logistic_r.R        logistic_local   500   0.0001
simulate_logistic_r.R        logistic_local   2000  0.0001
simulate_logistic_r.R        logistic_p01     100   0.0001
simulate_logistic_r.R        logistic_p01     500   0.0001
simulate_logistic_r.R        logistic_p01     2000  0.0001
simulate_logistic_r.R        logistic_p10     100   0.0035
simulate_logistic_r.R        logistic_p10     500   0.0063
simulate_logistic_r.R        logistic_p10     2000  0.0271
simulate_logistic_r.R        logistic_p50     100   0.0083
simulate_logistic_r.R        logistic_p50     500   1.0
simulate_logistic_r.R        logistic_p50     2000  1.0
simulate_logistic_r.R        logistic_p100    100   1.0
simulate_logistic_r.R        logistic_p100    500   1.0
simulate_logistic_r.R        logistic_p100    2000  1.0
simulate_logistic_python.py  logistic_global  100   0.0003
simulate_logistic_python.py  logistic_global  500   0.0004
simulate_logistic_python.py  logistic_global  2000  0.0009
simulate_logistic_python.py  logistic_local   100   0.0001
simulate_logistic_python.py  logistic_local   500   0.0004
simulate_logistic_python.py  logistic_local   2000  0.0009
simulate_logistic_python.py  logistic_p01     100   0.0001
simulate_logistic_python.py  logistic_p01     500   0.0004
simulate_logistic_python.py  logistic_p01     2000  0.0009
simulate_logistic_python.py  logistic_p10     100   0.0031
simulate_logistic_python.py  logistic_p10     500   0.0067
simulate_logistic_python.py  logistic_p10     2000  0.0076
simulate_logistic_python.py  logistic_p50     100   0.0652
simulate_logistic_python.py  logistic_p50     500   0.0831
simulate_logistic_python.py  logistic_p50     2000  0.0506
simulate_logistic_python.py  logistic_p100    100   0.3520
simulate_logistic_python.py  logistic_p100    500   0.4041
simulate_logistic_python.py  logistic_p100    2000  0.2128"

# write_job <name> <hours> <dependency> <command lines...>: write and submit one job.
write_job() {
    local name=$1 hours=$2 dep=$3; shift 3
    local script="$JOBDIR/${name}.sh"
    {
        cat << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=${hours}:00:00
#SBATCH --output=${ROOT}/logs/${name}_%j.out
#SBATCH --error=${ROOT}/logs/${name}_%j.err

cd "${ROOT}"
module load R/4.3.1
module load python/3.11.5   # wire/ was built with this Python
source "${ROOT}/wire/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EOF
        printf '%s\n' "$@"
    } > "$script"
    sbatch --parsable $dep "$script"
}

runner() {   # runner <script>: interpreter for a simulation script
    case $1 in *.R) echo "Rscript" ;; *) echo "python -u" ;; esac
}

job_count=0
for script in simulate_r.R simulate_python.py simulate_logistic_r.R simulate_logistic_python.py; do
    case $script in *logistic*) dep=$DEP_L ;; *) dep=$DEP_G ;; esac
    run=$(runner "$script")
    base=${script%.*}
    light_cmds=(); light_h=0; jids=()

    while read -r s task n hrep; do
        [ "$s" = "$script" ] || continue
        p=${task##*_p}; [ "$p" = "$task" ] && p=0
        if [ $((10#$p)) -ge 50 ]; then reps=$REPS_HEAVY; else reps=$REPS_LIGHT; fi
        total=$(awk -v r="$reps" -v h="$hrep" -v s="$SAFETY" 'BEGIN { print r * h * s }')

        if awk -v t="$total" 'BEGIN { exit !(t < 1) }'; then
            # Light cell: bundle into one sequential job per script.
            light_cmds+=("SIM_REP_START=1 SIM_REP_END=$reps $run public/$script $task $n")
            light_h=$(awk -v a="$light_h" -v t="$total" 'BEGIN { print a + t }')
            continue
        fi

        per=$(awk -v t="$TARGET_H" -v h="$hrep" -v s="$SAFETY" -v r="$reps" \
            'BEGIN { k = int(t / (h * s)); if (k < 1) k = 1; if (k > r) k = r
                     m = int((r + k - 1) / k); print int((r + m - 1) / m) }')   # balanced shards
        for ((start = 1; start <= reps; start += per)); do
            end=$((start + per - 1)); [ "$end" -gt "$reps" ] && end=$reps
            hours=$(awk -v k="$((end - start + 1))" -v h="$hrep" -v s="$SAFETY" \
                'BEGIN { x = k * h * s; t = int(x) + (x > int(x)) + 1; if (t < 2) t = 2; print t }')
            name="sim_${base#simulate_}_${task}_n${n}_r${start}-${end}"
            jid=$(write_job "$name" "$hours" "$dep" \
                "SIM_REP_START=$start SIM_REP_END=$end $run public/$script $task $n")
            jids+=("$jid"); job_count=$((job_count + 1))
            printf "  %-46s %3sh  job %s\n" "$name" "$hours" "$jid"
        done
    done <<< "$CELLS"

    if [ ${#light_cmds[@]} -gt 0 ]; then
        hours=$(awk -v x="$light_h" 'BEGIN { t = int(x) + (x > int(x)) + 1; if (t < 2) t = 2; print t }')
        name="sim_${base#simulate_}_light"
        jid=$(write_job "$name" "$hours" "$dep" "${light_cmds[@]}")
        jids+=("$jid"); job_count=$((job_count + 1))
        printf "  %-46s %3sh  job %s  (%d cells)\n" "$name" "$hours" "$jid" "${#light_cmds[@]}"
    fi

    # Merge after every shard of this script has ended (any state).
    deps=$(IFS=:; echo "${jids[*]}")
    name="sim_${base#simulate_}_merge"
    jid=$(write_job "$name" 2 "--dependency=afterany:$deps" "$run public/$script --merge")
    job_count=$((job_count + 1))
    printf "  %-46s %3sh  job %s  (after all %s shards)\n" "$name" 2 "$jid" "$base"
done

echo "All $job_count jobs submitted. Monitor with: squeue -u \$USER"
