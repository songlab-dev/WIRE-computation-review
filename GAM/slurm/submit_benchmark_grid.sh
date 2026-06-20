#!/bin/bash
# Submit one SLURM job per (scenario, n) cell of the benchmark grid.
#
#   families : additive, logistic
#   p        : 1, 5, 10, 50, 100
#   n        : 1000, 100000
#   => 2 x 5 x 2 = 20 jobs
#
# Each job runs R and Python in parallel on its own (scenario, n) cell, so
# there are no races on the per-cell chunk files.
#
# Run from the project root:
#   bash public/slurm/submit_benchmark_grid.sh

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/benchmark_chunks"

# Resources by (p, n): "<mem_GB> <time_hours>". n=100000 is the costly column.
get_resources() {
    local p=$1 n=$2
    if [ "$n" -le 1000 ]; then
        case $p in
            1|5)  echo "8 1"  ;;
            10)   echo "16 2" ;;
            50)   echo "24 3" ;;
            100)  echo "48 4" ;;
        esac
    else   # n = 100000
        case $p in
            1)    echo "16 3"  ;;
            5)    echo "16 4"  ;;
            10)   echo "16 8"  ;;
            50)   echo "24 24" ;;
            100)  echo "48 36" ;;
        esac
    fi
}

echo "Submitting benchmark grid jobs..."
echo "=========================================="

job_count=0
for family in additive logistic; do
    for p in 1 5 10 50 100; do
        scenario=$(printf "%s_p%02d" "$family" "$p")
        for n in 1000 100000; do
            job_count=$((job_count + 1))
            read mem time <<< "$(get_resources "$p" "$n")"
            # Job wall-clock budget (s) passed to the benchmark scripts so they
            # stop and checkpoint before SLURM kills them at the --time limit.
            budget_s=$((time * 3600))

            cat > /tmp/cell_${scenario}_n${n}.sh << EOF
#!/bin/bash
#SBATCH --job-name=bm_${scenario}_n${n}
#SBATCH --account=YOUR_ACCOUNT   # replace with your cluster account
#SBATCH --partition=standard     # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=${mem}G
#SBATCH --time=${time}:00:00
#SBATCH --output=${ROOT}/logs/${scenario}_n${n}_%j.out
#SBATCH --error=${ROOT}/logs/${scenario}_n${n}_%j.err

mkdir -p "${ROOT}/results/benchmark_chunks"

module load R/4.3.1   # adjust R version as needed
source "${ROOT}/wire/bin/activate"   # adjust if using a different venv / conda env

# Both R and Python stop launching new fits within JOB_MARGIN_S of this budget
# and checkpoint cleanly instead of being SIGKILLed.
export BENCH_JOB_BUDGET_S=${budget_s}

echo "Cell: $scenario n=$n (mem=${mem}GB, time=${time}h, budget=${budget_s}s)"
echo "=========================================="

Rscript "${ROOT}/public/benchmark_r.R" $scenario $n \
    > "${ROOT}/logs/r_${scenario}_n${n}_%j.log" 2>&1 &

python "${ROOT}/public/benchmark_python.py" $scenario $n \
    > "${ROOT}/logs/py_${scenario}_n${n}_%j.log" 2>&1 &

wait
EOF

            sbatch /tmp/cell_${scenario}_n${n}.sh
            echo "[$job_count/20] Submitted: $scenario n=$n (mem=${mem}GB, time=${time}h)"
        done
    done
done

echo "=========================================="
echo "All $job_count jobs submitted."
echo "Monitor with: squeue -u \$USER"
