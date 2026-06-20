#!/bin/bash
#SBATCH --job-name=gam_logistic_sim_py
#SBATCH --account=YOUR_ACCOUNT   # replace with your cluster account
#SBATCH --partition=standard     # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%D/logs/%x_%j.out
#SBATCH --error=%D/logs/%x_%j.err

# Submit from the project root: sbatch public/slurm/run_logistic_simulate_python.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"

source "$ROOT/wire/bin/activate"   # adjust if using a different venv / conda env

python "$ROOT/public/simulate_logistic_python.py"
