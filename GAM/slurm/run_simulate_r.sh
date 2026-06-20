#!/bin/bash
#SBATCH --job-name=gam_sim_r
#SBATCH --account=YOUR_ACCOUNT   # replace with your cluster account
#SBATCH --partition=standard     # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%D/logs/%x_%j.out
#SBATCH --error=%D/logs/%x_%j.err

# Submit from the project root: sbatch public/slurm/run_simulate_r.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"

module load R/4.3.1   # adjust R version as needed

Rscript "$ROOT/public/simulate_r.R"
