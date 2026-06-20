#!/bin/bash
#SBATCH --job-name=gam_logistic_gen
#SBATCH --account=YOUR_ACCOUNT   # replace with your cluster account
#SBATCH --partition=standard     # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%D/logs/%x_%j.out
#SBATCH --error=%D/logs/%x_%j.err

# Submit from the project root: sbatch public/slurm/run_logistic_generate.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/data"

module load R/4.3.1   # adjust R version as needed

Rscript "$ROOT/public/generate_logistic_data.R"
