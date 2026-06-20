#!/bin/bash
#SBATCH --job-name=gam_install
#SBATCH --account=YOUR_ACCOUNT   # replace with your cluster account
#SBATCH --partition=standard     # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%D/logs/%x_%j.out
#SBATCH --error=%D/logs/%x_%j.err

# Submit from the project root: sbatch public/slurm/run_install.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"

module load R/4.3.1   # adjust R version as needed

Rscript "$ROOT/public/install_packages.R"
