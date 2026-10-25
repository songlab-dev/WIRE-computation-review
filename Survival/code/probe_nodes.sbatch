#!/bin/bash
#SBATCH --job-name=surv_cpu_probe
#SBATCH --partition=standard
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:02:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
srun --ntasks=32 --ntasks-per-node=1 /usr/bin/python3 -c 'import socket,json; print(json.dumps({"host":socket.gethostname().split(".")[0],"cpu":next(l.split(":",1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))}))'
