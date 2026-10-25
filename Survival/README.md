# Survival analysis

Code and results for Section 4 of the WIREs review. The final experiment uses
100 independent datasets for each of two settings and three sample sizes:
600 datasets and 4,800 R/Python fits.

The checked results are in [results/summary](results/summary). Each row reports
IBS, C-index, fitting time, and Peak RSS. All 4,800 fits succeeded; one failed
startup attempt was retried. The earlier 50-replication study remains in Git history.

## Reproduce the experiment

Use Linux and the versions in [environment](environment). On Great Lakes, each
fit requests 4 CPUs and 32 GiB, uses one computational thread, and requires an
Intel Xeon Gold 6154 CPU. No GPU is used.

Run these commands from this directory. Set the two Python paths, your Slurm
account, and a new output directory:

```bash
export SURV_PYTHON="$HOME/.conda/envs/surv_py/bin/python"
export DEEP_PYTHON="$HOME/.conda/envs/deepsurv_env/bin/python"
export SBATCH_ACCOUNT=your_allocation
export EXPERIMENT_ROOT=/path/to/new_run
export CONCURRENCY=16

./run.sh prepare
# Wait for data preparation to finish.
./run.sh pilot
# Wait for the pilot and native checks to finish.
./run.sh validate
./run.sh run
# Wait for the formal array to finish.
./run.sh summary
```

Use a verified 6154 node pool; see [cluster setup](docs/cluster.md). Successful
fits are skipped on resubmission. To retry failed or missing fits:

```bash
./run.sh retry
```

Data are generated once and shared by all workers. Keep datasets, predictions,
and logs on the server. The repository contains compact results and seed
mappings, not the full data archive.

## Files

| File | Purpose |
| --- | --- |
| [code/prepare.py](code/prepare.py) | Generate datasets, splits, and task records |
| [code/fit_r.R](code/fit_r.R), [code/fit_python.py](code/fit_python.py) | Fit models and record time and memory |
| [code/common.py](code/common.py) | Shared Python calculations and evaluation |
| [code/run_task.py](code/run_task.py) | Run one fit and preserve its attempt history |
| [code/summarize.py](code/summarize.py) | Check completion and produce tables |
| [code/config.json](code/config.json) | Record the fixed experiment settings |

[Methods](docs/methods.md) describes the design and measurements.
[Parameters](docs/parameters.csv) lists explicit settings, defaults, and remaining
implementation differences. [Provenance](provenance/README.md) identifies the
exact source versions used for the paper.
