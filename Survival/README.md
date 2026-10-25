# Survival analysis

Code and results for the survival-analysis experiments in Section 4 of our
WIREs review. The study includes two simulation settings, three sample sizes,
and 100 independent datasets per configuration: 600 datasets and 4,800 fits.

The [per-fit results](results/summary/all_runs.csv) report IBS, C-index, fitting
time, and Peak RSS. The [summary table](results/summary/summary.csv) gives the
mean and sample standard deviation for each setting, sample size, model, and
implementation.

## Check the saved results

From this directory, use Python 3 to verify the saved summaries:

```bash
export SURV_PYTHON=python3
./run.sh verify-results
```

This checks all 384 reported means and standard deviations against the saved
per-fit results, without fitting any models. It requires neither Slurm nor a
specific CPU model.

## Reproduce the experiment

Use Linux and the versions in [environment](environment). The fitting runner
requires an **Intel Xeon Gold 6154** CPU and stops before fitting on other CPU
models. This check keeps the benchmark hardware consistent; fitting time and
Peak RSS measured on different hardware are not directly comparable.

On Great Lakes, each fit requests 4 CPUs and 32 GiB and uses one computational
thread. No GPU is used.

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
