# Poisson GLM benchmark

Reproducible R/Python code for the GLM section of the WIRE computational statistics review. Each scenario uses **100 independent datasets**, all timed. Each implementation receives **one untimed warm-up and five measured fits per dataset**. Running the pipeline generates the data, fit records, numerical checks and manuscript tables.

## Environment

Use Python 3.12 and R 4.4.0 for the recorded direct package versions:

```bash
cd GLM
GLM_PIN_R=1 bash install_dependencies.sh
```

Python dependencies are pinned in `requirements.txt`; R versions are in `environment/R-packages.csv`. `GLM_PIN_R=1` installs those R versions, potentially from source archives. Rcpp >= 1.1.0 is needed for fastglm. Set `PYTHON_FOR_VENV` to select another Python 3.12 executable. A complete transitive R dependency lock was not recorded; each new run saves its actual packages and BLAS. Omitting `GLM_PIN_R=1` installs available R packages for a separately recorded environment (R >= 4.3).

## Design and methods

`configs/full.json` defines the scenario grid, all 100 seeds, solver settings and tuning protocol.

| Scenario | Design | n | p | Entry density |
| --- | --- | ---: | ---: | ---: |
| D1 | Dense, AR(1) correlation 0.3 | 2,000 | 200 | 1 |
| D2 | Dense, AR(1) correlation 0.3 | 10,000 | 500 | 1 |
| D3 | Dense, AR(1) correlation 0.3 | 30,000 | 500 | 1 |
| S1 | Sparse | 10,000 | 500 | 0.01 |
| S2 | Sparse | 30,000 | 500 | 0.01 |
| S3 | Sparse | 10,000 | 500 | 0.05 |
| S4 | Sparse | 30,000 | 500 | 0.05 |
| H1 | Dense Gaussian, L1-penalized | 300 | 1,000 | 1 |
| H2 | Dense Gaussian, L1-penalized | 500 | 500 | 1 |
| SH1 | Sparse, L1-penalized | 300 | 1,000 | 0.01 |
| SH2 | Sparse, L1-penalized | 500 | 500 | 0.01 |

Unpenalized methods are R `stats::glm`, `fastglm`, `speedglm`, `biglm::bigglm`, and Python `statsmodels`, `scikit-learn`, `glum`, `spglm`. Penalized methods are R `glmnet` and Python `glum`, `skglm`. Methods fit the same saved data sequentially within each dataset. Sparse storage is preserved until an interface requires conversion; biglm uses resident-data chunks rather than disk streaming.

Unpenalized designs have 10 active positive slopes from 0.5 to 0.1. Penalized designs have 20 signed active slopes with magnitudes from 0.5 to 0.15. The true intercept is zero; the generating linear predictor is not clipped. Independent test sets have 2,000 observations.

Both languages use `mean(exp(eta) - y*eta)`, adding `lambda * sum(abs(beta))` for penalized fits. The intercept is unpenalized and predictors are not standardized. Penalized methods share five saved CV folds and 100 descending lambdas, select by observation-weighted Poisson deviance, and favor the larger lambda on an exact tie. Each measured workload includes the five fold paths and the full-data path. Path warm starts are allowed; each repetition constructs a fresh estimator.

## Local execution

```bash
.venv/bin/python test_protocol.py
bash run_all.sh smoke
.venv/bin/python check_results.py --out results/smoke --require-all-success
# After smoke passes:
bash run_all.sh pilot
.venv/bin/python estimate_runtime.py --pilot results/pilot
# After inspecting pilot failures and resources:
bash run_all.sh full --out results/full100x5
```

`smoke` uses small data to check interfaces. `pilot` uses full dimensions with a small repetition count. The optional `sensitivity` profile changes sparse density and the penalty range and is a separate study. Non-full table exports are labeled exploratory.

To resume, use the same output directory with `--resume`. Completed atomic method checkpoints are retained. Explicit solver errors, internal timeouts and quality failures remain completed attempts and are not automatically retried. Source, configuration or software changes require a fresh output directory. Results are never deleted by the runner. Replication counts are set in JSON, not a shell `REPS` variable.

## Great Lakes / Slurm

Configure the account, modules and resources before submitting:

```bash
cp slurm/site_config.example.sh slurm/site_config.sh
module spider python
module spider R
# Edit slurm/site_config.sh with your authorized account and installed modules.
bash slurm/submit_setup.sh
```

Wait for `Setup passed` in `slurm/setup-logs/`. Existing environments can instead be selected through `GLM_PYTHON` and `GLM_R_LIBS` after checking their dependencies.

| Setting in `slurm/site_config.sh` | Default / meaning |
| --- | --- |
| `GLM_ACCOUNT`, `GLM_MODULES` | Your authorized account and actual module names |
| `GLM_PARTITION` | `standard` |
| `GLM_PYTHON`, `GLM_R_LIBS` | Project `.venv/bin/python` and `.R-library` |
| `GLM_PIN_R` | `0`; set to `1` with R 4.4.0 for recorded direct R versions |
| `GLM_CPUS`, `GLM_MEM`, `GLM_TIME` | `2`, `8G`, `24:00:00`; adjust memory/time using the pilot |
| `GLM_MAX_PARALLEL` | `10` concurrent array elements |
| `GLM_CONSTRAINT`, `GLM_ALLOWED_NODES`, `GLM_EXPECTED_CPU` | Empty to allow different CPU models |

Submit smoke, inspect its completion, then pilot, inspect resources, then full:

```bash
bash slurm/submit.sh smoke
# After smoke completes:
.venv/bin/python check_results.py --out results/slurm/smoke/merged --require-all-success
bash slurm/submit.sh pilot
# After pilot completes and resources are checked:
bash slurm/submit.sh full --out results/slurm/full100x5
```

Full runs have 1,100 scenario-seed elements, 6,800 dataset-method tasks and 34,000 requested measured fits, plus 6,800 untimed warm-ups. Methods within each element run sequentially with one fitting thread; the second allocated core supports monitoring. Resource defaults are starting values, not cluster limits. Pilot fitting-hour estimates exclude queueing, startup, JIT and I/O.

```bash
.venv/bin/python slurm_workflow.py status --out results/slurm/full100x5
squeue -u "$USER"
bash slurm/submit.sh full --out results/slurm/full100x5 --resume
```

A preview is available with `--dry-run`; it saves a plan, so later submission to that directory needs `--resume`. An unfinished element resumes from its complete method checkpoints. Slurm-killed work without a checkpoint is incomplete; recorded method failures stay in the attempted denominator. `GLM_TIME` limits each array element; `task_timeout_seconds` in the JSON limits each method, including loading and warm-up. Changing the internal timeout changes the protocol and requires a new output directory.

Array jobs use `slurm_cpu_relaxed.py`. CPU models are recorded and may differ; software, BLAS, modules, architecture, source and thread settings remain checked. CV-score differences are informational, while selected-lambda, objective and completeness checks remain active. Use this merge entry point instead of the original strict merger retained in `slurm_workflow.py`:

```bash
.venv/bin/python slurm_cpu_relaxed.py merge --out results/slurm/full100x5
.venv/bin/python export_tables.py --results results/slurm/full100x5/merged
```

Merging and export normally run automatically after the array finishes. Missing elements block merging. Existing identical merges are reused; for a different/incomplete merged directory, preserve it and pass `--destination NEW_DIRECTORY` to the merger. Node-local input/communication files are temporary; shared checkpoints, logs, checksums and actual environment records are retained. Use a fresh directory for a new protocol rather than resuming an old 50-dataset experiment.

## Measurements and outputs

Time includes estimator construction, interface-specific conversion and fitting; penalized time includes CV scoring and paths. Imports, data generation/loading, warm-up and final evaluation are excluded. Each dataset contributes its median of five measured times. Means, sample SD and MCSE use successful independent datasets, with each metric's actual nonmissing count; the 500 measured fits are not 500 independent replications.

Memory is sampled peak worker RSS over the measured fitting block, including runtime and resident inputs. It is one peak per dataset-method block. Sampling is every 5 ms, so short peaks may be missed; absent samples remain missing. Successful tasks must complete the requested fits, satisfy available native solver checks, have finite evaluation results and pass the independent scaled KKT criterion of 1e-5. Failures and warnings remain recorded.

Each run saves `raw_results.csv`, `fit_trials.csv`, seeds, source fingerprints and environment records. Slurm results are under the run's `merged/` directory. The original `tables/` directory retains diagnostic summaries; manuscript exports are in `public_tables/`:

| Files | Contents |
| --- | --- |
| `table1_main.tex`, `table2_main.tex` | Mean (SD) for time, peak RSS, test deviance and penalized-model support F1 |
| `table1_unpenalized.csv`, `table2_penalized.csv` | Means, SD, MCSE and effective/success counts |
| `table1_time_memory.tex`, `table2_time_memory.tex` | Time/memory-only panels |
| `time_*.csv`, `memory_*.csv` | Wide mean panels |
| `failures.csv`, `scenarios.csv`, `run_summary.json` | Failed tasks, designs and verified run metadata |

The exporter checks identities, completeness, source fingerprints, trial counts and numerical criteria without editing inputs. Main TeX tables need `booktabs` and `caption`. Support F1 is `2 * |S intersect S_hat| / (|S| + |S_hat|)`, using the configured coefficient threshold. Timing/RSS can vary with hardware and operating conditions. Generated data/results and the local site configuration are ignored by Git.

## Code checks

Control tests need Python 3.12 and the reduced check dependencies, with no R installation or archived results:

```bash
python3.12 -m venv .venv-checks
.venv-checks/bin/python -m pip install -r requirements-checks.txt
.venv-checks/bin/python test_protocol.py
.venv-checks/bin/python -m unittest test_slurm test_slurm_cpu_relaxed test_cv_merge_policy test_merge_diagnostics test_node_selection test_collect_slurm_audit test_export_tables
```

GitHub Actions runs these synthetic protocol/controller/export checks; they do not fit a new benchmark. The numbered legacy entry points delegate to the common runner. Restricted-language runs are exploratory and must not be pooled into a formal paired comparison.
