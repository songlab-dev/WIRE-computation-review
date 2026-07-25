# GLMM Simulation Framework - Code Documentation

## File Overview

| File | Purpose |
|------|---------|
| `1_glmm_simulation_core.R` | Shared library: data generation, fitting, metrics |
| `2_generate_commands.R` | Generate batch commands for parallel execution |
| `3_run_lme4.R` | lme4 fitting method |
| `4_run_glmmtmb.R` | glmmTMB fitting method |
| `5_run_sjsdm.R` | sjSDM fitting method (q=1 only) |
| `6_generate_summary_table.R` | Aggregate results into summary statistics |

## Workflow

### Step 1: Generate Commands

```bash
cd code

# For q=1,5 with lme4
Rscript 2_generate_commands.R --method lme4 --q 1,5 --output commands_lme4.txt

# For q=15 with glmmTMB
Rscript 2_generate_commands.R --method glmmtmb --q 15 --output commands_glmmtmb.txt

# For q=1 with sjSDM
Rscript 2_generate_commands.R --method sjsdm --q 1 --output commands_sjsdm.txt
```

**Command count formula:** `q_values × (1 identity_config + 2 ar1_configs) × 100_replicates`

### Step 2: Run Simulations

```bash
# Run with parallel execution
cat commands_lme4.txt | xargs -I {} -P 10 bash -c '{}'
cat commands_glmmtmb.txt | xargs -I {} -P 10 bash -c '{}'
cat commands_sjsdm.txt | xargs -I {} -P 10 bash -c '{}'
```

### Step 3: Generate Summary Table

```bash
Rscript 6_generate_summary_table.R \
  --results_dirs ../results \
  --output ../summary_table.tex
```

## Key Components

### 1. Core Library (1_glmm_simulation_core.R)

**Functions:**
- `parse_args()` — Parse command-line arguments
- `simulate_dataset()` — Generate logistic GLMM data
- `split_dataset()` — Train/test split (cluster hold-out)
- `compute_metrics()` — Calculate 4 essential metrics
- `run_simulation()` — Main execution wrapper

**Design principle:** Shared core ensures consistent data generation and metrics across all methods.

### 2. Command Generator (2_generate_commands.R)

**Input parameters:**
- `--method` — lme4, glmmtmb, or sjsdm
- `--q` — Comma-separated dimension values (e.g., 1,5,15)
- `--output` — Output filename for commands
- `--replicates` — Number of replicates per config (default: 100)
- `--output_dir` — Output directory for results (default: results)

**Output:** Text file with one shell command per line

### 3. Method Runners (3_run_lme4.R, 4_run_glmmtmb.R, 5_run_sjsdm.R)

Each runner:
1. Sources `1_glmm_simulation_core.R`
2. Implements `fit_glmm_[method]()` function
3. Calls `run_simulation()` with method-specific fitting logic

**Do not call directly** — use `2_generate_commands.R` to generate proper commands.

### 4. Summary Aggregator (6_generate_summary_table.R)

**Input parameters:**
- `--results_dirs` — Comma-separated result directories
- `--output` — Output filename (LaTeX table)
- `--output_format` — tex (default) or csv

**Output:** Summary statistics table with mean/SD across replicates

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--q` | REQUIRED | Dimension: 1, 5, or 15 |
| `--cov_structure` | identity | identity or ar1 |
| `--rho` | varies | 0 for identity; 0.3, 0.7 for ar1 |
| `--replicate_id` | auto | Replicate number |
| `--output_dir` | results | Output directory |
| `--mc_sjsdm` | 10000 | MCMC samples for sjSDM |
| `--glmmtmb_iter_max` | 3000 | Iteration limit for glmmTMB |

## Metrics Calculated

1. **runtime_sec** — Wall-clock execution time
2. **convergence_status** — "ok" or "failed"
3. **beta_sup_norm_error** — max(|β̂ - β_true|)
4. **sigma_sup_norm_error** — max(|Σ̂ - Σ_true|)

## Output Files

**Result CSVs:** Each simulation creates one CSV file containing:
- Parameters (q, rho, method, replicate)
- Metrics (runtime, convergence, errors)

**Summary table:** Aggregated statistics (mean ± SD) across all replicates


## Configuration

All configuration via command-line arguments (no config files needed).

Default structure:
- 200 clusters × 50 observations per cluster
- 50 fixed effects (15 active)
- 100 replicates per configuration
- Fixed random-effect variance (σ²=1)
