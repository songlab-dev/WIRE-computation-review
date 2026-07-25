# GLMM Simulation Framework

Simulation and comparison of generalized linear mixed model (GLMM) fitting methods under various configurations.

## Overview

This framework simulates logistic GLMM data and compares three fitting methods:
- **lme4**: Laplace approximation with nlminb optimizer
- **glmmTMB**: Template Model Builder with configurable optimizers
- **sjSDM**: Neural network-based approach (q=1 only)

## Simulation Setup

- **Sample structure**: 200 clusters, 50 observations per cluster
- **Fixed effects**: 50 total (15 active, coefficient 0.5)
- **Random effects**: Dimension q ∈ {1, 5, 15}
- **Covariance**: Identity or AR(1) with ρ ∈ {0.3, 0.7}
- **Replicates**: 100 per configuration
- **Total configurations**: 60 (q × ρ × method combinations)

## Quick Start

```bash
# Step 1: Generate simulation commands
cd code
Rscript 2_generate_commands.R --method lme4 --q 1,5 --output commands_lme4.txt

# Step 2: Run in parallel (replace with your HPC scheduler if needed)
cat commands_lme4.txt | xargs -I {} -P 10 bash -c '{}'

# Step 3: Generate summary table
Rscript 6_generate_summary_table.R \
  --results_dirs ../results/lme4_q1_q5,../results/glmmtmb_q15,../results/lme4_q15,../results/sjsdm_q1 \
  --output ../summary_table.tex
```

## Project Structure

```
glmm-simulation/
├── code/                           # Source code
│   ├── 1_glmm_simulation_core.R   # Shared library (data gen + metrics)
│   ├── 2_generate_commands.R      # Command generator
│   ├── 3_run_lme4.R               # lme4 runner
│   ├── 4_run_glmmtmb.R            # glmmTMB runner
│   ├── 5_run_sjsdm.R              # sjSDM runner (q=1 only)
│   ├── 6_generate_summary_table.R # Results aggregator
│   └── README.md                  # Detailed documentation
├── results/                        # Output results directory (generated when running)
├── tables/                         # Output tables (generated when running)
└── README.md                       # This file
```

## Execution Order

The sequential numbering shows the recommended workflow:

1. **1_glmm_simulation_core.R** — Shared library (source, don't execute)
2. **2_generate_commands.R** — Generate batch commands
3. **3_run_lme4.R** — Automatically called via generated commands
4. **4_run_glmmtmb.R** — Automatically called via generated commands
5. **5_run_sjsdm.R** — Automatically called via generated commands
6. **6_generate_summary_table.R** — Aggregate results and create table

## Metrics

Each simulation computes:
- **runtime_sec**: Wall-clock execution time
- **convergence_status**: "ok" or "failed"
- **beta_sup_norm_error**: max(|β̂ - β_true|) across fixed effects
- **sigma_sup_norm_error**: max(|Σ̂ - Σ_true|) across random effect covariance

## Output Format

Results are saved as CSV files with one row per simulation, containing all parameters and metrics.

Each row contains:
- Simulation parameters (method, q, rho, replicate ID)
- Computed metrics (runtime, convergence status, error norms)

## Configuration Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--method` | REQUIRED | lme4, glmmtmb, or sjsdm |
| `--q` | REQUIRED | Random effect dimension: 1, 5, or 15 |
| `--cov_structure` | identity | identity or ar1 |
| `--rho` | 0.5 | AR(1) correlation (-1 to 1) |
| `--replicate_id` | 1 | Replicate number (1-100) |
| `--output_dir` | results | Output directory for results |
| `--mc_sjsdm` | 10000 | MCMC samples for sjSDM |
| `--glmmtmb_iter_max` | 3000 | Iteration limit for glmmTMB |

## Requirements

### R Packages
```r
install.packages(c("lme4", "glmmTMB", "sjSDM"))
```

### Python (for sjSDM)
- PyTorch (automatically installed with sjSDM)

## Documentation

For detailed architecture, design decisions, and usage examples, see [code/README.md](code/README.md).

## License

MIT

## Citation

If you use this framework, please cite:
```
@misc{glmm-simulation-framework,
  author = {Your Name},
  title = {GLMM Simulation Framework},
  year = {2026},
  url = {https://github.com/yourusername/glmm-simulation}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
