# Clustering Benchmarks

Benchmarking clustering algorithms across: Python (CPU), Python (GPU), and R.

## Overview

- **Goal**: Benchmark the cross-platform performance (runtime and accuracy) of specific clustering methods.
- **Data**: Synthetic blobs generated with `sklearn.datasets.make_blobs`, varying sample size and feature count.
- **Metrics**: Runtime (seconds), Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).

## Files

| File | Description |
| :--- | :--- |
| `generate_data.py` | Generates synthetic datasets (`.npz` format) |
| `benchmark_sklearn_cpu.py` | Runs clustering benchmarks using **scikit-learn** on CPU |
| `benchmark_cuml_gpu.py` | Runs clustering benchmarks using **RAPIDS cuML** on GPU |
| `benchmark_R_cpu.R` | Runs clustering benchmarks using **R** packages on CPU |
| `data/` | Input directory for generated datasets (`.npz` format) |
| `results/` | Output directory for benchmark CSV results |

## Algorithms

Five clustering algorithms are compared. Not all are available on every platform:

| Algorithm | Python (CPU) | Python (GPU) | R (CPU) |
| :--- | :---: | :---: | :---: |
| K-Means | ✓ | ✓ | ✓ |
| GMM | ✓ | — | ✓ |
| DBSCAN | ✓ | ✓ | ✓ |
| HDBSCAN | ✓ | ✓ | ✓ |
| Hierarchical Clustering (HC) | ✓ | ✓ | ✓ |

## Data Configurations

Each configuration generates **100 replicates** with **5 clusters**:

| Experiment | n (samples) | p (features) |
| :--- | :--- | :--- |
| Vary n | 1,000 / 5,000 / 10,000 | 50 |
| Vary p | 5,000 | 100 / 200 |

## Usage

**Step 1** — Generate the data (all outputs will be saved to the `data/` folder):

```bash
python generate_data.py
```

**Step 2** — Run the benchmarks:

```bash
# Python CPU
python benchmark_sklearn_cpu.py

# Python GPU (requires cuML + NVIDIA GPU + CUDA)
python benchmark_cuml_gpu.py

# R
Rscript benchmark_R_cpu.R
```

**Step 3** — Find the results in `results/` as CSV files (e.g., `python_kmeans_cpu_results.csv`, `r_hdbscan_results.csv`).

## Dependencies

**Python**: `numpy`, `pandas`, `scikit-learn`; additionally `cupy` and `cuml` for GPU benchmarks.

> Note for `cuML`: Installing `cuml` can be non-trivial due to strict hardware and driver alignments. We highly recommend using the official installation (https://github.com/rapidsai/cuml#installation). 

**R**: `reticulate`, `mclust`, `dbscan`, `aricode`, `jsonlite`.
