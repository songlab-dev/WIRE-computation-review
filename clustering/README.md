# Clustering Benchmark Suite

Benchmarking code for comparing clustering algorithms across **R** and **Python** (CPU via `scikit-learn`, GPU via `cuML`).

## Methods

| Method | R Implementation | Python (CPU) | Python (GPU) |
|--------|-----------------|-------------|-------------|
| K-Means | `stats::kmeans` | `sklearn.cluster.KMeans` | `cuml.cluster.KMeans` |
| GMM | `mclust::Mclust` | `sklearn.mixture.GaussianMixture` | — |
| HDBSCAN | `dbscan::hdbscan` | `sklearn.cluster.HDBSCAN` | `cuml.cluster.HDBSCAN` |
| DBSCAN | `dbscan::dbscan` | `sklearn.cluster.DBSCAN` | `cuml.cluster.DBSCAN` |
| Hierarchical Clustering | `stats::hclust` | `sklearn.cluster.AgglomerativeClustering` | `cuml.cluster.AgglomerativeClustering` |

## Metrics

- **Runtime** (seconds)
- **Adjusted Rand Index (ARI)** — pairwise agreement with ground truth
- **Normalized Mutual Information (NMI)** — information-theoretic agreement with ground truth

## Project Structure

```
wire/
├── generate_data.py          # Data generation (run FIRST)
├── data/
│   ├── config.json           # Shared experiment parameters
│   ├── replicate_0.npz       # Pre-generated datasets
│   ├── replicate_1.npz
│   └── ...
├── python_kmeans.py          # Python K-Means benchmark (sklearn + cuML)
├── python_gmm.py             # Python GMM benchmark (sklearn only)
├── python_hdbscan.py         # Python HDBSCAN benchmark (sklearn + cuML)
├── python_dbscan.py          # Python DBSCAN benchmark (sklearn + cuML)
├── python_hc.py              # Python HC benchmark (sklearn + cuML)
├── R_kmeans.R                # R K-Means benchmark
├── R_gmm.R                   # R GMM benchmark
├── R_hdbscan.R               # R HDBSCAN benchmark
├── R_dbscan.R                # R DBSCAN benchmark
├── R_hc.R                    # R HC benchmark
├── summarize_results.ipynb   # Summary notebook (mean, std, LaTeX export)
└── README.md
```

## Quick Start

### 1. Generate Data

Edit experiment parameters in `generate_data.py` (lines 19–22), then run:

```bash
python generate_data.py
```

This creates `data/config.json` and `.npz` files in `data/`. All benchmark scripts load parameters from `config.json`, so you only need to change them in one place.

### 2. Run Python Benchmarks (GPU node)

Requires an NVIDIA GPU with RAPIDS/cuML installed.

```bash
python python_kmeans.py
python python_gmm.py
python python_hdbscan.py
python python_dbscan.py
python python_hc.py
```

> **Note:** `python_gmm.py` is CPU-only (sklearn) since cuML does not support GMM.

### 3. Run R Benchmarks

```r
source("R_kmeans.R")
source("R_gmm.R")
source("R_hdbscan.R")
source("R_dbscan.R")
source("R_hc.R")
```

### 4. Results

Each script saves incremental results to a CSV file after every replicate:

| Script | Output File |
|--------|------------|
| `python_kmeans.py` | `python_kmeans_results.csv` |
| `python_gmm.py` | `python_gmm_results.csv` |
| `python_hdbscan.py` | `python_hdbscan_results.csv` |
| `python_dbscan.py` | `python_dbscan_results.csv` |
| `python_hc.py` | `python_hc_results.csv` |
| `R_kmeans.R` | `R_kmeans_results.csv` |
| `R_gmm.R` | `R_gmm_results.csv` |
| `R_hdbscan.R` | `R_hdbscan_results.csv` |
| `R_dbscan.R` | `R_dbscan_results.csv` |
| `R_hc.R` | `R_hc_results.csv` |

### 5. Summarize Results

After all benchmarks complete, open `summarize_results.ipynb` to:
- Compute **mean (std)** across replicates for each method and implementation
- Generate a **GPU speedup** table
- Export a **LaTeX table** for paper inclusion
- Save a consolidated `summary_results.csv`

## Data Generating Process

Synthetic data is generated using `sklearn.datasets.make_blobs` with:
- **Cluster standard deviations**: `np.linspace(1, n_clusters * 2, n_clusters)` — creates clusters of varying tightness
- **Center box**: `(-50.0, 50.0)` — range for random cluster center placement
- **Seed**: Each replicate uses `seed=i` for reproducibility

## Dependencies

### Python
- `numpy`, `pandas`, `scikit-learn` (≥1.3 for `HDBSCAN`)
- `cuml`, `cudf`, `cupy` (GPU benchmarks only; requires NVIDIA GPU + RAPIDS)

### R
- `reticulate` — Python interop for loading `.npz` files via numpy
- `mclust` — `adjustedRandIndex()`
- `aricode` — `NMI()`
- `dbscan` — DBSCAN and HDBSCAN (used in `R_dbscan.R` and `R_hdbscan.R`)
- `mclust` — GMM via `Mclust()` and `adjustedRandIndex()` (used in all R scripts)
- `jsonlite` — loading `config.json`

Install R packages:
```r
install.packages(c("reticulate", "mclust", "aricode", "dbscan", "jsonlite"))
```

## Scalability Notes

| Method | Time Complexity | Memory | Practical Limit |
|--------|----------------|--------|----------------|
| K-Means | O(n·p·k) | O(n·p) | Scales well to large n and p |
| HDBSCAN | O(n·log(n)) | O(n²) | Better than DBSCAN for varying density; memory-limited by mutual reachability graph |
| DBSCAN | O(n²·p) | O(n²) | Curse of dimensionality in high p; n² distances |
| HC | O(n²·p) | O(n²) | Distance matrix requires ~80GB for n=100K |
