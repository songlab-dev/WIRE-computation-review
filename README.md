# Cross-Language Benchmarking of Statistical Methods

A computational study benchmarking widely used statistical modeling implementations across Python and R.

## Overview

This repository contains code and analysis for a comprehensive cross-language benchmarking study comparing statistical modeling implementations across Python and R. We evaluate:

- **Generalized Linear Models (GLM)**: Linear regression and logistic models
- **Generalized Additive Models (GAM)**: Smooth function estimation
- **Generalized Linear Mixed Models (GLMM)**: Models with random effects
- **Survival Analysis**: Time-to-event modeling
- **Classification**: Supervised learning classification methods
- **Clustering**: Unsupervised grouping methods

Each directory is a self-contained analysis pipeline, containing simulation code (data generation, experiments) and scripts to reproduce the paper's figures and tables.

## Project Structure

One top-level directory per model class:

```text
Computation-toolbox-review/
├── GLM/              Generalized linear models
├── GAM/              Generalized additive models
├── GLMM/             Generalized linear mixed models
├── Survival/         Time-to-event models
├── Classification/   Supervised classification
└── Clustering/       Unsupervised clustering
```

Each directory is independent — no shared code or execution order between them — and follows the same four-stage layout:

1. Data generation
2. R implementations
3. Python implementations
4. Aggregation into paper figures/tables

**`README.md` inside each directory** documents the methods compared, dependencies, and the exact commands to reproduce that section of the paper.

## Citation

[Citation details to be added upon publication]
