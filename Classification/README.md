# Cross-language classification benchmark — replication code

A single unified simulation comparing R and Python training time, inference
time, memory, and AUROC across model families. One shared dataset is generated
per `(regime, n, p, seed)` using an AR(1) (ρ=0.5) data-generating process and is
loaded identically by every Python and R model, so timing differences reflect
the implementations rather than the data.

## Design

- **Regimes:** `dense` (all-numeric, `p ∈ {50,100,500}`, `n ∈ {10k,50k,100k}`)
  and `mixed` (10 categorical + numeric, `p=100`, `n ∈ {20k,50k,100k}`).
- **Seeds:** 1, 2, 3 (results average over seeds).
- **Families:** logistic (unpenalized / ridge / lasso), linear SVM, random
  forest, XGBoost, LightGBM, CatBoost, plus MLP and GLM variants; Python on CPU
  and GPU, R on CPU.
- One data-generating process for everything (no sparse regime).

## Files

| File | Role |
|------|------|
| `gen_data.py` | generate one dataset → `data/{regime}_n{n}_p{p}_s{seed}.csv.gz` |
| `generate_tasks.py` | build `tasks_data.csv`, `tasks_cpu.csv`, `tasks_gpu.csv`, `tasks_r.csv` |
| `run_task.py` → `bench_classification.py` | train one Python model, write one result row |
| `run_r_task.R` → `bench_r_models.R` | train one R model, write one result row |
| `merge_results.py` | concatenate per-task rows → `python_results.csv` / `r_results.csv` |
| `analyze_results.py` | summary + R-vs-Python comparison tables (`xlang_*.csv`) |
| `run_all.py` | run the whole pipeline on one machine |

Each task writes its own one-row file under `<out>_rows/` (atomic rename, no file
locks), so array jobs are safe to run concurrently; `merge_results.py` then
de-duplicates by `(run_id, hardware)`.

## Run on one machine

```bash
python run_all.py              # data → Python CPU → R → merge → analyze
python run_all.py --gpu        # also run the Python GPU stage
python run_all.py --max-tasks 3   # quick smoke test (3 tasks per stage)
```

Outputs land in `results/`: `python_results.csv`, `r_results.csv`,
`summary_by_group.csv`, and the `xlang_*.csv` comparison tables.

## Run on a SLURM cluster

Generate the task lists once, then submit each stage as an array (single-threaded
for fair timing: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`):

```bash
python generate_tasks.py        # prints the array bounds for each stage
# data:  run_task wrapper for gen_data.py over tasks_data.csv
# cpu:   run_task.py  --task-list tasks_cpu.csv --out results/python_results.csv
# gpu:   run_task.py  --task-list tasks_gpu.csv --out results/python_results.csv --gpu
# r:     run_r_task.R --task-list tasks_r.csv   --out results/r_results.csv
python merge_results.py results/python_results.csv results/r_results.csv
python analyze_results.py --py results/python_results.csv \
    --r results/r_results.csv --save --crosslang
```

## Requirements

- Python: numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, catboost,
  psutil; optional skglm, hummingbird-ml, tensorflow / tensorflow-probability /
  tensorflow-decision-forests, torch; GPU stage needs cupy + cuml + a CUDA torch.
- R: optparse, data.table, peakRAM, pROC, glmnet, LiblineaR, randomForest,
  ranger, speedglm, xgboost, lightgbm, catboost.
