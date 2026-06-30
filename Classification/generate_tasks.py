"""Generate the task CSVs for the unified cross-language benchmark.

One shared AR(1) dataset file per (regime, n, p, seed) is generated once
(gen_data.py) and consumed by every Python and R model. Emits:

  tasks_data.csv  datasets to generate
  tasks_cpu.csv   Python CPU models  (dense + mixed)
  tasks_gpu.csv   Python GPU models  (dense)
  tasks_r.csv     R models           (dense + mixed)
"""
import csv, os

N_DENSE = [10_000, 50_000, 100_000]
P_DENSE = [50, 100, 500]

N_MIXED = [20_000, 50_000, 100_000]
P_MIXED = [100]

SEEDS = [1, 2, 3]

DATA_DIR = "data"

CPU_DENSE = [
    "sklearn_logit_lbfgs", "sklearn_logit_newton", "sklearn_logit_saga",
    "sklearn_linearsvc", "sklearn_rf",
    "skglm_logit",
    "xgb", "lgbm", "catboost",
    "torch_mlp", "tfp_glm",
]

CPU_MIXED = [
    "sklearn_logit_lbfgs", "sklearn_logit_newton", "sklearn_logit_saga",
    "sklearn_linearsvc",
    "xgb", "lgbm", "catboost",
    "tfdf_rf",
    "sklearn_rf", "hummingbird_rf", "hummingbird_xgb",
]

GPU_MODELS = ["xgb", "catboost", "cuml_logit", "torch_mlp"]

R_DENSE = [
    "glm_logit", "speedglm_logit",
    "glmnet_ridge", "glmnet_lasso",
    "liblinear_svm",
    "rf_randomForest", "rf_ranger",
    "r_xgboost", "r_lgbm", "r_catboost",
]

R_MIXED = list(R_DENSE)


def data_path(regime, n, p, seed):
    return os.path.join(DATA_DIR, f"{regime}_n{n}_p{p}_s{seed}.csv.gz")


def _grid(models, regime, ns, ps):
    rows = []
    for model in models:
        for n in ns:
            for p in ps:
                for seed in SEEDS:
                    rows.append(dict(regime=regime, model=model,
                                     n=n, p=p, density="", seed=seed,
                                     data_file=data_path(regime, n, p, seed)))
    return rows


def data_tasks():
    rows = []
    for regime, ns, ps in [("dense", N_DENSE, P_DENSE), ("mixed", N_MIXED, P_MIXED)]:
        for n in ns:
            for p in ps:
                for seed in SEEDS:
                    rows.append(dict(regime=regime, n=n, p=p, density="", seed=seed))
    return rows


def cpu_tasks():
    return _grid(CPU_DENSE, "dense", N_DENSE, P_DENSE) + \
           _grid(CPU_MIXED, "mixed", N_MIXED, P_MIXED)


def gpu_tasks():
    return _grid(GPU_MODELS, "dense", N_DENSE, P_DENSE)


def r_tasks():
    return _grid(R_DENSE, "dense", N_DENSE, P_DENSE) + \
           _grid(R_MIXED, "mixed", N_MIXED, P_MIXED)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({"task_id": i, **row})
    print(f"Wrote {len(rows):4d} tasks → {path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    data = data_tasks()
    write_csv("tasks_data.csv", data,
              ["task_id", "regime", "n", "p", "density", "seed"])

    model_fields = ["task_id", "regime", "model", "n", "p", "density", "seed", "data_file"]
    cpu = cpu_tasks(); write_csv("tasks_cpu.csv", cpu, model_fields)
    gpu = gpu_tasks(); write_csv("tasks_gpu.csv", gpu, model_fields)
    r   = r_tasks();   write_csv("tasks_r.csv",   r,   model_fields)

    print("\nArray bounds:")
    print(f"  sbatch --array=0-{len(data)-1}  submit_data_gen.sh")
    print(f"  sbatch --array=0-{len(cpu)-1}   submit_cpu.sh")
    print(f"  sbatch --array=0-{len(gpu)-1}   submit_gpu.sh   # add --gpu in run_task.py")
    print(f"  sbatch --array=0-{len(r)-1}     submit_r.sh")
