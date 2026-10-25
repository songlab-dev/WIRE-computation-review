# Methods

## Data and seeds

The two settings are a sparse Cox model with 500 covariates and a smooth
nonlinear survival model with 50 covariates. Both use Gaussian AR(1) covariates
with correlation 0.3 and Weibull cumulative baseline hazard H0(t) = 0.1 t^1.5.

In the sparse model, the first 50 coefficients alternate in sign, with
magnitudes 0.93^(j−1). In the nonlinear model,

```text
eta = 0.4 X1 + 0.8 sin(X2) + 0.8 (X3² − 1)
      + 0.7 tanh(X4 + X5) + 0.5 exp(−X6²).
```

The risk score is standardized within each dataset. Independent uniform
censoring targets 30%; its upper limit is fixed using a separate 50,000-subject
calibration sample for each setting.

There are 100 datasets per setting at n = 500, 1000, and 1500. Python's
NumPy default_rng uses seeds 20226091–20226690, in setting, sample-size, then
replication order. Calibration seeds are 20226089 and 20226090. Training seeds
2026091–2026095 cycle across replications, with one fit per implementation per
dataset.

All implementations use the same ordered 60% training, 20% validation, and
20% test observations. The R helper records survivalmodels' validation sampling
rule; its unmodified fitting interface replays that split. Python reads the same
saved observations. The two languages use the same seed numbers without assuming
identical random-number streams.

## Models

The sparse setting includes penalized Cox, RSF, boosting, and DeepSurv.
The nonlinear setting replaces penalized Cox with Cox PH.
[parameters.csv](parameters.csv) records the full comparison.

- Cox PH uses Efron coefficient ties in both languages. Native baseline
  estimates still differ: Efron-adjusted in R and Breslow in Python.
- Penalized Cox uses the same 100 L1 penalties, training standardization,
  validation-only Breslow partial likelihood, and common training baseline.
  Native solver stopping criteria differ.
- RSF uses 200 trees, floor(sqrt(p)) candidate variables, log-rank splits,
  exhaustive candidate search, and n_train bootstrap draws with replacement.
  R nodesize=15 and Python min_samples_split=30, min_samples_leaf=1 bring the
  parent constraints closer. They remain unequal because R counts bootstrap
  draws and Python counts unique positive-weight observations. Cut-point
  representation and native OOB work also differ.
- Boosting uses 300 trees, learning rate 0.05, all training observations, and
  minimum child size 15. R allows three splits; Python uses depth 3 and at most
  four leaves. Their native terminal-node updates differ.
- DeepSurv uses a 32-unit ReLU layer, Adam at 0.001, batch size 150, at most
  150 epochs, and early-stopping patience 20. Dropout and batch normalization
  are disabled. Both interfaces use the same backend and ordered input data.

## Evaluation and resource measurement

All test predictions are evaluated by one Python routine. IBS uses 100 equally
spaced times between the training follow-up 10th and 90th percentiles, training
censoring Kaplan–Meier estimates, and all test observations. Harrell's C-index
uses risk-tie tolerance 1e−8 and minus the normalized survival integral as the
common risk score.

Each fit runs in a fresh process. Wall-clock time includes preprocessing,
fitting, necessary parameter selection, early stopping, and baseline estimation.
It excludes imports, CSV loading, test prediction, and evaluation.

Peak RSS is Linux VmHWM at the end of fitting, before test prediction. It
includes startup, imports, data loading, and fitting. R's embedded Python backend
shares the same process; computational child processes are checked. Raw KiB are
converted to MiB by dividing by 1024.

Tables report mean and sample SD across 100 datasets per configuration.
Failures retain their reason and attempt history and are never counted as
successful fits. Runtime and memory depend on the recorded environment;
reruns need not reproduce those measurements exactly.
