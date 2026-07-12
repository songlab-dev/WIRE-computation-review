# Generate simulation datasets for logistic univariate and additive tasks.
#
# Outputs (written to data/):
#   {task}_n{n}_b{b:03d}.csv  — one CSV per (task, n, bootstrap replicate)
#   tasks : logistic_global, logistic_local,
#           logistic_p01, logistic_p10, logistic_p50, logistic_p100
#   n     : 100, 500, 2000
#   B     : 500 replicates each
#
# Each CSV has columns: split, x / x1..xp, y, f0
#   split = "train" | "test"; f0 = true log-odds (LP) at test points; NA for train
#
# Run from the project root:
#   Rscript public/generate_logistic_data.R

set.seed(2024)

# Run from the project root directory.
DATA <- "data"
dir.create(DATA, showWarnings = FALSE, recursive = TRUE)

# ── True mean functions ───────────────────────────────────────────────────────

f_global <- function(x) sin(2 * pi * x) + 0.5 * x^2

f_local  <- function(x)
  3.0*exp(-150*(x-0.15)^2) - 2.0*exp(-120*(x-0.38)^2) +
  2.5*exp(-130*(x-0.72)^2) - 1.5*exp(-100*(x-0.88)^2) + 0.5*x

# 10-component library shared with the Gaussian scripts; x in [0, 1].
# f0 stored as the true linear predictor (log-odds) for LP-scale evaluation.
wood_bump <- function(x) 0.2*x^11*(10*(1-x))^6 + 10*(10*x)^3*(1-x)^10
COMP <- list(
  function(x) sin(2*pi*x),
  function(x) wood_bump(x),
  function(x) exp(2*x),
  function(x) (2*x - 1)^3,
  function(x) sin(4*pi*x),
  function(x) 40*x^2*(1-x)^2,
  function(x) abs(2*x - 1),
  function(x) log(1 + 5*x),
  function(x) 20*(x - 0.5)^2,
  function(x) x
)
# Components are recycled when p exceeds the library size, matching the
# recycling rule in benchmark_r.R / generate_benchmark_data.py.
f_comp <- function(X, p) {
  fns_p <- COMP[((seq_len(p) - 1L) %% length(COMP)) + 1L]
  Reduce(`+`, Map(function(f, j) f(X[, j]), fns_p, seq_len(p)))
}

# The raw component sum grows with p, which saturates the response (every
# y = 1 once p is large).  Standardise it to LP ~ N(0, 1.5^2) so the signal
# is comparable at every p — the same rule bench_logistic_p() uses in
# benchmark_r.R.  Calibration draws from its own seed and restores the main
# RNG stream so the simulation draws below are unaffected.
make_lp_fn <- function(p) {
  old <- if (exists(".Random.seed", .GlobalEnv))
    get(".Random.seed", .GlobalEnv) else NULL
  set.seed(1234L + p)
  raw <- f_comp(matrix(runif(20000 * p), nrow = 20000, ncol = p), p)
  mu  <- mean(raw)
  sdv <- sd(raw)
  if (!is.null(old)) assign(".Random.seed", old, envir = .GlobalEnv)
  function(X) 1.5 * (f_comp(X, p) - mu) / sdv
}

B      <- 500
n_test <- 1000

# ── Univariate logistic tasks (global / local) ────────────────────────────────

for (task in c("logistic_global", "logistic_local")) {
  f0 <- if (task == "logistic_global") f_global else f_local
  for (n in c(100, 500, 2000)) {
    n_new <- 0L
    for (b in seq_len(B)) {
      path <- sprintf("%s/%s_n%d_b%03d.csv", DATA, task, n, b)
      if (file.exists(path)) next
      x      <- runif(n);      lp   <- f0(x);      y  <- rbinom(n,      1, plogis(lp))
      x_test <- runif(n_test); lp_t <- f0(x_test); yt <- rbinom(n_test, 1, plogis(lp_t))
      write.csv(
        rbind(
          data.frame(split = "train", x = x,      y = y,  f0 = NA),
          data.frame(split = "test",  x = x_test, y = yt, f0 = lp_t)
        ),
        path, row.names = FALSE
      )
      n_new <- n_new + 1L
    }
    if (n_new == 0L) {
      cat(sprintf("  %s n=%d: all %d reps exist, skipping\n", task, n, B))
    } else {
      cat(sprintf("  %s n=%d: generated %d reps\n", task, n, n_new))
    }
  }
}

# ── Additive logistic tasks ───────────────────────────────────────────────────

for (p in c(1, 10, 50, 100)) {
  tag   <- sprintf("logistic_p%02d", p)
  lp_of <- make_lp_fn(p)   # calibrated once per p, as in bench_logistic_p()
  for (n in c(100, 500, 2000)) {
    n_new <- 0L
    for (b in seq_len(B)) {
      path <- sprintf("%s/%s_n%d_b%03d.csv", DATA, tag, n, b)
      if (file.exists(path)) next
      X  <- matrix(runif(n * p),      nrow = n,      ncol = p)
      Xt <- matrix(runif(n_test * p), nrow = n_test, ncol = p)
      colnames(X) <- colnames(Xt) <- paste0("x", seq_len(p))
      lp   <- lp_of(X)
      lp_t <- lp_of(Xt)
      y    <- rbinom(n,      1, plogis(lp))
      yt   <- rbinom(n_test, 1, plogis(lp_t))
      write.csv(
        rbind(
          cbind(split = "train", as.data.frame(X),  y = y,  f0 = NA),
          cbind(split = "test",  as.data.frame(Xt), y = yt, f0 = lp_t)
        ),
        path, row.names = FALSE
      )
      n_new <- n_new + 1L
    }
    if (n_new == 0L) {
      cat(sprintf("  %s n=%d: all %d reps exist, skipping\n", tag, n, B))
    } else {
      cat(sprintf("  %s n=%d: generated %d reps\n", tag, n, n_new))
    }
  }
}
