# Generate simulation datasets for Gaussian univariate and additive tasks.
#
# Outputs (written to data/):
#   {task}_n{n}_b{b:03d}.csv  — one CSV per (task, n, bootstrap replicate)
#   tasks : global, local, additive_p01, additive_p03, additive_p05, additive_p10
#   n     : 100, 500, 2000
#   B     : 500 replicates each
#
# Each CSV has columns: split, x / x1..xp, y, f0
#   split = "train" | "test"; f0 = NA for train rows (true mean at test points)
#
# Run from the project root:
#   Rscript public/generate_data.R

set.seed(2024)

# Run from the project root directory.
DATA <- "data"
dir.create(DATA, showWarnings = FALSE, recursive = TRUE)

# ── True mean functions ───────────────────────────────────────────────────────

f_global <- function(x) sin(2 * pi * x) + 0.5 * x^2

f_local  <- function(x)
  3.0*exp(-150*(x-0.15)^2) - 2.0*exp(-120*(x-0.38)^2) +
  2.5*exp(-130*(x-0.72)^2) - 1.5*exp(-100*(x-0.88)^2) + 0.5*x

# 10-component library shared with the benchmark scripts; all x in [0, 1].
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
f_comp <- function(X, p)
  Reduce(`+`, Map(function(f, j) f(X[, j]), COMP[seq_len(p)], seq_len(p)))

# ── Univariate tasks (global / local) ─────────────────────────────────────────

B      <- 500
n_test <- 1000

for (task in c("global", "local")) {
  f0 <- if (task == "global") f_global else f_local
  for (n in c(100, 500, 2000)) {
    for (b in seq_len(B)) {
      path <- sprintf("%s/%s_n%d_b%03d.csv", DATA, task, n, b)
      if (file.exists(path)) next
      x      <- runif(n);      y      <- f0(x)      + rnorm(n)
      x_test <- runif(n_test); y_test <- f0(x_test) + rnorm(n_test)
      write.csv(
        rbind(
          data.frame(split = "train", x = x,      y = y,      f0 = NA),
          data.frame(split = "test",  x = x_test, y = y_test, f0 = f0(x_test))
        ),
        path, row.names = FALSE
      )
    }
    cat(sprintf("generated %s n=%d\n", task, n))
  }
}

# ── Additive Gaussian tasks ───────────────────────────────────────────────────

for (p in c(1, 3, 5, 10)) {
  tag <- sprintf("additive_p%02d", p)
  for (n in c(100, 500, 2000)) {
    for (b in seq_len(B)) {
      path <- sprintf("%s/%s_n%d_b%03d.csv", DATA, tag, n, b)
      if (file.exists(path)) next
      X  <- matrix(runif(n * p),      nrow = n,      ncol = p)
      Xt <- matrix(runif(n_test * p), nrow = n_test, ncol = p)
      colnames(X) <- colnames(Xt) <- paste0("x", seq_len(p))
      f0_tr <- f_comp(X, p);  y  <- f0_tr + rnorm(n)
      f0_te <- f_comp(Xt, p); yt <- f0_te + rnorm(n_test)
      write.csv(
        rbind(
          cbind(split = "train", as.data.frame(X),  y = y,  f0 = NA),
          cbind(split = "test",  as.data.frame(Xt), y = yt, f0 = f0_te)
        ),
        path, row.names = FALSE
      )
    }
    cat(sprintf("generated %s n=%d\n", tag, n))
  }
}
