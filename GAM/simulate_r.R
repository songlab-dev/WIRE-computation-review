# Run Gaussian simulation (univariate and additive) using R.
#
# For each (task, n, bootstrap replicate b), reads the pre-generated CSV from
# data/ and fits every R method; writes one summary CSV per (task, n) to results/.
#
# Outputs (results/):
#   r_{task}_n{n}.csv   task in {global, local, additive_p01, ..., additive_p10}
#
# Prerequisites:
#   Rscript public/install_packages.R
#   Rscript public/generate_data.R
#
# Run from the project root:
#   Rscript public/simulate_r.R

suppressPackageStartupMessages({
  library(mgcv)
  library(gamlss)
})

# Run from the project root directory.
DATA    <- "data"
RESULTS <- "results"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

# ── Metric helpers ────────────────────────────────────────────────────────────

metrics <- function(p, se, f0, y_test) c(
  mise     = mean((p - f0)^2),
  test_mse = mean((y_test - p)^2),
  coverage = mean(abs(p - f0) <= 1.96 * se)
)

na_metrics <- c(mise = NA_real_, test_mse = NA_real_, coverage = NA_real_)
try_fit    <- function(expr) tryCatch(expr, error = function(e) NULL)

try_metrics <- function(m, df_t, f0, y_test) {
  if (is.null(m)) return(na_metrics)
  tryCatch({
    p <- predict(m, df_t, se.fit = TRUE)
    metrics(p$fit, p$se.fit, f0, y_test)
  }, error = function(e) na_metrics)
}

# gamlss::predict.gamlss needs the original training frame via `data=`
try_metrics_gamlss <- function(m, df, df_t, f0, y_test) {
  if (is.null(m)) return(na_metrics)
  tryCatch({
    p <- predict(m, what = "mu", newdata = df_t, type = "response",
                 se.fit = TRUE, data = df)
    fit <- if (is.list(p)) p$fit else p
    se  <- if (is.list(p)) p$se.fit else rep(NA_real_, length(fit))
    metrics(fit, se, f0, y_test)
  }, error = function(e) na_metrics)
}

GAMLSS_CTRL <- gamlss.control(trace = FALSE)

# ── Per-replicate fitters ─────────────────────────────────────────────────────

run_one_univariate <- function(d, task) {
  train  <- d[d$split == "train", ]; test <- d[d$split == "test", ]
  x      <- train$x;  y      <- train$y
  x_test <- test$x;   y_test <- test$y;  f0 <- test$f0
  df     <- data.frame(x = x, y = y);   df_t <- data.frame(x = x_test)

  # Cross-language comparisons: GCV, basis varies
  m_cr_gcv <- try_fit(gam(y ~ s(x, bs = "cr"), data = df, method = "GCV.Cp"))
  m_ps_gcv <- try_fit(gam(y ~ s(x, bs = "ps"), data = df, method = "GCV.Cp"))
  m_bs_gcv <- try_fit(gam(y ~ s(x, bs = "bs"), data = df, method = "GCV.Cp"))

  # Within-R algorithm comparison: GCV, basis = cr
  m_bam <- try_fit(bam(y ~ s(x, bs = "cr"), data = df, method = "GCV.Cp"))

  # P-splines group (within-R): gamlss with pb() — local ML for SP
  m_gamdist <- try_fit(gamlss(y ~ pb(x), data = df, family = NO(),
                              control = GAMLSS_CTRL))

  res <- list(
    gam_cr_gcv    = try_metrics(m_cr_gcv, df_t, f0, y_test),
    gam_ps_gcv    = try_metrics(m_ps_gcv, df_t, f0, y_test),
    gam_bs_gcv    = try_metrics(m_bs_gcv, df_t, f0, y_test),
    bam_cr_gcv    = try_metrics(m_bam,    df_t, f0, y_test),
    gamdist_ps_ml = try_metrics_gamlss(m_gamdist, df, df_t, f0, y_test)
  )

  # Cyclic cubic: global smooth only
  if (task == "global") {
    m_cc_gcv       <- try_fit(gam(y ~ s(x, bs = "cc"), data = df, method = "GCV.Cp"))
    res$gam_cc_gcv <- try_metrics(m_cc_gcv, df_t, f0, y_test)
  }

  res
}

run_one_additive <- function(d) {
  xcols  <- grep("^x[0-9]+$", names(d), value = TRUE)
  train  <- d[d$split == "train", ]; test <- d[d$split == "test", ]
  df     <- train[, c(xcols, "y")]
  df_t   <- test[,  xcols, drop = FALSE]
  y_test <- test$y; f0 <- test$f0

  fmla_cr <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='cr')", xcols), collapse = " + ")))
  fmla_ps <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='ps')", xcols), collapse = " + ")))
  fmla_bs <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='bs')", xcols), collapse = " + ")))
  fmla_pb <- as.formula(paste("y ~", paste(sprintf("pb(%s)",         xcols), collapse = " + ")))

  m_cr_gcv  <- try_fit(gam(fmla_cr, data = df, method = "GCV.Cp"))
  m_ps_gcv  <- try_fit(gam(fmla_ps, data = df, method = "GCV.Cp"))
  m_bs_gcv  <- try_fit(gam(fmla_bs, data = df, method = "GCV.Cp"))
  m_bam     <- try_fit(bam(fmla_cr, data = df, method = "GCV.Cp"))
  m_gamdist <- try_fit(gamlss(fmla_pb, data = df, family = NO(), control = GAMLSS_CTRL))

  list(
    gam_cr_gcv    = try_metrics(m_cr_gcv, df_t, f0, y_test),
    gam_ps_gcv    = try_metrics(m_ps_gcv, df_t, f0, y_test),
    gam_bs_gcv    = try_metrics(m_bs_gcv, df_t, f0, y_test),
    bam_cr_gcv    = try_metrics(m_bam,    df_t, f0, y_test),
    gamdist_ps_ml = try_metrics_gamlss(m_gamdist, df, df_t, f0, y_test)
  )
}

run_one <- function(path, task) {
  d <- read.csv(path)
  if ("x1" %in% names(d)) run_one_additive(d) else run_one_univariate(d, task)
}

run_mc <- function(task, n) {
  paths <- sort(Sys.glob(sprintf("%s/%s_n%d_b*.csv", DATA, task, n)))
  reps  <- lapply(paths, run_one, task = task)
  out   <- lapply(names(reps[[1]]), function(m)
    colMeans(do.call(rbind, lapply(reps, `[[`, m)), na.rm = TRUE))
  as.data.frame(do.call(rbind, out), row.names = names(reps[[1]]))
}

# ── Run ───────────────────────────────────────────────────────────────────────

for (task in c("global", "local")) {
  cat(sprintf("\n=== %s ===\n", task))
  for (n in c(100, 500, 2000)) {
    out <- sprintf("%s/r_%s_n%d.csv", RESULTS, task, n)
    if (file.exists(out)) {
      cat(sprintf("  n=%d: skipping (already done)\n", n)); next
    }
    cat(sprintf("\n%s n = %d\n", task, n))
    res <- run_mc(task, n)
    print(round(res, 3))
    write.csv(cbind(method = rownames(res), task = task, n = n, res),
              out, row.names = FALSE)
  }
}

cat("\n=== additive ===\n")
for (p in c(1, 3, 5, 10)) {
  subtask <- sprintf("additive_p%02d", p)
  for (n in c(100, 500, 2000)) {
    out <- sprintf("%s/r_%s_n%d.csv", RESULTS, subtask, n)
    if (file.exists(out)) {
      cat(sprintf("  %s n=%d: skipping (already done)\n", subtask, n)); next
    }
    cat(sprintf("\n%s n = %d\n", subtask, n))
    res <- run_mc(subtask, n)
    print(round(res, 3))
    write.csv(cbind(method = rownames(res), task = subtask, n = n, res),
              out, row.names = FALSE)
  }
}
