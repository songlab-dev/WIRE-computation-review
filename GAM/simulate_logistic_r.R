# Run logistic simulation (univariate and additive) using R.
#
# For each (task, n, bootstrap replicate b), reads the pre-generated CSV from
# data/ and fits every R method; writes one summary CSV per (task, n) to results/.
#
# Outputs (results/):
#   r_{task}_n{n}.csv
#   task in {logistic_global, logistic_local, logistic_p01, ..., logistic_p10}
#
# Prerequisites:
#   Rscript public/install_packages.R
#   Rscript public/generate_logistic_data.R
#
# Run from the project root:
#   Rscript public/simulate_logistic_r.R

suppressPackageStartupMessages({
  library(mgcv)
  library(gamlss)
})

# Run from the project root directory.
DATA    <- "data"
RESULTS <- "results"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

# ── Metric helpers ────────────────────────────────────────────────────────────

# All metrics on the linear-predictor (log-odds) scale:
#   mise     — MSE between predicted LP and true LP
#   brier    — Brier score: mean((y - p_hat)^2)
#   coverage — nominal 95% CI coverage on the LP scale
logistic_metrics <- function(lp_hat, se_lp, f0_lp, y_test) c(
  mise     = mean((lp_hat - f0_lp)^2),
  brier    = mean((as.numeric(y_test) - plogis(lp_hat))^2),
  coverage = mean(abs(lp_hat - f0_lp) <= 1.96 * se_lp)
)

na_logistic <- c(mise = NA_real_, brier = NA_real_, coverage = NA_real_)

# try_fit: evaluate expr with a hard wall-clock cap; increments env$timeout_count
# on each timeout so run_mc can skip the scenario once the threshold is reached.
FIT_TIMEOUT_S <- 3600
MAX_TIMEOUTS  <- 3

try_fit <- function(expr, env, timeout_s = FIT_TIMEOUT_S) {
  setTimeLimit(elapsed = timeout_s, transient = FALSE)
  on.exit(setTimeLimit(elapsed = Inf, transient = FALSE))
  tryCatch(expr, error = function(e) {
    if (grepl("elapsed time limit", conditionMessage(e), ignore.case = TRUE)) {
      env$timeout_count <- env$timeout_count + 1L
      message(sprintf("    [timeout > %ds, count=%d/%d]",
                      timeout_s, env$timeout_count, MAX_TIMEOUTS))
    }
    NULL
  })
}

GAMLSS_CTRL <- gamlss.control(trace = FALSE)

# mgcv: predict(type="link") returns LP with LP-scale SE.
try_logistic_metrics <- function(m, df_t, f0_lp, y_test) {
  if (is.null(m)) return(na_logistic)
  tryCatch({
    p <- predict(m, df_t, type = "link", se.fit = TRUE)
    logistic_metrics(p$fit, p$se.fit, f0_lp, y_test)
  }, error = function(e) na_logistic)
}

# gamlss BI(): predict returns probabilities; no LP-scale SE, so coverage = NA.
try_logistic_metrics_gamlss <- function(m, df, df_t, f0_lp, y_test) {
  if (is.null(m)) return(na_logistic)
  tryCatch({
    p_raw <- predict(m, what = "mu", newdata = df_t, type = "response", data = df)
    p_hat <- pmin(pmax(as.numeric(if (is.list(p_raw)) p_raw$fit else p_raw), 1e-8), 1 - 1e-8)
    lp_hat <- qlogis(p_hat)
    c(mise     = mean((lp_hat - f0_lp)^2),
      brier    = mean((as.numeric(y_test) - p_hat)^2),
      coverage = NA_real_)
  }, error = function(e) na_logistic)
}

# ── Per-replicate fitters ─────────────────────────────────────────────────────

run_one_logistic_univariate <- function(d, task, env) {
  train  <- d[d$split == "train", ]; test <- d[d$split == "test", ]
  df     <- data.frame(x = train$x, y = train$y)
  df_t   <- data.frame(x = test$x)
  y_test <- test$y;  f0_lp <- test$f0

  m_cr_gcv  <- try_fit(gam(y ~ s(x, bs = "cr"), data = df, family = binomial(), method = "GCV.Cp"), env)
  m_ps_gcv  <- try_fit(gam(y ~ s(x, bs = "ps"), data = df, family = binomial(), method = "GCV.Cp"), env)
  m_bs_gcv  <- try_fit(gam(y ~ s(x, bs = "bs"), data = df, family = binomial(), method = "GCV.Cp"), env)
  m_bam     <- try_fit(bam(y ~ s(x, bs = "cr"), data = df, family = binomial(), method = "GCV.Cp"), env)
  m_gamdist <- try_fit(gamlss(y ~ pb(x), data = df, family = BI(), control = GAMLSS_CTRL), env)

  res <- list(
    gam_cr_gcv    = try_logistic_metrics(m_cr_gcv,  df_t, f0_lp, y_test),
    gam_ps_gcv    = try_logistic_metrics(m_ps_gcv,  df_t, f0_lp, y_test),
    gam_bs_gcv    = try_logistic_metrics(m_bs_gcv,  df_t, f0_lp, y_test),
    bam_cr_gcv    = try_logistic_metrics(m_bam,     df_t, f0_lp, y_test),
    gamdist_ps_ml = try_logistic_metrics_gamlss(m_gamdist, df, df_t, f0_lp, y_test)
  )

  if (task == "logistic_global") {
    m_cc_gcv       <- try_fit(gam(y ~ s(x, bs = "cc"), data = df, family = binomial(), method = "GCV.Cp"), env)
    res$gam_cc_gcv <- try_logistic_metrics(m_cc_gcv, df_t, f0_lp, y_test)
  }

  res
}

run_one_logistic_additive <- function(d, env) {
  xcols  <- grep("^x[0-9]+$", names(d), value = TRUE)
  train  <- d[d$split == "train", ]; test <- d[d$split == "test", ]
  df     <- train[, c(xcols, "y")];  df_t <- test[, xcols, drop = FALSE]
  y_test <- test$y;  f0_lp <- test$f0

  fmla_cr <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='cr')", xcols), collapse = " + ")))
  fmla_ps <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='ps')", xcols), collapse = " + ")))
  fmla_bs <- as.formula(paste("y ~", paste(sprintf("s(%s,bs='bs')", xcols), collapse = " + ")))
  fmla_pb <- as.formula(paste("y ~", paste(sprintf("pb(%s)",         xcols), collapse = " + ")))

  m_cr_gcv  <- try_fit(gam(fmla_cr, data = df, family = binomial(), method = "GCV.Cp"), env)
  m_ps_gcv  <- try_fit(gam(fmla_ps, data = df, family = binomial(), method = "GCV.Cp"), env)
  m_bs_gcv  <- try_fit(gam(fmla_bs, data = df, family = binomial(), method = "GCV.Cp"), env)
  m_bam     <- try_fit(bam(fmla_cr, data = df, family = binomial(), method = "GCV.Cp"), env)
  m_gamdist <- try_fit(gamlss(fmla_pb, data = df, family = BI(), control = GAMLSS_CTRL), env)

  list(
    gam_cr_gcv    = try_logistic_metrics(m_cr_gcv,  df_t, f0_lp, y_test),
    gam_ps_gcv    = try_logistic_metrics(m_ps_gcv,  df_t, f0_lp, y_test),
    gam_bs_gcv    = try_logistic_metrics(m_bs_gcv,  df_t, f0_lp, y_test),
    bam_cr_gcv    = try_logistic_metrics(m_bam,     df_t, f0_lp, y_test),
    gamdist_ps_ml = try_logistic_metrics_gamlss(m_gamdist, df, df_t, f0_lp, y_test)
  )
}

run_one <- function(path, task, env) {
  d <- read.csv(path)
  if ("x1" %in% names(d)) run_one_logistic_additive(d, env) else run_one_logistic_univariate(d, task, env)
}

CHUNK_DIR <- file.path(RESULTS, "sim_chunks")
dir.create(CHUNK_DIR, showWarnings = FALSE, recursive = TRUE)

run_mc <- function(tag, n) {
  paths <- sort(Sys.glob(sprintf("%s/%s_n%d_b*.csv", DATA, tag, n)))
  env   <- new.env(parent = emptyenv())
  env$timeout_count <- 0L

  for (path in paths) {
    b     <- sub(".*_b([0-9]+)\\.csv$", "\\1", basename(path))
    chunk <- file.path(CHUNK_DIR, sprintf("r_%s_n%d_b%s.csv", tag, n, b))
    if (file.exists(chunk)) next

    if (env$timeout_count > MAX_TIMEOUTS) {
      cat(sprintf("    >%d timeouts — skipping remaining reps\n", MAX_TIMEOUTS))
      break
    }

    res    <- run_one(path, tag, env)
    rep_df <- as.data.frame(do.call(rbind, res))
    rep_df <- cbind(method = rownames(rep_df), rep_df)
    rownames(rep_df) <- NULL
    write.csv(rep_df, chunk, row.names = FALSE)
  }

  all_chunks <- list.files(CHUNK_DIR,
    pattern = sprintf("^r_%s_n%d_b[0-9]+\\.csv$", tag, n),
    full.names = TRUE)
  if (length(all_chunks) == 0) return(NULL)

  cat(sprintf("    averaging %d rep chunks\n", length(all_chunks)))
  all_reps    <- lapply(all_chunks, read.csv)
  methods     <- all_reps[[1]]$method
  metric_cols <- setdiff(names(all_reps[[1]]), "method")

  result <- as.data.frame(
    sapply(metric_cols, function(col) {
      mat <- sapply(all_reps, function(d) d[[col]])
      rowMeans(mat, na.rm = TRUE)
    })
  )
  rownames(result) <- methods
  result
}

# ── Run ───────────────────────────────────────────────────────────────────────

cat("\n=== logistic ===\n")
for (task in c("logistic_global", "logistic_local")) {
  for (n in c(100, 500, 2000)) {
    out <- sprintf("%s/r_%s_n%d.csv", RESULTS, task, n)
    if (file.exists(out)) {
      cat(sprintf("  %s n=%d: skipping (already done)\n", task, n)); next
    }
    cat(sprintf("\n%s n = %d\n", task, n))
    res <- run_mc(task, n)
    if (is.null(res)) { cat("  no reps completed\n"); next }
    print(round(res, 3))
    write.csv(cbind(method = rownames(res), task = task, n = n, res),
              out, row.names = FALSE)
  }
}

for (p in c(1, 10, 50, 100)) {
  tag <- sprintf("logistic_p%02d", p)
  for (n in c(100, 500, 2000)) {
    out <- sprintf("%s/r_%s_n%d.csv", RESULTS, tag, n)
    if (file.exists(out)) {
      cat(sprintf("  %s n=%d: skipping (already done)\n", tag, n)); next
    }
    cat(sprintf("\n%s n = %d\n", tag, n))
    res <- run_mc(tag, n)
    if (is.null(res)) { cat("  no reps completed\n"); next }
    print(round(res, 3))
    write.csv(cbind(method = rownames(res), task = tag, n = n, res),
              out, row.names = FALSE)
  }
}
