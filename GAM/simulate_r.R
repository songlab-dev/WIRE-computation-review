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
# GAM_RESULTS / GAM_MAX_REPS: pilot overrides (separate output dir, fewer reps).
RESULTS <- Sys.getenv("GAM_RESULTS", "results")
MAX_REPS <- as.integer(Sys.getenv("GAM_MAX_REPS", "0"))   # 0 = all
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

# ── Metric helpers ────────────────────────────────────────────────────────────

metrics <- function(p, se, f0, y_test) c(
  mise     = mean((p - f0)^2),
  test_mse = mean((y_test - p)^2),
  coverage = mean(abs(p - f0) <= 1.96 * se)
)

na_metrics <- c(mise = NA_real_, test_mse = NA_real_, coverage = NA_real_)

# try_fit: evaluate expr (the fit for method `name`) under a hard wall-clock
# cap; NULL on timeout or error. After MAX_METHOD_TIMEOUTS timeouts in this job
# the method is skipped (expr is never evaluated) and recorded as NA.
FIT_TIMEOUT_S       <- 3600
MAX_METHOD_TIMEOUTS <- 5L
TIMEOUTS            <- new.env()

try_fit <- function(expr, name, timeout_s = FIT_TIMEOUT_S) {
  n_to <- get0(name, envir = TIMEOUTS, ifnotfound = 0L)
  if (n_to >= MAX_METHOD_TIMEOUTS) return(NULL)
  setTimeLimit(elapsed = timeout_s, transient = FALSE)
  on.exit(setTimeLimit(elapsed = Inf, transient = FALSE))
  tryCatch(expr, error = function(e) {
    if (grepl("elapsed time limit", conditionMessage(e), ignore.case = TRUE)) {
      assign(name, n_to + 1L, envir = TIMEOUTS)
      message(sprintf("    [%s timeout > %ds (%d/%d)]",
                      name, timeout_s, n_to + 1L, MAX_METHOD_TIMEOUTS))
    }
    NULL
  })
}

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
  m_cr_gcv <- try_fit(gam(y ~ s(x, bs = "cr"), data = df, method = "GCV.Cp"), "gam_cr_gcv")
  m_ps_gcv <- try_fit(gam(y ~ s(x, bs = "ps"), data = df, method = "GCV.Cp"), "gam_ps_gcv")
  m_bs_gcv <- try_fit(gam(y ~ s(x, bs = "bs"), data = df, method = "GCV.Cp"), "gam_bs_gcv")

  # Within-R algorithm comparison: GCV, basis = cr
  m_bam <- try_fit(bam(y ~ s(x, bs = "cr"), data = df, method = "GCV.Cp"), "bam_cr_gcv")

  # P-splines group (within-R): gamlss with pb() — local ML for SP
  m_gamlss <- try_fit(gamlss(y ~ pb(x), data = df, family = NO(),
                              control = GAMLSS_CTRL), "gamlss_ps_ml")

  res <- list(
    gam_cr_gcv    = try_metrics(m_cr_gcv, df_t, f0, y_test),
    gam_ps_gcv    = try_metrics(m_ps_gcv, df_t, f0, y_test),
    gam_bs_gcv    = try_metrics(m_bs_gcv, df_t, f0, y_test),
    bam_cr_gcv    = try_metrics(m_bam,    df_t, f0, y_test),
    gamlss_ps_ml = try_metrics_gamlss(m_gamlss, df, df_t, f0, y_test)
  )

  # Cyclic cubic: global smooth only
  if (task == "global") {
    m_cc_gcv       <- try_fit(gam(y ~ s(x, bs = "cc"), data = df, method = "GCV.Cp"), "gam_cc_gcv")
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

  m_cr_gcv  <- try_fit(gam(fmla_cr, data = df, method = "GCV.Cp"), "gam_cr_gcv")
  m_ps_gcv  <- try_fit(gam(fmla_ps, data = df, method = "GCV.Cp"), "gam_ps_gcv")
  m_bs_gcv  <- try_fit(gam(fmla_bs, data = df, method = "GCV.Cp"), "gam_bs_gcv")
  m_bam     <- try_fit(bam(fmla_cr, data = df, method = "GCV.Cp"), "bam_cr_gcv")
  m_gamlss <- try_fit(gamlss(fmla_pb, data = df, family = NO(), control = GAMLSS_CTRL), "gamlss_ps_ml")

  list(
    gam_cr_gcv    = try_metrics(m_cr_gcv, df_t, f0, y_test),
    gam_ps_gcv    = try_metrics(m_ps_gcv, df_t, f0, y_test),
    gam_bs_gcv    = try_metrics(m_bs_gcv, df_t, f0, y_test),
    bam_cr_gcv    = try_metrics(m_bam,    df_t, f0, y_test),
    gamlss_ps_ml = try_metrics_gamlss(m_gamlss, df, df_t, f0, y_test)
  )
}

run_one <- function(path, task) {
  d <- read.csv(path)
  if ("x1" %in% names(d)) run_one_additive(d) else run_one_univariate(d, task)
}

CHUNK_DIR <- file.path(RESULTS, "sim_chunks")
dir.create(CHUNK_DIR, showWarnings = FALSE, recursive = TRUE)

chunk_path <- function(task, n, b)
  file.path(CHUNK_DIR, sprintf("r_%s_n%d_b%03d.csv", task, n, b))

# run_cell: fit every rep of one (task, n) cell in [REP_START, REP_END] that has
# no chunk yet; one chunk file per rep. Returns the number of reps fitted.
run_cell <- function(task, n) {
  paths <- sort(Sys.glob(sprintf("%s/%s_n%d_b*.csv", DATA, task, n)))
  b     <- as.integer(sub(".*_b([0-9]+)\\.csv$", "\\1", paths))
  keep  <- b >= REP_START & b <= REP_END
  paths <- paths[keep]; b <- b[keep]
  if (MAX_REPS > 0) { paths <- head(paths, MAX_REPS); b <- head(b, MAX_REPS) }
  k <- 0L
  for (i in seq_along(paths)) {
    chunk <- chunk_path(task, n, b[i])
    if (file.exists(chunk)) next
    res    <- run_one(paths[i], task)
    rep_df <- as.data.frame(do.call(rbind, res))
    write.csv(cbind(method = rownames(rep_df), rep_df), chunk, row.names = FALSE)
    k <- k + 1L
  }
  k
}

# merge_cell: average all rep chunks of one cell into the summary CSV.
merge_cell <- function(task, n, out) {
  chunks <- list.files(CHUNK_DIR, full.names = TRUE,
                       pattern = sprintf("^r_%s_n%d_b[0-9]+\\.csv$", task, n))
  if (length(chunks) == 0) {
    cat(sprintf("  %s n=%d: no rep chunks\n", task, n)); return(invisible(NULL))
  }
  reps <- do.call(rbind, lapply(chunks, read.csv))
  res  <- aggregate(. ~ method, data = reps, FUN = mean, na.rm = TRUE,
                    na.action = na.pass)
  res  <- res[match(unique(reps$method), res$method), ]
  cat(sprintf("  %s n=%d: averaged %d rep chunks\n", task, n, length(chunks)))
  print(data.frame(res[-1], row.names = res$method), digits = 3)
  write.csv(cbind(method = res$method, task = task, n = n, res[-1]),
            out, row.names = FALSE)
}

# ── Run ───────────────────────────────────────────────────────────────────────
# Rscript <script> [--merge] [<task> <n>]
#   <task> <n>  run one cell (default: every cell)
#   --merge     no fitting; average existing rep chunks into the summary CSVs
# SIM_REP_START / SIM_REP_END restrict fitting to dataset files b<START>..b<END>
# (one shard of a cell). Sharded runs only write rep chunks; a later --merge
# run averages them.

args  <- commandArgs(trailingOnly = TRUE)
MERGE <- length(args) >= 1 && args[1] == "--merge"
if (MERGE) args <- args[-1]
REP_START <- as.integer(Sys.getenv("SIM_REP_START", "1"))
REP_END   <- as.integer(Sys.getenv("SIM_REP_END", "999999"))
SHARDED   <- nzchar(Sys.getenv("SIM_REP_START")) || nzchar(Sys.getenv("SIM_REP_END"))

cells <- expand.grid(n = c(100, 500, 2000), task = c("global", "local", sprintf("additive_p%02d", c(1, 10, 50, 100))),
                     stringsAsFactors = FALSE)
if (length(args) >= 2) cells <- data.frame(task = args[1], n = as.numeric(args[2]))

for (i in seq_len(nrow(cells))) {
  task <- cells$task[i]; n <- cells$n[i]
  out  <- sprintf("%s/r_%s_n%d.csv", RESULTS, task, n)
  if (!MERGE) {
    if (!SHARDED && file.exists(out)) {
      cat(sprintf("  %s n=%d: skipping (already done)\n", task, n)); next
    }
    cat(sprintf("\n%s n = %d\n", task, n))
    t0 <- proc.time()[["elapsed"]]
    k  <- run_cell(task, n)
    cat(sprintf("  [%s n=%d: %.1fs for %d reps]\n", task, n,
                proc.time()[["elapsed"]] - t0, k))
  }
  if (MERGE || !SHARDED) merge_cell(task, n, out)
}
