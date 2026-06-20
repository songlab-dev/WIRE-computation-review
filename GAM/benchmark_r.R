# Scalability benchmark — R methods.
#
# For each (scenario, n) cell, runs REPS replicates of fit + predict for all R
# methods and records elapsed time, peak memory, and MISE. Results are
# checkpointed after every rep so a wall-clock kill never loses completed work.
#
# Outputs (results/benchmark_chunks/):
#   r_{scenario}_n{n}.csv   per-cell chunk files
# Outputs (results/):
#   r_benchmark.csv          combined file (all chunks)
#
# CLI usage (optional — selects one cell):
#   Rscript public/benchmark_r.R <scenario> <n>
#   e.g.  Rscript public/benchmark_r.R additive_p10 100000
#
# Without arguments, runs all (scenario, n) cells in the default grid.
#
# SLURM job-budget integration: set BENCH_JOB_BUDGET_S to the job's wall-clock
# limit in seconds. The script stops launching new fits within JOB_MARGIN_S of
# the budget and checkpoints cleanly rather than being SIGKILLed mid-run.
#
# Run from the project root:
#   Rscript public/benchmark_r.R

suppressPackageStartupMessages({
  library(mgcv)
  library(gamlss)
})
set.seed(2024)

GAMLSS_CTRL <- gamlss.control(trace = FALSE)

# Run from the project root directory.
RESULTS <- "results"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

REPS    <- 100
N_TEST  <- 2000          # held-out points for the MISE estimate
EPS     <- 1e-6

BENCH_TIMEOUT_S  <- 1800   # per-fit wall-clock cap (s)
MAX_ALLFAIL_REPS <- 1000   # effectively disabled (REPS=100)

# Global job wall-clock budget (set by SLURM scripts via BENCH_JOB_BUDGET_S).
# Defaults to 0 (disabled). When set, the script stops before being SIGKILLed.
JOB_START    <- proc.time()[["elapsed"]]
JOB_BUDGET_S <- as.numeric(Sys.getenv("BENCH_JOB_BUDGET_S", "0"))
JOB_MARGIN_S <- as.numeric(Sys.getenv("BENCH_JOB_MARGIN_S",
                                      as.character(BENCH_TIMEOUT_S + 300)))

past_deadline <- function() {
  JOB_BUDGET_S > 0 &&
    (proc.time()[["elapsed"]] - JOB_START) > (JOB_BUDGET_S - JOB_MARGIN_S)
}

# time_mem: run fn() with a hard wall-clock cap.
# Returns list(elapsed_s, mem_mb, model); model = NULL on timeout/error.
time_mem <- function(fn, timeout_s = BENCH_TIMEOUT_S) {
  setTimeLimit(elapsed = timeout_s, transient = FALSE)
  on.exit(setTimeLimit(elapsed = Inf, transient = FALSE))
  gc(reset = TRUE)
  tryCatch({
    model   <- NULL
    elapsed <- system.time({ model <- fn() })[["elapsed"]]
    mem_mb  <- sum(gc()[, 6])
    list(elapsed_s = elapsed, mem_mb = mem_mb, model = model)
  }, error = function(e) {
    if (grepl("elapsed time limit", conditionMessage(e), ignore.case = TRUE)) {
      message(sprintf("    [timeout > %ds]", timeout_s))
    } else {
      message(sprintf("    [error: %s]", conditionMessage(e)))
    }
    list(elapsed_s = NA_real_, mem_mb = NA_real_, model = NULL)
  })
}

# bench_scenario: run REPS replicates for one (scenario, n) cell.
# Checkpoints after every completed rep; stops on job-budget or all-fail deferral.
bench_scenario <- function(spec, n, done = 0L, checkpoint = NULL) {
  rows <- list(); allfail <- 0L; deferred <- FALSE
  for (rep in seq_len(REPS)) {
    d <- spec$make_rep(n)        # always draw so the RNG stream matches a full run
    if (rep <= done) next
    if (past_deadline()) {
      cat(sprintf("  [job budget reached before rep %d: stopping cleanly (%d/%d done)]\n",
                  rep, done, REPS))
      break
    }
    rep_rows <- list(); rep_all_failed <- TRUE; aborted <- FALSE
    for (nm in names(spec$methods)) {
      if (past_deadline()) {
        cat(sprintf("  [job budget reached mid-rep %d: discarding partial rep, stopping (%d/%d done)]\n",
                    rep, done, REPS))
        aborted <- TRUE
        break
      }
      mth <- spec$methods[[nm]]
      tm  <- time_mem(function() mth$fit(d$train))
      mise <- NA_real_
      if (!is.null(tm$model)) {
        rep_all_failed <- FALSE
        mise <- tryCatch({
          vals <- mth$pred(tm$model, d$train, d$test)
          mean((as.numeric(vals) - d$f0)^2)
        }, error = function(e) {
          message(sprintf("    [%s predict failed: %s]", nm, conditionMessage(e)))
          NA_real_
        })
      }
      rep_rows[[length(rep_rows) + 1]] <- data.frame(
        method = nm, n = n, rep = rep,
        elapsed_s = tm$elapsed_s, mem_mb = tm$mem_mb, mise = mise
      )
    }
    if (aborted) break
    rows <- c(rows, rep_rows)
    done <- rep
    if (!is.null(checkpoint)) checkpoint(do.call(rbind, rows))
    if (rep_all_failed) {
      allfail <- allfail + 1L
      cat(sprintf("    rep %d: all methods failed (%d/%d)\n",
                  rep, allfail, MAX_ALLFAIL_REPS))
      if (allfail >= MAX_ALLFAIL_REPS) {
        cat(sprintf("    deferring after %d all-fail reps\n", MAX_ALLFAIL_REPS))
        deferred <- TRUE
        break
      }
    }
  }
  list(data = if (length(rows) > 0) do.call(rbind, rows) else NULL,
       deferred = deferred)
}

# ── 10-component signal library ───────────────────────────────────────────────
wood_bump <- function(x) 0.2*x^11*(10*(1-x))^6 + 10*(10*x)^3*(1-x)^10

COMPONENT_FNS <- list(
  function(x) sin(2*pi*x),
  function(x) wood_bump(x),
  function(x) exp(2*x),
  function(x) (2*x - 1)^3,
  function(x) sin(4*pi*x),
  function(x) 40*x^2*(1 - x)^2,
  function(x) abs(2*x - 1),
  function(x) log(1 + 5*x),
  function(x) 20*(x - 0.5)^2,
  function(x) x
)

.scenario_parts <- function(p) {
  fns_p <- COMPONENT_FNS[((seq_len(p) - 1L) %% length(COMPONENT_FNS)) + 1L]
  xcols <- paste0("x", seq_len(p))
  list(
    fns_p   = fns_p,
    xcols   = xcols,
    signal  = function(X) Reduce(`+`, Map(function(f, j) f(X[, j]), fns_p, seq_len(p))),
    fmla_cr = as.formula(paste("y ~", paste(sprintf("s(x%d,bs='cr')", seq_len(p)), collapse = " + "))),
    fmla_ps = as.formula(paste("y ~", paste(sprintf("s(x%d,bs='ps')", seq_len(p)), collapse = " + "))),
    fmla_bs = as.formula(paste("y ~", paste(sprintf("s(x%d,bs='bs')", seq_len(p)), collapse = " + "))),
    fmla_pb = as.formula(paste("y ~", paste(sprintf("pb(x%d)",         seq_len(p)), collapse = " + ")))
  )
}

# ── Gaussian additive scenario ────────────────────────────────────────────────
bench_additive_p <- function(p) {
  P <- .scenario_parts(p)
  make_rep <- function(n) {
    X <- matrix(runif(n * p), nrow = n, ncol = p); colnames(X) <- P$xcols
    y <- P$signal(X) + rnorm(n)
    Xt <- matrix(runif(N_TEST * p), nrow = N_TEST, ncol = p); colnames(Xt) <- P$xcols
    list(train = as.data.frame(cbind(X, y = y)),
         test  = as.data.frame(Xt),
         f0    = P$signal(Xt))
  }
  pred_mgcv  <- function(m, tr, te) as.numeric(predict(m, newdata = te))
  pred_gamls <- function(m, tr, te)
    as.numeric(predict(m, what = "mu", newdata = te, data = tr, type = "response"))
  methods <- list(
    gam_cr     = list(fit = function(df) gam(P$fmla_cr, data = df, method = "GCV.Cp"), pred = pred_mgcv),
    bam_cr     = list(fit = function(df) bam(P$fmla_cr, data = df, method = "GCV.Cp",
                                             discrete = TRUE, nthreads = 1), pred = pred_mgcv),
    gam_ps     = list(fit = function(df) gam(P$fmla_ps, data = df, method = "GCV.Cp"), pred = pred_mgcv),
    gamdist_ps = list(fit = function(df) gamlss(P$fmla_pb, data = df, family = NO(),
                                                control = GAMLSS_CTRL), pred = pred_gamls),
    gam_bs     = list(fit = function(df) gam(P$fmla_bs, data = df, method = "GCV.Cp"), pred = pred_mgcv)
  )
  list(make_rep = make_rep, methods = methods)
}

# ── Logistic additive scenario ────────────────────────────────────────────────
# True LP = additive signal scaled to LP ~ N(0, 1.5^2).
# Calibration saves/restores the RNG so the main stream is unaffected.
bench_logistic_p <- function(p) {
  P <- .scenario_parts(p)
  old <- if (exists(".Random.seed", .GlobalEnv)) get(".Random.seed", .GlobalEnv) else NULL
  set.seed(1234L + p)
  raw <- P$signal(matrix(runif(20000 * p), nrow = 20000, ncol = p))
  mu <- mean(raw); sdv <- sd(raw)
  if (!is.null(old)) assign(".Random.seed", old, envir = .GlobalEnv)
  lp_of <- function(X) 1.5 * (P$signal(X) - mu) / sdv

  make_rep <- function(n) {
    X <- matrix(runif(n * p), nrow = n, ncol = p); colnames(X) <- P$xcols
    y <- rbinom(n, 1, plogis(lp_of(X)))
    Xt <- matrix(runif(N_TEST * p), nrow = N_TEST, ncol = p); colnames(Xt) <- P$xcols
    list(train = as.data.frame(cbind(X, y = y)),
         test  = as.data.frame(Xt),
         f0    = lp_of(Xt))
  }
  pred_mgcv  <- function(m, tr, te) as.numeric(predict(m, newdata = te, type = "link"))
  pred_gamls <- function(m, tr, te) {
    ph <- pmin(pmax(predict(m, what = "mu", newdata = te, data = tr, type = "response"),
                    EPS), 1 - EPS)
    qlogis(ph)
  }
  methods <- list(
    gam_cr     = list(fit = function(df) gam(P$fmla_cr, data = df, family = binomial(),
                                             method = "GCV.Cp"), pred = pred_mgcv),
    bam_cr     = list(fit = function(df) bam(P$fmla_cr, data = df, family = binomial(),
                                             method = "GCV.Cp", discrete = TRUE, nthreads = 1), pred = pred_mgcv),
    gam_ps     = list(fit = function(df) gam(P$fmla_ps, data = df, family = binomial(),
                                             method = "GCV.Cp"), pred = pred_mgcv),
    gamdist_ps = list(fit = function(df) gamlss(P$fmla_pb, data = df, family = BI(),
                                                control = GAMLSS_CTRL), pred = pred_gamls),
    gam_bs     = list(fit = function(df) gam(P$fmla_bs, data = df, family = binomial(),
                                             method = "GCV.Cp"), pred = pred_mgcv)
  )
  list(make_rep = make_rep, methods = methods)
}

P_VALUES  <- c(1, 5, 10, 50, 100)
SCENARIOS <- c(
  setNames(lapply(P_VALUES, bench_additive_p),  sprintf("additive_p%02d",  P_VALUES)),
  setNames(lapply(P_VALUES, bench_logistic_p),  sprintf("logistic_p%02d", P_VALUES))
)

CHUNKS_DIR <- file.path(RESULTS, "benchmark_chunks")
dir.create(CHUNKS_DIR, showWarnings = FALSE, recursive = TRUE)

OUT_COLS <- c("method", "n", "rep", "elapsed_s", "mem_mb", "mise", "scenario")

# Optional CLI args select one cell: Rscript public/benchmark_r.R <scenario> <n>
args   <- commandArgs(trailingOnly = TRUE)
sel_sc <- if (length(args) >= 1) args[1] else names(SCENARIOS)
sel_n  <- if (length(args) >= 2) as.numeric(args[2]) else c(1e3, 1e5)

# ── Run/resume one (scenario, n) cell ────────────────────────────────────────
process_cell <- function(sc, n) {
  chunk <- file.path(CHUNKS_DIR, sprintf("r_%s_n%g.csv", sc, n))
  prev <- NULL; done <- 0L
  if (file.exists(chunk)) {
    prev <- read.csv(chunk, stringsAsFactors = FALSE)
    if (!"mise" %in% names(prev))     prev$mise <- NA_real_
    if (!"scenario" %in% names(prev)) prev$scenario <- sc
    prev <- prev[, OUT_COLS]
    done <- if (nrow(prev) > 0) max(prev$rep) else 0L
    if (done >= REPS) {
      cat(sprintf("  %s n=%g: complete (%d reps) — skipping\n", sc, n, done))
      return(FALSE)
    }
    cat(sprintf("  %s n=%g: resuming at rep %d (have %d/%d)\n",
                sc, n, done + 1L, done, REPS))
  } else {
    cat(sprintf("  %s n=%g\n", sc, n))
  }
  checkpoint <- function(bm) {
    bm$scenario <- sc
    bm <- bm[, OUT_COLS]
    combined <- if (!is.null(prev)) rbind(prev, bm) else bm
    write.csv(combined, chunk, row.names = FALSE)
  }
  res <- bench_scenario(SCENARIOS[[sc]], n, done, checkpoint)
  bm  <- res$data
  if (!is.null(bm)) { bm$scenario <- sc; bm <- bm[, OUT_COLS] }
  combined <- prev
  if (!is.null(bm)) combined <- if (!is.null(prev)) rbind(prev, bm) else bm
  if (is.null(combined)) { cat("  no results — all fits failed\n"); return(res$deferred) }
  if (is.null(bm))       { cat(sprintf("  %s n=%g: no new reps added\n", sc, n)); return(res$deferred) }
  for (nm in unique(combined$method)) {
    sel <- combined$method == nm
    cat(sprintf("    %-10s reps=%d  median=%.3fs  mem=%.1fMb  mise=%.4g\n",
                nm, length(unique(combined$rep[sel])),
                median(combined$elapsed_s[sel], na.rm = TRUE),
                median(combined$mem_mb[sel],   na.rm = TRUE),
                median(combined$mise[sel],     na.rm = TRUE)))
  }
  write.csv(combined, chunk, row.names = FALSE)
  cat(sprintf("  saved %s\n", chunk))
  res$deferred
}

# Process all viable cells first; revisit deferred (all-fail) cells at the end.
deferred <- list()
for (sc in sel_sc) {
  cat(sprintf("\n=== %s ===\n", sc))
  for (n in sel_n) {
    if (process_cell(sc, n)) deferred[[length(deferred) + 1]] <- list(sc = sc, n = n)
  }
}
if (length(deferred) > 0) {
  cat(sprintf("\n=== revisiting %d deferred cell(s) ===\n", length(deferred)))
  for (d in deferred) process_cell(d$sc, d$n)
}

# Combine all chunks into the final CSV.
chunks <- list.files(CHUNKS_DIR, pattern = "^r_.*\\.csv$", full.names = TRUE)
if (length(chunks) > 0) {
  dfs <- lapply(chunks, function(f) {
    d <- read.csv(f, stringsAsFactors = FALSE)
    if (!"mise" %in% names(d)) d$mise <- NA_real_
    if (!"scenario" %in% names(d)) {
      d$scenario <- sub("^r_(.*)_n[0-9]+\\.csv$", "\\1", basename(f))
    }
    d[, OUT_COLS]
  })
  combined <- do.call(rbind, dfs)
  write.csv(combined, file.path(RESULTS, "r_benchmark.csv"), row.names = FALSE)
  cat(sprintf("\ncombined %d chunks -> r_benchmark.csv\n", length(chunks)))
} else {
  cat("No chunks found — nothing written.\n")
}
