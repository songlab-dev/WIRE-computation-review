# Scalability benchmark — R methods.
#
# For each (scenario, n) cell, runs REPS replicates of fit + predict for all R
# methods and records elapsed time, itr/sec, total time, peak memory (max RSS),
# and MISE. Results are checkpointed after every rep so a wall-clock kill never
# loses completed work.
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
# Each fit runs in its own worker process under `timeout` and `/usr/bin/time`,
# as in benchmark_python.py, so max_rss_mb is measured the same way in both
# languages. The "baseline" method loads the data without fitting (interpreter
# + data floor). Timings come from bench::mark: one untimed fit, then
# BENCH_ITERS (default 1) timed fits. Set BENCH_TIMEOUT_BIN / BENCH_TIME_BIN to
# override the tool paths.
#
# Datasets are drawn once by `Rscript public/benchmark_r.R --gen <scenario> <n>`
# into BENCH_DATA_DIR (default results/benchmark_data) and read by both this
# script and benchmark_python.py.
#
# Run from the project root:
#   Rscript public/benchmark_r.R

suppressPackageStartupMessages({
  library(mgcv)
  library(gamlss)
  library(bench)
})
set.seed(2024)

GAMLSS_CTRL <- gamlss.control(trace = FALSE)

# Run from the project root directory.
# GAM_RESULTS / GAM_MAX_REPS: pilot overrides (separate output dir, fewer reps).
RESULTS  <- Sys.getenv("GAM_RESULTS", "results")
MAX_REPS <- as.integer(Sys.getenv("GAM_MAX_REPS", "0"))   # 0 = all
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

REPS    <- if (MAX_REPS > 0) MAX_REPS else 100L
N_TEST  <- 2000          # held-out points for the MISE estimate
EPS     <- 1e-6

BENCH_TIMEOUT_S  <- 1800   # per-fit wall-clock cap (s)
MAX_ALLFAIL_REPS <- 1000   # effectively disabled (REPS=100)
# BENCH_REP_START / BENCH_REP_END: run only this rep range (one shard of a cell).
REP_START <- as.integer(Sys.getenv("BENCH_REP_START", "1"))
REP_END   <- as.integer(Sys.getenv("BENCH_REP_END", as.character(REPS)))
SHARDED   <- REP_START != 1L || REP_END != REPS
# After this many timeouts in a cell (per job), skip that method for the
# remaining reps; skipped reps are recorded as not completed (NA).
MAX_METHOD_TIMEOUTS <- 5L
BENCH_ITERS      <- as.integer(Sys.getenv("BENCH_ITERS", "1"))
STARTUP_SLACK_S  <- 300    # worker start-up allowance (s)
BASELINE         <- "baseline"

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

NA_RESULT <- list(elapsed_s = NA_real_, itr_per_sec = NA_real_,
                  total_time_s = NA_real_, max_rss_mb = NA_real_, mise = NA_real_)

# bench_fit: one untimed fit supplies the model, then bench::mark timings.
# (bench::mark skips its own untimed run when check = FALSE and memory = FALSE,
# so its `result` column is empty here.)
bench_fit <- function(fn) {
  model <- fn()
  bm <- bench::mark(fn(), iterations = BENCH_ITERS, check = FALSE,
                    filter_gc = FALSE, memory = FALSE)
  list(elapsed_s    = as.numeric(bm$median),
       itr_per_sec  = as.numeric(bm$`itr/sec`),
       total_time_s = as.numeric(bm$total_time),
       model        = model)
}

# find_tools: locate `timeout` and /usr/bin/time; detect GNU vs BSD time.
find_tools <- function() {
  to <- Sys.getenv("BENCH_TIMEOUT_BIN")
  if (!nzchar(to)) to <- Sys.which("timeout")
  if (!nzchar(to)) to <- Sys.which("gtimeout")
  tb <- Sys.getenv("BENCH_TIME_BIN", "/usr/bin/time")
  if (!nzchar(to) || !file.exists(tb))
    stop("benchmark needs coreutils `timeout` and /usr/bin/time ",
         "(set BENCH_TIMEOUT_BIN / BENCH_TIME_BIN)")
  gnu <- system2(tb, c("-v", "-o", "/dev/null", "true"),
                 stdout = FALSE, stderr = FALSE) == 0
  list(timeout = unname(to), time = tb, gnu = gnu)
}

# parse_max_rss_mb: GNU time reports kB, BSD time bytes.
parse_max_rss_mb <- function(path, gnu) {
  if (!file.exists(path)) return(NA_real_)
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  if (gnu) {
    m <- regmatches(txt, regexec("Maximum resident set size \\(kbytes\\):\\s*([0-9]+)", txt))[[1]]
    if (length(m) == 2) as.numeric(m[2]) / 1024 else NA_real_
  } else {
    m <- regmatches(txt, regexec("([0-9]+)\\s+maximum resident set size", txt))[[1]]
    if (length(m) == 2) as.numeric(m[2]) / 1024^2 else NA_real_
  }
}

# run_isolated: run one method in a worker process; all-NA result on failure.
run_isolated <- function(sc, method, data_path, tmpdir) {
  out_path <- file.path(tmpdir, paste0(method, ".csv"))
  rss_path <- file.path(tmpdir, paste0(method, ".time"))
  cap <- format(BENCH_TIMEOUT_S * (1 + BENCH_ITERS) + STARTUP_SLACK_S, scientific = FALSE)
  rc <- system2(TOOLS$timeout,
                c("-k", "30", cap, TOOLS$time, if (TOOLS$gnu) "-v" else "-l",
                  "-o", shQuote(rss_path), shQuote(RSCRIPT), shQuote(SCRIPT),
                  "--worker", sc, method, shQuote(data_path), shQuote(out_path)))
  if (rc %in% c(124, 137)) {
    message(sprintf("    [%s timeout > %ss]", method, cap))
    return(c(NA_RESULT, timed_out = TRUE))
  }
  if (rc != 0 || !file.exists(out_path)) {
    message(sprintf("    [%s worker failed, exit status %d]", method, rc))
    return(NA_RESULT)
  }
  res <- as.list(read.csv(out_path))
  res$max_rss_mb <- if (isTRUE(res$fit_ok)) parse_max_rss_mb(rss_path, TOOLS$gnu) else NA_real_
  res$fit_ok <- NULL
  res[names(NA_RESULT)]
}

# run_worker: fit + predict one method on one replicate; write results to CSV.
run_worker <- function(sc, method, data_path, out_path) {
  d   <- readRDS(data_path)
  res <- NA_RESULT[setdiff(names(NA_RESULT), "max_rss_mb")]
  res$fit_ok <- method == BASELINE
  # Cell given up (marker written by monitor/giveup.py after >= 10 failed fits in
  # this cell): record a failed fit without running it. Workers are fresh processes,
  # so this also applies to jobs that were started before the marker existed.
  flag <- file.path(RESULTS, "benchmark_giveup",
                    sprintf("%s_n%d_%s.flag", sc, nrow(d$train), method))
  if (method != BASELINE && file.exists(flag)) {
    message(sprintf("    [%s: cell given up, skipping]", method))
    write.csv(as.data.frame(res), out_path, row.names = FALSE)
    return(invisible(NULL))
  }
  if (method != BASELINE) {
    mth <- SCENARIOS[[sc]]$methods[[method]]
    tm  <- tryCatch(bench_fit(function() mth$fit(d$train)), error = function(e) {
      message(sprintf("    [%s error: %s]", method, conditionMessage(e)))
      NULL
    })
    if (!is.null(tm) && !is.null(tm$model)) {
      res[c("elapsed_s", "itr_per_sec", "total_time_s")] <-
        tm[c("elapsed_s", "itr_per_sec", "total_time_s")]
      res$fit_ok <- TRUE
      res$mise <- tryCatch({
        vals <- mth$pred(tm$model, d$train, d$test)
        mean((as.numeric(vals) - d$f0)^2)
      }, error = function(e) {
        message(sprintf("    [%s predict failed: %s]", method, conditionMessage(e)))
        NA_real_
      })
    }
  }
  write.csv(as.data.frame(res), out_path, row.names = FALSE)
}

# ── Shared datasets ───────────────────────────────────────────────────────────
# R draws each rep once (`Rscript benchmark_r.R --gen <scenario> <n>`) and both
# benchmark_r.R and benchmark_python.py read it, so every method in both
# languages fits the same data. One raw float64 file per rep, column-major:
# X (n x p), y (n), Xt (N_TEST x p), f0 (N_TEST).
DATA_DIR <- Sys.getenv("BENCH_DATA_DIR", file.path(RESULTS, "benchmark_data"))

rep_path <- function(sc, n, rep)
  file.path(DATA_DIR, sprintf("%s_n%d_rep%03d.bin", sc, as.integer(n), rep))

write_reps <- function(sc, n) {
  dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
  for (rep in seq_len(REP_END)) {
    d <- SCENARIOS[[sc]]$make_rep(n)   # draw every rep: same RNG stream as a full run
    if (rep < REP_START) next
    xc <- setdiff(names(d$train), "y")
    writeBin(c(as.double(as.matrix(d$train[xc])), as.double(d$train$y),
               as.double(as.matrix(d$test[xc])), as.double(d$f0)),
             rep_path(sc, n, rep))
  }
}

read_rep <- function(sc, n, rep) {
  p  <- as.integer(sub(".*_p", "", sc)); xc <- paste0("x", seq_len(p))
  v  <- readBin(rep_path(sc, n, rep), "double", n = n * p + n + N_TEST * (p + 1))
  o  <- n * p + n
  X  <- matrix(v[seq_len(n * p)], n, p, dimnames = list(NULL, xc))
  Xt <- matrix(v[o + seq_len(N_TEST * p)], N_TEST, p, dimnames = list(NULL, xc))
  list(train = as.data.frame(cbind(X, y = v[n * p + seq_len(n)])),
       test  = as.data.frame(Xt),
       f0    = v[o + N_TEST * p + seq_len(N_TEST)])
}

# bench_scenario: run REPS replicates for one (scenario, n) cell.
# Checkpoints after every completed rep; stops on job-budget or all-fail deferral.
bench_scenario <- function(sc, n, done = 0L, checkpoint = NULL) {
  spec <- SCENARIOS[[sc]]
  rows <- list(); allfail <- 0L; deferred <- FALSE
  timeouts <- list()
  for (rep in seq_len(REP_END)) {
    if (rep <= done) next
    if (!file.exists(rep_path(sc, n, rep)))
      stop(sprintf("missing %s — run `Rscript benchmark_r.R --gen %s %g` first",
                   rep_path(sc, n, rep), sc, n))
    d <- read_rep(sc, n, rep)
    if (past_deadline()) {
      cat(sprintf("  [job budget reached before rep %d: stopping cleanly (%d/%d done)]\n",
                  rep, done, REP_END))
      break
    }
    tmpdir <- tempfile("bench_"); dir.create(tmpdir)
    data_path <- file.path(tmpdir, "data.rds")
    saveRDS(d, data_path)
    rep_rows <- list(); rep_all_failed <- TRUE; aborted <- FALSE
    for (nm in c(BASELINE, names(spec$methods))) {
      if (past_deadline()) {
        cat(sprintf("  [job budget reached mid-rep %d: discarding partial rep, stopping (%d/%d done)]\n",
                    rep, done, REP_END))
        aborted <- TRUE
        break
      }
      n_to <- if (is.null(timeouts[[nm]])) 0L else timeouts[[nm]]
      if (n_to >= MAX_METHOD_TIMEOUTS) {
        res <- NA_RESULT   # skipped: too many timeouts
      } else {
        res <- run_isolated(sc, nm, data_path, tmpdir)
        if (isTRUE(res$timed_out)) {
          timeouts[[nm]] <- n_to + 1L
          if (timeouts[[nm]] == MAX_METHOD_TIMEOUTS)
            cat(sprintf("    [%s: %d timeouts — skipping it for the remaining reps]\n",
                        nm, MAX_METHOD_TIMEOUTS))
        }
        res$timed_out <- NULL
      }
      if (nm != BASELINE && !is.na(res$elapsed_s)) rep_all_failed <- FALSE
      rep_rows[[length(rep_rows) + 1]] <- data.frame(method = nm, n = n, rep = rep, res)
    }
    unlink(tmpdir, recursive = TRUE)
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
    gamlss_ps  = list(fit = function(df) gamlss(P$fmla_pb, data = df, family = NO(),
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
    gamlss_ps  = list(fit = function(df) gamlss(P$fmla_pb, data = df, family = BI(),
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

# Worker mode: Rscript benchmark_r.R --worker <scenario> <method> <data> <out>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && args[1] == "--worker") {
  run_worker(args[2], args[3], args[4], args[5])
  quit(save = "no", status = 0)
}
# Data mode: Rscript benchmark_r.R --gen <scenario> <n>  (reps REP_START..REP_END)
if (length(args) >= 1 && args[1] == "--gen") {
  write_reps(args[2], as.numeric(args[3]))
  quit(save = "no", status = 0)
}

TOOLS   <- find_tools()
RSCRIPT <- file.path(R.home("bin"), "Rscript")
SCRIPT  <- normalizePath(sub("^--file=", "",
                             grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))

CHUNKS_DIR <- file.path(RESULTS, "benchmark_chunks")
dir.create(CHUNKS_DIR, showWarnings = FALSE, recursive = TRUE)

OUT_COLS <- c("method", "n", "rep", "elapsed_s", "itr_per_sec", "total_time_s",
              "max_rss_mb", "mise", "scenario")

# Older chunks may lack some columns; fill with NA.
fill_cols <- function(d) {
  for (c in setdiff(OUT_COLS, c("scenario", names(d)))) d[[c]] <- NA_real_
  d
}

# Optional CLI args select one cell: Rscript public/benchmark_r.R <scenario> <n>
sel_sc <- if (length(args) >= 1) args[1] else names(SCENARIOS)
sel_n  <- if (length(args) >= 2) as.numeric(args[2]) else c(1e3, 1e5)

# ── Run/resume one (scenario, n) cell ────────────────────────────────────────
process_cell <- function(sc, n) {
  shard <- if (SHARDED) sprintf("_r%d-%d", REP_START, REP_END) else ""
  chunk <- file.path(CHUNKS_DIR, sprintf("r_%s_n%g%s.csv", sc, n, shard))
  prev <- NULL; done <- REP_START - 1L
  if (file.exists(chunk)) {
    prev <- fill_cols(read.csv(chunk, stringsAsFactors = FALSE))
    if (!"scenario" %in% names(prev)) prev$scenario <- sc
    prev <- prev[, OUT_COLS]
    done <- if (nrow(prev) > 0) max(prev$rep) else done
    if (done >= REP_END) {
      cat(sprintf("  %s n=%g: complete (%d reps) — skipping\n", sc, n, done))
      return(FALSE)
    }
    cat(sprintf("  %s n=%g: resuming at rep %d (have %d/%d)\n",
                sc, n, done + 1L, done, REP_END))
  } else {
    cat(sprintf("  %s n=%g\n", sc, n))
  }
  checkpoint <- function(bm) {
    bm$scenario <- sc
    bm <- bm[, OUT_COLS]
    combined <- if (!is.null(prev)) rbind(prev, bm) else bm
    write.csv(combined, chunk, row.names = FALSE)
  }
  res <- bench_scenario(sc, n, done, checkpoint)
  bm  <- res$data
  if (!is.null(bm)) { bm$scenario <- sc; bm <- bm[, OUT_COLS] }
  combined <- prev
  if (!is.null(bm)) combined <- if (!is.null(prev)) rbind(prev, bm) else bm
  if (is.null(combined)) { cat("  no results — all fits failed\n"); return(res$deferred) }
  if (is.null(bm))       { cat(sprintf("  %s n=%g: no new reps added\n", sc, n)); return(res$deferred) }
  for (nm in unique(combined$method)) {
    sel <- combined$method == nm
    cat(sprintf("    %-10s reps=%d  median=%.3fs  %.3g itr/s  max_rss=%.1fMb  mise=%.4g\n",
                nm, length(unique(combined$rep[sel])),
                median(combined$elapsed_s[sel],    na.rm = TRUE),
                median(combined$itr_per_sec[sel],  na.rm = TRUE),
                median(combined$max_rss_mb[sel],   na.rm = TRUE),
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
    d <- fill_cols(read.csv(f, stringsAsFactors = FALSE))
    if (!"scenario" %in% names(d)) {
      d$scenario <- sub("^r_(.*)_n[0-9]+\\.csv$", "\\1", basename(f))
    }
    d[, OUT_COLS]
  })
  combined <- do.call(rbind, dfs)
  out <- file.path(RESULTS, "r_benchmark.csv")
  tmp <- sprintf("%s.%d.tmp", out, Sys.getpid())   # shards finish concurrently
  write.csv(combined, tmp, row.names = FALSE)
  file.rename(tmp, out)
  cat(sprintf("\ncombined %d chunks -> r_benchmark.csv\n", length(chunks)))
} else {
  cat("No chunks found — nothing written.\n")
}
