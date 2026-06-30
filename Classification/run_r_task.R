#!/usr/bin/env Rscript
# Read one row from an R task CSV by --task-id and call bench_r_models.R.

suppressPackageStartupMessages(library(optparse))

opt_list <- list(
  make_option("--task-id",   type = "integer"),
  make_option("--task-list", type = "character", default = "tasks_r.csv"),
  make_option("--out",       type = "character", default = "results/r_results.csv")
)
opt <- parse_args(OptionParser(option_list = opt_list))

task_id   <- opt[["task-id"]]
task_list <- opt[["task-list"]]
out_path  <- opt$out

tasks <- read.csv(task_list, stringsAsFactors = FALSE)
row   <- tasks[tasks$task_id == task_id, ]
if (nrow(row) == 0) stop(sprintf("task_id %d not found in %s", task_id, task_list))
row   <- row[1, ]

cat(sprintf("[run_r_task] task_id=%d model=%s regime=%s n=%s p=%s seed=%s\n",
            task_id, row$model, row$regime, row$n, row$p, row$seed))

script <- tryCatch(
  file.path(dirname(normalizePath(commandArgs(FALSE)[4])), "bench_r_models.R"),
  error = function(e) "bench_r_models.R"
)
if (!file.exists(script)) script <- "bench_r_models.R"

cmd <- c(
  "Rscript", script,
  "--model",     row$model,
  "--data-file", row$data_file,
  "--regime",    row$regime,
  "--n",         as.character(row$n),
  "--p",         as.character(row$p),
  "--seed",      as.character(row$seed),
  "--out",       out_path
)

cat(sprintf("  cmd: %s\n", paste(cmd, collapse = " ")))
ret <- system2(cmd[1], args = cmd[-1])
quit(status = ret)
