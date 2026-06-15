#!/usr/bin/env Rscript
# ============================================================
# 04_summarize_results.R
# Robust summarizer:
#   - Handles different columns in R and Python result files.
#   - Keeps both R TRUE and Python True convergence flags.
#   - Preserves raw rows, including timeout/error rows.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option("--r-results", default = "results_R.csv"),
  make_option("--py-results", default = "results_python.csv"),
  make_option("--out-raw", default = "results_all_raw.csv"),
  make_option("--out-summary", default = "results_all_summary.csv")
)
opt <- parse_args(OptionParser(option_list = option_list))

read_if_exists <- function(path) {
  if (file.exists(path)) read.csv(path, stringsAsFactors = FALSE, check.names = FALSE) else NULL
}

normalize_converged <- function(x) {
  z <- tolower(trimws(as.character(x)))
  z %in% c("true", "t", "1", "yes")
}

is_empty_error <- function(x) {
  is.na(x) | trimws(as.character(x)) == "" | trimws(as.character(x)) == "NA"
}

bind_rows_fill <- function(dfs) {
  dfs <- Filter(Negate(is.null), dfs)
  if (length(dfs) == 0) return(data.frame())

  all_names <- unique(unlist(lapply(dfs, names)))
  aligned <- lapply(dfs, function(d) {
    missing <- setdiff(all_names, names(d))
    for (nm in missing) d[[nm]] <- NA
    d <- d[, all_names, drop = FALSE]
    d
  })
  do.call(rbind, aligned)
}

rres <- read_if_exists(opt$`r-results`)
pres <- read_if_exists(opt$`py-results`)

allres <- bind_rows_fill(list(rres, pres))

if (is.null(allres) || nrow(allres) == 0) {
  write.csv(data.frame(), opt$`out-raw`, row.names = FALSE)
  write.csv(data.frame(), opt$`out-summary`, row.names = FALSE)
  quit(save = "no")
}

# Ensure required columns exist even if one side failed to produce them.
required_cols <- c("regime", "n", "p", "density", "method",
                   "time", "loss", "mean_dev", "converged", "error")
for (nm in required_cols) {
  if (!nm %in% names(allres)) allres[[nm]] <- NA
}

allres$converged_clean <- normalize_converged(allres$converged)
allres$error_clean <- is_empty_error(allres$error)

write.csv(allres, opt$`out-raw`, row.names = FALSE)

ok <- subset(allres, converged_clean & error_clean)

if (nrow(ok) == 0) {
  write.csv(data.frame(), opt$`out-summary`, row.names = FALSE)
  cat("No converged rows retained for summary.\n")
  quit(save = "no")
}

# Coerce numeric columns defensively.
for (nm in c("n", "p", "density", "time", "loss", "mean_dev", "selected",
             "recall", "precision", "f1")) {
  if (!nm %in% names(ok)) ok[[nm]] <- NA_real_
  ok[[nm]] <- suppressWarnings(as.numeric(ok[[nm]]))
}

ok$density_key <- ifelse(is.na(ok$density), "__NA__", as.character(ok$density))

summarize_metric <- function(metric, prefix) {
  tmp <- aggregate(ok[[metric]],
                   by = list(regime = ok$regime,
                             n = ok$n,
                             p = ok$p,
                             density_key = ok$density_key,
                             method = ok$method),
                   FUN = function(z) c(mean = mean(z, na.rm = TRUE),
                                       sd = sd(z, na.rm = TRUE)))
  out <- data.frame(tmp[, 1:5], tmp$x)
  names(out)[6:7] <- paste0(prefix, c("_mean", "_sd"))
  out
}

time_sum <- summarize_metric("time", "time")
loss_sum <- summarize_metric("loss", "loss")
dev_sum <- summarize_metric("mean_dev", "mean_dev")
selected_sum <- summarize_metric("selected", "selected")
recall_sum <- summarize_metric("recall", "recall")
precision_sum <- summarize_metric("precision", "precision")
f1_sum <- summarize_metric("f1", "f1")

merge_keys <- c("regime", "n", "p", "density_key", "method")
summary <- Reduce(function(a, b) merge(a, b, by = merge_keys, all = TRUE),
                  list(time_sum, loss_sum, dev_sum, selected_sum,
                       recall_sum, precision_sum, f1_sum))

summary$density <- suppressWarnings(as.numeric(ifelse(summary$density_key == "__NA__", NA, summary$density_key)))
summary$density_key <- NULL

summary <- summary[, c("regime", "n", "p", "density", "method",
                       "time_mean", "time_sd",
                       "loss_mean", "loss_sd",
                       "mean_dev_mean", "mean_dev_sd",
                       "selected_mean", "selected_sd",
                       "recall_mean", "recall_sd",
                       "precision_mean", "precision_sd",
                       "f1_mean", "f1_sd")]

write.csv(summary, opt$`out-summary`, row.names = FALSE)

cat("Rows in R results:", ifelse(is.null(rres), 0, nrow(rres)), "\n")
cat("Rows in Python results:", ifelse(is.null(pres), 0, nrow(pres)), "\n")
cat("Rows in all results:", nrow(allres), "\n")
cat("Rows retained for summary:", nrow(ok), "\n")
cat("Methods retained:\n")
print(sort(unique(ok$method)))
cat("Saved raw results to", opt$`out-raw`, "\n")
cat("Saved summary results to", opt$`out-summary`, "\n")
