#!/usr/bin/env Rscript

# Run GLMM simulations with sjSDM (q=1 only)
# Usage: Rscript run_sjsdm.R --q 1 --mc_sjsdm 10000 --replicate_id 1

suppressPackageStartupMessages({
  library(sjSDM)
})

source("1_glmm_simulation_core.R")

fit_glmm_sjsdm <- function(train_df, test_df, args) {
  # sjSDM only supports q=1 (random intercept)
  if (args$q != 1L) {
    stop("sjSDM only supports q=1 (random intercept model)")
  }

  # Prepare data for sjSDM
  X_train <- as.matrix(train_df[, paste0("x", seq_len(args$p))])
  y_train <- train_df$y
  groups_train <- as.numeric(train_df$group)

  fit <- tryCatch({
    sjSDM::sjSDM(
      Y = y_train,
      X = X_train,
      env_formula = ~group,
      sigma = 1,
      mc = args$mc_sjsdm,
      iter = args$iter_sjsdm,
      verbose = FALSE
    )
  }, error = function(e) {
    return(NULL)
  })

  outcome <- list(status = if (!is.null(fit)) "ok" else "failed")

  list(fit = fit, outcome = outcome)
}

# Run main simulation
args <- parse_args(commandArgs(trailingOnly = TRUE))

# Verify q=1 for sjSDM
if (args$q != 1L) {
  stop("sjSDM only supports q=1. Requested q=", args$q)
}

run_simulation(args, fit_glmm_sjsdm)
