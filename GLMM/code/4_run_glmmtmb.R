#!/usr/bin/env Rscript

# Run GLMM simulations with glmmTMB
# Usage: Rscript run_glmmtmb.R --q 5 --cov_structure ar1 --rho 0.3 --replicate_id 1

suppressPackageStartupMessages({
  library(glmmTMB)
})

source("1_glmm_simulation_core.R")

fit_glmm_glmmtmb <- function(train_df, test_df, args) {
  # Build random effects formula
  if (args$q == 1L) {
    random_formula <- "(1 | group)"
  } else {
    slopes <- paste0("x", seq_len(args$q - 1L))
    random_formula <- paste0("(1 + ", paste(slopes, collapse = " + "), " | group)")
  }

  # Build full formula
  fixed_vars <- paste0("x", seq_len(args$p), collapse = " + ")
  formula_str <- paste0("y ~ ", fixed_vars, " + ", random_formula)

  fit <- tryCatch({
    glmmTMB::glmmTMB(
      as.formula(formula_str),
      data = train_df,
      family = binomial(link = "logit"),
      control = glmmTMB::glmmTMBControl(
        optCtrl = list(
          iter.max = args$glmmtmb_iter_max,
          eval.max = args$glmmtmb_eval_max
        )
      )
    )
  }, error = function(e) {
    return(NULL)
  })

  outcome <- list(status = if (!is.null(fit)) "ok" else "failed")

  list(fit = fit, outcome = outcome)
}

# Run main simulation
args <- parse_args(commandArgs(trailingOnly = TRUE))
run_simulation(args, fit_glmm_glmmtmb)
