#!/usr/bin/env Rscript

# GLMM Simulation Core Engine
# Fits logistic GLMM and evaluates performance metrics
#
# Design:
#   Y_ij ~ Bernoulli(logit^{-1}(X_ij beta + Z_ij b_i))
#   n_groups = 200, m = 50, p = 50 by default
#   q in {1, 5, 15}
#   covariance: identity or AR(1) with rho in {0.3, 0.7}
#   q = 1 uses random intercepts; q > 1 uses (1, x1, ..., x_{q-1})
#
# Metrics computed: runtime, convergence rate, fixed-effect sup-norm error,
#                   random-effect sup-norm error

suppressPackageStartupMessages({
  library(stats)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

parse_args <- function(args) {
  out <- list(
    method = "lme4",
    n_groups = 200L,
    m = 50L,
    p = 50L,
    q = 1L,
    sigma_b2 = 1,
    beta_active = 0.5,
    cov_structure = "identity",
    rho = 0.5,
    fit_cov_structure = "auto",
    replicate_id = 1L,
    seed = 1L,
    test_frac = 0.2,
    split_strategy = "within_cluster",
    output_dir = "results",
    skip_existing = 0L,
    mc_sjsdm = 10000L,
    iter_sjsdm = 200L,
    glmmtmb_iter_max = 3000L,
    glmmtmb_eval_max = 3000L
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key))
    }
    key <- substring(key, 3L)
    if (i == length(args)) {
      stop(sprintf("Missing value for --%s", key))
    }
    out[[key]] <- args[[i + 1L]]
    i <- i + 2L
  }

  int_keys <- c("n_groups", "m", "p", "q", "replicate_id", "seed",
                "skip_existing", "mc_sjsdm", "iter_sjsdm",
                "glmmtmb_iter_max", "glmmtmb_eval_max")
  dbl_keys <- c("test_frac", "sigma_b2", "rho", "beta_active")
  for (nm in int_keys) out[[nm]] <- as.integer(out[[nm]])
  for (nm in dbl_keys) out[[nm]] <- as.numeric(out[[nm]])

  out$method <- tolower(out$method)
  out$cov_structure <- tolower(out$cov_structure)
  out$fit_cov_structure <- tolower(out$fit_cov_structure)
  out$split_strategy <- tolower(out$split_strategy)

  if (!out$split_strategy %in% c("within_cluster", "cluster")) {
    stop("split_strategy must be one of: within_cluster, cluster")
  }

  out
}

make_sigma <- function(q, sigma_b2, cov_structure, rho) {
  if (cov_structure == "identity") {
    return(diag(sigma_b2, q))
  }
  if (cov_structure == "ar1") {
    idx <- seq_len(q)
    return(sigma_b2 * rho^abs(outer(idx, idx, "-")))
  }
  stop("cov_structure must be one of: identity, ar1")
}

simulate_random_effects <- function(n_groups, sigma) {
  q <- ncol(sigma)
  chol_sigma <- chol(sigma)
  matrix(rnorm(n_groups * q), nrow = n_groups, ncol = q) %*% chol_sigma
}

simulate_dataset <- function(n_groups, m, p, q, sigma_b2, beta_active, cov_structure, rho, seed) {
  set.seed(seed)
  n_obs <- n_groups * m
  group <- rep(seq_len(n_groups), each = m)
  x <- matrix(rnorm(n_obs * p), nrow = n_obs, ncol = p)
  colnames(x) <- paste0("x", seq_len(p))
  beta <- c(rep(beta_active, min(15L, p)), rep(0, max(0L, p - 15L)))

  if (q == 1L) {
    z <- matrix(1, nrow = n_obs, ncol = 1L)
    colnames(z) <- "(Intercept)"
  } else {
    z <- cbind("(Intercept)" = 1, x[, seq_len(q - 1L), drop = FALSE])
  }

  sigma <- make_sigma(q, sigma_b2, cov_structure, rho)
  b <- simulate_random_effects(n_groups, sigma)
  colnames(b) <- colnames(z)

  eta <- as.numeric(x %*% beta) + rowSums(z * b[group, , drop = FALSE])
  prob <- plogis(pmax(pmin(eta, 30), -30))
  y <- rbinom(n_obs, size = 1L, prob = prob)

  list(data = data.frame(y = y, group = factor(group), x, check.names = FALSE),
       beta = beta, b = b, sigma = sigma)
}

split_by_cluster <- function(df, test_frac, seed) {
  set.seed(seed)
  group_levels <- levels(df$group)
  n_test_groups <- max(1L, floor(length(group_levels) * test_frac))
  test_groups <- sample(group_levels, size = n_test_groups, replace = FALSE)
  test_index <- which(df$group %in% test_groups)
  train_index <- which(!(df$group %in% test_groups))
  list(train_index = train_index, test_index = test_index)
}

split_within_cluster <- function(df, test_frac, seed) {
  set.seed(seed)
  group_index <- split(seq_len(nrow(df)), df$group)
  test_index <- unlist(lapply(group_index, function(idx) {
    n_test <- max(1L, floor(length(idx) * test_frac))
    sample(idx, size = n_test, replace = FALSE)
  }), use.names = FALSE)
  train_index <- setdiff(seq_len(nrow(df)), test_index)
  list(train_index = train_index, test_index = test_index)
}

split_dataset <- function(df, test_frac, seed, split_strategy) {
  if (split_strategy == "cluster") {
    split_by_cluster(df, test_frac, seed)
  } else {
    split_within_cluster(df, test_frac, seed)
  }
}

extract_beta_sup_norm <- function(fit, method, p) {
  expected <- paste0("x", seq_len(p))
  beta_hat <- rep(NA_real_, p)
  names(beta_hat) <- expected

  raw <- tryCatch({
    if (method == "lme4") {
      lme4::fixef(fit)
    } else if (method == "glmmtmb") {
      glmmTMB::fixef(fit)$cond
    } else if (method == "sjsdm") {
      coef_obj <- coef(fit)
      env_coef <- if (is.list(coef_obj)) coef_obj$env else coef_obj
      if (is.list(env_coef) && length(env_coef) == 1L) env_coef <- env_coef[[1L]]
      if (is.matrix(env_coef) || is.data.frame(env_coef)) {
        as.numeric(env_coef)[seq_len(min(length(env_coef), p))]
      } else {
        as.numeric(env_coef)
      }
    } else {
      NULL
    }
  }, error = function(e) NULL)

  if (is.null(raw)) return(NA_real_)
  if (!is.null(names(raw))) {
    matched <- raw[intersect(expected, names(raw))]
    beta_hat[names(matched)] <- as.numeric(matched)
  } else if (length(raw) >= p) {
    beta_hat[] <- as.numeric(raw[seq_len(p)])
  }

  truth_beta <- c(rep(0.5, min(15L, p)), rep(0, max(0L, p - 15L)))
  beta_diff <- beta_hat - truth_beta
  finite_beta_diff <- beta_diff[is.finite(beta_diff)]
  if (length(finite_beta_diff) > 0) max(abs(finite_beta_diff)) else NA_real_
}

extract_sigma_sup_norm <- function(fit, method, q) {
  vc <- tryCatch({
    if (method == "lme4") {
      lme4::VarCorr(fit)
    } else if (method == "glmmtmb") {
      glmmTMB::VarCorr(fit)$cond
    } else {
      NULL
    }
  }, error = function(e) NULL)

  if (is.null(vc)) return(NA_real_)

  sigma_hat <- matrix(NA_real_, q, q)
  mats <- lapply(vc, as.matrix)
  if (length(mats) == 1L && all(dim(mats[[1L]]) >= q)) {
    sigma_hat <- mats[[1L]][seq_len(q), seq_len(q), drop = FALSE]
  }

  sigma_true <- diag(1, q)
  sigma_diff <- sigma_hat - sigma_true
  finite_sigma_diff <- sigma_diff[is.finite(sigma_diff)]
  if (length(finite_sigma_diff) > 0) max(abs(finite_sigma_diff)) else NA_real_
}

extract_convergence <- function(fit, method) {
  if (method == "lme4") {
    opt <- fit@optinfo
    messages <- unlist(opt$conv$lme4$messages %||% character(), use.names = FALSE)
    opt_code <- opt$conv$opt %||% 0L
    identical(as.integer(opt_code), 0L) && length(messages) == 0L
  } else if (method == "glmmtmb") {
    !is.null(fit) && isTRUE(fit$fit$convergence == 0)
  } else if (method == "sjsdm") {
    !is.null(fit)
  } else {
    NA
  }
}

compute_metrics <- function(train_df, test_df, truth_beta, truth_sigma, fit, method, p, q) {
  converged <- extract_convergence(fit, method)

  if (!isTRUE(converged)) {
    return(list(
      converged = converged,
      beta_sup_norm_error = NA_real_,
      sigma_sup_norm_error = NA_real_
    ))
  }

  beta_sup_norm_error <- extract_beta_sup_norm(fit, method, p)
  sigma_sup_norm_error <- extract_sigma_sup_norm(fit, method, q)

  list(
    converged = converged,
    beta_sup_norm_error = beta_sup_norm_error,
    sigma_sup_norm_error = sigma_sup_norm_error
  )
}

format_results <- function(args, outcome, metrics, runtime_sec) {
  data.frame(
    method = args$method,
    n_groups = args$n_groups,
    m = args$m,
    p = args$p,
    q = args$q,
    sigma_b2 = args$sigma_b2,
    beta_active = args$beta_active,
    cov_structure = args$cov_structure,
    rho = args$rho,
    split_strategy = args$split_strategy,
    replicate_id = args$replicate_id,
    seed = args$seed,
    mc_sjsdm = args$mc_sjsdm,
    glmmtmb_iter_max = args$glmmtmb_iter_max,
    glmmtmb_eval_max = args$glmmtmb_eval_max,
    runtime_sec = runtime_sec,
    convergence_status = if (isTRUE(metrics$converged)) "ok" else "failed",
    converged = metrics$converged,
    beta_sup_norm_error = metrics$beta_sup_norm_error,
    sigma_sup_norm_error = metrics$sigma_sup_norm_error,
    fit_status = outcome$status,
    stringsAsFactors = FALSE
  )
}

save_results <- function(results, output_dir, args) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  filename <- sprintf(
    "result_%s_%s_q%d_%s_sb%.3f_rho%.4f_%s_rep%03d_mc%d.csv",
    args$method,
    gsub(" ", "_", args$cov_structure),
    args$q,
    args$split_strategy,
    args$sigma_b2,
    args$rho,
    format(Sys.Date(), "%Y%m%d"),
    args$replicate_id,
    args$mc_sjsdm
  )

  filepath <- file.path(output_dir, filename)
  write.csv(results, filepath, row.names = FALSE)
  filepath
}

# Main execution wrapper - to be called from method-specific runners
run_simulation <- function(args, fit_func) {
  output_dir <- args$output_dir
  out_filename <- sprintf(
    "result_%s_q%d_%s_sb%.3f_rho%.4f_rep%03d.csv",
    args$method, args$q, args$cov_structure,
    args$sigma_b2, args$rho, args$replicate_id
  )
  out_path <- file.path(output_dir, out_filename)

  if (args$skip_existing == 1L && file.exists(out_path)) {
    cat("Skipping existing:", out_filename, "\n")
    return(invisible())
  }

  # Generate data
  sim <- simulate_dataset(args$n_groups, args$m, args$p, args$q,
                          args$sigma_b2, args$beta_active,
                          args$cov_structure, args$rho, args$seed)

  # Split data
  split <- split_dataset(sim$data, args$test_frac, args$seed, args$split_strategy)
  train_df <- sim$data[split$train_index, ]
  test_df <- sim$data[split$test_index, ]

  # Fit model and time it
  fit_time <- system.time({
    fit_result <- fit_func(train_df, test_df, args)
    fit <- fit_result$fit
    outcome <- fit_result$outcome
  })
  runtime_sec <- as.numeric(fit_time["elapsed"])

  # Compute metrics
  metrics <- compute_metrics(train_df, test_df, sim$beta, sim$sigma,
                             fit, args$method, args$p, args$q)

  # Format and save results
  results <- format_results(args, outcome, metrics, runtime_sec)
  save_results(results, output_dir, args)

  invisible()
}
