#!/usr/bin/env Rscript
# ============================================================
# 01_generate_data.R
# Parallel data generation for GLM benchmark.
# Parallelization is over independent dataset designs.
# ============================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(MASS)
  library(optparse)
  library(parallel)
})

option_list <- list(
  make_option("--outdir", default = "data_glm"),
  make_option("--seed", type = "integer", default = 20260427),
  make_option("--reps", type = "integer", default = 5),
  make_option("--workers", type = "integer", default = 1)
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

stable_log_link <- function(eta) {
  # Clamp eta so that exp(eta) does not overflow and remains numerically reasonable.
  # exp(8) ~ 2981, exp(-8) ~ 3e-4
  exp(pmin(pmax(eta, -8), 8))
}

gen_dense <- function(n, p, rho = 0.3, s0 = 10) {
  Sigma <- rho ^ abs(outer(seq_len(p), seq_len(p), "-"))
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  beta <- rep(0, p)
  beta[seq_len(min(s0, p))] <- seq(0.5, 0.1, length.out = min(s0, p))
  eta <- as.numeric(X %*% beta)
  y <- rpois(n, stable_log_link(eta))
  list(X = X, y = y, beta = beta)
}

gen_sparse <- function(n, p, density = 0.01, s0 = 10) {
  X <- Matrix::rsparsematrix(n, p, density = density)
  beta <- rep(0, p)
  beta[seq_len(min(s0, p))] <- seq(0.5, 0.1, length.out = min(s0, p))
  eta <- as.numeric(X %*% beta)
  y <- rpois(n, stable_log_link(eta))
  list(X = X, y = y, beta = beta)
}

gen_highdim <- function(n, p, s0 = 20) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta <- rep(0, p)
  beta[seq_len(s0)] <- sample(c(-1, 1), s0, replace = TRUE) * seq(0.5, 0.15, length.out = s0)
  eta <- as.numeric(X %*% beta)
  y <- rpois(n, stable_log_link(eta))
  list(X = X, y = y, beta = beta)
}

gen_sparse_highdim <- function(n, p, density = 0.01, s0 = 20) {
  X <- Matrix::rsparsematrix(n, p, density = density)
  beta <- rep(0, p)
  beta[seq_len(s0)] <- sample(c(-1, 1), s0, replace = TRUE) * seq(0.5, 0.15, length.out = s0)
  eta <- as.numeric(X %*% beta)
  y <- rpois(n, stable_log_link(eta))
  list(X = X, y = y, beta = beta)
}

write_dataset <- function(obj, outdir, id, regime, n, p, rep_id, density = NA_real_) {
  rds_file <- file.path(outdir, paste0(id, ".rds"))
  csv_x <- file.path(outdir, paste0(id, "_X.csv"))
  csv_y <- file.path(outdir, paste0(id, "_y.csv"))
  csv_beta <- file.path(outdir, paste0(id, "_beta.csv"))

  saveRDS(obj, rds_file)
  write.csv(as.matrix(obj$X), csv_x, row.names = FALSE)
  write.csv(data.frame(y = obj$y), csv_y, row.names = FALSE)
  write.csv(data.frame(beta = obj$beta), csv_beta, row.names = FALSE)

  data.frame(
    id = id, regime = regime, n = n, p = p, rep = rep_id, density = density,
    rds_file = rds_file, x_csv = csv_x, y_csv = csv_y, beta_csv = csv_beta,
    stringsAsFactors = FALSE
  )
}

make_design_table <- function(reps) {
  dense_settings <- data.frame(
    n = c(2000, 10000, 30000),
    p = c(200, 500, 500)
  )
  dense <- merge(dense_settings, data.frame(rep = seq_len(reps)))
  dense$regime <- "dense"
  dense$density <- NA_real_

  sparse_settings <- data.frame(
    n = c(10000, 30000, 10000, 30000),
    p = c(500, 500, 500, 500),
    density = c(0.01, 0.01, 0.05, 0.05)
  )
  sparse <- merge(sparse_settings, data.frame(rep = seq_len(reps)))
  sparse$regime <- "sparse"

  highdim_settings <- data.frame(
    n = c(300, 500),
    p = c(1000, 500)
  )
  highdim <- merge(highdim_settings, data.frame(rep = seq_len(reps)))
  highdim$regime <- "highdim"
  highdim$density <- NA_real_

  sparse_highdim_settings <- data.frame(
    n = c(300, 500),
    p = c(1000, 500),
    density = c(0.01, 0.01)
  )
  sparse_highdim <- merge(sparse_highdim_settings, data.frame(rep = seq_len(reps)))
  sparse_highdim$regime <- "sparse_highdim"

  common_cols <- c("regime", "n", "p", "density", "rep")
  rbind(dense[, common_cols], sparse[, common_cols],
        highdim[, common_cols], sparse_highdim[, common_cols])
}

make_id <- function(d) {
  if (d$regime == "dense") {
    sprintf("dense_n%d_p%d_rep%02d", d$n, d$p, d$rep)
  } else if (d$regime == "sparse") {
    den_tag <- gsub("\\.", "", as.character(d$density))
    sprintf("sparse_n%d_p%d_den%s_rep%02d", d$n, d$p, den_tag, d$rep)
  } else if (d$regime == "sparse_highdim") {
    den_tag <- gsub("\\.", "", as.character(d$density))
    sprintf("sparse_highdim_n%d_p%d_den%s_rep%02d", d$n, d$p, den_tag, d$rep)
  } else {
    sprintf("highdim_n%d_p%d_rep%02d", d$n, d$p, d$rep)
  }
}

run_one_generation <- function(task) {
  d <- task$design
  outdir <- task$outdir
  base_seed <- task$seed

  # Deterministic per-dataset seed independent of parallel scheduling.
  set.seed(base_seed + as.integer(d$rep) * 100000 + as.integer(d$n) + as.integer(d$p))

  id <- make_id(d)

  obj <- if (d$regime == "dense") {
    gen_dense(d$n, d$p)
  } else if (d$regime == "sparse") {
    gen_sparse(d$n, d$p, density = d$density)
  } else if (d$regime == "sparse_highdim") {
    gen_sparse_highdim(d$n, d$p, density = d$density)
  } else {
    gen_highdim(d$n, d$p)
  }

  cat("Generated", id, "\n")
  write_dataset(obj, outdir, id, d$regime, d$n, d$p, d$rep, d$density)
}

designs <- make_design_table(opt$reps)

tasks <- lapply(seq_len(nrow(designs)), function(i) {
  list(design = designs[i, ], outdir = opt$outdir, seed = opt$seed)
})

cat("Generation tasks:", length(tasks), " Workers:", opt$workers, "\n")

if (length(tasks) == 0) {
  manifest_df <- data.frame()
} else if (opt$workers <= 1 || .Platform$OS.type == "windows") {
  manifest_df <- do.call(rbind, lapply(tasks, run_one_generation))
} else {
  # Use mc.set.seed=FALSE because each task sets its own deterministic seed.
  manifest_df <- do.call(rbind, parallel::mclapply(tasks, run_one_generation,
                                                   mc.cores = opt$workers,
                                                   mc.set.seed = FALSE))
}

manifest_df <- manifest_df[order(manifest_df$regime, manifest_df$n, manifest_df$p,
                                 manifest_df$density, manifest_df$rep), ]

write.csv(manifest_df, file.path(opt$outdir, "manifest.csv"), row.names = FALSE)
cat("Saved manifest to", file.path(opt$outdir, "manifest.csv"), "\n")
