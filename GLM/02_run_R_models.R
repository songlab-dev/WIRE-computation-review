#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(glmnet)
  library(Matrix)
  library(optparse)
  library(parallel)
})

option_list <- list(
  make_option("--manifest", default = "data_glm/manifest.csv"),
  make_option("--out", default = "results_R.csv"),
  make_option("--regime", default = "all"),
  make_option("--workers", type = "integer", default = 1)
)
opt <- parse_args(OptionParser(option_list = option_list))

negloglik <- function(y, mu) {
  # Poisson negative log-likelihood (per observation), dropping the
  # log(y!) term that does not depend on the model parameters.
  eps <- 1e-8
  mu <- pmax(as.numeric(mu), eps)
  y <- as.numeric(y)
  -mean(y * log(mu) - mu)
}

mean_poisson_deviance <- function(y, mu) {
  eps <- 1e-8
  y <- as.numeric(y)
  mu <- pmax(as.numeric(mu), eps)
  # 2 * sum( y*log(y/mu) - (y - mu) )  with the convention 0*log(0) = 0
  term <- ifelse(y > 0, y * log(y / mu), 0) - (y - mu)
  mean(2 * term)
}

# Support-recovery metrics: recall, precision, F1.
# Returns NAs when no true support is available (e.g. for non-highdim regimes).
support_metrics <- function(beta_hat, beta_true, tol_hat = 1e-8, tol_true = 1e-12) {
  if (is.null(beta_true) || all(is.na(beta_true))) {
    return(list(recall = NA_real_, precision = NA_real_, f1 = NA_real_))
  }
  S_true <- which(abs(beta_true) > tol_true)
  S_hat  <- which(abs(beta_hat)  > tol_hat)
  tp <- length(intersect(S_hat, S_true))
  fp <- length(setdiff(S_hat, S_true))
  fn <- length(setdiff(S_true, S_hat))
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall    <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  list(recall = recall, precision = precision, f1 = f1)
}

run_stats_glm <- function(obj) {
  X <- as.matrix(obj$X)
  y <- obj$y
  df <- data.frame(y = y, X)
  t <- system.time({
    fit <- glm(y ~ ., data = df, family = poisson(), control = glm.control(maxit = 50))
  })
  pred <- predict(fit, type = "response")
  list(method = "R_stats_glm", time = as.numeric(t["elapsed"]),
       loss = negloglik(y, pred),
       mean_dev = mean_poisson_deviance(y, pred),
       selected = NA_integer_,
       recall = NA_real_, precision = NA_real_, f1 = NA_real_,
       converged = fit$converged, error = NA_character_)
}

run_glmnet <- function(obj) {
  X <- as.matrix(obj$X)
  y <- obj$y
  t <- system.time({
    fit <- cv.glmnet(X, y, family = "poisson", alpha = 1, nfolds = 5, parallel = FALSE)
  })
  pred <- as.numeric(predict(fit, X, s = "lambda.min", type = "response"))
  beta_hat <- as.numeric(coef(fit, s = "lambda.min"))[-1]
  sm <- support_metrics(beta_hat, obj$beta)
  list(method = "R_glmnet_lasso", time = as.numeric(t["elapsed"]),
       loss = negloglik(y, pred),
       mean_dev = mean_poisson_deviance(y, pred),
       selected = sum(beta_hat != 0),
       recall = sm$recall, precision = sm$precision, f1 = sm$f1,
       converged = TRUE, error = NA_character_)
}

make_tasks <- function(manifest) {
  tasks <- list()
  for (i in seq_len(nrow(manifest))) {
    m <- manifest[i, ]
    if (m$regime %in% c("dense", "sparse")) {
      tasks[[length(tasks) + 1]] <- list(row = m, method = "stats_glm")
    }
    if (m$regime %in% c("highdim", "sparse_highdim")) {
      tasks[[length(tasks) + 1]] <- list(row = m, method = "glmnet")
    }
  }
  tasks
}

run_one_task <- function(task) {
  m <- task$row
  method <- task$method
  cat("Running", method, "on", m$id, "\n")
  tryCatch({
    obj <- readRDS(m$rds_file)
    res <- if (method == "stats_glm") run_stats_glm(obj) else run_glmnet(obj)
    data.frame(id = m$id, regime = m$regime, n = m$n, p = m$p,
               rep = m$rep, density = m$density,
               method = res$method, time = res$time,
               loss = res$loss, mean_dev = res$mean_dev,
               selected = res$selected,
               recall = res$recall, precision = res$precision, f1 = res$f1,
               converged = res$converged,
               error = res$error)
  }, error = function(e) {
    data.frame(id = m$id, regime = m$regime, n = m$n, p = m$p,
               rep = m$rep, density = m$density,
               method = paste0("R_", method), time = NA_real_,
               loss = NA_real_, mean_dev = NA_real_,
               selected = NA_integer_,
               recall = NA_real_, precision = NA_real_, f1 = NA_real_,
               converged = FALSE,
               error = conditionMessage(e))
  })
}

manifest <- read.csv(opt$manifest, stringsAsFactors = FALSE)
if (opt$regime != "all") manifest <- subset(manifest, regime == opt$regime)

tasks <- make_tasks(manifest)
cat("R tasks:", length(tasks), " Workers:", opt$workers, "\n")

if (length(tasks) == 0) {
  results <- data.frame()
} else if (opt$workers <= 1 || .Platform$OS.type == "windows") {
  results <- do.call(rbind, lapply(tasks, run_one_task))
} else {
  results <- do.call(rbind, parallel::mclapply(tasks, run_one_task, mc.cores = opt$workers))
}

write.csv(results, opt$out, row.names = FALSE)
cat("Saved R results to", opt$out, "\n")
