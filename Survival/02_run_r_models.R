rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
  library(dplyr)
  library(readr)
  library(randomForestSRC)
  library(gbm)
})

out_dir <- Sys.getenv("EXPERIMENT_DIR", "outputs/main_n500")
data_dir <- file.path(out_dir, "data")
results_dir <- file.path(out_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

choose_time_grid <- function(train, test, n_grid = 100) {
  lower <- as.numeric(quantile(test$time, 0.10, na.rm = TRUE))
  upper <- as.numeric(quantile(test$time, 0.90, na.rm = TRUE))
  train_max <- max(train$time, na.rm = TRUE)
  test_max <- max(test$time, na.rm = TRUE)
  train_events <- train$time[train$event == 1]
  train_event_max <- if (length(train_events) > 0) max(train_events, na.rm = TRUE) else train_max
  upper <- min(upper, train_event_max * 0.99, train_max * 0.99, test_max * 0.99)
  if (upper <= lower) {
    lower <- max(min(test$time, na.rm = TRUE), 1e-8)
    upper <- min(train_event_max * 0.99, train_max * 0.99, test_max * 0.99)
  }
  if (upper <= lower) {
    stop("Invalid time grid for IBS.")
  }
  seq(lower, upper, length.out = n_grid)
}

restrict_test_to_train_support <- function(train, test) {
  out <- test[test$time < max(train$time, na.rm = TRUE), , drop = FALSE]
  if (nrow(out) == 0) {
    stop("No test observations inside training follow-up support.")
  }
  out
}

get_censor_surv_function <- function(train) {
  fit_g <- survival::survfit(Surv(time, 1 - event) ~ 1, data = train)
  function(t) {
    s <- summary(fit_g, times = t, extend = TRUE)$surv
    pmax(s, 1e-6)
  }
}

compute_ipcw_ibs <- function(surv_mat, train, test, time_grid) {
  g_fun <- get_censor_surv_function(train)
  y_time <- test$time
  y_event <- test$event
  bs_vec <- numeric(length(time_grid))
  for (k in seq_along(time_grid)) {
    t0 <- time_grid[k]
    s_hat <- surv_mat[, k]
    g_t <- g_fun(t0)
    g_y <- g_fun(y_time)
    term1 <- as.numeric(y_time > t0) * (1 - s_hat)^2 / g_t
    term2 <- as.numeric(y_time <= t0 & y_event == 1) * (s_hat)^2 / g_y
    bs_vec[k] <- mean(term1 + term2, na.rm = TRUE)
  }
  if (length(time_grid) < 2) {
    return(mean(bs_vec, na.rm = TRUE))
  }
  sum(diff(time_grid) * (head(bs_vec, -1) + tail(bs_vec, -1)) / 2) /
    (max(time_grid) - min(time_grid))
}

compute_cindex <- function(test, risk_score) {
  cdat <- data.frame(
    time = test$time,
    event = test$event,
    risk_score = as.numeric(risk_score)
  )
  tryCatch(
    survival::concordance(
      Surv(time, event) ~ risk_score,
      data = cdat,
      reverse = TRUE
    )$concordance,
    error = function(e) NA_real_
  )
}

cox_partial_loglik <- function(time, event, eta) {
  ord <- order(time, decreasing = TRUE)
  event_ord <- as.integer(event[ord])
  eta_ord <- as.numeric(eta[ord])
  max_eta <- max(eta_ord)
  exp_eta <- exp(eta_ord - max_eta)
  cum_risk <- cumsum(exp_eta)
  event_idx <- which(event_ord == 1)
  if (length(event_idx) == 0) {
    return(-Inf)
  }
  sum(eta_ord[event_idx] - (log(cum_risk[event_idx]) + max_eta))
}

choose_lambda_by_validation <- function(fit, x_val, val) {
  eta_path <- predict(fit, newx = x_val, s = fit$lambda, type = "link")
  eta_path <- as.matrix(eta_path)
  pll <- apply(
    eta_path,
    2,
    function(eta) cox_partial_loglik(val$time, val$event, eta)
  )
  best_idx <- which.max(pll)
  list(lambda = fit$lambda[best_idx], validation_partial_loglik = pll[best_idx])
}

read_split <- function(setting, rep, split) {
  read_csv(
    file.path(data_dir, sprintf("%s_rep%03d_%s.csv", setting, rep, split)),
    show_col_types = FALSE
  )
}

predict_surv_cox <- function(fit, test, time_grid) {
  bh <- survival::basehaz(fit, centered = FALSE)
  lp <- as.numeric(predict(fit, newdata = test, type = "lp"))
  h0_t <- approx(
    x = bh$time,
    y = bh$hazard,
    xout = time_grid,
    method = "constant",
    rule = 2,
    f = 0
  )$y
  outer(exp(lp), h0_t, function(risk, h0) exp(-risk * h0))
}

run_coxph <- function(setting, rep, p) {
  train <- read_split(setting, rep, "train")
  test <- read_split(setting, rep, "test")
  x_cols <- paste0("x", seq_len(p))
  form <- as.formula(paste0("Surv(time, event) ~ ", paste(x_cols, collapse = " + ")))

  start_time <- proc.time()[["elapsed"]]
  fit <- survival::coxph(form, data = train, ties = "breslow", x = TRUE, y = TRUE, model = TRUE)
  runtime_sec <- proc.time()[["elapsed"]] - start_time

  test_eval <- restrict_test_to_train_support(train, test)
  time_grid <- choose_time_grid(train, test_eval)
  surv_mat <- predict_surv_cox(fit, test_eval, time_grid)
  ibs <- compute_ipcw_ibs(surv_mat, train, test_eval, time_grid)
  risk_score <- as.numeric(predict(fit, newdata = test, type = "lp"))

  data.frame(
    method = "survival_coxph",
    model_class = "CoxPH",
    language = "R",
    ibs = ibs,
    cindex_td = compute_cindex(test, risk_score),
    runtime_sec = runtime_sec,
    stringsAsFactors = FALSE
  )
}

predict_surv_rsf <- function(fit, test, time_grid) {
  pred <- predict(fit, newdata = test)
  pred_times <- pred$time.interest
  pred_surv <- pred$survival
  t(apply(pred_surv, 1, function(s) {
    approx(
      x = pred_times,
      y = s,
      xout = time_grid,
      method = "constant",
      rule = 2,
      f = 0
    )$y
  }))
}

run_rsf <- function(setting, rep, p) {
  train <- read_split(setting, rep, "train")
  test <- read_split(setting, rep, "test")
  x_cols <- paste0("x", seq_len(p))
  form <- as.formula(paste0("Surv(time, event) ~ ", paste(x_cols, collapse = " + ")))
  set.seed(100000 + rep)

  start_time <- proc.time()[["elapsed"]]
  fit <- randomForestSRC::rfsrc(
    formula = form,
    data = train,
    ntree = 200,
    mtry = max(1, floor(sqrt(p))),
    nodesize = 15,
    nsplit = 10,
    importance = FALSE,
    forest = TRUE,
    seed = 100000 + rep
  )
  runtime_sec <- proc.time()[["elapsed"]] - start_time

  test_eval <- restrict_test_to_train_support(train, test)
  time_grid <- choose_time_grid(train, test_eval)
  surv_mat <- predict_surv_rsf(fit, test_eval, time_grid)
  ibs <- compute_ipcw_ibs(surv_mat, train, test_eval, time_grid)
  risk_score <- as.numeric(predict(fit, newdata = test)$predicted)

  data.frame(
    method = "randomForestSRC_rfsrc",
    model_class = "RSF",
    language = "R",
    ibs = ibs,
    cindex_td = compute_cindex(test, risk_score),
    runtime_sec = runtime_sec,
    stringsAsFactors = FALSE
  )
}

fit_offset_cox <- function(lp_train, train) {
  train$lp_boost <- as.numeric(lp_train)
  survival::coxph(
    Surv(time, event) ~ offset(lp_boost),
    data = train,
    ties = "breslow",
    x = TRUE,
    y = TRUE,
    model = TRUE
  )
}

predict_surv_boosting <- function(offset_fit, lp_test, time_grid) {
  bh <- survival::basehaz(offset_fit, centered = FALSE)
  h0_t <- approx(
    x = bh$time,
    y = bh$hazard,
    xout = time_grid,
    method = "constant",
    rule = 2,
    f = 0
  )$y
  outer(exp(lp_test), h0_t, function(risk, h0) exp(-risk * h0))
}

run_boosting <- function(setting, rep, p) {
  train <- read_split(setting, rep, "train")
  test <- read_split(setting, rep, "test")
  x_cols <- paste0("x", seq_len(p))
  form <- as.formula(paste0("Surv(time, event) ~ ", paste(x_cols, collapse = " + ")))
  set.seed(100000 + rep)

  start_time <- proc.time()[["elapsed"]]
  fit <- gbm::gbm(
    formula = form,
    data = train,
    distribution = "coxph",
    n.trees = 300,
    interaction.depth = 3,
    shrinkage = 0.05,
    n.minobsinnode = 15,
    bag.fraction = 1.0,
    train.fraction = 1.0,
    verbose = FALSE
  )
  runtime_sec <- proc.time()[["elapsed"]] - start_time

  lp_train <- as.numeric(predict(fit, newdata = train, n.trees = 300, type = "link"))
  lp_test <- as.numeric(predict(fit, newdata = test, n.trees = 300, type = "link"))
  test_eval <- restrict_test_to_train_support(train, test)
  lp_eval <- as.numeric(predict(fit, newdata = test_eval, n.trees = 300, type = "link"))
  offset_fit <- fit_offset_cox(lp_train, train)
  time_grid <- choose_time_grid(train, test_eval)
  surv_mat <- predict_surv_boosting(offset_fit, lp_eval, time_grid)
  ibs <- compute_ipcw_ibs(surv_mat, train, test_eval, time_grid)

  data.frame(
    method = "gbm_coxph",
    model_class = "Boosting",
    language = "R",
    ibs = ibs,
    cindex_td = compute_cindex(test, lp_test),
    runtime_sec = runtime_sec,
    stringsAsFactors = FALSE
  )
}

run_glmnet <- function(setting, rep, p) {
  train <- read_split(setting, rep, "train")
  val <- read_split(setting, rep, "val")
  test <- read_split(setting, rep, "test")
  x_cols <- paste0("x", seq_len(p))
  x_train <- as.matrix(train[, x_cols])
  x_val <- as.matrix(val[, x_cols])
  y_train <- survival::Surv(train$time, train$event)
  set.seed(100000 + rep)

  start_time <- proc.time()[["elapsed"]]
  fit <- glmnet::glmnet(
    x = x_train,
    y = y_train,
    family = "cox",
    alpha = 1,
    nlambda = 100,
    lambda.min.ratio = 0.01,
    standardize = TRUE
  )
  selected <- choose_lambda_by_validation(fit, x_val, val)
  runtime_sec <- proc.time()[["elapsed"]] - start_time
  beta_hat <- as.numeric(coef(fit, s = selected$lambda))
  names(beta_hat) <- x_cols

  train$lp_glmnet <- as.numeric(as.matrix(train[, x_cols]) %*% beta_hat)
  offset_fit <- survival::coxph(
    Surv(time, event) ~ offset(lp_glmnet),
    data = train,
    ties = "breslow",
    x = TRUE,
    y = TRUE,
    model = TRUE
  )
  test_eval <- restrict_test_to_train_support(train, test)
  time_grid <- choose_time_grid(train, test_eval)
  bh <- survival::basehaz(offset_fit, centered = FALSE)
  h0_t <- approx(
    x = bh$time,
    y = bh$hazard,
    xout = time_grid,
    method = "constant",
    rule = 2,
    f = 0
  )$y
  lp_eval <- as.numeric(as.matrix(test_eval[, x_cols]) %*% beta_hat)
  surv_mat <- outer(exp(lp_eval), h0_t, function(risk, h0) exp(-risk * h0))
  ibs <- compute_ipcw_ibs(surv_mat, train, test_eval, time_grid)
  risk_score <- as.numeric(as.matrix(test[, x_cols]) %*% beta_hat)

  data.frame(
    method = "glmnet_validation_glmnet",
    model_class = "Penalized Cox",
    language = "R",
    ibs = ibs,
    cindex_td = compute_cindex(test, risk_score),
    runtime_sec = runtime_sec,
    stringsAsFactors = FALSE
  )
}

meta <- read_csv(file.path(data_dir, "metadata.csv"), show_col_types = FALSE) %>%
  distinct(setting, setting_type, rep, n, p, s, rho)

all_results <- list()
for (i in seq_len(nrow(meta))) {
  cfg <- meta[i, ]
  setting <- cfg$setting
  rep <- cfg$rep
  p <- cfg$p
  runners <- if (cfg$setting_type == "highdim") {
    list(run_glmnet, run_rsf, run_boosting)
  } else {
    list(run_coxph, run_rsf, run_boosting)
  }

  for (runner in runners) {
    runner_name <- deparse(substitute(runner))
    cat(sprintf("running R %s %s rep=%03d\n", runner_name, setting, rep))
    result <- tryCatch(
      runner(setting, rep, p),
      error = function(e) {
        data.frame(
          method = runner_name,
          model_class = "unknown",
          language = "R",
          ibs = NA_real_,
          cindex_td = NA_real_,
          runtime_sec = NA_real_,
          status = "failed",
          error_message = e$message,
          stringsAsFactors = FALSE
        )
      }
    )
    if (!("status" %in% names(result))) {
      result$status <- "success"
      result$error_message <- ""
    }
    all_results[[length(all_results) + 1]] <- bind_cols(cfg, result)
  }
}

results <- bind_rows(all_results)
write_csv(results, file.path(results_dir, "r_models.csv"))
cat("Wrote", file.path(results_dir, "r_models.csv"), "\n")
