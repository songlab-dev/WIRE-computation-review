rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(survivalmodels)
  library(reticulate)
  library(dplyr)
  library(readr)
})

out_dir <- normalizePath(
  Sys.getenv("EXPERIMENT_DIR", "outputs/main_n500"),
  mustWork = FALSE
)
data_dir <- file.path(out_dir, "data")
results_dir <- file.path(out_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
# Keep temporary pycox checkpoints separate for concurrent experiments.
setwd(results_dir)

set_all_seeds <- function(seed) {
  seed <- as.integer(seed)
  set.seed(seed)
  py_random <- reticulate::import("random")
  py_numpy <- reticulate::import("numpy")
  py_torch <- reticulate::import("torch")
  py_random$seed(seed)
  py_numpy$random$seed(seed)
  py_torch$manual_seed(seed)
  if (py_torch$cuda$is_available()) {
    py_torch$cuda$manual_seed_all(seed)
  }
}

choose_time_grid <- function(train, test, n_grid = 100) {
  lower <- as.numeric(quantile(test$time, 0.10, na.rm = TRUE))
  upper <- as.numeric(quantile(test$time, 0.90, na.rm = TRUE))
  train_max <- max(train$time, na.rm = TRUE)
  test_max <- max(test$time, na.rm = TRUE)
  train_events <- train$time[train$event == 1]
  train_event_max <- if (length(train_events) > 0) max(train_events, na.rm = TRUE) else train_max
  upper <- min(
    upper,
    train_event_max * 0.99,
    train_max * 0.99,
    test_max * 0.99
  )
  if (upper <= lower) {
    lower <- max(min(test$time, na.rm = TRUE), 1e-8)
    upper <- min(
      train_event_max * 0.99,
      train_max * 0.99,
      test_max * 0.99
    )
  }
  if (upper <= lower) {
    stop("Invalid time grid for IBS evaluation.")
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

standardize_x <- function(train, val, test, x_cols) {
  x_train <- as.matrix(train[, x_cols])
  x_val <- as.matrix(val[, x_cols])
  x_test <- as.matrix(test[, x_cols])
  x_mean <- colMeans(x_train)
  x_sd <- apply(x_train, 2, sd)
  x_sd[x_sd == 0] <- 1
  train[, x_cols] <- scale(x_train, center = x_mean, scale = x_sd)
  val[, x_cols] <- scale(x_val, center = x_mean, scale = x_sd)
  test[, x_cols] <- scale(x_test, center = x_mean, scale = x_sd)
  list(train = train, val = val, test = test)
}

get_censor_surv_function <- function(train) {
  fit_g <- survival::survfit(Surv(time, 1 - event) ~ 1, data = train)
  function(t) {
    s <- summary(fit_g, times = t, extend = TRUE)$surv
    pmax(s, 1e-6)
  }
}

compute_ibs_manual <- function(surv_mat, train, test, time_grid) {
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
    term2 <- as.numeric(y_time <= t0 & y_event == 1) * s_hat^2 / g_y
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

format_survival_matrix <- function(pred, n_test, time_grid) {
  pred_df <- as.data.frame(pred)
  surv_mat_raw <- as.matrix(pred_df)
  if (length(dim(surv_mat_raw)) != 2) {
    stop("Prediction object cannot be converted to a 2D survival matrix.")
  }
  if (nrow(surv_mat_raw) == n_test) {
    # Already subjects x internal times.
  } else if (ncol(surv_mat_raw) == n_test) {
    surv_mat_raw <- t(surv_mat_raw)
  } else {
    stop(sprintf(
      "Unexpected survival prediction dimension: %s; expected n_test=%d.",
      paste(dim(surv_mat_raw), collapse = " x "),
      n_test
    ))
  }

  pred_times <- suppressWarnings(as.numeric(colnames(pred_df)))
  if (all(!is.na(pred_times)) && length(pred_times) == ncol(surv_mat_raw)) {
    surv_mat <- t(apply(surv_mat_raw, 1, function(s) {
      approx(
        x = pred_times,
        y = s,
        xout = time_grid,
        method = "constant",
        rule = 2,
        f = 0
      )$y
    }))
  } else {
    internal_grid <- seq_len(ncol(surv_mat_raw))
    target_grid <- seq(
      from = 1,
      to = ncol(surv_mat_raw),
      length.out = length(time_grid)
    )
    surv_mat <- t(apply(surv_mat_raw, 1, function(s) {
      approx(
        x = internal_grid,
        y = s,
        xout = target_grid,
        method = "constant",
        rule = 2,
        f = 0
      )$y
    }))
  }

  surv_mat <- pmin(pmax(surv_mat, 0), 1)
  if (nrow(surv_mat) != n_test || ncol(surv_mat) != length(time_grid)) {
    stop(sprintf(
      "Final survival matrix has wrong dimension: %s; expected %d x %d.",
      paste(dim(surv_mat), collapse = " x "),
      n_test,
      length(time_grid)
    ))
  }
  surv_mat
}

read_split <- function(setting, rep, split) {
  read_csv(
    file.path(data_dir, sprintf("%s_rep%03d_%s.csv", setting, rep, split)),
    show_col_types = FALSE
  )
}

run_one <- function(cfg) {
  setting <- cfg$setting
  rep <- cfg$rep
  p <- cfg$p
  set_all_seeds(880000 + rep)
  x_cols <- paste0("x", seq_len(p))

  train <- read_split(setting, rep, "train")
  val <- read_split(setting, rep, "val")
  test <- read_split(setting, rep, "test")

  std <- standardize_x(train, val, test, x_cols)
  train <- std$train
  val <- std$val
  test <- std$test
  train_val <- bind_rows(train, val)
  y_train_val <- survival::Surv(train_val$time, train_val$event)

  start_time <- proc.time()[["elapsed"]]
  fit <- survivalmodels::deepsurv(
    x = train_val[, x_cols],
    y = y_train_val,
    frac = 0.25,
    activation = "relu",
    num_nodes = c(32L),
    batch_norm = FALSE,
    dropout = 0.0,
    early_stopping = TRUE,
    best_weights = TRUE,
    patience = 20L,
    batch_size = 128L,
    epochs = 150L,
    learning_rate = 1e-3,
    verbose = FALSE,
    num_workers = 0L,
    shuffle = TRUE
  )
  runtime_sec <- proc.time()[["elapsed"]] - start_time

  test_eval <- restrict_test_to_train_support(train, test)
  time_grid <- choose_time_grid(train, test_eval)
  pred <- predict(
    fit,
    newdata = test[, x_cols],
    times = time_grid,
    type = "survival"
  )
  surv_mat <- format_survival_matrix(pred, nrow(test), time_grid)
  eval_mask <- test$time < max(train$time, na.rm = TRUE)
  ibs <- compute_ibs_manual(
    surv_mat[eval_mask, , drop = FALSE],
    train,
    test_eval,
    time_grid
  )
  risk_score <- rowMeans(1 - surv_mat, na.rm = TRUE)

  data.frame(
    setting = setting,
    setting_type = cfg$setting_type,
    rep = rep,
    n = cfg$n,
    p = p,
    s = cfg$s,
    rho = cfg$rho,
    method = "survivalmodels_deepsurv",
    model_class = "DeepSurv",
    language = "R",
    ibs = ibs,
    cindex_td = compute_cindex(test, risk_score),
    runtime_sec = runtime_sec,
    status = "success",
    error_message = "",
    stringsAsFactors = FALSE
  )
}

meta <- read_csv(file.path(data_dir, "metadata.csv"), show_col_types = FALSE) %>%
  distinct(setting, setting_type, rep, n, p, s, rho)

cat("Reticulate Python configuration:\n")
print(reticulate::py_config())

all_results <- list()
for (i in seq_len(nrow(meta))) {
  cfg <- meta[i, ]
  cat(sprintf("running R DeepSurv %s rep=%03d\n", cfg$setting, cfg$rep))
  result <- tryCatch(
    run_one(cfg),
    error = function(e) {
      data.frame(
        setting = cfg$setting,
        setting_type = cfg$setting_type,
        rep = cfg$rep,
        n = cfg$n,
        p = cfg$p,
        s = cfg$s,
        rho = cfg$rho,
        method = "survivalmodels_deepsurv",
        model_class = "DeepSurv",
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
  all_results[[length(all_results) + 1]] <- result
}

results <- bind_rows(all_results)
write_csv(results, file.path(results_dir, "r_deepsurv.csv"))
cat("Wrote", file.path(results_dir, "r_deepsurv.csv"), "\n")
