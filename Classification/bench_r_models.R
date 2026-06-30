#!/usr/bin/env Rscript
# Train one R model on a unified CSV.gz dataset and write a one-row result file.
#
#   Rscript bench_r_models.R --model glm_logit \
#     --data-file data/dense_n10000_p50_s1.csv.gz \
#     --regime dense --n 10000 --p 50 --seed 1 \
#     --out results/run_v2/r_results.csv

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(peakRAM)
  library(pROC)
})

opt_list <- list(
  make_option("--model",     type = "character"),
  make_option("--data-file", type = "character"),
  make_option("--regime",    type = "character"),
  make_option("--n",         type = "integer"),
  make_option("--p",         type = "integer"),
  make_option("--seed",      type = "integer", default = 1L),
  make_option("--out",       type = "character", default = "results/r_results.csv")
)
opt <- parse_args(OptionParser(option_list = opt_list))

model_key <- opt$model
data_file <- opt[["data-file"]]
regime    <- opt$regime
n_obs     <- opt$n
p_feat    <- opt$p
seed      <- opt$seed
out_path  <- opt$out

set.seed(seed)

cat(sprintf("[bench_r] model=%s regime=%s n=%d p=%d seed=%d\n",
            model_key, regime, n_obs, p_feat, seed))

load_data <- function(data_file, regime) {
  df <- as.data.frame(data.table::fread(data_file))
  y  <- as.integer(df$y)
  df$y <- NULL
  if (regime == "mixed") {
    cat_cols <- grep("^cat_", names(df), value = TRUE)
    for (col in cat_cols) df[[col]] <- as.character(df[[col]])
  }
  list(X = df, y = y)
}

dat  <- load_data(data_file, regime)
X_df <- dat$X
y_all <- dat$y

cat(sprintf("  loaded: %d rows x %d cols  y_mean=%.3f\n",
            nrow(X_df), ncol(X_df), mean(y_all)))

stratified_split <- function(y, frac = 0.8, seed = 1L) {
  set.seed(seed)
  idx0 <- which(y == 0L); idx1 <- which(y == 1L)
  n0 <- round(frac * length(idx0)); n1 <- round(frac * length(idx1))
  tr <- c(sample(idx0, n0), sample(idx1, n1))
  list(tr = tr, te = setdiff(seq_along(y), tr))
}

sp   <- stratified_split(y_all, seed = seed)
tr   <- sp$tr
te   <- sp$te
y_tr <- y_all[tr]
y_te <- y_all[te]

# One-hot encode cat_* columns, pass num_* through (for linear models).
make_design <- function(X_df, fit_enc = NULL) {
  num_cols <- grep("^num_", names(X_df), value = TRUE)
  cat_cols <- grep("^cat_", names(X_df), value = TRUE)
  X_num <- as.matrix(X_df[, num_cols, drop = FALSE])
  if (length(cat_cols) == 0) return(list(mat = X_num, enc = NULL))
  col_enc <- if (is.null(fit_enc)) {
    lapply(cat_cols, function(col) list(col = col, levs = sort(unique(X_df[[col]]))))
  } else {
    fit_enc
  }
  oh_blocks <- lapply(col_enc, function(e) {
    col_f <- factor(X_df[[e$col]], levels = e$levs)
    model.matrix(~ col_f - 1)
  })
  list(mat = cbind(X_num, do.call(cbind, oh_blocks)), enc = col_enc)
}

# Convert cat_* columns to integer codes (for tree models).
factor_to_int <- function(X_df) {
  df2 <- X_df
  cat_cols <- grep("^cat_", names(df2), value = TRUE)
  for (col in cat_cols) df2[[col]] <- as.integer(factor(df2[[col]])) - 1L
  as.matrix(df2)
}

compute_metrics <- function(y_true, pred, proba = NULL, score = NULL) {
  tn <- sum(pred == 0L & y_true == 0L)
  fp <- sum(pred == 1L & y_true == 0L)
  fn <- sum(pred == 0L & y_true == 1L)
  tp <- sum(pred == 1L & y_true == 1L)

  accuracy    <- (tp + tn) / length(y_true)
  recall      <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  prec        <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  prec0       <- if ((tn + fn) > 0) tn / (tn + fn) else 0
  recall0     <- specificity
  f1_1 <- if ((prec + recall)   > 0) 2 * prec  * recall  / (prec  + recall)  else 0
  f1_0 <- if (!is.na(recall0) && (prec0 + recall0) > 0)
            2 * prec0 * recall0 / (prec0 + recall0) else 0
  f1   <- (f1_0 + f1_1) / 2

  ll  <- NA_real_
  auc <- NA_real_
  p1  <- if (!is.null(proba)) {
    if (is.matrix(proba) || is.data.frame(proba)) proba[, 2] else as.numeric(proba)
  } else if (!is.null(score)) {
    as.numeric(score)
  } else NULL

  if (!is.null(p1)) {
    p1  <- pmin(pmax(p1, 1e-6), 1 - 1e-6)
    ll  <- -mean(y_true * log(p1) + (1 - y_true) * log(1 - p1))
    roc_obj <- suppressMessages(pROC::roc(as.factor(y_true), p1, quiet = TRUE))
    auc <- as.numeric(pROC::auc(roc_obj))
  }

  list(accuracy = accuracy, recall = recall, specificity = specificity,
       f1 = f1, log_loss = ll, auroc = auc)
}

# Atomic per-task one-row file: write to tmp then rename. Merge afterward.
write_row <- function(row, out) {
  fields <- c("run_id", "regime", "method", "package", "implementation", "solver",
              "hardware", "n", "p", "sparsity", "categorical_levels", "seed",
              "train_time_s", "predict_time_s", "peak_memory_mb", "gpu_peak_memory_mb",
              "accuracy", "recall", "specificity", "f1", "log_loss", "auroc", "notes")
  rows_dir <- paste0(sub("\\.csv$", "", out), "_rows")
  dir.create(rows_dir, recursive = TRUE, showWarnings = FALSE)
  row_dt <- as.data.table(lapply(row, function(x) if (is.null(x)) NA else x))
  row_dt <- row_dt[, fields[fields %in% names(row_dt)], with = FALSE]
  for (f in fields[!fields %in% names(row_dt)]) row_dt[[f]] <- NA
  setcolorder(row_dt, fields)
  rid   <- gsub("/", "_", as.character(row[["run_id"]]))
  hw    <- tolower(gsub("/", "_", as.character(row[["hardware"]])))
  if (is.na(hw) || hw == "") hw <- "cpu"
  final <- file.path(rows_dir, paste0(rid, "__", hw, ".csv"))
  tmp   <- paste0(final, ".", Sys.getpid(), ".tmp")
  fwrite(row_dt, tmp)
  file.rename(tmp, final)
}

cat_levels_str <- function(X_df) {
  cat_cols <- grep("^cat_", names(X_df), value = TRUE)
  if (length(cat_cols) == 0) return("")
  paste(sapply(cat_cols, function(col) length(unique(X_df[[col]]))), collapse = ",")
}

cat_levels <- if (regime == "mixed") cat_levels_str(X_df) else ""

run_id <- sprintf("%s_%s_n%d_p%d_s%d", model_key, regime, n_obs, p_feat, seed)

emit <- function(method, package, impl, solver,
                 train_t, pred_t, mem_mb, mets, note = "") {
  row <- list(
    run_id             = run_id,
    regime             = regime,
    method             = method,
    package            = package,
    implementation     = impl,
    solver             = solver,
    hardware           = "cpu",
    n                  = n_obs,
    p                  = p_feat,
    sparsity           = "",
    categorical_levels = cat_levels,
    seed               = seed,
    train_time_s       = round(train_t, 6),
    predict_time_s     = round(pred_t, 6),
    peak_memory_mb     = round(mem_mb, 2),
    gpu_peak_memory_mb = "",
    accuracy           = round(mets$accuracy,    6),
    recall             = round(mets$recall,      6),
    specificity        = if (is.na(mets$specificity)) NA else round(mets$specificity, 6),
    f1                 = round(mets$f1,          6),
    log_loss           = if (is.na(mets$log_loss)) NA else round(mets$log_loss, 6),
    auroc              = if (is.na(mets$auroc))   NA else round(mets$auroc,    6),
    notes              = note
  )
  write_row(row, out_path)
  cat(sprintf("  done  train=%.2fs pred=%.2fs acc=%.4f auroc=%s\n",
              train_t, pred_t,
              mets$accuracy,
              if (is.na(mets$auroc)) "NA" else sprintf("%.4f", mets$auroc)))
  invisible(row)
}

tryCatch({

if (model_key == "glm_logit") {
  suppressPackageStartupMessages(library(stats))
  X_tr_df <- X_df[tr, ]; X_tr_df$y <- y_tr
  X_te_df <- X_df[te, ]
  if (regime == "mixed") {
    cat_cols <- grep("^cat_", names(X_tr_df), value = TRUE)
    for (col in cat_cols) {
      levs <- sort(unique(X_tr_df[[col]]))
      X_tr_df[[col]] <- factor(X_tr_df[[col]], levels = levs)
      X_te_df[[col]] <- factor(X_te_df[[col]], levels = levs)
    }
  }
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- glm(y ~ ., data = X_tr_df, family = binomial())
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, newdata = X_te_df, type = "response")
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("logistic", "base_R", "cpu", "irls",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "speedglm_logit") {
  suppressPackageStartupMessages(library(speedglm))
  X_tr_df <- X_df[tr, ]; X_tr_df$y <- y_tr
  X_te_df <- X_df[te, ]
  if (regime == "mixed") {
    cat_cols <- grep("^cat_", names(X_tr_df), value = TRUE)
    for (col in cat_cols) {
      levs <- sort(unique(X_tr_df[[col]]))
      X_tr_df[[col]] <- factor(X_tr_df[[col]], levels = levs)
      X_te_df[[col]] <- factor(X_te_df[[col]], levels = levs)
    }
  }
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- speedglm(y ~ ., data = X_tr_df, family = binomial())
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, newdata = X_te_df, type = "response")
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("logistic", "speedglm", "cpu", "speedglm_irls",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key %in% c("glmnet_ridge", "glmnet_lasso")) {
  suppressPackageStartupMessages(library(glmnet))
  alpha_val <- if (model_key == "glmnet_ridge") 0 else 1
  enc <- make_design(X_df[tr, ])
  X_tr_mat <- enc$mat
  X_te_mat <- make_design(X_df[te, ], fit_enc = enc$enc)$mat
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- cv.glmnet(X_tr_mat, y_tr, alpha = alpha_val, family = "binomial",
                     nfolds = 5, parallel = FALSE)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- as.numeric(predict(fit, newx = X_te_mat,
                              s = "lambda.min", type = "response"))
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  penalty <- if (alpha_val == 0) "ridge" else "lasso"
  emit("logistic", "glmnet", "cpu", paste0("glmnet_", penalty),
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "liblinear_svm") {
  suppressPackageStartupMessages(library(LiblineaR))
  enc <- make_design(X_df[tr, ])
  X_tr_mat <- enc$mat
  X_te_mat <- make_design(X_df[te, ], fit_enc = enc$enc)$mat
  col_means <- colMeans(X_tr_mat)
  col_sds   <- apply(X_tr_mat, 2, sd)
  col_sds[col_sds == 0] <- 1
  X_tr_sc <- scale(X_tr_mat, center = col_means, scale = col_sds)
  X_te_sc <- scale(X_te_mat, center = col_means, scale = col_sds)
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- LiblineaR(data = X_tr_sc, target = factor(y_tr), type = 1L)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    res <- predict(fit, newx = X_te_sc, decisionValues = TRUE)
    pred_t <- proc.time()[3] - t2
  })
  pred  <- as.integer(as.character(res$predictions))
  score <- as.numeric(res$decisionValues[, 1])
  mets  <- compute_metrics(y_te, pred, score = score)
  emit("linear_svm", "LiblineaR", "cpu", "liblinear_l2",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "rf_randomForest") {
  suppressPackageStartupMessages(library(randomForest))
  X_tr_mat <- factor_to_int(X_df[tr, ])
  X_te_mat <- factor_to_int(X_df[te, ])
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- randomForest(x = X_tr_mat, y = factor(y_tr), ntree = 300)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, X_te_mat, type = "prob")[, "1"]
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("random_forest", "randomForest", "cpu", "bagged_trees",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "rf_ranger") {
  suppressPackageStartupMessages(library(ranger))
  X_tr_df2 <- X_df[tr, ]
  X_te_df2 <- X_df[te, ]
  if (regime == "mixed") {
    cat_cols <- grep("^cat_", names(X_tr_df2), value = TRUE)
    for (col in cat_cols) {
      levs <- sort(unique(X_tr_df2[[col]]))
      X_tr_df2[[col]] <- factor(X_tr_df2[[col]], levels = levs)
      X_te_df2[[col]] <- factor(X_te_df2[[col]], levels = levs)
    }
  }
  X_tr_df2$y <- factor(y_tr)
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- ranger(y ~ ., data = X_tr_df2, num.trees = 300,
                  probability = TRUE, seed = seed, num.threads = 1)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, data = X_te_df2)$predictions[, "1"]
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("random_forest", "ranger", "cpu", "bagged_trees_oob",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "r_xgboost") {
  suppressPackageStartupMessages(library(xgboost))
  X_tr_mat <- factor_to_int(X_df[tr, ])
  X_te_mat <- factor_to_int(X_df[te, ])
  dtrain <- xgb.DMatrix(X_tr_mat, label = y_tr)
  dtest  <- xgb.DMatrix(X_te_mat)
  params <- list(objective = "binary:logistic", max_depth = 6L,
                 eta = 0.05, subsample = 0.8, colsample_bytree = 0.8,
                 nthread = 1L, seed = seed)
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- xgb.train(params = params, data = dtrain, nrounds = 100L, verbose = 0)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, dtest)
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("gbdt", "xgboost", "cpu", "hist",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "r_lgbm") {
  suppressPackageStartupMessages(library(lightgbm))
  X_tr_mat <- factor_to_int(X_df[tr, ])
  X_te_mat <- factor_to_int(X_df[te, ])
  dtrain <- lgb.Dataset(X_tr_mat, label = y_tr)
  params <- list(objective = "binary", learning_rate = 0.05,
                 num_leaves = 63L, subsample = 0.8, colsample_bytree = 0.8,
                 num_threads = 1L, verbose = -1L, seed = seed)
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- lgb.train(params = params, data = dtrain, nrounds = 100L)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- predict(fit, X_te_mat)
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("gbdt", "lightgbm", "cpu", "histogram",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else if (model_key == "r_catboost") {
  suppressPackageStartupMessages(library(catboost))
  cat_cols <- grep("^cat_", names(X_df), value = TRUE)
  X_tr_df2 <- X_df[tr, ]
  X_te_df2 <- X_df[te, ]
  if (length(cat_cols) > 0) {
    levs_list <- lapply(cat_cols, function(col) sort(unique(X_tr_df2[[col]])))
    for (i in seq_along(cat_cols)) {
      col <- cat_cols[i]
      X_tr_df2[[col]] <- factor(X_tr_df2[[col]], levels = levs_list[[i]])
      X_te_df2[[col]] <- factor(X_te_df2[[col]], levels = levs_list[[i]])
    }
  }
  pool_tr <- catboost.load_pool(data = X_tr_df2, label = y_tr)
  pool_te <- catboost.load_pool(data = X_te_df2)
  params  <- list(iterations = 300L, learning_rate = 0.05, depth = 8L,
                  loss_function = "Logloss", eval_metric = "AUC",
                  logging_level = "Silent", random_seed = seed)
  pm <- peakRAM({
    t1  <- proc.time()[3]
    fit <- catboost.train(learn_pool = pool_tr, params = params)
    train_t <- proc.time()[3] - t1
    t2  <- proc.time()[3]
    pr  <- catboost.predict(fit, pool_te, prediction_type = "Probability")
    pred_t <- proc.time()[3] - t2
  })
  pred <- as.integer(pr >= 0.5)
  mets <- compute_metrics(y_te, pred, proba = pr)
  emit("gbdt", "catboost", "cpu", "ordered_boosting",
       train_t, pred_t, pm$Peak_RAM_Used_MiB[1], mets)

} else {
  stop(sprintf("Unknown model: %s", model_key))
}

}, error = function(e) {
  msg <- conditionMessage(e)
  cat(sprintf("  ERROR: %s\n", msg))
  row <- list(
    run_id = run_id, regime = regime, method = "", package = model_key,
    implementation = "cpu", solver = "", hardware = "cpu",
    n = n_obs, p = p_feat, sparsity = "", categorical_levels = cat_levels,
    seed = seed, train_time_s = NA, predict_time_s = NA,
    peak_memory_mb = NA, gpu_peak_memory_mb = "",
    accuracy = NA, recall = NA, specificity = NA, f1 = NA,
    log_loss = NA, auroc = NA,
    notes = paste0("ERROR:", gsub("\n", " ", msg))
  )
  write_row(row, out_path)
})

cat(sprintf("Done. Row written to %s\n", out_path))
