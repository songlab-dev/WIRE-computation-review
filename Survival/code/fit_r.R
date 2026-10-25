args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(jsonlite))
request <- fromJSON(args[1])
d <- request$data_dir
out <- request$output_dir
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)))
cfg <- fromJSON(file.path(dirname(script), "config.json"))
model <- request$model
seed <- as.integer(request$training_seed)
required <- cfg$required_r_versions
actual <- lapply(names(required), function(p) as.character(packageVersion(p)))
names(actual) <- names(required)
stopifnot(all(unlist(actual) == unlist(required)))
suppressPackageStartupMessages(library(survival))
if (model == "Coxnet") suppressPackageStartupMessages(library(glmnet))
if (model == "RSF") suppressPackageStartupMessages(library(randomForestSRC))
if (model == "Boosting") suppressPackageStartupMessages(library(gbm))
if (model == "DeepSurv") {
    suppressPackageStartupMessages(library(survivalmodels))
    suppressPackageStartupMessages(library(reticulate))
    py_run_string(paste0("import sys\nsys.path.insert(0, ", toJSON(dirname(script), auto_unbox = TRUE),
        ")"))
    py_run_string("import random, os, numpy as np, torch, pycox, torchtuples\nfrom common import versions\ntorch.set_num_threads(1)\ntorch.set_num_interop_threads(1)\ntorch.use_deterministic_algorithms(True)")
    deep_versions <- py$versions(cfg$required_deep_versions)
}
read_part <- function(name) read.csv(file.path(d, paste0(name, ".csv")), check.names = FALSE)
train <- read_part("train")
val <- read_part("val")
test <- read_part("test")
scaling <- read_part("scaling")
grid <- scan(file.path(d, "grid.csv"), quiet = TRUE)
features <- paste0("x", seq_len(request$p))
if (model == "Coxnet") penalties <- scan(file.path(d, "penalties.csv"), quiet = TRUE)
if (model == "DeepSurv") pool <- read_part("pool")
form <- as.formula(paste("Surv(time,event) ~", paste(features, collapse = "+")))
status <- function() {
    z <- readLines("/proc/self/status")
    pick <- function(key) as.numeric(strsplit(trimws(sub(paste0("^", key, ":"), "", grep(paste0("^",
        key, ":"), z, value = TRUE))), " +")[[1]][1])
    list(VmHWM = pick("VmHWM"), VmRSS = pick("VmRSS"), Threads = pick("Threads"), pid = Sys.getpid())
}
step_surv <- function(times, values, grid, initial = 1) {
    idx <- findInterval(grid, times)
    out <- matrix(initial, nrow = nrow(values), ncol = length(grid))
    out[, idx > 0] <- values[, idx[idx > 0], drop = FALSE]
    out
}
pll <- function(times, events, eta) {
    ord <- order(times, decreasing = TRUE)
    times <- times[ord]
    events <- events[ord]
    eta <- eta[ord]
    lengths <- rle(times)$lengths
    ends <- rep(cumsum(lengths), lengths)
    shift <- max(eta)
    risk <- cumsum(exp(eta - shift))
    use <- events == 1
    sum(eta[use] - log(risk[ends[use]]) - shift)
}
breslow <- function(times, events, eta, grid) {
    shift <- max(eta)
    w <- exp(eta - shift)
    ts <- sort(unique(times[events == 1]))
    h <- cumsum(vapply(ts, function(t) sum(times == t & events == 1)/sum(w[times >= t]), numeric(1)))
    idx <- findInterval(grid, ts)
    list(shift = shift, h = c(0, h)[idx + 1])
}
from_lp <- function(eta, bh) exp(-outer(exp(eta - bh$shift), bh$h))
standardize <- function(x) sweep(sweep(x, 2, scaling$mean, "-"), 2, scaling$scale, "/")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
options(rf.cores = 1L)
if (model == "DeepSurv") {
    py$random$seed(seed)
    py$np$random$seed(seed)
    py$torch$manual_seed(seed)
}
before <- status()
details <- list()
file.create(file.path(out, "fit_active"))
start <- proc.time()[["elapsed"]]
x <- as.matrix(train[, features])
xv <- as.matrix(val[, features])
if (model %in% c("Coxnet", "DeepSurv")) {
    stopifnot(max(abs(colMeans(x) - scaling$mean)) < 1e-12)
    stopifnot(max(abs(sqrt(colMeans(sweep(x, 2, colMeans(x), "-")^2)) - scaling$scale)) < 1e-12)
    x <- standardize(x)
    xv <- standardize(xv)
}
if (model == "CoxPH") {
    fit <- coxph(form, data = train, ties = "efron", x = TRUE, y = TRUE, model = TRUE, control = coxph.control(iter.max = 500L,
        eps = 1e-07))
    bh <- basehaz(fit, centered = FALSE)
    details <- list(ties = "Efron", baseline = "Efron adjusted", iterations = fit$iter)
} else if (model == "Coxnet") {
    ord <- order(train$time, decreasing = TRUE)
    xs <- x[ord, ]
    es <- train$event[ord] == 1
    lengths <- rle(train$time[ord])$lengths
    ends <- rep(cumsum(lengths), lengths)
    cumulative <- apply(xs, 2, cumsum)
    risk_mean <- cumulative[ends, , drop = FALSE]/ends
    maximum <- max(abs(colSums((xs - risk_mean)[es, , drop = FALSE])))/nrow(xs)
    derived <- exp(seq(log(maximum), log(maximum * 0.01), length.out = 100))
    stopifnot(max(abs(derived - penalties)) < 1e-12)
    glmnet.control(fdev = 0, devmax = 1)
    fit <- glmnet(x, Surv(train$time, train$event), family = "cox", alpha = 1, lambda = penalties,
        standardize = FALSE, thresh = cfg$coxnet$tolerance, maxit = as.integer(cfg$coxnet$max_iterations))
    stopifnot(length(fit$lambda) == length(penalties), max(abs(fit$lambda - penalties)) < 1e-12)
    coef <- as.matrix(fit$beta)
    eta_path <- xv %*% coef
    scores <- apply(eta_path, 2, function(eta) pll(val$time, val$event, eta))
    best <- which.max(scores)
    selected <- coef[, best]
    bh <- breslow(train$time, train$event, as.numeric(x %*% selected), grid)
    details <- list(selected_penalty = penalties[best], selected_index = best - 1, penalty_candidates = length(penalties),
        validation_partial_loglik = scores[best], native_path_length = length(fit$lambda),
        jerr = fit$jerr)
    stopifnot(fit$jerr == 0)
} else if (model == "RSF") {
    fit <- rfsrc(form, data = train[, c(features, "time", "event")], ntree = 200L, mtry = as.integer(floor(sqrt(request$p))),
        nodesize = 15L, nsplit = 0L, splitrule = "logrank", bootstrap = "by.root", samptype = "swr",
        sampsize = nrow(train), ntime = sort(unique(train$time[train$event == 1])), importance = FALSE,
        forest = TRUE, seed = seed, perf.type = "none")
    details <- list(trees = fit$ntree, mtry = fit$mtry, nodesize = 15, nsplit = 0, bootstrap_draws = nrow(train),
        rf_cores = randomForestSRC:::get.rf.cores(), prediction_times = length(fit$time.interest))
} else if (model == "Boosting") {
    fit <- gbm(form, data = train[, c(features, "time", "event")], distribution = "coxph",
        n.trees = 300L, interaction.depth = 3L, shrinkage = 0.05, n.minobsinnode = 15L, bag.fraction = 1,
        train.fraction = 1, cv.folds = 0L, n.cores = 1L, keep.data = TRUE, verbose = FALSE)
    eta <- as.numeric(predict(fit, newdata = train, n.trees = 300L, type = "link"))
    bh <- breslow(train$time, train$event, eta, grid)
    details <- list(trees = fit$n.trees, bag_fraction = fit$bag.fraction, interaction_depth = fit$interaction.depth,
        min_node = fit$n.minobsinnode)
} else if (model == "DeepSurv") {
    xp <- standardize(as.matrix(pool[, features]))
    fit <- deepsurv(x = as.data.frame(xp), y = Surv(pool$time, pool$event), frac = 0.25, activation = "relu",
        num_nodes = c(32L), batch_norm = FALSE, dropout = 0, device = "cpu", early_stopping = TRUE,
        best_weights = TRUE, min_delta = 0, patience = 20L, batch_size = as.integer(cfg$deepsurv$batch_size),
        epochs = 150L, learning_rate = 0.001, optimizer = "adam", betas = c(0.9, 0.999), eps = 1e-08,
        weight_decay = 0, amsgrad = FALSE, verbose = FALSE, num_workers = 0L, shuffle = TRUE)
    # No data argument: native pycox uses only its actual training_data.
    fit$model$compute_baseline_hazards()
    details <- list(training_rows = nrow(train), validation_rows = nrow(val), batch_size = cfg$deepsurv$batch_size,
        torch_threads = py$torch$get_num_threads(), torch_interop_threads = py$torch$get_num_interop_threads(),
        backend_pid = py$os$getpid())
}
elapsed <- proc.time()[["elapsed"]] - start
memory <- status()
unlink(file.path(out, "fit_active"))
record <- list(fitting_time_s = elapsed, peak_rss_kib = memory$VmHWM, peak_rss_mib = memory$VmHWM/1024,
    prefit_process = before, postfit_process = memory, environment = list(R = as.character(getRversion()),
        packages = actual), details = details)
if (model == "DeepSurv") record$environment$python <- deep_versions
write_json(record, file.path(out, "fit.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
if (model == "Coxnet") {
    write.csv(data.frame(penalty = penalties, validation_partial_loglik = scores), file.path(out,
        "penalty_selection.csv"), row.names = FALSE)
    write.table(selected, file.path(out, "selected_coefficients.csv"), row.names = FALSE, col.names = FALSE)
}
xt <- as.matrix(test[, features])
if (model %in% c("Coxnet", "DeepSurv")) xt <- standardize(xt)
if (model == "CoxPH") {
    eta <- as.numeric(predict(fit, newdata = test, type = "lp", reference = "zero"))
    h <- c(0, bh$hazard)[findInterval(grid, bh$time) + 1]
    pred <- exp(-outer(exp(eta), h))
} else if (model == "Coxnet") {
    pred <- from_lp(as.numeric(xt %*% selected), bh)
} else if (model == "RSF") {
    pr <- predict(fit, newdata = test[, c(features, "time", "event")], perf.type = "none")
    pred <- step_surv(pr$time.interest, pr$survival, grid)
} else if (model == "Boosting") {
    pred <- from_lp(as.numeric(predict(fit, newdata = test, n.trees = 300L, type = "link")),
        bh)
} else {
    py$native_model <- fit$model
    py$expected_x <- x
    py$expected_xv <- xv
    py$expected_time <- train$time
    py$expected_event <- train$event
    py$expected_val_time <- val$time
    py$expected_val_event <- val$event
    py$audit_path <- file.path(out, "deep_training_audit.npz")
    py$log_path <- file.path(out, "training_log.csv")
    py$test_x <- xt
    py$grid <- grid
    py$prediction_path <- file.path(out, "survival.csv")
    py_run_string(paste("from common import step_matrix", "td = native_model.training_data",
        "assert np.array_equal(td[0], np.asarray(expected_x, dtype='float32'))", "assert np.array_equal(td[1][0], np.asarray(expected_time, dtype='float32'))",
        "assert np.array_equal(td[1][1], np.asarray(expected_event, dtype='float32'))", "history = native_model.log.to_pandas()",
        "history.to_csv(log_path, index=False)", "assert np.isfinite(history.to_numpy()).all(), 'Non-finite DeepSurv training/validation loss despite restored checkpoint'",
        "np.savez(audit_path, x=td[0], xv=np.asarray(expected_xv,dtype='float32'), time=td[1][0], event=td[1][1], val_time=np.asarray(expected_val_time,dtype='float32'),val_event=np.asarray(expected_val_event,dtype='float32'))",
        "sf = native_model.predict_surv_df(np.asarray(test_x, dtype='float32'))", "pr = step_matrix(sf.index.to_numpy(), sf.to_numpy().T, np.asarray(grid))",
        "np.savetxt(prediction_path, pr, delimiter=',', fmt='%.17g')", sep = "\n"))
}
if (model != "DeepSurv") write.table(pred, file.path(out, "survival.csv"), sep = ",", row.names = FALSE,
    col.names = FALSE)
write.table(test$row_id, file.path(out, "test_row_ids.csv"), row.names = FALSE, col.names = FALSE)
capture.output(sessionInfo(), file = file.path(out, "R_sessionInfo.txt"))
