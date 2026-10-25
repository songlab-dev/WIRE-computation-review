args <- commandArgs(TRUE)
out <- args[1]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages({
    library(survival)
    library(randomForestSRC)
    library(jsonlite)
})
options(rf.cores = 1)
set.seed(2026091)
a <- data.frame(x = 1:42, time = c(1, rep(100, 41)), event = c(1, rep(0, 41)))
one <- rfsrc(Surv(time, event) ~ x, a, ntree = 1, mtry = 1, nodesize = 15, nsplit = 0, bootstrap = "none",
    splitrule = "logrank", membership = TRUE, perf.type = "none", seed = 2026091)
leaf_sizes <- as.integer(table(one$membership[, 1]))
stopifnot(min(leaf_sizes) == 1, length(leaf_sizes) == 2)
b <- a[1:20, ]
samp <- matrix(c(rep(2L, 15), rep(0L, 5)), ncol = 1)
two <- rfsrc(Surv(time, event) ~ x, b, ntree = 1, mtry = 1, nodesize = 15, nsplit = 0, bootstrap = "by.user",
    samp = samp, samptype = "swr", sampsize = 30L, splitrule = "logrank", membership = TRUE,
    perf.type = "none", seed = 2026091)
weighted_leaves <- length(unique(two$membership[, 1]))
stopifnot(weighted_leaves > 1)
write_json(list(R = as.character(getRversion()), randomForestSRC = as.character(packageVersion("randomForestSRC")),
    parent_42_child_sizes = leaf_sizes, weighted_30_draws_15_unique_leaves = weighted_leaves,
    conclusion = "R nodesize constrains bootstrap draw counts in the parent; children can have fewer than nodesize observations."),
    file.path(out, "rsf_native_R.json"), pretty = TRUE, auto_unbox = TRUE)
# The native validation helper is called without fitting, to verify both ordered training
# and validation data, not just complementary row sets.
Sys.setenv(RETICULATE_PYTHON = path.expand(Sys.getenv("DEEP_PYTHON", "~/.conda/envs/deepsurv_env/bin/python")))
library(reticulate)
library(survivalmodels)
z <- data.frame(row_identifier = 1:400, x = seq_len(400)/10)
y <- data.frame(time = seq_len(400), status = rep(c(0, 1), 200))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(2026091)
val_idx <- sample(seq_len(400), 100)
set.seed(2026091)
native <- survivalmodels:::pycox_prepare_train_data(z, y, frac = 0.25, model = "deepsurv")
py$native_split <- native
py$expect_train <- as.matrix(z[-val_idx, ])
py$expect_val <- as.matrix(z[val_idx, ])
py_run_string("import numpy as np\nassert np.array_equal(native_split['x_train'], np.asarray(expect_train,dtype='float32'))\nassert np.array_equal(native_split['val'][0], np.asarray(expect_val,dtype='float32'))")
write_json(list(exact_ordered_training = TRUE, exact_ordered_validation = TRUE, n_train = 300,
    n_val = 100), file.path(out, "native_deep_split.json"), auto_unbox = TRUE, pretty = TRUE)
library(gbm)
set.seed(2026091)
fixture <- data.frame(time = rexp(300), event = rbinom(300, 1, 0.7), x1 = rnorm(300), x2 = rnorm(300))
fit <- gbm(Surv(time, event) ~ x1 + x2, fixture, distribution = "coxph", n.trees = 10, interaction.depth = 3,
    n.minobsinnode = 15, bag.fraction = 1, shrinkage = 0.05, train.fraction = 1, cv.folds = 0,
    n.cores = 1, verbose = FALSE)
checks <- lapply(1:10, function(i) {
    tree <- pretty.gbm.tree(fit, i.tree = i)
    # gbm creates fallback missing-value branches with inherited weights even when no
    # observation is missing. Count only ordinary reachable leaves.
    missing_ids <- tree$MissingNode[tree$MissingNode >= 0]
    leaf <- tree$SplitVar == -1 & !(seq_len(nrow(tree)) - 1L) %in% missing_ids
    list(nonempty_leaves = sum(leaf), min_nonempty_leaf_weight = min(tree$Weight[leaf]), splits = sum(tree$SplitVar >=
        0))
})
stopifnot(all(vapply(checks, function(x) x$nonempty_leaves <= 4 && x$splits <= 3 && x$min_nonempty_leaf_weight >=
    15, logical(1))))
write_json(checks, file.path(out, "gbm_native_tree_constraints.json"), auto_unbox = TRUE, pretty = TRUE)
