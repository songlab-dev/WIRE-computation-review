#!/usr/bin/env Rscript
# ============================================================
# 05_generate_paper_tables.R
# Generate publication-ready tables (booktabs) from benchmark results.
# The regime tables go into the main paper:
#   table_dense_regime.tex   (all dense settings)
#   table_sparse_regime.tex  (all sparse settings)
#   table_highdim_regime.tex (all highdim settings, with Selected and F1)
#   table_sparse_highdim_regime.tex
#       (high-dimensional sparse-design settings, with Selected and F1)
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option("--summary", default = "results_all_summary.csv"),
  make_option("--raw", default = "results_all_raw.csv"),
  make_option("--out-dir", default = "paper_tables"),
  make_option("--reps", type = "integer", default = NA_integer_,
              help = "Number of repetitions (printed in table notes). If NA, inferred from raw results.")
)
opt <- parse_args(OptionParser(option_list = option_list))

dir.create(opt$`out-dir`, showWarnings = FALSE, recursive = TRUE)

# ----- Formatting helpers -----------------------------------------------------

format_num <- function(x, digits = 3) {
  ifelse(is.na(x), "--",
         ifelse(abs(x) < 0.001 & x != 0, sprintf("%.2e", x),
                sprintf(sprintf("%%.%df", digits), x)))
}

format_time <- function(x) {
  # Always seconds, three significant-ish digits, to keep columns narrow.
  ifelse(is.na(x), "--",
         ifelse(x >= 100, sprintf("%.1f", x),
                ifelse(x >= 1, sprintf("%.2f", x),
                       sprintf("%.3f", x))))
}

format_int <- function(x) {
  ifelse(is.na(x), "--", sprintf("%d", as.integer(round(x))))
}

# Map raw method ids to short, paper-friendly labels.
nice_method <- function(m) {
  map <- c(
    "R_stats_glm"            = "R glm",
    "R_glmnet_lasso"         = "glmnet",
    "PY_statsmodels_glm"     = "statsmodels",
    "PY_sklearn_unpenalized" = "sklearn",
    "PY_glum_unpenalized"    = "glum",
    "PY_glum_lasso"          = "glum (Lasso, 5-fold CV)",
    "PY_skglm_default"       = "skglm (Lasso, 5-fold CV)",
    "PY_spglm"               = "spglm"
  )
  out <- map[m]
  out[is.na(out)] <- m[is.na(out)]
  unname(out)
}

method_order <- function(regime) {
  # Order in which we want methods to appear inside each (n, p)/(n, p, delta) block.
  if (regime %in% c("highdim", "sparse_highdim")) {
    c("glmnet", "glum (Lasso, 5-fold CV)", "skglm (Lasso, 5-fold CV)")
  } else {
    c("R glm", "statsmodels", "sklearn", "glum", "spglm")
  }
}

# Read inputs ------------------------------------------------------------------

summary_df <- read.csv(opt$summary, stringsAsFactors = FALSE, check.names = FALSE)

raw_df <- if (file.exists(opt$raw)) {
  read.csv(opt$raw, stringsAsFactors = FALSE, check.names = FALSE)
} else NULL

if (nrow(summary_df) == 0) {
  cat("No results to summarize.\n")
  quit(save = "no")
}

# Infer REPS for the table notes if not provided.
reps_for_notes <- opt$reps
if (is.na(reps_for_notes) && !is.null(raw_df) && nrow(raw_df) > 0 && "rep" %in% names(raw_df)) {
  reps_for_notes <- length(unique(raw_df$rep))
}
if (is.na(reps_for_notes)) reps_for_notes <- 5L

# Add display labels and apply ordering helper.
summary_df$method_label <- nice_method(summary_df$method)

order_block <- function(df, regime) {
  ord <- method_order(regime)
  rank <- match(df$method_label, ord)
  rank[is.na(rank)] <- length(ord) + 1L
  df[order(rank, df$method_label), , drop = FALSE]
}

# ----- Table builders ---------------------------------------------------------

# Compact column block for runtime + loss + deviance (used in dense/sparse).
fit_cols_unpen <- function(d) {
  data.frame(
    runtime  = format_time(d$time_mean),
    loss     = format_num(d$loss_mean, 3),
    dev      = format_num(d$mean_dev_mean, 3),
    stringsAsFactors = FALSE
  )
}

# Dense table: rows grouped by (n, p), one method per row.
build_dense_latex <- function(df) {
  df <- df[df$regime == "dense", , drop = FALSE]
  if (nrow(df) == 0) return(NULL)
  df <- df[order(df$n, df$p), ]

  out <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\small",
    "\\caption{Dense moderate-dimensional Poisson regression: unpenalized GLM benchmark.}",
    "\\label{tab:dense}",
    "\\begin{tabular}{rrlrrr}",
    "\\toprule",
    "$n$ & $p$ & Method & Runtime (s) & Loss & Mean dev. \\\\",
    "\\midrule"
  )

  blocks <- split(df, list(df$n, df$p), drop = TRUE)
  # Re-sort the block keys by numeric (n, p).
  block_keys <- do.call(rbind, lapply(names(blocks), function(k) {
    sub <- blocks[[k]][1, ]
    data.frame(key = k, n = sub$n, p = sub$p, stringsAsFactors = FALSE)
  }))
  block_keys <- block_keys[order(block_keys$n, block_keys$p), ]

  for (i in seq_len(nrow(block_keys))) {
    sub <- order_block(blocks[[block_keys$key[i]]], "dense")
    fmt <- fit_cols_unpen(sub)
    n_lab <- sub$n
    p_lab <- sub$p
    for (j in seq_len(nrow(sub))) {
      n_print <- if (j == 1) format(n_lab[j], big.mark = ",") else ""
      p_print <- if (j == 1) sprintf("%d", p_lab[j]) else ""
      out <- c(out, sprintf("%s & %s & %s & %s & %s & %s \\\\",
                            n_print, p_print, sub$method_label[j],
                            fmt$runtime[j], fmt$loss[j], fmt$dev[j]))
    }
    if (i < nrow(block_keys)) out <- c(out, "\\midrule")
  }

  note <- sprintf(
    paste("\\multicolumn{6}{p{0.95\\linewidth}}{\\footnotesize",
          "\\textit{Note.} Runtime is in seconds; Loss is the per-observation",
          "Poisson negative log-likelihood (without the $\\log(y!)$ term);",
          "Mean dev.\\ is the mean Poisson deviance.",
          "Results are averaged over %d repetitions.}"),
    reps_for_notes
  )

  out <- c(out,
           "\\bottomrule",
           "\\addlinespace[0.4ex]",
           paste0(note, " \\\\"),
           "\\end{tabular}",
           "\\end{table}",
           "")
  out
}

# Sparse table: rows grouped by (n, p, density). Includes all sparse settings
# present in the summary (the design grid is controlled by 01_generate_data.R).
build_sparse_latex <- function(df) {
  df <- df[df$regime == "sparse", , drop = FALSE]
  if (nrow(df) == 0) return(NULL)
  df <- df[order(df$density, df$n), ]

  out <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\small",
    "\\caption{Sparse-design Poisson regression: unpenalized GLM benchmark on sparse $X$.}",
    "\\label{tab:sparse}",
    "\\begin{tabular}{rrrlrrr}",
    "\\toprule",
    "$n$ & $p$ & $\\delta$ & Method & Runtime (s) & Loss & Mean dev. \\\\",
    "\\midrule"
  )

  blocks <- split(df, list(df$density, df$n), drop = TRUE)
  block_keys <- do.call(rbind, lapply(names(blocks), function(k) {
    sub <- blocks[[k]][1, ]
    data.frame(key = k, n = sub$n, density = sub$density, stringsAsFactors = FALSE)
  }))
  block_keys <- block_keys[order(block_keys$density, block_keys$n), ]

  for (i in seq_len(nrow(block_keys))) {
    sub <- order_block(blocks[[block_keys$key[i]]], "sparse")
    fmt <- fit_cols_unpen(sub)
    for (j in seq_len(nrow(sub))) {
      n_print   <- if (j == 1) format(sub$n[j], big.mark = ",") else ""
      p_print   <- if (j == 1) sprintf("%d", sub$p[j])         else ""
      den_print <- if (j == 1) sprintf("%.2f", sub$density[j]) else ""
      out <- c(out, sprintf("%s & %s & %s & %s & %s & %s & %s \\\\",
                            n_print, p_print, den_print, sub$method_label[j],
                            fmt$runtime[j], fmt$loss[j], fmt$dev[j]))
    }
    if (i < nrow(block_keys)) out <- c(out, "\\midrule")
  }

  note <- sprintf(
    paste("\\multicolumn{7}{p{0.95\\linewidth}}{\\footnotesize",
          "\\textit{Note.} Runtime is in seconds; Loss is the per-observation",
          "Poisson negative log-likelihood (without the $\\log(y!)$ term);",
          "Mean dev.\\ is the mean Poisson deviance;",
          "$\\delta$ is the nonzero entry probability of the sparse design matrix.",
          "Results are averaged over %d repetitions.}"),
    reps_for_notes
  )

  out <- c(out,
           "\\bottomrule",
           "\\addlinespace[0.4ex]",
           paste0(note, " \\\\"),
           "\\end{tabular}",
           "\\end{table}",
           "")
  out
}

# Highdim table: rows grouped by (n, p), with Selected and F1.
build_highdim_latex <- function(df) {
  df <- df[df$regime == "highdim", , drop = FALSE]
  if (nrow(df) == 0) return(NULL)
  df <- df[order(df$n, df$p), ]

  has_f1 <- "f1_mean" %in% names(df) && any(!is.na(df$f1_mean))

  if (has_f1) {
    out <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\small",
      "\\caption{High-dimensional penalized Poisson regression: support recovery and computational cost.}",
      "\\label{tab:highdim}",
      "\\begin{tabular}{rrlrrrrr}",
      "\\toprule",
      "$n$ & $p$ & Method & Runtime (s) & Loss & Mean dev. & Selected & F1 \\\\",
      "\\midrule"
    )
  } else {
    out <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\small",
      "\\caption{High-dimensional penalized Poisson regression: support recovery and computational cost.}",
      "\\label{tab:highdim}",
      "\\begin{tabular}{rrlrrrr}",
      "\\toprule",
      "$n$ & $p$ & Method & Runtime (s) & Loss & Mean dev. & Selected \\\\",
      "\\midrule"
    )
  }

  blocks <- split(df, list(df$n, df$p), drop = TRUE)
  block_keys <- do.call(rbind, lapply(names(blocks), function(k) {
    sub <- blocks[[k]][1, ]
    data.frame(key = k, n = sub$n, p = sub$p, stringsAsFactors = FALSE)
  }))
  block_keys <- block_keys[order(block_keys$n, block_keys$p), ]

  for (i in seq_len(nrow(block_keys))) {
    sub <- order_block(blocks[[block_keys$key[i]]], "highdim")
    runtime <- format_time(sub$time_mean)
    loss    <- format_num(sub$loss_mean, 3)
    dev     <- format_num(sub$mean_dev_mean, 3)
    selected <- format_int(sub$selected_mean)
    f1       <- format_num(sub$f1_mean, 3)
    for (j in seq_len(nrow(sub))) {
      n_print <- if (j == 1) format(sub$n[j], big.mark = ",") else ""
      p_print <- if (j == 1) format(sub$p[j], big.mark = ",") else ""
      if (has_f1) {
        out <- c(out, sprintf("%s & %s & %s & %s & %s & %s & %s & %s \\\\",
                              n_print, p_print, sub$method_label[j],
                              runtime[j], loss[j], dev[j],
                              selected[j], f1[j]))
      } else {
        out <- c(out, sprintf("%s & %s & %s & %s & %s & %s & %s \\\\",
                              n_print, p_print, sub$method_label[j],
                              runtime[j], loss[j], dev[j], selected[j]))
      }
    }
    if (i < nrow(block_keys)) out <- c(out, "\\midrule")
  }

  base_note <- paste(
    "\\textit{Note.} Runtime is in seconds; Loss is the per-observation",
    "Poisson negative log-likelihood (without the $\\log(y!)$ term);",
    "Mean dev.\\ is the mean Poisson deviance;",
    "Selected denotes the number of nonzero coefficients.",
    "The regularization parameter $\\lambda$ is chosen by 5-fold cross-validation",
    "for each method.",
    sprintf("Results are averaged over %d repetitions.", reps_for_notes)
  )

  if (has_f1) {
    full_note <- paste(
      base_note,
      "F1 is the harmonic mean of precision and recall of the selected support",
      "against the true support of $\\beta^\\ast$."
    )
    note <- sprintf(
      "\\multicolumn{8}{p{0.95\\linewidth}}{\\footnotesize %s}",
      full_note
    )
  } else {
    full_note <- paste(
      base_note,
      "F1 score is not reported because true/false selection counts were",
      "not saved by the current benchmark pipeline."
    )
    note <- sprintf(
      "\\multicolumn{7}{p{0.95\\linewidth}}{\\footnotesize %s}",
      full_note
    )
  }

  out <- c(out,
           "\\bottomrule",
           "\\addlinespace[0.4ex]",
           paste0(note, " \\\\"),
           "\\end{tabular}",
           "\\end{table}",
           "")
  out
}

# Sparse highdim table: rows grouped by (n, p, density), with Selected and F1.
build_sparse_highdim_latex <- function(df) {
  df <- df[df$regime == "sparse_highdim", , drop = FALSE]
  if (nrow(df) == 0) return(NULL)
  df <- df[order(df$density, df$n, df$p), ]

  has_f1 <- "f1_mean" %in% names(df) && any(!is.na(df$f1_mean))

  if (has_f1) {
    out <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\small",
      "\\caption{High-dimensional penalized Poisson regression with sparse design matrices.}",
      "\\label{tab:sparse-highdim}",
      "\\begin{tabular}{rrrlrrrrr}",
      "\\toprule",
      "$n$ & $p$ & $\\delta$ & Method & Runtime (s) & Loss & Mean dev. & Selected & F1 \\\\",
      "\\midrule"
    )
  } else {
    out <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\small",
      "\\caption{High-dimensional penalized Poisson regression with sparse design matrices.}",
      "\\label{tab:sparse-highdim}",
      "\\begin{tabular}{rrrlrrrr}",
      "\\toprule",
      "$n$ & $p$ & $\\delta$ & Method & Runtime (s) & Loss & Mean dev. & Selected \\\\",
      "\\midrule"
    )
  }

  blocks <- split(df, list(df$density, df$n, df$p), drop = TRUE)
  block_keys <- do.call(rbind, lapply(names(blocks), function(k) {
    sub <- blocks[[k]][1, ]
    data.frame(key = k, n = sub$n, p = sub$p, density = sub$density,
               stringsAsFactors = FALSE)
  }))
  block_keys <- block_keys[order(block_keys$density, block_keys$n, block_keys$p), ]

  for (i in seq_len(nrow(block_keys))) {
    sub <- order_block(blocks[[block_keys$key[i]]], "sparse_highdim")
    runtime <- format_time(sub$time_mean)
    loss    <- format_num(sub$loss_mean, 3)
    dev     <- format_num(sub$mean_dev_mean, 3)
    selected <- format_int(sub$selected_mean)
    f1       <- format_num(sub$f1_mean, 3)
    for (j in seq_len(nrow(sub))) {
      n_print   <- if (j == 1) format(sub$n[j], big.mark = ",") else ""
      p_print   <- if (j == 1) format(sub$p[j], big.mark = ",") else ""
      den_print <- if (j == 1) sprintf("%.2f", sub$density[j]) else ""
      if (has_f1) {
        out <- c(out, sprintf("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\",
                              n_print, p_print, den_print, sub$method_label[j],
                              runtime[j], loss[j], dev[j],
                              selected[j], f1[j]))
      } else {
        out <- c(out, sprintf("%s & %s & %s & %s & %s & %s & %s & %s \\\\",
                              n_print, p_print, den_print, sub$method_label[j],
                              runtime[j], loss[j], dev[j], selected[j]))
      }
    }
    if (i < nrow(block_keys)) out <- c(out, "\\midrule")
  }

  base_note <- paste(
    "\\textit{Note.} Runtime is in seconds; Loss is the per-observation",
    "Poisson negative log-likelihood (without the $\\log(y!)$ term);",
    "Mean dev.\\ is the mean Poisson deviance;",
    "$\\delta$ is the nonzero entry probability of the sparse design matrix;",
    "Selected denotes the number of nonzero coefficients.",
    "The regularization parameter $\\lambda$ is chosen by 5-fold cross-validation",
    "for each method.",
    sprintf("Results are averaged over %d repetitions.", reps_for_notes)
  )

  if (has_f1) {
    full_note <- paste(
      base_note,
      "F1 is the harmonic mean of precision and recall of the selected support",
      "against the true support of $\\beta^\\ast$."
    )
    note <- sprintf(
      "\\multicolumn{9}{p{0.95\\linewidth}}{\\footnotesize %s}",
      full_note
    )
  } else {
    full_note <- paste(
      base_note,
      "F1 score is not reported because true/false selection counts were",
      "not saved by the current benchmark pipeline."
    )
    note <- sprintf(
      "\\multicolumn{8}{p{0.95\\linewidth}}{\\footnotesize %s}",
      full_note
    )
  }

  out <- c(out,
           "\\bottomrule",
           "\\addlinespace[0.4ex]",
           paste0(note, " \\\\"),
           "\\end{tabular}",
           "\\end{table}",
           "")
  out
}

# ----- Write the tables -------------------------------------------------------

write_table <- function(lines, path) {
  if (is.null(lines)) return(invisible(NULL))
  writeLines(lines, path)
  cat("Wrote", path, "\n")
}

dense_lines   <- build_dense_latex(summary_df)
sparse_lines  <- build_sparse_latex(summary_df)
highdim_lines <- build_highdim_latex(summary_df)
sparse_highdim_lines <- build_sparse_highdim_latex(summary_df)

write_table(dense_lines,   file.path(opt$`out-dir`, "table_dense_regime.tex"))
write_table(sparse_lines,  file.path(opt$`out-dir`, "table_sparse_regime.tex"))
write_table(highdim_lines, file.path(opt$`out-dir`, "table_highdim_regime.tex"))
write_table(sparse_highdim_lines,
            file.path(opt$`out-dir`, "table_sparse_highdim_regime.tex"))

# ----- Master document with the main-text tables -----------------------------

master <- c(
  "% Auto-generated. Compiles the main-text benchmark tables.",
  sprintf("%% Generated: %s", Sys.time()),
  "\\documentclass{article}",
  "\\usepackage{booktabs}",
  "\\usepackage{amsmath}",
  "\\usepackage[a4paper,margin=1in]{geometry}",
  "\\title{GLM Benchmark Tables}",
  "\\author{}",
  sprintf("\\date{%s}", format(Sys.Date(), "%Y-%m-%d")),
  "\\begin{document}",
  "\\maketitle"
)

include_or_inline <- function(master, tex_file) {
  if (file.exists(tex_file)) {
    c(master, "", paste(readLines(tex_file, warn = FALSE), collapse = "\n"))
  } else master
}

master <- include_or_inline(master, file.path(opt$`out-dir`, "table_dense_regime.tex"))
master <- include_or_inline(master, file.path(opt$`out-dir`, "table_sparse_regime.tex"))
master <- include_or_inline(master, file.path(opt$`out-dir`, "table_highdim_regime.tex"))
master <- include_or_inline(master, file.path(opt$`out-dir`, "table_sparse_highdim_regime.tex"))

master <- c(master, "\\end{document}", "")

writeLines(master, file.path(opt$`out-dir`, "all_tables.tex"))
cat("Wrote", file.path(opt$`out-dir`, "all_tables.tex"), "\n")

cat("\n=== Paper tables generation complete ===\n")
