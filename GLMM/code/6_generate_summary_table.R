#!/usr/bin/env Rscript

# Generate summary statistics table from GLMM simulation results
# Usage: Rscript generate_summary_table.R --results_dirs dir1,dir2,dir3 --output table.tex

suppressPackageStartupMessages({
  library(stats)
})

parse_args <- function(args) {
  out <- list(
    results_dirs = NULL,
    output = "summary_table.tex",
    output_format = "tex"
  )

  i <- 1L
  while (i <= length(args)) {
    if (!startsWith(args[[i]], "--")) break

    key <- substring(args[[i]], 3L)
    i <- i + 1L

    if (i > length(args)) stop(sprintf("Missing value for --%s", key))

    if (key == "results_dirs") {
      out$results_dirs <- trimws(unlist(strsplit(args[[i]], ",")))
    } else if (key == "output") {
      out$output <- args[[i]]
    } else if (key == "output_format") {
      out$output_format <- args[[i]]
    }

    i <- i + 1L
  }

  if (is.null(out$results_dirs)) stop("--results_dirs required")

  out
}

read_results <- function(dir) {
  files <- list.files(dir, pattern = "^result_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    cat("Warning: No result files in", dir, "\n")
    return(data.frame())
  }

  results_list <- lapply(files, function(f) {
    tryCatch(read.csv(f, stringsAsFactors = FALSE), error = function(e) NULL)
  })

  do.call(rbind, results_list[!sapply(results_list, is.null)])
}

summarize_results <- function(all_results) {
  # Filter for cluster split strategy only
  all_results <- all_results[all_results$split_strategy == "cluster", ]

  # Remove redundant q=1 rows with rho != 0
  q1_identity <- (all_results$q == 1L & all_results$cov_structure == "identity")
  q_not_1 <- (all_results$q != 1L)
  all_results <- all_results[q1_identity | q_not_1, ]

  # Get unique configurations
  configs <- unique(all_results[, c("q", "cov_structure", "rho", "method")])
  configs <- configs[order(configs$q, configs$cov_structure, configs$rho, configs$method), ]

  summary_list <- list()

  for (i in seq_len(nrow(configs))) {
    cfg <- configs[i, ]

    mask <- (all_results$q == cfg$q &
             all_results$cov_structure == cfg$cov_structure &
             all_results$rho == cfg$rho &
             all_results$method == cfg$method)

    cfg_data <- all_results[mask, ]

    if (nrow(cfg_data) > 0) {
      n_converged <- sum(cfg_data$converged == TRUE | cfg_data$converged == 1, na.rm = TRUE)
      conv_rate <- n_converged / nrow(cfg_data)

      summary_list[[i]] <- data.frame(
        q = cfg$q,
        cov_structure = cfg$cov_structure,
        rho = cfg$rho,
        method = cfg$method,
        n_runs = nrow(cfg_data),
        convergence_rate = conv_rate,
        runtime_mean = mean(cfg_data$runtime_sec, na.rm = TRUE),
        runtime_sd = sd(cfg_data$runtime_sec, na.rm = TRUE),
        beta_sup_mean = mean(cfg_data$beta_sup_norm_error, na.rm = TRUE),
        beta_sup_sd = sd(cfg_data$beta_sup_norm_error, na.rm = TRUE),
        sigma_sup_mean = mean(cfg_data$sigma_sup_norm_error, na.rm = TRUE),
        sigma_sup_sd = sd(cfg_data$sigma_sup_norm_error, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, summary_list)
}

format_stat <- function(mean_val, sd_val) {
  if (is.na(mean_val)) return("--")
  if (is.na(sd_val)) return(sprintf("%.3f", mean_val))
  sprintf("%.3f (%.3f)", mean_val, sd_val)
}

format_rate <- function(rate) {
  if (is.na(rate)) return("--")
  sprintf("%.3f", rate)
}

cov_label <- function(cov, rho) {
  if (cov == "identity") return("$\\rho=0$")
  if (cov == "ar1") return(sprintf("$\\rho=%.1f$", rho))
  return(cov)
}

generate_latex_table <- function(summary_stats) {
  lines <- c(
    "\\begin{table}[H]",
    "\\centering",
    "\\caption{GLMM simulation results: runtime, algorithm success rate, fixed-effect and random-effect sup-norm errors. Results are summarized as mean (sd) across 100 replicates per configuration.}",
    "\\label{tab:glmm_summary}",
    "\\resizebox{\\textwidth}{!}{",
    "\\begin{tabular}{llcccc}",
    "\\toprule",
    "Setting & Software & Success rate & Runtime (s) & Fixed sup. err. & RE sup. err. \\\\",
    "\\midrule"
  )

  q_values <- sort(unique(summary_stats$q))

  for (q in q_values) {
    q_indices <- which(summary_stats$q == q)
    q_data <- summary_stats[q_indices, ]

    lines <- c(lines, sprintf("\\multicolumn{6}{c}{Random-effect dimension $q = %d$} \\\\", q))
    lines <- c(lines, "\\midrule")

    configs_q <- unique(q_data[, c("cov_structure", "rho")])
    rho_order <- order(configs_q$rho)
    configs_q <- configs_q[rho_order, ]

    for (cfg_idx in seq_len(nrow(configs_q))) {
      cfg <- configs_q[cfg_idx, ]
      cfg_mask <- (q_data$cov_structure == cfg$cov_structure &
                   q_data$rho == cfg$rho)
      cfg_data <- q_data[cfg_mask, ]

      setting <- cov_label(cfg$cov_structure, cfg$rho)

      for (method_idx in seq_len(nrow(cfg_data))) {
        method_data <- cfg_data[method_idx, ]

        conv_str <- format_rate(method_data$convergence_rate)
        runtime_str <- format_stat(method_data$runtime_mean, method_data$runtime_sd)
        beta_sup_str <- format_stat(method_data$beta_sup_mean, method_data$beta_sup_sd)
        sigma_sup_str <- format_stat(method_data$sigma_sup_mean, method_data$sigma_sup_sd)

        if (method_idx == 1) {
          line <- sprintf("%s & \\texttt{%s} & %s & %s & %s & %s  \\\\",
                         setting, method_data$method, conv_str, runtime_str,
                         beta_sup_str, sigma_sup_str)
        } else {
          line <- sprintf(" & \\texttt{%s} & %s & %s & %s & %s  \\\\",
                         method_data$method, conv_str, runtime_str,
                         beta_sup_str, sigma_sup_str)
        }

        lines <- c(lines, line)
      }

      if (cfg_idx < nrow(configs_q)) {
        lines <- c(lines, "\\addlinespace")
      }
    }

    if (q != q_values[length(q_values)]) {
      lines <- c(lines, "\\addlinespace", "\\midrule")
    }
  }

  lines <- c(lines, "\\bottomrule", "\\end{tabular}", "}", "\\vspace{0.5em}")
  lines <- c(lines,
    "\\begin{minipage}{0.98\\textwidth}",
    "\\footnotesize Note: Results show cluster split strategy only. For sjSDM, random-effect errors are omitted (method does not estimate covariance matrices like lme4/glmmTMB).",
    "\\end{minipage}",
    "\\end{table}"
  )

  lines
}

# Main
args <- parse_args(commandArgs(trailingOnly = TRUE))

cat("Reading results from", length(args$results_dirs), "directories...\n")
all_results <- do.call(rbind, lapply(args$results_dirs, read_results))

cat("Total records read:", nrow(all_results), "\n")

summary_stats <- summarize_results(all_results)
cat("Configurations summarized:", nrow(summary_stats), "\n")

if (args$output_format == "tex") {
  lines <- generate_latex_table(summary_stats)
  writeLines(lines, args$output)
  cat("LaTeX table written to:", args$output, "\n")
}

cat("Done!\n")
