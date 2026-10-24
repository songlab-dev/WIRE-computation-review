#!/usr/bin/env Rscript

# Generate simulation commands for GLMM studies
# Usage: Rscript generate_commands.R --method lme4 --q 1 5 --output commands.txt

suppressPackageStartupMessages({
  library(stats)
})

parse_args <- function(args) {
  out <- list(
    method = NULL,
    q_values = NULL,
    output = "commands.txt",
    replicates = 100L,
    output_dir = "results"
  )

  i <- 1L
  while (i <= length(args)) {
    if (!startsWith(args[[i]], "--")) break

    key <- substring(args[[i]], 3L)
    i <- i + 1L

    if (i > length(args)) stop(sprintf("Missing value for --%s", key))

    if (key == "method") {
      out$method <- args[[i]]
    } else if (key == "q") {
      out$q_values <- as.integer(unlist(strsplit(args[[i]], ",")))
    } else if (key == "output") {
      out$output <- args[[i]]
    } else if (key == "replicates") {
      out$replicates <- as.integer(args[[i]])
    } else if (key == "output_dir") {
      out$output_dir <- args[[i]]
    }

    i <- i + 1L
  }

  if (is.null(out$method)) stop("--method required (lme4, glmmtmb, or sjsdm)")
  if (is.null(out$q_values)) stop("--q required (e.g., --q 1 or --q 1,5,15)")

  out$method <- tolower(out$method)
  if (!out$method %in% c("lme4", "glmmtmb", "sjsdm")) {
    stop("--method must be lme4, glmmtmb, or sjsdm")
  }

  if (out$method == "sjsdm" && !all(out$q_values == 1)) {
    stop("sjSDM only supports q=1")
  }

  out
}

generate_commands <- function(method, q_values, replicates, output_dir) {
  commands <- character()

  rho_values <- c(0, 0.3, 0.7)
  cov_structures <- c("identity", "ar1")

  seed_base <- 20260518L

  for (q in q_values) {
    for (rep in seq_len(replicates)) {
      for (cov_struct in cov_structures) {
        rho_list <- if (cov_struct == "identity") 0 else c(0.3, 0.7)

        for (rho in rho_list) {
          seed <- seed_base + (q - 1L) * 10000L + rep

          cmd <- sprintf(
            "Rscript run_%s.R --q %d --cov_structure %s --rho %.1f --replicate_id %d --seed %d --output_dir %s",
            method, q, cov_struct, rho, rep, seed, output_dir
          )

          commands <- c(commands, cmd)
        }
      }
    }
  }

  commands
}

# Main
args <- parse_args(commandArgs(trailingOnly = TRUE))
commands <- generate_commands(args$method, args$q_values, args$replicates, args$output_dir)

writeLines(commands, args$output)
cat("Generated", length(commands), "commands\n")
cat("Output:", args$output, "\n")
