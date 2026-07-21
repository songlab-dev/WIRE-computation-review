# R clustering benchmark with per-slice summaries and checkpoints

# Python interoperability and metric dependencies
library(reticulate)

py_require(c(
  "numpy",
  "scikit-learn"
))

np <- import("numpy")
sk_metrics <- import("sklearn.metrics")

getwd()
#setwd("wire_clustering")
getwd()

# R clustering and configuration packages
library(mclust)
library(dbscan)
library(jsonlite)

# Input and output directories
base_data_dir <- "data"
out_dir <- "results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Locate and order all data slices by n and then p
slice_folders <- list.dirs(
  base_data_dir,
  full.names = FALSE,
  recursive = FALSE
)
slice_folders <- slice_folders[startsWith(slice_folders, "n_")]

if (length(slice_folders) == 0L) {
  stop(sprintf("No data slice folders were found under '%s'.", base_data_dir))
}

folder_parts <- strsplit(slice_folders, "_")
n_vals <- as.numeric(vapply(folder_parts, function(x) x[2], character(1)))
p_vals <- as.numeric(vapply(folder_parts, function(x) x[4], character(1)))
slice_folders <- slice_folders[order(n_vals, p_vals)]

methods <- c("KMeans", "GMM", "HDBSCAN", "DBSCAN", "HC")

cat(sprintf(
  "Starting R Benchmark Suite across %d slices...\n",
  length(slice_folders)
))

for (method in methods) {
  cat("\n============================================================\n")
  cat(sprintf("Evaluating Method: %s (R)\n", method))
  cat("============================================================\n")

  # Cumulative replicate-level and slice-level results for this method
  all_results <- data.frame()
  summary_results <- data.frame()

  # These files are updated after every completed (n, p) slice.
  cumulative_out_file <- file.path(
    out_dir,
    sprintf("r_%s_results.csv", tolower(method))
  )
  summary_out_file <- file.path(
    out_dir,
    sprintf("r_%s_summary.csv", tolower(method))
  )

  for (slice_folder in slice_folders) {
    slice_dir <- file.path(base_data_dir, slice_folder)
    config_path <- file.path(slice_dir, "config.json")

    if (!file.exists(config_path)) {
      warning(sprintf("Missing configuration file: %s; slice skipped.", config_path))
      next
    }

    config <- fromJSON(config_path)
    n_samples <- as.integer(config$n_samples)
    n_features <- as.integer(config$n_features)
    n_clusters <- as.integer(config$n_clusters)
    n_replicates <- as.integer(config$n_replicates)

    cat(sprintf(
      "\n--- Processing Slice: n=%d, p=%d ---\n",
      n_samples,
      n_features
    ))

    # Store only successful replicate results from the current slice.
    slice_results <- data.frame()

    for (i in 0:(n_replicates - 1L)) {
      data_path <- file.path(
        slice_dir,
        sprintf("replicate_%d.npz", i)
      )

      if (!file.exists(data_path)) {
        cat(sprintf(
          "  Replicate %d missing; skipped (%s)\n",
          i + 1L,
          data_path
        ))
        next
      }

      # Read the pre-generated dataset shared with the Python benchmarks.
      data <- np$load(data_path)
      X <- data["X"]
      y_true <- as.integer(data["y"])
      data$close()

      X <- as.matrix(X)
      y_true <- as.integer(y_true)

      if (nrow(X) != length(y_true)) {
        cat(sprintf(
          "  Replicate %d has mismatched X/y lengths; skipped.\n",
          i + 1L
        ))
        next
      }

      # Reproducible within-R seed for this replicate.
      set.seed(i)

      # Runtime includes initialization and fitting, but excludes ARI/NMI.
      start_time <- proc.time()
      r_labels <- NULL
      fit_error <- NULL

      if (method == "KMeans") {
        res <- kmeans(
          X,
          centers = n_clusters,
          iter.max = 100,
          nstart = 1,
          algorithm = "Lloyd"
        )
        r_labels <- res$cluster

      } else if (method == "GMM") {
        res <- tryCatch(
          {
            # Step 1: use K-means to generate initial cluster assignments
            kmeans_init <- kmeans(
              X,
              centers = n_clusters,
              iter.max = 100,
              nstart = 1,
              algorithm = "Lloyd"
            )
            
            # Step 2: convert hard K-means labels into an initial
            # membership-probability matrix for GMM
            z_init <- unmap(kmeans_init$cluster)
            
            # Step 3: fit the VII Gaussian mixture with EM
            me(
              data = X,
              modelName = "VII",
              z = z_init,
              control = emControl(itmax = 100)
            )
          },
          error = function(e) {
            fit_error <<- conditionMessage(e)
            NULL
          }
        )
        
        if (
          is.null(res) ||
          is.null(res$z) ||
          nrow(res$z) != length(y_true)
        ) {
          elapsed_failed <- unname(
            (proc.time() - start_time)["elapsed"]
          )
          
          error_suffix <- if (is.null(fit_error)) {
            ""
          } else {
            paste0("; ", fit_error)
          }
          
          cat(sprintf(
            "  Replicate %d failed after %.4fs; skipped%s\n",
            i + 1L,
            elapsed_failed,
            error_suffix
          ))
          
          next
        }
        
        # Convert posterior probabilities to final cluster labels
        r_labels <- max.col(
          res$z,
          ties.method = "first"
        )

      } else if (method == "HDBSCAN") {
        res <- hdbscan(X, minPts = 10L)
        r_labels <- res$cluster

      } else if (method == "DBSCAN") {
        res <- dbscan(X, eps = 75.0, minPts = 10L)
        r_labels <- res$cluster

      } else if (method == "HC") {
        dist_matrix <- dist(X, method = "euclidean")
        res <- hclust(dist_matrix, method = "ward.D2")
        r_labels <- cutree(res, k = n_clusters)

      } else {
        stop(sprintf("Unknown clustering method: %s", method))
      }

      elapsed <- unname((proc.time() - start_time)["elapsed"])
      r_labels <- as.integer(r_labels)

      if (length(r_labels) != length(y_true)) {
        cat(sprintf(
          "  Replicate %d returned %d labels for %d observations; skipped.\n",
          i + 1L,
          length(r_labels),
          length(y_true)
        ))
        next
      }

      # Use sklearn definitions for identical ARI/NMI metric normalization.
      ari <- as.numeric(
        sk_metrics$adjusted_rand_score(y_true, r_labels)
      )
      nmi_val <- as.numeric(
        sk_metrics$normalized_mutual_info_score(
          y_true,
          r_labels,
          average_method = "arithmetic"
        )
      )

      row <- data.frame(
        n_samples = n_samples,
        n_features = n_features,
        Replicate = i + 1L,
        R_Runtime_sec = elapsed,
        R_ARI = ari,
        R_NMI = nmi_val
      )

      all_results <- rbind(all_results, row)
      slice_results <- rbind(slice_results, row)

      cat(sprintf(
        "  Replicate %d/%d done in %.4fs\n",
        i + 1L,
        n_replicates,
        elapsed
      ))
    }

    # Print and retain a summary for the completed (n, p) slice.
    n_success <- nrow(slice_results)
    n_failed <- n_replicates - n_success

    if (n_success > 0L) {
      avg_runtime <- mean(slice_results$R_Runtime_sec)
      avg_ari <- mean(slice_results$R_ARI)
      avg_nmi <- mean(slice_results$R_NMI)
    } else {
      avg_runtime <- NA_real_
      avg_ari <- NA_real_
      avg_nmi <- NA_real_
    }

    slice_summary <- data.frame(
      Method = method,
      n_samples = n_samples,
      n_features = n_features,
      n_requested = n_replicates,
      n_success = n_success,
      n_failed = n_failed,
      Avg_Runtime_sec = avg_runtime,
      Avg_ARI = avg_ari,
      Avg_NMI = avg_nmi
    )
    summary_results <- rbind(summary_results, slice_summary)

    if (n_success > 0L) {
      cat(sprintf(
        paste0(
          "\n>>> %s summary: n=%d, p=%d\n",
          "    successful: %d/%d\n",
          "    failed:     %d\n",
          "    avg runtime: %.4f sec\n",
          "    avg ARI:     %.6f\n",
          "    avg NMI:     %.6f\n"
        ),
        method,
        n_samples,
        n_features,
        n_success,
        n_replicates,
        n_failed,
        avg_runtime,
        avg_ari,
        avg_nmi
      ))
    } else {
      cat(sprintf(
        "\n>>> %s summary: n=%d, p=%d: all %d replicates failed\n",
        method,
        n_samples,
        n_features,
        n_replicates
      ))
    }

    # Save this individual slice.
    slice_out_file <- file.path(
      out_dir,
      sprintf(
        "r_%s_n_%d_p_%d_results.csv",
        tolower(method),
        n_samples,
        n_features
      )
    )
    write.csv(slice_results, slice_out_file, row.names = FALSE)

    # Update cumulative raw and summary checkpoints after every slice.
    write.csv(all_results, cumulative_out_file, row.names = FALSE)
    write.csv(summary_results, summary_out_file, row.names = FALSE)

    cat(sprintf("-> Saved slice results to %s\n", slice_out_file))
    cat(sprintf("-> Updated cumulative results at %s\n", cumulative_out_file))
    cat(sprintf("-> Updated summary results at %s\n", summary_out_file))
  }

  # Final write for the completed method.
  write.csv(all_results, cumulative_out_file, row.names = FALSE)
  write.csv(summary_results, summary_out_file, row.names = FALSE)

  cat(sprintf(
    "\nCompleted %s. Final results saved to %s\n",
    method,
    cumulative_out_file
  ))
}

cat("\nAll R benchmarks completed successfully.\n")
