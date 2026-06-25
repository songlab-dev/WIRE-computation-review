# Import the reticulate package to call Python from R
library(reticulate)
# Import the mclust package for Gaussian Mixture Modeling and Adjusted Rand Index
library(mclust)
# Import the dbscan package for HDBSCAN and DBSCAN clustering algorithms
library(dbscan)
# Import the aricode package to calculate Normalized Mutual Information (NMI)
library(aricode)
# Import the jsonlite package to read configuration files in JSON format
library(jsonlite)

#setwd("wire_clustering/")
getwd()

# Load the Python numpy library to read pre-generated .npz datasets
np <- import("numpy")

# Define the base directory where the data slices are stored
base_data_dir <- "data"
# Define the output directory where the result CSV files will be saved
out_dir <- "results"
# Create the output directory if it does not already exist, suppress warnings
dir.create(out_dir, showWarnings = FALSE)

# List all subdirectories in the base data directory without full paths
slice_folders <- list.dirs(base_data_dir, full.names = FALSE, recursive = FALSE)
# Filter the list to only include folders that start with "n_"
slice_folders <- slice_folders[startsWith(slice_folders, "n_")]

# Split the folder names by the underscore character to extract numerical values
folder_parts <- strsplit(slice_folders, "_")
# Extract the sample size (n) from the second element of each split string and cast to numeric
n_vals <- as.numeric(sapply(folder_parts, function(x) x[2]))
# Extract the feature size (p) from the fourth element of each split string and cast to numeric
p_vals <- as.numeric(sapply(folder_parts, function(x) x[4]))
# Reorder the slice folders primarily by sample size (n), then by feature size (p)
slice_folders <- slice_folders[order(n_vals, p_vals)]

# Define the list of clustering methods to evaluate
#methods <- c("KMeans", "GMM", "HDBSCAN", "DBSCAN", "HC")
methods <- c("HDBSCAN", "DBSCAN", "HC")

# Print a starting message indicating the total number of slices to process
cat(sprintf("Starting R Benchmark Suite across %d slices...\n", length(slice_folders)))

# Iterate over each clustering method defined in the methods vector
for (method in methods) {
  # Print a formatted top border for the current method being evaluated
  cat(sprintf("\n============================================================\n"))
  # Print the current method name
  cat(sprintf("Evaluating Method: %s (R)\n", method))
  # Print a formatted bottom border for the current method being evaluated
  cat(sprintf("============================================================\n"))
  
  # Initialize an empty data frame to collect all results for this specific method
  all_results <- data.frame()
  
  # Iterate over each data slice folder 
  for (slice_folder in slice_folders) {
    # Construct the full path to the current slice directory
    slice_dir <- file.path(base_data_dir, slice_folder)
    
    # Read the config.json file from the current slice directory
    config <- fromJSON(file.path(slice_dir, "config.json"))
    # Extract the number of samples and cast to integer
    n_samples    <- as.integer(config$n_samples)
    # Extract the number of features and cast to integer
    n_features   <- as.integer(config$n_features)
    # Extract the number of clusters and cast to integer
    n_clusters   <- as.integer(config$n_clusters)
    # Extract the number of replicates and cast to integer
    n_replicates <- as.integer(config$n_replicates)
    
    # Memory Protection Mechanism: HC calculates an n x n distance matrix
    # Skip Hierarchical Clustering if the number of samples exceeds 20,000 to prevent RAM overflow
    if (method == "HC" && n_samples > 20000) {
      # Print the slice information
      cat(sprintf("\n--- Processing Slice: n=%d, p=%d ---\n", n_samples, n_features))
      # Print a warning that the method is being skipped due to memory constraints
      cat(sprintf("  [Skipped] HC requires O(n^2) memory. Skipping for n > 20000.\n"))
      # Move to the next slice folder
      next
    }
    
    # Print a message indicating the start of processing for the current slice
    cat(sprintf("\n--- Processing Slice: n=%d, p=%d ---\n", n_samples, n_features))
    
    # Loop through each replicate (from 0 to n_replicates - 1 to match Python indexing)
    for (i in 0:(n_replicates - 1)) {
      
      # Construct the path to the specific .npz data file for this replicate
      data_path <- file.path(slice_dir, sprintf("replicate_%d.npz", i))
      
      # Load the .npz file using the imported numpy library
      data <- np$load(data_path)
      # Extract the feature matrix X
      X <- data["X"]
      # Extract the true labels y and cast them to integers
      y_true <- as.integer(data["y"])
      # Explicitly close the file handle via reticulate to prevent resource leaks
      data$close()
      
      # Set the random seed to ensure reproducibility (using i + 1 as R is 1-indexed)
      set.seed(i + 1)
      
      # Record the start time before running the clustering algorithm
      start_time <- proc.time()
      
      # Initialize a variable to hold the predicted labels
      r_labels <- NULL
      
      # Check if the current method is KMeans
      if (method == "KMeans") {
        # Run K-Means with the specified number of clusters and a maximum of 100 iterations
        res <- kmeans(X, centers = n_clusters, iter.max = 100, nstart = 1)
        # Extract the cluster assignments
        r_labels <- res$cluster
        
        # Check if the current method is GMM
      } else if (method == "GMM") {
        # Create a small random subset of indices to speed up initialization
        subset_idx <- sample(1:nrow(X), min(500, nrow(X)))
        #
        res <- Mclust(X, G = n_clusters, modelNames = "EII", initialization = list(subset = subset_idx))
        # Extract the classification labels and cast to integers
        r_labels <- as.integer(res$classification)
        
        # Check if the current method is HDBSCAN
      } else if (method == "HDBSCAN") {
        # Run HDBSCAN with a minimum cluster size of 10
        res <- hdbscan(X, minPts = 10L)
        # Extract the cluster assignments
        r_labels <- res$cluster
        
        # Check if the current method is DBSCAN
      } else if (method == "DBSCAN") {
        # Run standard DBSCAN with an epsilon of 75.0 and minimum points of 10
        res <- dbscan(X, eps = 75.0, minPts = 10L)
        # Extract the cluster assignments
        r_labels <- res$cluster
        
        # Check if the current method is HC (Hierarchical Clustering)
      } else if (method == "HC") {
        # Compute the pairwise Euclidean distance matrix
        dist_matrix <- dist(X)
        # Run hierarchical clustering using Ward's minimum variance method
        res <- hclust(dist_matrix, method = "ward.D2")
        # Cut the resulting dendrogram to yield the desired number of clusters
        r_labels <- cutree(res, k = n_clusters)
      }
      
      # Calculate the elapsed time by subtracting the start time from the current time
      elapsed <- (proc.time() - start_time)["elapsed"]
      
      # Calculate the Adjusted Rand Index (ARI) comparing true labels and predictions
      ari <- adjustedRandIndex(y_true, r_labels)
      # Calculate the Normalized Mutual Information (NMI) score
      nmi_val <- NMI(y_true, r_labels)
      
      # Create a single-row data frame with the results for the current replicate
      row <- data.frame(
        n_samples = n_samples,
        n_features = n_features,
        Replicate = i + 1,
        R_Runtime_sec = elapsed,
        R_ARI = ari,
        R_NMI = nmi_val
      )
      # Append the current row to the main results data frame
      all_results <- rbind(all_results, row)
      
      # Print a short status update indicating completion and elapsed time for this replicate
      cat(sprintf("  Replicate %d/%d done in %.4fs\n", i + 1, n_replicates, elapsed))
    }
  }
  
  # Construct the output filename for the current method's results
  out_file <- file.path(out_dir, sprintf("r_%s_results.csv", tolower(method)))
  # Write the accumulated results to a CSV file without row names
  write.csv(all_results, file = out_file, row.names = FALSE)
  # Print a confirmation message that the file has been saved
  cat(sprintf("-> Saved all results for %s to %s\n", method, out_file))
}

# Print a final message indicating all benchmarks are complete
cat("\nAll R benchmarks completed successfully.\n")

