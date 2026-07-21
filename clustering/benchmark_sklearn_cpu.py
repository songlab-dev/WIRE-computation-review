# Import the os module for file and directory operations
import os
# Import the json module to read the configuration files
import json
# Import the time module to measure execution time
import time
# Import numpy for numerical computations and array handling
import numpy as np
# Import pandas for data manipulation and saving results to CSV
import pandas as pd

# Import KMeans from scikit-learn
from sklearn.cluster import KMeans
# Import AgglomerativeClustering (HC) from scikit-learn
from sklearn.cluster import AgglomerativeClustering
# Import DBSCAN from scikit-learn
from sklearn.cluster import DBSCAN
# Import HDBSCAN from scikit-learn
from sklearn.cluster import HDBSCAN
# Import GaussianMixture from scikit-learn
from sklearn.mixture import GaussianMixture

# Import Adjusted Rand Score for evaluation
from sklearn.metrics import adjusted_rand_score
# Import Normalized Mutual Information for evaluation
from sklearn.metrics import normalized_mutual_info_score

# Define the base directory where all the slice folders are stored
base_data_dir = "data"

# Find all subdirectories in the base directory that start with 'n_' to identify slice folders
slice_folders = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith("n_")]
# Sort the folders alphabetically for consistent execution order
slice_folders.sort()

# Define a list of method names to iterate over (including GMM)
methods = ["KMeans", "GMM", "HDBSCAN", "HC", "DBSCAN"]


# Define a function to instantiate the correct model based on the method name, seed, and number of clusters
def get_cpu_model(method_name, seed, n_clusters):
    # Check if the requested method is KMeans
    if method_name == "KMeans":
        # Return a KMeans instance with random initialization and specified random state
        return KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=100, random_state=seed)
    # Check if the requested method is GMM
    elif method_name == "GMM":
        # Return a GaussianMixture instance with spherical covariance and specified random state
        return GaussianMixture(n_components=n_clusters, covariance_type='spherical', n_init=1, init_params='random', max_iter=100, random_state=seed)
    # Check if the requested method is HDBSCAN
    elif method_name == "HDBSCAN":
        # Return an HDBSCAN instance with specified min_cluster_size and min_samples
        return HDBSCAN(min_cluster_size=10, min_samples=10)
    # Check if the requested method is DBSCAN
    elif method_name == "DBSCAN":
        # Return a DBSCAN instance with eps=75.0 and min_samples=10
        return DBSCAN(eps=75.0, min_samples=10)
    # Check if the requested method is HC (Agglomerative Clustering)
    elif method_name == "HC":
        # Return an AgglomerativeClustering instance with the specified number of clusters
        return AgglomerativeClustering(n_clusters=n_clusters)
        #return AgglomerativeClustering(n_clusters=n_clusters, linkage="single", metric="euclidean")
    # Raise an error if the method name is not recognized
    else:
        raise ValueError(f"Unknown method: {method_name}")

# Print a starting message for the benchmark suite
print(f"Starting CPU (scikit-learn) Benchmark Suite across {len(slice_folders)} slices...")

# Iterate over each method defined in the methods list
for method in methods:
    # Print a header for the current method being evaluated
    print(f"\n{'='*60}\nEvaluating Method: {method} (CPU)\n{'='*60}")

    # Initialize an empty list to store all results for this specific method
    all_results = []
    
    # Track if warm-up has been performed for this method
    warmed_up = False

    # Iterate over each slice folder (e.g., n_10000_p_100)
    for slice_folder in slice_folders:
        # Construct the full path to the current slice directory
        slice_dir = os.path.join(base_data_dir, slice_folder)
        
        # Construct the path to the config.json file in this slice
        config_path = os.path.join(slice_dir, "config.json")
        # Open the config.json file in read mode
        with open(config_path, "r") as f:
            # Load the JSON data into a configuration dictionary
            config = json.load(f)

        # Extract the sample size from the slice configuration
        n_samples = config["n_samples"]
        # Extract the feature size from the slice configuration
        n_features = config["n_features"]
        # Extract the number of clusters from the slice configuration
        n_clusters = config["n_clusters"]
        # Extract the number of replicates from the slice configuration
        n_replicates = config["n_replicates"]

        # Print the current configuration slice being processed
        print(f"\n--- Processing Slice: n={n_samples}, p={n_features} ---")

        # Check if the method needs a warm-up phase (only done once per method)
        if not warmed_up:
            # Print a message indicating the start of the warm-up phase
            print("  Performing CPU Warm-up...")
            # Load the first replicate dataset from the current slice to use for warm-up
            warm_data = np.load(os.path.join(slice_dir, "replicate_0.npz"))
            # Extract the first 1000 samples
            X_warm = warm_data["X"][:1000].astype(np.float64)
            # Instantiate a warm-up model using seed 0 and current cluster count
            warm_model = get_cpu_model(method, seed=0, n_clusters=n_clusters)
            # Fit the warm-up model to initialize internal scikit-learn libraries
            _ = warm_model.fit_predict(X_warm)
            # Set the flag to True so warm-up is skipped for subsequent slices
            warmed_up = True
            # Print a message indicating warm-up is complete
            print("  Warm-up complete.")

        # Loop through the number of replicates specified in the config
        for i in range(n_replicates):
            # Construct the path to the current replicate's .npz file
            data_path = os.path.join(slice_dir, f"replicate_{i}.npz")
            # Load the full dataset for the current replicate
            data = np.load(data_path)
            # Extract the feature matrix X
            X = data["X"]
            # Extract the true labels y
            y_true = data["y"]

            # Cast the feature matrix to float as expected by the algorithms
            X_cpu = X.astype(np.float64)

            # Instantiate the model for the current replicate
            model = get_cpu_model(method, seed=i, n_clusters=n_clusters)

            # Record the start time before fitting the model
            start_time = time.perf_counter()
            # Fit the model to the data and return the predicted cluster labels
            labels = model.fit_predict(X_cpu)
            # Calculate the elapsed time
            elapsed_time = time.perf_counter() - start_time

            # Calculate the Adjusted Rand Score
            ari = adjusted_rand_score(y_true, labels)
            # Calculate the Normalized Mutual Information score
            nmi = normalized_mutual_info_score(y_true, labels, average_method="arithmetic")

            # Append a dictionary with the results for this replicate to the all_results list
            all_results.append({
                # Store the sample size for this run
                "n_samples": n_samples,
                # Store the feature size for this run
                "n_features": n_features,
                # Store the replicate index
                "Replicate": i + 1,
                # Store the execution time
                "CPU_Runtime_sec": elapsed_time,
                # Store the ARI score
                "CPU_ARI": ari,
                # Store the NMI score
                "CPU_NMI": nmi
            })
            
            # Print a short status update for the current replicate
            print(f"  Replicate {i + 1}/{n_replicates} done in {elapsed_time:.4f}s")

    # Convert the collected results for all slices into a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    # Define the output CSV filename for the current method
    output_filename = f"results/python_{method.lower()}_cpu_results.csv"
    # Save the DataFrame to a CSV file without writing the row indices
    results_df.to_csv(output_filename, index=False)
    # Print a confirmation that the CSV file was saved
    print(f"-> Saved all results for {method} to {output_filename}")

print("\nAll CPU benchmarks completed successfully.")
