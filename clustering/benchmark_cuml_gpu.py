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

# Import cupy for GPU array operations
import cupy as cp
# Import cuml library for GPU-accelerated machine learning
import cuml
# Import KMeans from cuml
from cuml.cluster import KMeans
# Import AgglomerativeClustering (HC) from cuml
from cuml.cluster import AgglomerativeClustering
# Import DBSCAN from cuml
from cuml.cluster import DBSCAN
# Import HDBSCAN from cuml
from cuml.cluster import HDBSCAN

# Import Adjusted Rand Score for evaluation (computed on CPU)
from sklearn.metrics import adjusted_rand_score
# Import Normalized Mutual Information for evaluation (computed on CPU)
from sklearn.metrics import normalized_mutual_info_score

# Define the base directory where all the slice folders are stored
base_data_dir = "new_data"

# Find all subdirectories in the base directory that start with 'n_'
slice_folders = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith("n_")]
# Sort the folders alphabetically for consistent execution order
slice_folders.sort()

# Define a list of method names to iterate over (excluding GMM)
#methods = ["KMeans", "HDBSCAN", "DBSCAN", "HC"]
methods = ["HC"]

# Define a function to instantiate the correct GPU model based on the method name, seed, and cluster count
def get_gpu_model(method_name, seed, n_clusters):
    # Check if the requested method is KMeans
    if method_name == "KMeans":
        # Return a KMeans instance with random initialization and specified random state
        return KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=100, random_state=seed)
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
        # Strategy A: Use 'pairwise' for exact computation since N=10000 fits easily in GPU memory (~400MB)
        # This skips KNN approximation and can be extremely fast for datasets under 30k samples
        return AgglomerativeClustering(
            # Specify the number of target clusters to find
            n_clusters=n_clusters,
            # Compute the full N x N distance matrix for absolute exactness
            connectivity='pairwise',
            # With 'pairwise', we are no longer restricted to euclidean; 'cosine' or 'manhattan' can be tested
            metric='euclidean',
        )
    # Raise an error if the method name is not recognized
    else:
        # Raise an exception for unsupported models
        raise ValueError(f"Unknown method: {method_name}")

# Print a starting message for the benchmark suite along with the cuML version
print(f"Starting GPU (cuML {cuml.__version__}) Benchmark Suite across {len(slice_folders)} slices...")

# Iterate over each method defined in the methods list
for method in methods:
    # Print a header for the current method being evaluated
    print(f"\n{'='*60}\nEvaluating Method: {method} (GPU)\n{'='*60}")

    # Initialize an empty list to store all results for this specific method
    all_results = []
    
    # Track if warm-up has been performed for this method
    warmed_up = False

    # Iterate over each slice folder
    for slice_folder in slice_folders:
        # Construct the full path to the current slice directory
        slice_dir = os.path.join(base_data_dir, slice_folder)
        
        # Construct the path to the config.json file
        config_path = os.path.join(slice_dir, "config.json")
        # Open and load the JSON configuration
        with open(config_path, "r") as f:
            # Parse the JSON content into a dictionary
            config = json.load(f)

        # Extract parameters from the configuration
        n_samples = config["n_samples"]
        # Extract the number of features
        n_features = config["n_features"]
        # Extract the true number of clusters
        n_clusters = config["n_clusters"]
        # Extract the number of technical replicates
        n_replicates = config["n_replicates"]

        # Print the current configuration slice being processed
        print(f"\n--- Processing Slice: n={n_samples}, p={n_features} ---")

        # Check if the method needs a warm-up phase
        if not warmed_up:
            # Print a message indicating the start of the warm-up phase
            print("  Performing GPU Warm-up...")
            # Load the first replicate dataset to act as warm-up data
            warm_data = np.load(os.path.join(slice_dir, "replicate_0.npz"))
            # Extract the first 1000 samples, cast to float32, and transfer to GPU using cupy
            X_warm_gpu = cp.asarray(warm_data["X"][:1000].astype(np.float32))
            
            # Instantiate a warm-up model dynamically using the unified function
            warm_model = get_gpu_model(method, seed=0, n_clusters=n_clusters)
            # Fit the model to initialize CUDA context and kernels
            _ = warm_model.fit_predict(X_warm_gpu)
                
            # Set the flag to True so warm-up is skipped in future iterations
            warmed_up = True
            # Print a message indicating warm-up is complete
            print("  Warm-up complete.")

        # Loop through the number of replicates
        for i in range(n_replicates):
            # Construct the path to the current replicate's .npz file
            data_path = os.path.join(slice_dir, f"replicate_{i}.npz")
            # Load the dataset arrays from the file
            data = np.load(data_path)
            # Extract feature matrix
            X = data["X"]
            # Extract true labels
            y_true = data["y"]

            # Cast the feature matrix to float32 and transfer to the GPU
            X_gpu = cp.asarray(X.astype(np.float32))

            # Record the start time right before the modeling process
            start_time = time.time()
            
            # Instantiate the model for the current replicate using the unified function
            model = get_gpu_model(method, seed=i, n_clusters=n_clusters)
            # Fit the model and get predictions on the GPU
            labels_gpu = model.fit_predict(X_gpu)
                
            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Transfer the predicted labels back to the CPU memory for scoring
            labels_cpu = labels_gpu.get()

            # Calculate ARI using the CPU labels
            ari = adjusted_rand_score(y_true, labels_cpu)
            # Calculate NMI using the CPU labels
            nmi = normalized_mutual_info_score(y_true, labels_cpu)

            # Append the results for this replicate into the list
            all_results.append({
                # Store sample size
                "n_samples": n_samples,
                # Store feature size
                "n_features": n_features,
                # Store replicate index
                "Replicate": i + 1,
                # Store execution time
                "GPU_Runtime_sec": elapsed_time,
                # Store ARI score
                "GPU_ARI": ari,
                # Store NMI score
                "GPU_NMI": nmi
            })
            
            # Print a short status update indicating progress
            print(f"  Replicate {i + 1}/{n_replicates} done in {elapsed_time:.4f}s")

    # Convert the collected list of dictionaries into a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Ensure the 'results' directory exists
    os.makedirs("results", exist_ok=True)
    
    # Define the output CSV filename dynamically
    output_filename = f"new_results/python_{method.lower()}_gpu_results.csv"
    # Save the DataFrame to a CSV without row indices
    results_df.to_csv(output_filename, index=False)
    # Print confirmation that the file was saved
    print(f"-> Saved all results for {method} to {output_filename}")

# Print final message after all benchmark sweeps are done
print("\nAll GPU benchmarks completed successfully.")