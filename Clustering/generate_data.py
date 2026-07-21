# Import the os module for operating system dependent functionality
import os
# Import the json module to read and write configuration files
import json
# Import numpy for numerical operations and array saving
import numpy as np
# Import make_blobs from scikit-learn for synthetic data generation
from sklearn.datasets import make_blobs

# Define the number of clusters for all datasets
n_clusters = 5
# Define the number of replicate datasets per configuration
n_replicates = 100
# Define the base output directory name
base_output_dir = "new_data"

# Create the base directory, ignoring errors if it already exists
os.makedirs(base_output_dir, exist_ok=True)

# Define Slice 1: Varying sample size (n) while fixing feature size (p)
slice_vary_n = [(1000, 50), (5000, 50), (10000, 50)]

# Define Slice 2: Varying feature size (p) while fixing sample size (n)
slice_vary_p = [(5000, 100), (5000, 200)]

# Combine the two slices into a single list
all_configs = slice_vary_n + slice_vary_p

# Remove any duplicate parameter combinations by converting to a set and back to a list
unique_configs = list(set(all_configs))
# Sort the configurations by n first, then p, for organized output
unique_configs.sort(key=lambda x: (x[0], x[1]))

# Define the function to generate a dataset
def generate_dataset(n_samples, n_features, n_clusters, seed=0, noise_level=1): #
    std_array = np.linspace(start=1, stop=(n_clusters * 2), num=n_clusters)
    
    # Generate the feature matrix X and true labels y
    X, y = make_blobs(n_samples=n_samples, 
                      # Set the number of features
                      n_features=n_features, 
                      # Set the number of clusters
                      centers=n_clusters, 
                      # Set the random seed for reproducibility
                      random_state=seed,
                      # Apply the reduced standard deviations to the clusters
                      cluster_std=std_array,
                      center_box=(-20.0, 20.0)
                     )

    # Initialize a random number generator with the same seed for consistent noise generation
    rng = np.random.RandomState(seed)
    
    # Generate a Gaussian noise matrix with the exact same dimensions as X
    # 'loc=0.0' means the noise is centered at zero, 'scale' controls the noise amplitude
    noise_matrix = rng.normal(loc=0.0, scale=noise_level, size=X.shape)
    
    # Add the generated noise matrix element-wise to the original feature matrix X
    X_noisy = X + noise_matrix
    
    # Return the new noisy feature matrix and the original unchanged cluster labels
    return X_noisy, y

# Loop through each unique combination of n and p
for n_samples, n_features in unique_configs:
    
    # Construct a folder name indicating whether this is a varying n or p experiment
    combo_folder = f"n_{n_samples}_p_{n_features}"
    # Construct the full path for the current configuration
    combo_dir = os.path.join(base_output_dir, combo_folder)
    # Create the directory for this configuration
    os.makedirs(combo_dir, exist_ok=True)
    
    # Create a dictionary holding the current parameters
    config = {
        # Save the current sample size
        "n_samples": n_samples,
        # Save the current feature size
        "n_features": n_features,
        # Save the number of clusters
        "n_clusters": n_clusters,
        # Save the number of replicates
        "n_replicates": n_replicates
    }
    
    # Construct the path for the configuration JSON file
    config_path = os.path.join(combo_dir, "config.json")
    # Open the JSON file for writing
    with open(config_path, "w") as f:
        # Write the dictionary to the file with indentation
        json.dump(config, f, indent=2)
        
    # Print progress to the console
    print(f"Generating benchmark slice: n={n_samples}, p={n_features} -> {combo_dir}/")
    
    # Loop over the number of replicates to generate data
    for i in range(n_replicates):
        # Construct the file name for the current replicate
        filepath = os.path.join(combo_dir, f"replicate_{i}.npz")
        
        # Generate the dataset using the defined function
        X, y = generate_dataset(n_samples=n_samples, 
                                # Pass feature size
                                n_features=n_features,
                                # Pass cluster size
                                n_clusters=n_clusters, 
                                # Use the loop index as the random seed
                                seed=i)
                                
        # Save the dataset to disk in compressed format
        np.savez_compressed(filepath, X=X, y=y)
        
        # Print a small progress indicator for replicates
        print(f"  Saved replicate {i + 1}/{n_replicates}")
        
    # Print a newline for visual separation in the console
    print("")

# Print completion message
print("Done. Datasets generated.")