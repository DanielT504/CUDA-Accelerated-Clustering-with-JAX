import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import os
import ctypes
import time
import subprocess

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Resolve the absolute path to the CUDA shared library
cuda_lib_path = os.path.abspath(os.path.join(script_dir, "../cuda/kmeans_cuda.so"))
print(f"Expected CUDA library path: {cuda_lib_path}")

if not os.path.exists(cuda_lib_path):
    raise FileNotFoundError(f"CUDA library not found at {cuda_lib_path}")

# Load the CUDA library
cuda_lib = ctypes.CDLL(cuda_lib_path)

# Define compute_distances for CUDA
def compute_distances_cuda(data, centroids):
    n_points, dim = data.shape
    n_clusters = centroids.shape[0]
    distances = np.zeros((n_points, n_clusters), dtype=np.float32)

    # Call CUDA function
    cuda_lib.compute_distances(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_points),
        ctypes.c_int(n_clusters),
        ctypes.c_int(dim),
    )
    return distances

# Define compute_distances for Python
def compute_distances_python(data, centroids):
    return np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)

# Define compute_distances for JAX
def compute_distances_jax(data, centroids):
    return jnp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)

# Clustering logic
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters, n_clusters):
    return np.array([data[clusters == k].mean(axis=0) for k in range(n_clusters)])

# Generate random data
n_points = 10_000  # Adjust for testing purposes
data = np.random.rand(n_points, 2).astype(np.float32)
centroids = np.random.rand(5, 2).astype(np.float32)

# Timing Python implementation
start = time.time()
distances_python = compute_distances_python(data, centroids)
end = time.time()
time_python = end - start
print(f"Python Execution Time (CPU, no parallelism): {time_python:.6f} seconds")

# Timing JAX implementation
data_jax = jnp.array(data)
centroids_jax = jnp.array(centroids)
start = time.time()
distances_jax = compute_distances_jax(data_jax, centroids_jax)
distances_jax.block_until_ready()  # Ensure computation is complete
end = time.time()
time_jax = end - start
print(f"JAX Execution Time (CPU, parallelism): {time_jax:.6f} seconds")

# Timing CUDA implementation
start = time.time()
distances_cuda = compute_distances_cuda(data, centroids)
end = time.time()
time_cuda = end - start
print(f"CUDA Execution Time (GPU (Nvidia Quadro K2000), high parallelism): {time_cuda:.6f} seconds")

# Calculate speedups
speedup_jax_vs_python = time_python / time_jax
speedup_cuda_vs_python = time_python / time_cuda
speedup_cuda_vs_jax = time_jax / time_cuda

# Print speedups
print(f"Speedup (JAX vs Python): {speedup_jax_vs_python:.2f}x")
print(f"Speedup (CUDA vs Python): {speedup_cuda_vs_python:.2f}x")
print(f"Speedup (CUDA vs JAX): {speedup_cuda_vs_jax:.2f}x")

# Save clustering results
clusters = assign_clusters(distances_cuda)
output_dir = os.path.abspath(os.path.join(script_dir, "../r"))
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "cluster_results.csv")
results = pd.DataFrame(data, columns=["X1", "X2"])
results["Cluster"] = np.array(clusters)
results.to_csv(output_file, index=False)
print(f"Clustering results saved to {output_file}")

# Call the R script for analysis and visualization
try:
    r_script_path = os.path.abspath(os.path.join(script_dir, "../r/analyze_results.R"))
    subprocess.run(["Rscript", r_script_path], check=True)
    print("R analysis and visualization completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"R script execution failed: {e}")
