import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# Generate a random dataset (replace with real dataset if needed)
data = np.random.rand(1000, 2).astype(np.float32)  # 1000 points, 2 dimensions
data = jnp.array(data)

# Initialize centroids randomly
key = jax.random.PRNGKey(0)
centroids = jax.random.choice(key, data, shape=(5,))  # 5 centroids

# Define functions
def compute_distances(data, centroids):
    """Compute Euclidean distances from points to centroids."""
    return jnp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)

def assign_clusters(distances):
    """Assign points to the nearest centroid."""
    return jnp.argmin(distances, axis=1)

def update_centroids(data, clusters, n_centroids):
    """Update centroids as the mean of assigned points."""
    return jnp.array([
        data[clusters == k].mean(axis=0) if jnp.any(clusters == k) else data[jnp.random.randint(len(data))]
        for k in range(n_centroids)
    ])

# Perform k-Means
print("Starting JAX k-Means...")
start = time.time()
for i in range(10):
    distances = compute_distances(data, centroids)
    clusters = assign_clusters(distances)
    centroids = update_centroids(data, clusters, 5)
end = time.time()
print(f"JAX k-Means completed in {end - start:.4f} seconds")

# Save results
results = pd.DataFrame(np.array(data), columns=["X1", "X2"])
results["Cluster"] = np.array(clusters)
output_path = os.path.join(os.path.dirname(__file__), "../r/cluster_results.csv")
results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Optional visualization
plt.figure(figsize=(8, 6))
for cluster in range(5):
    cluster_points = results[results["Cluster"] == cluster]
    plt.scatter(cluster_points["X1"], cluster_points["X2"], label=f"Cluster {cluster}")
plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c="black", marker="x", s=100, label="Centroids")
plt.legend()
plt.title("k-Means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
