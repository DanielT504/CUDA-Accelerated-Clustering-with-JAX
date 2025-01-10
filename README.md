# CUDA-Accelerated-Clustering-with-JAX

```
Python Execution Time (CPU, no parallelism): 0.171278 seconds
JAX Execution Time (CPU, parallelism): 0.109112 seconds
CUDA Execution Time (GPU (Nvidia Quadro K2000), high parallelism): 0.086840 seconds
Speedup (JAX vs Python): 1.57x
Speedup (CUDA vs Python): 1.97x
Speedup (CUDA vs JAX): 1.26x
```

An implementation of k-Means clustering (visualized with R) that uses CUDA for GPU-accelerated distance calculations and JAX for efficient centroid updates and cluster assignments.

Demonstrates the performance of clustering computations across three implementations:
Python (Baseline, single-threaded CPU).
JAX (CPU with parallelism).
CUDA (GPU acceleration on a 2013 Nvidia Quadro K2000).

NOTE: The clustering algorithm is not yet categorizing points properly; the primary goal is to showcase speedup comparisons across implementations.

## Prerequisites:

### Python: Version 3.8 or higher.

Required Python libraries:
    jax, jaxlib
    pandas
    numpy
    matplotlib
Install with:
pip install jax jaxlib pandas numpy matplotlib

CUDA Toolkit: Ensure a compatible version of the CUDA Toolkit is installed, and the nvcc compiler is available in your PATH (for Windows).

### R:

Required R libraries:
ggplot2
Install with:
install.packages("ggplot2")

## Execution

#Compile cuda library and run kmeans_jax.py with:
./build_and_run.sh
on bash
or
.\build_and_run.ps1
on PowerShell (no logging available)

PNG output to /r/cluster_plot.png will give a hexbin clustering of the randomized data

### Example console output:

```
$ ./build_and_run.sh
Compiling CUDA library...
CUDA library should be here: /path/to/cuda/kmeans_cuda.so
Running JAX clustering with CUDA acceleration...
Expected CUDA library path: C:\path\to\cuda\kmeans_cuda.so
Python Execution Time (CPU, no parallelism): 0.171278 seconds
JAX Execution Time (CPU, parallelism): 0.109112 seconds
CUDA Execution Time (GPU (Nvidia Quadro K2000), high parallelism): 0.086840 seconds
Speedup (JAX vs Python): 1.57x
Speedup (CUDA vs Python): 1.97x
Speedup (CUDA vs JAX): 1.26x
Saving clustering results to: C:\path\to\r\cluster_results.csv
Clustering results saved successfully to C:\path\to\r\cluster_results.csv
[1] "Plot saved to: C:\\path\\to\\r\\cluster_plot.png"
[1] "Cluster visualization saved to /path/to/r/cluster_plot.png"
R analysis and visualization completed successfully.
Script completed successfully!
```
