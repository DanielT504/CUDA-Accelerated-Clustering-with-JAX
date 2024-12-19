#!/bin/bash

# Define project root directory as the current directory
project_dir="$(pwd)"
log_file="build_and_run.log"

# Ensure the script is running in the project root
if [ ! -d "$project_dir" ]; then
    echo "Project directory not found: $project_dir"
    exit 1
fi

# Compile the CUDA shared library
echo "Compiling CUDA library..."
nvcc -shared -o cuda/kmeans_cuda.so cuda/kmeans_cuda.cu || { echo "CUDA compilation failed"; exit 1; }

# Verify the .so file
if [ ! -f "cuda/kmeans_cuda.so" ]; then
    echo "CUDA library compilation failed or file not found!"
    exit 1
fi
echo "CUDA library should be here: $(pwd)/cuda/kmeans_cuda.so"
# Run the Python script
echo "Running JAX clustering with CUDA acceleration..."
python jax/kmeans_jax.py || { echo "Python script execution failed"; exit 1; }

echo "Script completed successfully!"
