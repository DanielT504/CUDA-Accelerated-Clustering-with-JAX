# CUDA-Accelerated-Clustering-with-JAX

An implementation of k-Means clustering (visualized with R) that uses CUDA for GPU-accelerated distance calculations and JAX for efficient centroid updates and cluster assignments.

cmake -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2" ..

nvcc --version
cd C:\Users\User\Desktop\Not backed up\CUDA-Accelerated-Clustering-with-JAX
Remove-Item -Recurse -Force build
New-Item -ItemType Directory -Name build
cd build
cmake -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2" ..
cmake --build . --config Release
cd Release
nvprof .\kmeans_cuda.exe

export CUDAPATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2"


better data, decide on hexbin

Comprehensive Readme
