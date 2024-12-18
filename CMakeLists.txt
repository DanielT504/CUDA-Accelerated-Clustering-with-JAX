cmake_minimum_required(VERSION 3.18)  # Ensure compatibility with CUDA support in CMake
project(kmeans_cuda LANGUAGES CUDA)

# Specify the target executable and source file
add_executable(kmeans_cuda cuda/kmeans_cuda.cu)

# Set CUDA compilation options
set_target_properties(kmeans_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 30  # Adjust this to your GPU's compute capability
)
