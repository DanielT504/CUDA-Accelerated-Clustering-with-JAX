#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // For measuring execution time

__global__ void compute_distances_gpu(float *data, float *centroids, float *distances, int n_points, int n_clusters, int dim) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n_points) {
        for (int c = 0; c < n_clusters; ++c) {
            float dist = 0;
            for (int d = 0; d < dim; ++d) {
                float diff = data[point_idx * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            distances[point_idx * n_clusters + c] = sqrtf(dist);
        }
    }
}

void compute_distances_cpu(const std::vector<float>& data, const std::vector<float>& centroids, std::vector<float>& distances, int n_points, int n_clusters, int dim) {
    for (int i = 0; i < n_points; ++i) {
        for (int c = 0; c < n_clusters; ++c) {
            float dist = 0;
            for (int d = 0; d < dim; ++d) {
                float diff = data[i * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            distances[i * n_clusters + c] = std::sqrt(dist);
        }
    }
}

int main() {
    const int n_points = 1000; // Adjustable for testing
    const int n_clusters = 5;
    const int dim = 2;

    // Host data
    std::vector<float> h_data(n_points * dim);
    std::vector<float> h_centroids(n_clusters * dim);
    std::vector<float> h_distances_cpu(n_points * n_clusters);
    std::vector<float> h_distances_gpu(n_points * n_clusters);

    // Fill data with random values
    for (int i = 0; i < h_data.size(); ++i) h_data[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < h_centroids.size(); ++i) h_centroids[i] = rand() / float(RAND_MAX);

    // Measure CPU computation time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    compute_distances_cpu(h_data, h_centroids, h_distances_cpu, n_points, n_clusters, dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time: " << cpu_time << " ms\n";

    // Allocate device memory
    float *d_data, *d_centroids, *d_distances;
    cudaMalloc(&d_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_centroids, h_centroids.size() * sizeof(float));
    cudaMalloc(&d_distances, h_distances_gpu.size() * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), h_centroids.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU computation time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    int threads_per_block = 256;
    int num_blocks = (n_points + threads_per_block - 1) / threads_per_block;
    compute_distances_gpu<<<num_blocks, threads_per_block>>>(d_data, d_centroids, d_distances, n_points, n_clusters, dim);
    cudaDeviceSynchronize(); // Ensure kernel completion
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    std::cout << "GPU time: " << gpu_time << " ms\n";

    // Copy results back to host
    cudaMemcpy(h_distances_gpu.data(), d_distances, h_distances_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    for (size_t i = 0; i < h_distances_cpu.size(); ++i) {
        if (std::abs(h_distances_cpu[i] - h_distances_gpu[i]) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": " << h_distances_cpu[i] << " vs " << h_distances_gpu[i] << "\n";
            return -1;
        }
    }

    std::cout << "Results are consistent.\n";

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);

    return 0;
}
