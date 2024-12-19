#include <cuda_runtime.h>
#include <cmath>

extern "C" __declspec(dllexport) void compute_distances(float *data, float *centroids, float *distances, int n_points, int n_clusters, int dim);

__global__ void compute_distances_kernel(float *data, float *centroids, float *distances, int n_points, int n_clusters, int dim) {
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

extern "C" __declspec(dllexport) void compute_distances(float *data, float *centroids, float *distances, int n_points, int n_clusters, int dim) {
    int threads_per_block = 256;
    int num_blocks = (n_points + threads_per_block - 1) / threads_per_block;

    float *d_data, *d_centroids, *d_distances;
    cudaMalloc(&d_data, n_points * dim * sizeof(float));
    cudaMalloc(&d_centroids, n_clusters * dim * sizeof(float));
    cudaMalloc(&d_distances, n_points * n_clusters * sizeof(float));

    cudaMemcpy(d_data, data, n_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);

    compute_distances_kernel<<<num_blocks, threads_per_block>>>(d_data, d_centroids, d_distances, n_points, n_clusters, dim);

    cudaMemcpy(distances, d_distances, n_points * n_clusters * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);
}
