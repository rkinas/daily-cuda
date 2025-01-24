// Day 2: Element-wise Operations - Perform element-wise addition and scaling for tensors

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition and scaling
__global__ void elementWiseAddAndScale(float* a, float* b, float* result, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (a[idx] + b[idx]) * scalar;
    }
}

int main() {
    const int N = 100;
    const float scalar = 2.5f;

    // Host arrays
    float h_a[N], h_b[N], h_result[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Device arrays
    float *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with enough threads to cover all elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementWiseAddAndScale<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, scalar, N);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some of the results
    std::cout << "Element-wise addition and scaling results (first 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "h_result[" << i << "] = " << h_result[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
