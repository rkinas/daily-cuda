#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CPU implementation of ReLU
void reluCPU(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = std::fmax(input[i], 0.0f);
    }
}

// CUDA kernel for ReLU
__global__ void reluCUDA(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

int main() {
    const int N = 1000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_input = new float[N];
    float *h_output_cpu = new float[N];
    float *h_output_gpu = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i) / N - 0.5f; // Values from -0.5 to 0.5
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run CPU implementation
    reluCPU(h_input, h_output_cpu, N);

    // Run CUDA implementation
    reluCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    std::cout << "Comparing CPU and GPU results:\n";
    for (int i = 0; i < 10; i++) { // Compare first 10 elements
        std::cout << "Index " << i << ": CPU = " << h_output_cpu[i]
                  << ", GPU = " << h_output_gpu[i]
                  << ", Difference = " << fabs(h_output_cpu[i] - h_output_gpu[i]) << "\n";
    }

    // Free memory
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
