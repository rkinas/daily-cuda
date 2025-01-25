#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CPU implementation of Softmax
void softmaxCPU(const float* input, float* output, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += expf(input[i] - max_val);
    }

    for (int i = 0; i < n; ++i) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

// CUDA kernel for Softmax
__global__ void softmaxCUDA(const float* input, float* output, int n) {
    extern __shared__ float shared_data[];

    int idx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Compute the maximum value in the input (for numerical stability)
    float max_val = -INFINITY;
    if (tid < n) {
        shared_data[idx] = input[tid];
    } else {
        shared_data[idx] = -INFINITY;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (idx < stride) {
            shared_data[idx] = fmaxf(shared_data[idx], shared_data[idx + stride]);
        }
        __syncthreads();
    }

    max_val = shared_data[0];

    // Step 2: Compute the exponentials and their sum
    float sum = 0.0f;
    if (tid < n) {
        shared_data[idx] = expf(input[tid] - max_val);
    } else {
        shared_data[idx] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (idx < stride) {
            shared_data[idx] += shared_data[idx + stride];
        }
        __syncthreads();
    }

    sum = shared_data[0];

    // Step 3: Compute the softmax output
    if (tid < n) {
        output[tid] = expf(input[tid] - max_val) / sum;
    }
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_input = new float[N];
    float *h_output_cpu = new float[N];
    float *h_output_gpu = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i) / N;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run CPU implementation
    softmaxCPU(h_input, h_output_cpu, N);

    // Run CUDA kernel
    softmaxCUDA<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    std::cout << "Comparing CPU and GPU results:\n";
    for (int i = 0; i < 10; i++) {
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
