#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

/**
 * @brief CPU implementation of the Softmax function
 * @param input Pointer to the input array
 * @param output Pointer to the output array where softmax results will be stored
 * @param n Number of elements in the input array
 */
void softmaxCPU(const float* input, float* output, int n) {
    // Step 1: Find the maximum value for numerical stability
    float max_val = *std::max_element(input, input + n);

    // Step 2: Compute the exponentials and their sum
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Step 3: Normalize the output array
    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

/**
 * @brief CUDA kernel for the Softmax function
 * @param input Pointer to the input array on the device
 * @param output Pointer to the output array on the device where softmax results will be stored
 * @param n Number of elements in the input array
 */
__global__ void softmaxCUDA(const float* input, float* output, int n) {
    extern __shared__ float shared_data[]; // Shared memory for intermediate calculations

    int idx = threadIdx.x;                // Thread index within a block
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // Step 1: Load input data into shared memory and find the maximum value
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
    const int N = 1024;                      // Number of elements
    const int threadsPerBlock = 256;         // Number of threads per block
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks

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
