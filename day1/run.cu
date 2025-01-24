// Day 1: "Hello, GPU World!"

#include <iostream>

// CUDA kernel function
__global__ void helloFromGPU() {
    printf("Hello, GPU World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 10 threads per block
    helloFromGPU<<<1, 10>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello, GPU World from the CPU!" << std::endl;

    return 0;
}
