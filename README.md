# CUDA 30-Day Challenge 🚀

Welcome to the **CUDA 30-Day Challenge**! 🎉 This repository is dedicated to exploring and implementing CUDA-based solutions over the next 30 days. Each day introduces a new challenge, starting with beginner-friendly tasks and gradually focusing on optimizing AI and deep learning workflows.

The goal? To learn, experiment, and build a strong foundation in GPU programming and its applications in AI.

---

## Table of Contents
- [About](#about)
- [Challenges](#challenges)
- [How to Use](#how-to-use)
- [Prerequisites](#prerequisites)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

---

## About
This repository contains:
- **Daily challenges**: A set of 30 tasks tailored to AI and deep learning optimizations.
- **Solutions**: CUDA implementations for each challenge.
- **Documentation**: Notes and explanations for every solution.

---

## Challenges
### Day 1-10: Basics of CUDA for AI 🛠️
1. "Hello, GPU World!" - Write and execute your first CUDA kernel.
2. Element-wise Operations - Perform element-wise addition and scaling for tensors.
3. Activation Functions - Implement ReLU, Sigmoid, and Tanh activations in CUDA.
4. Softmax Function - Compute the softmax for a 1D tensor.
5. Matrix-Vector Multiplication - Multiply a matrix and a vector efficiently.
6. Shared Memory Basics - Use shared memory to optimize tensor operations.
7. Prefix Sum (Scan) - Implement an inclusive prefix sum for a small tensor.
8. Loss Function: MSE - Compute the Mean Squared Error loss for two tensors.
9. Cross-Entropy Loss - Implement the categorical cross-entropy loss function.
10. Memory Coalescing - Optimize memory access for tensor addition.

### Day 11-20: Intermediate AI Optimization 🌀
11. Batch Matrix Multiplication - Multiply batches of matrices in parallel.
12. Tiled Matrix Multiplication - Optimize matrix multiplication using shared memory.
13. Forward Pass: Dense Layer - Implement a dense (fully connected) layer forward pass.
14. Backpropagation: Gradient of MSE - Compute the gradient of MSE loss with respect to weights.
15. Convolution: 1D Signal - Apply a 1D convolution to a signal.
16. Convolution: 2D Images - Implement a 2D convolution for image data.
17. Max Pooling - Apply max pooling to a 2D image.
18. Dropout - Simulate dropout regularization in a forward pass.
19. Layer Normalization - Implement layer normalization for a batch of inputs.
20. Data Augmentation - Apply basic data augmentation (flipping, cropping) on images.

### Day 21-30: Advanced AI Workflows 🚀
21. Backpropagation: Dense Layer - Compute gradients for a dense layer.
22. Softmax with Logits - Implement a numerically stable softmax function with logits.
23. Sparse Matrix Operations - Multiply sparse matrices with dense tensors.
24. Custom CUDA Kernel for Attention - Implement a simplified attention mechanism.
25. Optimizer: SGD - Implement Stochastic Gradient Descent in CUDA.
26. Optimizer: Adam - Implement the Adam optimizer for model training.
27. Embedding Lookup - Optimize embedding lookups for NLP tasks.
28. Transformer Encoder - Implement the core computations of a transformer encoder block.
29. Mixed Precision Training - Use half-precision (FP16) arithmetic for tensor computations.
30. Neural Network Inference - Optimize a simple neural network for real-time inference.

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cuda-30-day-challenge.git
   ```
2. Navigate to the day-specific folder:
   ```bash
   cd day-01
   ```
3. Compile and run the code:
   ```bash
   nvcc main.cu -o main && ./main
   ```

---

## Prerequisites
- Basic knowledge of C/C++ programming
- NVIDIA GPU with CUDA support
- Installed CUDA Toolkit (compatible version for your GPU)

---

## Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Deep Learning with CUDA](https://developer.nvidia.com/deep-learning)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [Accelerated Computing Documentation](https://developer.nvidia.com/documentation/)

---

## Contributing
Contributions are welcome! If you want to improve the solutions or suggest new challenges, feel free to open an issue or a pull request.

---

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding! 🚀
