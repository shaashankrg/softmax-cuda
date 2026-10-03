// softmax_kernels.cu
// All softmax kernel versions + their host-side "launcher" functions.
// A launcher runs on the CPU: it allocates the output, picks grid/block sizes,
// and launches the kernel on the GPU.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// V1: naive (one thread per row)
// ---------------------------------------------------------------------------

// Each thread owns one entire row and walks it three times.
__global__ void softmax_v1_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;  // last block may have more threads than rows left

    const float* x_row = x + (size_t)row * cols;
    float* y_row = y + (size_t)row * cols;

    // Pass 1: row max (for numerical stability)
    float max_val = -INFINITY;
    for (int c = 0; c < cols; ++c) {
        max_val = fmaxf(max_val, x_row[c]);
    }

    // Pass 2: sum of exp(x - max)
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        sum += expf(x_row[c] - max_val);
    }

    // Pass 3: normalize and write
    for (int c = 0; c < cols; ++c) {
        y_row[c] = expf(x_row[c] - max_val) / sum;
    }
}

torch::Tensor softmax_v1_cuda(torch::Tensor x) {
    const int rows = x.size(0);
    const int cols = x.size(1);
    auto y = torch::empty_like(x);
    if (x.numel() == 0) return y;  // launching 0 blocks is a CUDA error

    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;  // ceil(rows / threads)

    softmax_v1_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
