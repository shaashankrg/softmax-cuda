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

// ---------------------------------------------------------------------------
// V2: one block per row + shared-memory tree reduction
// ---------------------------------------------------------------------------

constexpr int V2_THREADS = 256;  // must be a power of 2 for the tree reduction

// A block of 256 threads cooperates on one row.
__global__ void softmax_v2_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int cols) {
    __shared__ float smem[V2_THREADS];  // one slot per thread, visible to the whole block

    const int tid = threadIdx.x;
    const float* x_row = x + (size_t)blockIdx.x * cols;
    float* y_row = y + (size_t)blockIdx.x * cols;

    // Pass 1a: each thread finds the max of its strided elements: tid, tid+256, ...
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += V2_THREADS) {
        local_max = fmaxf(local_max, x_row[c]);
    }
    smem[tid] = local_max;
    __syncthreads();  // (A) all partial maxes written before anyone reads them

    // Pass 1b: tree reduction 256 -> 128 -> ... -> 1; result ends up in smem[0]
    for (int s = V2_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();  // (B) finish this level before starting the next
    }
    const float row_max = smem[0];
    __syncthreads();  // (C) everyone has read smem[0] before it is overwritten below

    // Pass 2: same pattern for the sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += V2_THREADS) {
        local_sum += expf(x_row[c] - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();  // (D)

    for (int s = V2_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();  // (E)
    }
    const float row_sum = smem[0];

    // Pass 3: normalize and write, coalesced
    for (int c = tid; c < cols; c += V2_THREADS) {
        y_row[c] = expf(x_row[c] - row_max) / row_sum;
    }
}

torch::Tensor softmax_v2_cuda(torch::Tensor x) {
    const int rows = x.size(0);
    const int cols = x.size(1);
    auto y = torch::empty_like(x);
    if (x.numel() == 0) return y;

    softmax_v2_kernel<<<rows, V2_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
