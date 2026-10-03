// softmax_kernels.cu
// All softmax kernel versions + their host-side "launcher" functions.
// A launcher runs on the CPU: it allocates the output, picks grid/block sizes,
// and launches the kernel on the GPU.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// V3: one block per row + warp-shuffle reductions
// ---------------------------------------------------------------------------

constexpr unsigned FULL_MASK = 0xffffffff;  // all 32 lanes of the warp participate
constexpr int V3_THREADS = 256;
constexpr int V3_WARPS = V3_THREADS / 32;   // 8 warps per block

// Reduce 32 values held in registers (one per lane) in 5 steps.
// After the loop, every lane holds the result (butterfly pattern).
__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, offset);
    }
    return v;
}

// Block-wide reductions: warp-reduce, lane 0 of each warp parks its result in
// shared memory, then every warp reduces those 8 values itself so the final
// answer ends up in all 256 threads without a second trip through shared memory.
__device__ __forceinline__ float block_max(float v, float* smem) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();  // all 8 warp results written before anyone reads them
    v = (lane < V3_WARPS) ? smem[lane] : -INFINITY;
    return warp_max(v);
}

__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (lane < V3_WARPS) ? smem[lane] : 0.0f;
    return warp_sum(v);
}

__global__ void softmax_v3_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int cols) {
    // Separate buffers for max and sum, so a fast warp writing its partial sum
    // can never overwrite a partial max that a slow warp hasn't read yet.
    __shared__ float smem_max[V3_WARPS];
    __shared__ float smem_sum[V3_WARPS];

    const int tid = threadIdx.x;
    const float* x_row = x + (size_t)blockIdx.x * cols;
    float* y_row = y + (size_t)blockIdx.x * cols;

    // Pass 1: max
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += V3_THREADS) {
        local_max = fmaxf(local_max, x_row[c]);
    }
    const float row_max = block_max(local_max, smem_max);

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += V3_THREADS) {
        local_sum += expf(x_row[c] - row_max);
    }
    const float row_sum = block_sum(local_sum, smem_sum);

    // Pass 3: normalize and write
    for (int c = tid; c < cols; c += V3_THREADS) {
        y_row[c] = expf(x_row[c] - row_max) / row_sum;
    }
}

torch::Tensor softmax_v3_cuda(torch::Tensor x) {
    const int rows = x.size(0);
    const int cols = x.size(1);
    auto y = torch::empty_like(x);
    if (x.numel() == 0) return y;

    softmax_v3_kernel<<<rows, V3_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---------------------------------------------------------------------------
// V4: fused online softmax (max and sum in one pass)
// ---------------------------------------------------------------------------

constexpr int V4_THREADS = 256;
constexpr int V4_WARPS = V4_THREADS / 32;

// Running state of online softmax over some set of elements:
//   m = max of the elements seen so far
//   d = sum of exp(x - m) over those elements
struct MD {
    float m;
    float d;
};

// Merge the states of two disjoint sets of elements. Each d is rescaled to the
// shared max before adding, so the result is exactly the state of the union.
__device__ __forceinline__ MD md_combine(MD a, MD b) {
    const float m = fmaxf(a.m, b.m);
    return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
}

__device__ __forceinline__ MD warp_md(MD v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        MD other = {__shfl_xor_sync(FULL_MASK, v.m, offset),
                    __shfl_xor_sync(FULL_MASK, v.d, offset)};
        v = md_combine(v, other);
    }
    return v;
}

__device__ __forceinline__ MD block_md(MD v, MD* smem) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_md(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (lane < V4_WARPS) ? smem[lane] : MD{-FLT_MAX, 0.0f};
    return warp_md(v);
}

// Online-softmax update: fold one new element into the running state.
__device__ __forceinline__ void md_update(MD& state, float v) {
    const float new_m = fmaxf(state.m, v);
    state.d = state.d * __expf(state.m - new_m) + __expf(v - new_m);
    state.m = new_m;
}

// VEC = true: each thread loads 4 floats at once (one 128-bit float4 load).
// Requires cols % 4 == 0 and a 16-byte-aligned base pointer; the launcher
// checks this and falls back to VEC = false otherwise.
template <bool VEC>
__global__ void softmax_v4_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int cols) {
    __shared__ MD smem[V4_WARPS];

    const int tid = threadIdx.x;
    const float* x_row = x + (size_t)blockIdx.x * cols;
    float* y_row = y + (size_t)blockIdx.x * cols;

    // Pass 1: running max and running sum together, one read of the row.
    // -FLT_MAX, not -INFINITY: if a thread has no elements, combining two
    // empty states must give exp(-FLT_MAX - -FLT_MAX) = exp(0), not exp(NaN).
    MD state = {-FLT_MAX, 0.0f};
    if constexpr (VEC) {
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (int i = tid; i < cols / 4; i += V4_THREADS) {
            const float4 v = x4[i];
            // Max of the 4 first (cheap, independent), then ONE rescale of d.
            const float new_m = fmaxf(state.m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            state.d = state.d * __expf(state.m - new_m)
                    + __expf(v.x - new_m) + __expf(v.y - new_m)
                    + __expf(v.z - new_m) + __expf(v.w - new_m);
            state.m = new_m;
        }
    } else {
        for (int c = tid; c < cols; c += V4_THREADS) {
            md_update(state, x_row[c]);
        }
    }
    const MD row = block_md(state, smem);
    const float inv_d = 1.0f / row.d;  // one divide per thread, then multiplies

    // Pass 2: normalize and write (second and final read of the row)
    if constexpr (VEC) {
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        float4* y4 = reinterpret_cast<float4*>(y_row);
        for (int i = tid; i < cols / 4; i += V4_THREADS) {
            const float4 v = x4[i];
            float4 out;
            out.x = __expf(v.x - row.m) * inv_d;
            out.y = __expf(v.y - row.m) * inv_d;
            out.z = __expf(v.z - row.m) * inv_d;
            out.w = __expf(v.w - row.m) * inv_d;
            y4[i] = out;
        }
    } else {
        for (int c = tid; c < cols; c += V4_THREADS) {
            y_row[c] = __expf(x_row[c] - row.m) * inv_d;
        }
    }
}

torch::Tensor softmax_v4_cuda(torch::Tensor x) {
    const int rows = x.size(0);
    const int cols = x.size(1);
    auto y = torch::empty_like(x);
    if (x.numel() == 0) return y;

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream();

    // float4 needs every row start 16-byte aligned: base aligned + cols % 4 == 0.
    const bool can_vec = cols % 4 == 0 &&
                         reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(y_ptr) % 16 == 0;
    if (can_vec) {
        softmax_v4_kernel<true><<<rows, V4_THREADS, 0, stream>>>(x_ptr, y_ptr, cols);
    } else {
        softmax_v4_kernel<false><<<rows, V4_THREADS, 0, stream>>>(x_ptr, y_ptr, cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
