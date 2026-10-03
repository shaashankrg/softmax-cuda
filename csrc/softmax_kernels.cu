// softmax_kernels.cu
// All softmax kernel versions + their host-side "launcher" functions.
// A launcher runs on the CPU: it allocates the output, picks grid/block sizes,
// and launches the kernel on the GPU.

#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// V1: naive (one thread per row)
// ---------------------------------------------------------------------------

// PLACEHOLDER (Task 2): calls PyTorch's own softmax so we can prove the
// build + Python binding pipeline works end to end. Replaced by a real
// kernel in Task 4.
torch::Tensor softmax_v1_cuda(torch::Tensor x) {
    return torch::softmax(x, /*dim=*/-1);
}
