// binding.cpp
// Exposes the CUDA launchers to Python via pybind11, after validating inputs.

#include <torch/extension.h>

// Declarations: the definitions live in softmax_kernels.cu (compiled by nvcc).
torch::Tensor softmax_v1_cuda(torch::Tensor x);
torch::Tensor softmax_v2_cuda(torch::Tensor x);
torch::Tensor softmax_v3_cuda(torch::Tensor x);

// Every kernel assumes a 2D, contiguous, float32 tensor on the GPU.
// Checking here turns a silent wrong answer or crash into a clear Python error.
static void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.dim() == 2, "input must be 2D (rows x cols)");
}

torch::Tensor softmax_v1(torch::Tensor x) {
    check_input(x);
    return softmax_v1_cuda(x);
}

torch::Tensor softmax_v2(torch::Tensor x) {
    check_input(x);
    return softmax_v2_cuda(x);
}

torch::Tensor softmax_v3(torch::Tensor x) {
    check_input(x);
    return softmax_v3_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_v1", &softmax_v1, "Softmax V1: naive, one thread per row");
    m.def("softmax_v2", &softmax_v2, "Softmax V2: block per row, shared-memory reduction");
    m.def("softmax_v3", &softmax_v3, "Softmax V3: block per row, warp-shuffle reduction");
}
