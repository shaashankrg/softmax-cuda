"""Compiles csrc/ into a Python module (cached; only rebuilds when sources change)."""
import os

from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_ext(verbose=False):
    return load(
        name="softmax_cuda_ext",
        sources=[
            os.path.join(ROOT, "csrc", "binding.cpp"),
            os.path.join(ROOT, "csrc", "softmax_kernels.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
