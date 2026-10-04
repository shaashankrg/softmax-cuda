"""Temporarily routes torch.nn.functional.softmax through our CUDA kernel."""
import contextlib

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def custom_softmax(ext):
    """Inside `with custom_softmax(ext) as stats:`, every F.softmax call our kernel
    can handle runs softmax_v4; everything else falls back to PyTorch.
    `stats` counts both, so you can confirm the swap actually happened."""
    original = F.softmax
    stats = {"kernel_calls": 0, "fallback_calls": 0}

    # Same signature as torch.nn.functional.softmax, so callers can't tell the difference.
    def wrapper(input, dim=None, _stacklevel=3, dtype=None):
        x = input
        if (x.is_cuda
                and x.dtype == torch.float32               # kernel is float32-only
                and dtype in (None, torch.float32)         # caller isn't asking for a different output dtype
                and x.dim() >= 1 and x.numel() > 0
                and dim is not None and dim in (-1, x.dim() - 1)  # kernel only does the last dim
                and not (torch.is_grad_enabled() and x.requires_grad)):  # kernel has no backward pass
            stats["kernel_calls"] += 1
            shape = x.shape
            # (batch, heads, q_len, k_len) -> (batch*heads*q_len, k_len): every row is
            # still one softmax row, and view() on contiguous memory copies nothing.
            return ext.softmax_v4(x.contiguous().view(-1, shape[-1])).view(shape)
        stats["fallback_calls"] += 1
        return original(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    F.softmax = wrapper
    try:
        yield stats
    finally:
        F.softmax = original  # restore even if the code inside the `with` raised
