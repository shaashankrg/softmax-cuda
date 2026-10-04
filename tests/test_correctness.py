"""Compares every softmax_vN in the extension against torch.softmax.

Run from the repo root:  python tests/test_correctness.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_ext import load_ext  # noqa: E402

SHAPES = [
    (1, 1),        # smallest possible input
    (7, 33),       # rows and cols both odd, cols = 32 + 1
    (128, 1024),   # "nice" power-of-two shape
    (4096, 1000),  # cols not a multiple of 32 or 256
    (1024, 4096),  # typical LLM hidden size
    (64, 32768),   # very long rows (bigger than one block can hold in shared memory)
]
STABILITY_SHAPE = (128, 1024)


def check(fn, name):
    torch.manual_seed(0)
    cases = [(shape, 10.0) for shape in SHAPES] + [(STABILITY_SHAPE, 1000.0)]
    all_ok = True
    for shape, scale in cases:
        x = torch.randn(*shape, device="cuda") * scale
        out = fn(x)
        ref = torch.softmax(x, dim=-1)
        # allclose is False if `out` has NaN/inf, so overflow bugs fail here too.
        ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
        max_err = (out - ref).abs().max().item()
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  shape={str(shape):<13} "
              f"scale={scale:<6} max_abs_err={max_err:.2e}")

    # Contiguous but starting 4 bytes into its buffer: not 16-byte aligned, so
    # kernels with float4 loads must detect it and take their scalar path.
    x = (torch.randn(128 * 1024 + 1, device="cuda") * 10)[1:].view(128, 1024)
    out, ref = fn(x), torch.softmax(x, dim=-1)
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    all_ok &= ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  misaligned (128, 1024) "
          f"max_abs_err={(out - ref).abs().max().item():.2e}")

    # Causal attention mask: row i may only see columns 0..i; the rest are filled
    # with -inf or the most negative float, like Llama's attention mask does.
    # Row 0 has a single visible element, so its output must be exactly [1, 0, 0, ...].
    x = torch.randn(256, 1024, device="cuda") * 10
    hidden = torch.ones(256, 1024, dtype=torch.bool, device="cuda").triu(diagonal=1)
    for fill_name, fill in (("-inf", float("-inf")), ("finfo.min", torch.finfo(torch.float32).min)):
        xm = x.masked_fill(hidden, fill)
        out, ref = fn(xm), torch.softmax(xm, dim=-1)
        ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  causal mask ({fill_name:<9}) "
              f"max_abs_err={(out - ref).abs().max().item():.2e}")
    return all_ok


if __name__ == "__main__":
    ext = load_ext()
    # Test every version that exists, so adding softmax_v2 later needs no edit here.
    versions = [v for v in ("softmax_v1", "softmax_v2", "softmax_v3", "softmax_v4")
                if hasattr(ext, v)]
    results = {v: check(getattr(ext, v), v) for v in versions}
    print()
    for v, ok in results.items():
        print(f"{v}: {'ALL PASSED' if ok else 'FAILED'}")
    sys.exit(0 if all(results.values()) else 1)
