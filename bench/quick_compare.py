"""Quick timing of every softmax version vs torch.softmax on one shape.

Run from the repo root:  python bench/quick_compare.py [rows cols]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_ext import load_ext  # noqa: E402


def time_ms(fn, x, warmup=10, iters=100, trials=7):
    for _ in range(warmup):  # first calls include one-time setup costs
        fn(x)
    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x)
        end.record()
        torch.cuda.synchronize()  # kernel launches are async; wait before reading the timer
        times.append(start.elapsed_time(end) / iters)
    # Median of several trials: one slow trial (clock change, background work) can't skew it.
    return sorted(times)[len(times) // 2]


if __name__ == "__main__":
    rows, cols = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (4096, 4096)
    ext = load_ext()
    x = torch.randn(rows, cols, device="cuda")
    min_bytes = 2 * x.numel() * x.element_size()  # read input once + write output once

    fns = [("torch", lambda t: torch.softmax(t, dim=-1))]
    fns += [(v, getattr(ext, v)) for v in ("softmax_v1", "softmax_v2", "softmax_v3", "softmax_v4")
            if hasattr(ext, v)]

    print(f"shape=({rows}, {cols})")
    for name, fn in fns:
        ms = time_ms(fn, x)
        gbps = min_bytes / (ms * 1e-3) / 1e9
        print(f"  {name:<11} {ms:8.3f} ms   {gbps:6.1f} GB/s effective")
