"""Benchmarks V1-V4 against torch.softmax and writes results/benchmarks.csv.

Run from the repo root:  python bench/bench_kernels.py
"""
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from load_ext import load_ext  # noqa: E402

SHAPES = [(4096, 512), (4096, 1024), (4096, 4096), (1024, 16384)]
T4_PEAK_GBPS = 320.0
WARMUP = 10
RUNS = 100


def median_ms(fn, x):
    for _ in range(WARMUP):
        fn(x)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
    for i in range(RUNS):
        # Events are recorded *into the GPU's queue*, so they timestamp when the
        # GPU actually reaches them, not when Python called record().
        starts[i].record()
        fn(x)
        ends[i].record()
    torch.cuda.synchronize()  # wait until every queued kernel and event is done
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[RUNS // 2]


def main():
    ext = load_ext()
    kernels = [("torch", lambda t: torch.softmax(t, dim=-1))]
    kernels += [(f"v{i}", getattr(ext, f"softmax_v{i}")) for i in range(1, 5)]

    rows_out = []
    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda")
        ref = torch.softmax(x, dim=-1)
        min_bytes = 2 * rows * cols * 4  # read input once + write output once (float32)

        results = {}
        for name, fn in kernels:
            # Never report a speed for a wrong answer.
            assert torch.allclose(fn(x), ref, atol=1e-5, rtol=1e-4), f"{name} wrong at {(rows, cols)}"
            results[name] = median_ms(fn, x)

        print(f"shape=({rows}, {cols})")
        for name, ms in results.items():
            gbps = min_bytes / (ms * 1e-3) / 1e9
            pct_peak = gbps / T4_PEAK_GBPS * 100
            pct_torch = results["torch"] / ms * 100
            print(f"  {name:<6} {ms:8.3f} ms  {gbps:6.1f} GB/s  "
                  f"{pct_peak:5.1f}% of peak  {pct_torch:6.1f}% of torch")
            rows_out.append({
                "shape": f"{rows}x{cols}", "rows": rows, "cols": cols, "kernel": name,
                "median_ms": round(ms, 4), "gbps": round(gbps, 1),
                "pct_peak": round(pct_peak, 1), "pct_torch": round(pct_torch, 1),
            })

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results", "benchmarks.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nGPU: {torch.cuda.get_device_name()}  ->  wrote {out_path}")


if __name__ == "__main__":
    main()
