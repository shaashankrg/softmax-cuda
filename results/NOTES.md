# Benchmark notes (T4, float32, median of 100 runs)

## Headline numbers

- **V4 vs PyTorch:** 60% to 114% of PyTorch's throughput, depending on row length. Best case 114% at 4096×1024; worst 60% at 4096×512. In an earlier quick run V4 reached 124% at 1024×32768.
- **V4 speedup over V1:** 5.5× (4096×4096) to 15.7× (1024×16384).
- **V4 bandwidth:** 75% of the T4's 320 GB/s peak at best (241 GB/s, 4096×1024).

## Where V4 wins and loses

| Shape | Row size | V4 vs torch | Likely reason |
|---|---|---|---|
| 4096×512 | 2 KB | 60% | 256 threads for 512 elements: too little work per block, so the fixed costs per row (reductions, barriers) dominate |
| 4096×1024 | 4 KB | **114%** | Enough work per block; V4 at 75% of peak |
| 4096×4096 | 16 KB | 77% | V4 slower than V3 here; cause unknown, needs a profiler |
| 1024×16384 | 64 KB | 91% | Fewer reads helps (V4 is 1.28× V3), but PyTorch is still ahead |

## Next optimizations

1. One warp per row for short rows, holding the whole row in registers so it's read from global memory once.
2. Caching medium rows in shared memory so the second pass doesn't go back to global memory.
3. Profiling V4 at 4096×4096 with Nsight Compute to find the actual cause.
4. float16/bfloat16 support, which halves the bytes moved.
