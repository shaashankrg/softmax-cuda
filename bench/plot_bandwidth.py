"""Reads results/benchmarks.csv and saves results/bandwidth.png (grouped bar chart).

Run from the repo root:  python bench/plot_bandwidth.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")  # render straight to a file; no display needed
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T4_PEAK_GBPS = 320.0

KERNELS = ["v1", "v2", "v3", "v4", "torch"]
LABELS = {
    "v1": "V1 naive",
    "v2": "V2 shared memory",
    "v3": "V3 warp shuffles",
    "v4": "V4 online softmax",
    "torch": "PyTorch",
}
# Greys for the stepping-stone versions, one strong color for V4, a contrasting one for PyTorch.
COLORS = {"v1": "#d0d0d0", "v2": "#a8a8a8", "v3": "#7f7f7f", "v4": "#76b900", "torch": "#3b6ea8"}


def main():
    with open(os.path.join(ROOT, "results", "benchmarks.csv")) as f:
        data = list(csv.DictReader(f))

    shapes = list(dict.fromkeys(r["shape"] for r in data))  # keep CSV order, drop duplicates
    gbps = {(r["shape"], r["kernel"]): float(r["gbps"]) for r in data}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.16
    for i, k in enumerate(KERNELS):
        xs = [s + (i - (len(KERNELS) - 1) / 2) * width for s in range(len(shapes))]
        ax.bar(xs, [gbps[(shape, k)] for shape in shapes], width, label=LABELS[k], color=COLORS[k])

    ax.axhline(T4_PEAK_GBPS, linestyle="--", color="black", linewidth=1)
    ax.text(len(shapes) - 0.5, T4_PEAK_GBPS + 5, "T4 peak (320 GB/s)", ha="right", va="bottom")

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([s.replace("x", " rows × ") + " cols" for s in shapes])
    ax.set_xlabel("Input shape")
    ax.set_ylabel("Effective bandwidth (GB/s) — higher is better")
    ax.set_title("Softmax CUDA kernels vs PyTorch on an NVIDIA T4")
    ax.set_ylim(0, T4_PEAK_GBPS * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()

    out_path = os.path.join(ROOT, "results", "bandwidth.png")
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
