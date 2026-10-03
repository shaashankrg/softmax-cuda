# softmax-cuda

Softmax CUDA kernels optimized step by step (naive → shared memory → warp shuffles → fused online softmax), benchmarked against `torch.softmax` and plugged into Llama 3.2 1B.

## Run (Colab, T4 GPU)

```python
!git clone https://github.com/<you>/softmax-cuda.git
%cd softmax-cuda
!python tests/test_correctness.py
```

*Results coming soon.*
