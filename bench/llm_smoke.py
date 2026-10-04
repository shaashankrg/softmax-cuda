"""Loads Llama 3.2 1B, checks it generates text, and shows where attention calls softmax.

Run from the repo root:  python bench/llm_smoke.py
"""
import inspect

import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama

# Ungated mirror of meta-llama/Llama-3.2-1B (same weights; Meta's license still applies).
MODEL_ID = "unsloth/Llama-3.2-1B"


def main():
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")

    # Newer transformers renamed torch_dtype -> dtype.
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",  # plain PyTorch attention, so softmax is a visible, swappable call
        **{dtype_kw: torch.float32},  # our kernels only accept float32
    ).to("cuda").eval()

    inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"forward pass OK, logits shape {tuple(logits.shape)}  (batch, tokens, vocab)")

    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print("generated:", repr(tok.decode(out[0], skip_special_tokens=True)))

    # Find every softmax call in the installed Llama implementation.
    path = inspect.getsourcefile(modeling_llama)
    print(f"\nsoftmax calls in {path}:")
    for i, line in enumerate(inspect.getsource(modeling_llama).splitlines(), 1):
        if "softmax" in line:
            print(f"  line {i}: {line.strip()}")

    if hasattr(modeling_llama, "eager_attention_forward"):
        print("\n--- eager_attention_forward ---")
        print(inspect.getsource(modeling_llama.eager_attention_forward))


if __name__ == "__main__":
    main()
