"""Runs Llama 3.2 1B with and without our softmax kernel: checks the outputs match,
then measures end-to-end speed and softmax's share of GPU time.

Run from the repo root:  python bench/bench_llm.py
"""
import json
import os
import sys

import torch
import transformers
from packaging import version
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from load_ext import load_ext  # noqa: E402
from softmax_patch import custom_softmax  # noqa: E402

MODEL_ID = "unsloth/Llama-3.2-1B"  # ungated mirror of meta-llama/Llama-3.2-1B
PROMPT = "The capital of France is"
SEQ_LEN = 2048
WARMUP = 3
RUNS = 20


def load_model():
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager", **{dtype_kw: torch.float32}
    ).to("cuda").eval()
    return tok, model


def time_one_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def softmax_share(fn):
    """Fraction of GPU kernel time spent in kernels with 'softmax' in their name."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = prof.key_averages()
    self_time = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    total = sum(self_time(e) for e in events)
    softmax = sum(self_time(e) for e in events if "softmax" in e.key.lower())
    top = sorted(events, key=self_time, reverse=True)[:6]
    return softmax / total * 100, [(e.key[:70], self_time(e) / total * 100) for e in top]


@torch.no_grad()
def main():
    ext = load_ext()
    tok, model = load_model()
    n_layers = model.config.num_hidden_layers
    results = {"model": MODEL_ID, "gpu": torch.cuda.get_device_name()}

    # --- Task 1: does the swap reach every attention layer? -----------------------
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    with custom_softmax(ext) as stats:
        model(**inputs)
    print(f"[swap] kernel calls in one forward pass: {stats['kernel_calls']} "
          f"(expected {n_layers}), fallbacks: {stats['fallback_calls']}")
    results["kernel_calls_per_forward"] = stats["kernel_calls"]

    # --- Task 2: same logits, same generated text ---------------------------------
    ids = torch.randint(0, model.config.vocab_size, (1, 512), device="cuda")
    ref = model(input_ids=ids).logits
    ref_again = model(input_ids=ids).logits  # PyTorch vs itself: the run-to-run noise floor
    with custom_softmax(ext):
        ours = model(input_ids=ids).logits
    max_diff = (ours - ref).abs().max().item()
    match = torch.allclose(ours, ref, atol=1e-4)
    print(f"[logits] 512 tokens: max |diff| = {max_diff:.2e} "
          f"(PyTorch vs itself: {(ref_again - ref).abs().max().item():.2e}), "
          f"allclose(atol=1e-4): {match}")
    results.update(logits_max_abs_diff=max_diff, logits_allclose=bool(match))
    del ref, ref_again, ours

    gen = dict(max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
    text_ref = tok.decode(model.generate(**inputs, **gen)[0], skip_special_tokens=True)
    with custom_softmax(ext) as stats:
        text_ours = tok.decode(model.generate(**inputs, **gen)[0], skip_special_tokens=True)
    print(f"[generate] identical text: {text_ref == text_ours}, kernel calls: {stats['kernel_calls']}")
    print(f"  PyTorch: {text_ref!r}")
    print(f"  ours:    {text_ours!r}")
    results.update(generated_identical=text_ref == text_ours, generate_kernel_calls=stats["kernel_calls"])

    # --- Task 3: end-to-end speed + softmax's share of the time -------------------
    for batch in (4, 2, 1):  # back off if the T4's 16 GB runs out
        try:
            ids = torch.randint(0, model.config.vocab_size, (batch, SEQ_LEN), device="cuda")
            # logits_to_keep=1: only compute logits for the last token. Full logits at
            # this size would be batch*2048*128256 floats (4 GB at batch 4).
            fwd = lambda: model(input_ids=ids, use_cache=False, logits_to_keep=1)

            def fwd_ours():
                with custom_softmax(ext):
                    fwd()

            for _ in range(WARMUP):
                fwd()
                fwd_ours()
            base, ours = [], []
            for _ in range(RUNS):  # interleaved, so clock/temperature drift hits both equally
                base.append(time_one_ms(fwd))
                ours.append(time_one_ms(fwd_ours))
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[speed] batch {batch} ran out of memory, trying smaller")
            torch.cuda.empty_cache()

    base_ms = sorted(base)[RUNS // 2]
    ours_ms = sorted(ours)[RUNS // 2]
    speedup = (base_ms / ours_ms - 1) * 100
    print(f"[speed] batch {batch}, seq {SEQ_LEN}: PyTorch {base_ms:.1f} ms, ours {ours_ms:.1f} ms "
          f"-> speedup {speedup:+.2f}%")

    share_torch, top_torch = softmax_share(fwd)
    share_ours, top_ours = softmax_share(fwd_ours)
    print(f"[profile] softmax share of GPU time: PyTorch run {share_torch:.1f}%, our run {share_ours:.1f}%")
    print("  top kernels (PyTorch run):")
    for name, pct in top_torch:
        print(f"    {pct:5.1f}%  {name}")

    results.update(batch=batch, seq_len=SEQ_LEN, torch_ms=round(base_ms, 2), ours_ms=round(ours_ms, 2),
                   speedup_pct=round(speedup, 2), softmax_share_torch_pct=round(share_torch, 2),
                   softmax_share_ours_pct=round(share_ours, 2))
    out_path = os.path.join(ROOT, "results", "llm_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
