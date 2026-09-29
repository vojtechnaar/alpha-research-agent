"""Inference smoke test: load Qwen3-8B in bfloat16 on a CUDA GPU and answer one prompt.

Usage (on the GPU server, from the repository root):
    python src/agents/test_llm.py
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = "Propose one testable quantitative hypothesis for predicting BTC hourly returns using OHLCV data."
MAX_NEW_TOKENS = 300
GIB = 1024**3


def check_cuda() -> None:
    """Print CUDA/GPU info and exit with a clear error if no GPU is usable."""
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        sys.exit(
            "ERROR: CUDA is not available. Run this on the GPU server and check `nvidia-smi` "
            "and that the installed torch build has CUDA support (torch.version.cuda="
            f"{torch.version.cuda})."
        )
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def print_gpu_memory(label: str) -> None:
    """Print allocated/reserved memory for each visible GPU."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / GIB
        reserved = torch.cuda.memory_reserved(i) / GIB
        print(f"[{label}] GPU {i}: allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB")


def main() -> None:
    check_cuda()

    print(f"Loading tokenizer and model {MODEL_ID} ...")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"Loaded in {time.perf_counter() - start:.1f}s")
    print_gpu_memory("after load")

    messages = [{"role": "user", "content": PROMPT}]
    # Thinking mode off: otherwise Qwen3 spends the 300-token budget on <think> reasoning.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        # Sampling settings recommended by Qwen for non-thinking mode.
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.8, top_k=20
        )
    elapsed = time.perf_counter() - start

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    print(f"\nPrompt: {PROMPT}\n\nResponse:\n{tokenizer.decode(new_tokens, skip_special_tokens=True).strip()}\n")
    print(f"Generated {len(new_tokens)} tokens in {elapsed:.1f}s ({len(new_tokens) / elapsed:.1f} tok/s)")
    print_gpu_memory("after generate")


if __name__ == "__main__":
    main()
