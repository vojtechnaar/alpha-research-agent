"""Qwen3 chat generation on a CUDA GPU, optionally with a LoRA adapter.

Smoke test (on the GPU server):
    python -m src.models.llm "Propose one testable hypothesis for BTC hourly returns."
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 1024**3


def require_cuda() -> None:
    """Print the visible GPUs; exit with a clear error if CUDA is unavailable."""
    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA is not available. Run this on the GPU server and check `nvidia-smi` and that "
            f"torch was installed with CUDA support (torch.version.cuda={torch.version.cuda})."
        )
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def gpu_memory() -> str:
    """Allocated/reserved memory per GPU, for logging."""
    return ", ".join(
        f"GPU {i} {torch.cuda.memory_allocated(i) / GIB:.1f}/{torch.cuda.memory_reserved(i) / GIB:.1f} GiB"
        for i in range(torch.cuda.device_count())
    )


class QwenGenerator:
    """Loads the model once and samples chat completions."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        adapter_path: str | Path | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        require_cuda()
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        self.enable_thinking = enable_thinking
        self.generation_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
        )
        print(f"Loaded {model_id}{f' + {adapter_path}' if adapter_path else ''} in "
              f"{time.perf_counter() - start:.0f}s ({gpu_memory()})")

    def generate(self, messages: list[dict[str, str]], n: int = 1) -> list[str]:
        """Sample `n` independent assistant replies to the chat `messages`."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, num_return_sequences=n, **self.generation_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip() for seq in output]


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Propose one testable quantitative hypothesis for predicting BTC hourly returns using OHLCV data."
    generator = QwenGenerator(max_new_tokens=300)
    start = time.perf_counter()
    reply = generator.generate([{"role": "user", "content": prompt}])[0]
    print(f"\n{reply}\n\n({time.perf_counter() - start:.1f}s, {gpu_memory()})")
