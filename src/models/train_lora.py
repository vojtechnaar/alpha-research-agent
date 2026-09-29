"""Fine-tune Qwen3-8B with LoRA on the SFT dataset from src/models/sft_data.py.

    python -m src.models.train_lora --data results/sft.jsonl --output checkpoints/lora/v1

Loss is computed on the assistant reply (the spec JSON) only. The adapter (~100 MB) is saved to
--output and loaded by the agent with `--adapter`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.llm import gpu_memory, require_cuda

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_example(tokenizer: Any, example: dict[str, Any], max_length: int) -> dict[str, list[int]] | None:
    """input_ids and labels with the prompt masked (-100); None if the example is too long."""
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = prompt + messages[-1]["content"] + "<|im_end|>\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(input_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of Qwen3 on strategy specs.")
    parser.add_argument("--data", type=Path, default=Path("results/sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/lora/v1"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    require_cuda()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    raw = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    examples = [e for e in (tokenize_example(tokenizer, r, args.max_length) for r in raw) if e]
    if not examples:
        raise SystemExit(f"No usable examples in {args.data}")
    print(f"{len(examples)} training examples ({len(raw) - len(examples)} dropped as too long)")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=TARGET_MODULES, task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = max(1, args.epochs * len(examples) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / (total_steps + 1))

    model.train()
    step, start = 0, time.perf_counter()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        running = 0.0
        for i, example in enumerate(examples, 1):
            input_ids = torch.tensor([example["input_ids"]], device=model.device)
            labels = torch.tensor([example["labels"]], device=model.device)
            loss = model(input_ids=input_ids, labels=labels).loss / args.grad_accum
            loss.backward()
            running += loss.item()
            if i % args.grad_accum == 0 or i == len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running:.4f} "
                      f"({time.perf_counter() - start:.0f}s, {gpu_memory()})")
                running = 0.0

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
