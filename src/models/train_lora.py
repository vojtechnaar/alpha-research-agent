"""Fine-tune Qwen3-8B with LoRA on the research examples from src/models/lora_data.py.

    python -m src.models.train_lora --device cuda:0                   # data/lora/{train,val}.jsonl
    python -m src.models.train_lora --device cuda:0 --epochs 3 --name v1

The base model stays frozen (bf16); only small low-rank matrices added to the attention and MLP
projections are trained (~0.5% of the parameters). The loss is on the assistant reply (the
proposal JSON) only. The adapter with the lowest validation loss is saved to
checkpoints/lora/<name>/ and used with:  python -m src.agents.research ... --adapter checkpoints/lora/<name>
Fits on one A6000 (48 GB) with gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.models.lora_data import DEFAULT_OUT_DIR, tokenize_example

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reply_loss(model: Any, input_ids: list[int], labels: list[int]) -> Any:
    """Cross-entropy on the assistant reply only, computing logits ONLY for the reply.

    The prompt (rules + feedback) is thousands of tokens but carries no loss; asking the model for
    logits at just the last `reply + 1` positions (`logits_to_keep`) avoids a [tokens x 152k-vocab]
    logits tensor for the whole sequence, which is most of the activation memory. Equal to the
    standard shifted causal-LM loss with the prompt masked out.
    """
    import torch

    n_prompt = next(i for i, label in enumerate(labels) if label != -100)
    keep = len(input_ids) - n_prompt + 1  # from the last prompt token: it predicts the first reply token
    ids = torch.tensor([input_ids], device=model.device)
    logits = model(input_ids=ids, logits_to_keep=keep).logits  # [1, keep, vocab]
    targets = torch.tensor([labels[n_prompt:]], device=model.device)
    return torch.nn.functional.cross_entropy(logits[0, :-1].float(), targets[0], ignore_index=-100)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LoRA fine-tuning of Qwen3 on research examples.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--name", default="v1", help="adapter name -> checkpoints/lora/<name>")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=8, help="examples per optimizer step")
    p.add_argument("--max-length", type=int, default=8192, help="longer examples are skipped (prompts are ~3-6k tokens)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live even when redirected to a log file

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models.llm import gpu_memory, require_cuda

    require_cuda()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = PROJECT_ROOT / "checkpoints" / "lora" / args.name

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    splits = {}
    for split in ("train", "val"):
        rows = load_jsonl(args.data_dir / f"{split}.jsonl")
        tokenized = [tokenize_example(tokenizer, row["messages"], args.max_length) for row in rows]
        splits[split] = [t for t in tokenized if t is not None]
        lengths = [len(t["input_ids"]) for t in splits[split]] or [0]
        print(f"{split}: {len(splits[split])} examples ({len(rows) - len(splits[split])} dropped: longer than "
              f"{args.max_length} tokens); tokens per example median {statistics.median(lengths):.0f}, max {max(lengths)}")
    if not splits["train"]:
        raise SystemExit("No training examples; run python -m src.models.lora_data first.")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map={"": args.device})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(splits["train"]) / args.grad_accum) * args.epochs)
    warmup = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / total_steps))

    def loss_of(example: dict) -> torch.Tensor:
        return reply_loss(model, example["input_ids"], example["labels"])

    def validation_loss() -> float:
        if not splits["val"]:
            return float("nan")
        model.eval()
        with torch.no_grad():
            losses = [loss_of(e).item() for e in splits["val"]]
        model.train()
        return sum(losses) / len(losses)

    history = [{"epoch": 0, "val_loss": validation_loss()}]
    print(f"epoch 0: validation loss {history[0]['val_loss']:.4f} (base model, before training)")
    best, started, step = history[0]["val_loss"], time.perf_counter(), 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(splits["train"])))
        random.shuffle(order)
        running = []
        for i, index in enumerate(order, 1):
            loss = loss_of(splits["train"][index]) / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            if i % args.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                elapsed = time.perf_counter() - started
                print(f"  epoch {epoch}/{args.epochs}, step {step}/{total_steps}: {i}/{len(order)} examples, "
                      f"train loss {statistics.mean(running):.4f}, {elapsed / 60:.1f} min, "
                      f"~{elapsed / step * (total_steps - step) / 60:.0f} min left")
        val = validation_loss()
        history.append({"epoch": epoch, "train_loss": sum(running) / len(running), "val_loss": val})
        print(f"epoch {epoch}/{args.epochs}: train loss {history[-1]['train_loss']:.4f}, validation loss {val:.4f} "
              f"({time.perf_counter() - started:.0f}s, {gpu_memory(args.device)})")
        if not splits["val"] or val < best:  # keep the adapter that generalises best
            best = val
            model.save_pretrained(out_dir)
            print(f"  saved adapter to {out_dir}")

    (out_dir / "training_info.json").write_text(json.dumps({
        "model_id": args.model_id, "data_dir": str(args.data_dir), "examples": {k: len(v) for k, v in splits.items()},
        "settings": {k: v for k, v in vars(args).items() if k not in ("data_dir",)}, "history": history,
    }, indent=2, default=str))
    print(f"Done. Best validation loss {best:.4f}. Use it with: --adapter {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
