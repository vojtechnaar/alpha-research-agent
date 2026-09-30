"""Build the LoRA training set from saved research runs.

Each example is one research step as a chat:
    system     the run's system prompt (rules, features, typical values)       run.json
    user       what Qwen saw: earlier results, exploration, feedback          record.llm.context
    assistant  the validated proposal it ended up with, as compact JSON       record fields

Only GOOD research steps become examples (a high train Sharpe alone is not enough):
  - completed, and "useful": held up on validation, traded often enough, not buy-and-hold
    in disguise (src.research.runs.experiment_flags)
  - no condition that is (almost) never or always true (unit mistakes)
  - the hypothesis text does not contradict the long/short rule (keyword heuristic)
  - not a repeat of an example already taken
Held-out markets (default ETH/USD and GLD) are never used, so base vs LoRA can be compared on them.

    python -m src.models.lora_data                     # -> data/lora/train.jsonl, data/lora/val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.research.records import ExperimentRecord, load_records
from src.research.runs import DEFAULT_RUNS_DIR, experiment_flags
from src.strategies.schema import StrategySpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "lora"
DEFAULT_HELD_OUT = ("ETH/USD", "GLD")

# Crude but explicit: words that commit the hypothesis to a direction.
UP_WORDS = ("rebound", "upward", "rise", "rally", "bullish", "higher", "upside", "recover")
DOWN_WORDS = ("downward", "decline", "bearish", "lower", "downside", "sell-off", "fall further", "drop further")


def direction_mismatch(hypothesis: str, spec: StrategySpec) -> bool:
    """True if the text only talks about one direction and the rule trades the other way."""
    text = hypothesis.lower()
    says_up = any(word in text for word in UP_WORDS)
    says_down = any(word in text for word in DOWN_WORDS)
    if says_up == says_down:  # neither or both: can't tell
        return False
    return (says_up and spec.true_position == -1) or (says_down and spec.true_position == 1)


def target_json(record: ExperimentRecord) -> str:
    """The proposal the model should have written: validated, auto-corrected, compact, one line."""
    proposal = {
        "hypothesis": record.hypothesis,
        "rationale": record.rationale,
        "strategy": record.strategy_spec,
        "parameter_space": record.requested_parameter_space or record.parameter_space,
    }
    return json.dumps(proposal, separators=(",", ":"))


def rejection_reason(record: ExperimentRecord) -> str | None:
    """Why a record is not a good training example (None = keep)."""
    if record.status != "completed" or not record.strategy_spec:
        return "not completed"
    if not (record.llm or {}).get("context"):
        return "no LLM context"
    flags = experiment_flags(record)
    if not flags["useful"]:
        return "not useful"
    if flags["unit_problem"]:
        return "unit problem"
    if direction_mismatch(record.hypothesis, StrategySpec.from_dict(record.strategy_spec)):
        return "hypothesis contradicts rule"
    return None


def build_examples(runs_dir: Path, held_out: tuple[str, ...] = DEFAULT_HELD_OUT) -> tuple[list[dict], Counter]:
    """(examples, counts of kept/dropped reasons) from every LLM run under `runs_dir`."""
    examples: list[dict] = []
    reasons: Counter = Counter()
    seen: set[str] = set()
    for run_dir in sorted(d for d in Path(runs_dir).iterdir() if (d / "experiments.jsonl").exists()):
        info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
        system = info.get("system_prompt")
        if not system or "replay" in info.get("generator", {}):
            reasons["no system prompt / replay run"] += 1
            continue
        for record in load_records(run_dir / "experiments.jsonl"):
            if record.dataset in held_out:
                reasons["held-out market"] += 1
                continue
            reason = rejection_reason(record)
            key = ""
            if reason is None:
                space = record.requested_parameter_space or record.parameter_space
                key = StrategySpec.from_dict(record.strategy_spec).identity() + json.dumps(space, sort_keys=True)
                if key in seen:
                    reason = "duplicate example"
            if reason:
                reasons[reason] += 1
                continue
            seen.add(key)
            reasons["kept"] += 1
            examples.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": record.llm["context"]},
                    {"role": "assistant", "content": target_json(record)},
                ],
                "meta": {"run": run_dir.name, "experiment_id": record.experiment_id, "dataset": record.dataset,
                         "validation_median_sharpe": record.validation_summary.get("median")},
            })
    return examples, reasons


def split(examples: list[dict], val_fraction: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Deterministic split BY RUN (a run's examples share context, so they stay on one side)."""
    def in_val(run: str) -> bool:
        return int(hashlib.sha256(run.encode()).hexdigest(), 16) % 1000 < val_fraction * 1000
    train = [e for e in examples if not in_val(e["meta"]["run"])]
    val = [e for e in examples if in_val(e["meta"]["run"])]
    return train, val


def tokenize_example(tokenizer: Any, messages: list[dict], max_length: int = 4096) -> dict[str, list[int]] | None:
    """input_ids and labels for supervised fine-tuning; the loss is only on the assistant reply.

    Uses the same chat template (thinking off) as generation, so training matches inference.
    Returns None if the example is longer than `max_length` tokens.
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = prompt + messages[-1]["content"] + "<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(input_ids) > max_length or input_ids[: len(prompt_ids)] != prompt_ids:
        return None
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the LoRA training set from saved research runs.")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--held-out", nargs="*", default=list(DEFAULT_HELD_OUT), help="markets never used for training")
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args(argv)

    examples, reasons = build_examples(args.runs_dir, tuple(args.held_out))
    train, val = split(examples, args.val_fraction)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    per_market = Counter(e["meta"]["dataset"] for e in examples)
    print(f"Examples: {len(examples)} kept ({len(train)} train, {len(val)} validation), per market {dict(per_market)}")
    print(f"Decisions: {dict(reasons)}")
    print(f"Held out (never used): {', '.join(args.held_out)}")
    print(f"Written to {args.out_dir}/train.jsonl and val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
