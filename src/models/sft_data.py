"""Build a LoRA fine-tuning dataset from agent runs.

Keeps strategies that did well on train AND validation (never test), deduplicated, and turns each
into a chat example: system prompt + base request -> the spec as JSON.

    python -m src.models.sft_data results/2026* --out results/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agents.prompts import BASE_REQUEST, system_prompt
from src.agents.report import load_attempts, mean_sharpe, rank_by_validation
from src.strategy.dsl import canonical_signal


def select_attempts(
    attempts: list[dict[str, Any]], min_train_sharpe: float, min_validation_sharpe: float, min_trades: int
) -> list[dict[str, Any]]:
    """Unique strategies passing both Sharpe thresholds, best validation first."""
    selected, seen = [], set()
    for attempt in rank_by_validation(attempts, min_trades):
        key = canonical_signal(attempt["spec"])
        if key in seen or attempt["score"] < min_train_sharpe or mean_sharpe(attempt["validation"]) < min_validation_sharpe:
            continue
        seen.add(key)
        selected.append(attempt)
    return selected


def to_example(attempt: dict[str, Any], cost_bps: float) -> dict[str, Any]:
    """Chat-format training example whose target is the spec JSON."""
    spec = {key: attempt["spec"][key] for key in ("name", "hypothesis", "signal")}
    return {
        "messages": [
            {"role": "system", "content": system_prompt(cost_bps)},
            {"role": "user", "content": BASE_REQUEST},
            {"role": "assistant", "content": json.dumps(spec)},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LoRA SFT dataset from agent runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/sft.jsonl"))
    parser.add_argument("--min-train-sharpe", type=float, default=0.5)
    parser.add_argument("--min-validation-sharpe", type=float, default=0.3)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=10.0, help="must match the agent's cost")
    args = parser.parse_args()

    selected = select_attempts(load_attempts(args.runs), args.min_train_sharpe, args.min_validation_sharpe, args.min_trades)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for attempt in selected:
            f.write(json.dumps(to_example(attempt, args.cost_bps)) + "\n")
    print(f"Wrote {len(selected)} examples to {args.out}")


if __name__ == "__main__":
    main()
