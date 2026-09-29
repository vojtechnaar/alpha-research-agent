"""Research agent loop: the LLM proposes strategy specs, the backtester scores them, results feed back.

Usage (on the GPU server, from the repository root):
    python -m src.agents.loop                          # settings from configs/agent.yaml
    python -m src.agents.loop --iterations 5 --backend cuda
    python -m src.agents.loop --adapter checkpoints/lora/v1

Every attempt (valid or not) is appended to results/<run>/attempts.jsonl with its train and
validation metrics. The LLM only ever sees train metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.agents.config import DEFAULT_CONFIG, build_evaluator, load_config
from src.agents.prompts import build_messages, parse_response, score
from src.backtest.evaluate import PROJECT_ROOT, Evaluator
from src.strategy.dsl import SpecError, canonical_signal


class Generator(Protocol):
    """Anything that turns chat messages into `n` sampled replies (QwenGenerator, or a fake in tests)."""

    def generate(self, messages: list[dict[str, str]], n: int = 1) -> list[str]: ...


def run_agent(
    generator: Generator,
    evaluator: Evaluator,
    run_dir: Path,
    iterations: int,
    candidates_per_iteration: int = 4,
    min_trades: int = 20,
    history_best: int = 5,
    history_recent: int = 5,
) -> list[dict[str, Any]]:
    """Run the propose -> backtest -> feedback loop and return all attempts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "attempts.jsonl"
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for iteration in range(iterations):
        start = time.perf_counter()
        messages = build_messages(attempts, evaluator.cost_bps, history_best, history_recent)
        batch = []
        for text in generator.generate(messages, n=candidates_per_iteration):
            attempt: dict[str, Any] = {"id": len(attempts) + len(batch), "iteration": iteration, "response": text}
            try:
                spec = parse_response(text)
                key = canonical_signal(spec)
                if key in seen:
                    raise SpecError(f"duplicate of an earlier strategy ({spec['name']})")
                seen.add(key)
                attempt["spec"] = spec
            except SpecError as exc:
                attempt["error"] = str(exc)
            batch.append(attempt)

        valid = [a for a in batch if "spec" in a]
        if valid:
            specs = [a["spec"] for a in valid]
            for attempt, train, validation in zip(
                valid, evaluator.evaluate(specs, "train"), evaluator.evaluate(specs, "validation")
            ):
                attempt.update(train=train, validation=validation, score=score(train, min_trades))

        attempts.extend(batch)
        with log_path.open("a") as f:
            for attempt in batch:
                f.write(json.dumps(attempt) + "\n")

        best = max((a["score"] for a in attempts if a.get("score") is not None), default=math.nan)
        print(f"[{iteration + 1}/{iterations}] {len(valid)}/{len(batch)} valid, "
              f"best train Sharpe {best:.2f} ({time.perf_counter() - start:.0f}s)")
    return attempts


def summarize(attempts: list[dict[str, Any]], top: int = 10) -> list[dict[str, Any]]:
    """Top attempts by train score, with mean train and validation Sharpe."""
    scored = sorted((a for a in attempts if a.get("score") is not None), key=lambda a: a["score"], reverse=True)
    return [
        {
            "id": a["id"],
            "name": a["spec"]["name"],
            "train_sharpe": a["score"],
            "validation_sharpe": sum(m["sharpe"] for m in a["validation"].values()) / len(a["validation"]),
            "hypothesis": a["spec"]["hypothesis"],
        }
        for a in scored[:top]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM strategy research loop.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--iterations", type=int, help="override agent.iterations")
    parser.add_argument("--backend", choices=["python", "cuda"], help="override backtest.backend")
    parser.add_argument("--adapter", help="LoRA adapter directory (overrides model.adapter_path)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.iterations is not None:
        config["agent"]["iterations"] = args.iterations
    if args.backend:
        config["backtest"]["backend"] = args.backend
    if args.adapter:
        config["model"]["adapter_path"] = args.adapter

    from src.models.llm import QwenGenerator  # imported here so tests don't need torch

    evaluator = build_evaluator(config)
    generator = QwenGenerator(**config["model"])
    agent = config["agent"]
    run_dir = PROJECT_ROOT / agent["output_dir"] / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    attempts = run_agent(
        generator, evaluator, run_dir,
        iterations=agent["iterations"],
        candidates_per_iteration=agent["candidates_per_iteration"],
        min_trades=agent["min_trades"],
        history_best=agent["history_best"],
        history_recent=agent["history_recent"],
    )
    summary = summarize(attempts)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTop strategies (train vs validation Sharpe), full log in {run_dir}:")
    for row in summary:
        print(f"  #{row['id']:<4} {row['name']:<32} train {row['train_sharpe']:6.2f}   "
              f"validation {row['validation_sharpe']:6.2f}")


if __name__ == "__main__":
    main()
