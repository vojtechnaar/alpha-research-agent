"""Re-evaluate the best strategies of a run on another split, e.g. the final test holdout.

Strategies are ranked by validation Sharpe (never by test), then evaluated on --split.
    python -m src.agents.report results/20260929-120000 --split test --top 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.agents.config import build_evaluator, load_config


def load_attempts(paths: list[Path]) -> list[dict[str, Any]]:
    """Read attempts.jsonl from run directories (or direct .jsonl paths)."""
    attempts = []
    for path in paths:
        file = path / "attempts.jsonl" if path.is_dir() else path
        attempts.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    return attempts


def mean_sharpe(metrics: dict[str, dict[str, float]]) -> float:
    return sum(m["sharpe"] for m in metrics.values()) / len(metrics)


def rank_by_validation(attempts: list[dict[str, Any]], min_trades: int) -> list[dict[str, Any]]:
    """Scored attempts with enough validation trades, best mean validation Sharpe first."""
    eligible = [
        a for a in attempts
        if a.get("score") is not None
        and all(m["n_trades"] >= min_trades for m in a["validation"].values())
        and not math.isnan(mean_sharpe(a["validation"]))
    ]
    return sorted(eligible, key=lambda a: mean_sharpe(a["validation"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories under results/")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--backend", choices=["python", "cuda"])
    args = parser.parse_args()

    config = load_config(args.runs[0] / "config.yaml")
    ranked = rank_by_validation(load_attempts(args.runs), config["agent"]["min_trades"])[: args.top]
    if not ranked:
        raise SystemExit("No scored strategies with enough validation trades.")
    results = build_evaluator(config, args.backend).evaluate([a["spec"] for a in ranked], args.split)

    print(f"{'strategy':<32} {'train':>7} {'valid':>7} {args.split:>7}   per-asset {args.split} Sharpe")
    for attempt, metrics in zip(ranked, results):
        per_asset = ", ".join(f"{asset} {m['sharpe']:.2f}" for asset, m in metrics.items())
        print(f"{attempt['spec']['name']:<32} {attempt['score']:7.2f} {mean_sharpe(attempt['validation']):7.2f} "
              f"{mean_sharpe(metrics):7.2f}   {per_asset}")


if __name__ == "__main__":
    main()
