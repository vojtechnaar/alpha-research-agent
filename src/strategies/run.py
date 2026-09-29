"""Evaluate one StrategySpec, or sweep its parameters, on an OHLCV Parquet file.

    python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
        --data data/raw/bitstamp_BTC-USD_1h.parquet --start 2023-01-01 --end 2024-01-01

    python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
        --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.backtest.engine import DEFAULT_COST_BPS
from src.strategies.evaluator import evaluate_strategy
from src.strategies.schema import SpecError, StrategySpec
from src.strategies.sweep import DEFAULT_MAX_CANDIDATES, count_candidates, run_sweep, summarize_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a StrategySpec or a parameter sweep.")
    parser.add_argument("strategy", type=Path, help="StrategySpec JSON file")
    parser.add_argument("--data", type=Path, required=True, help="OHLCV Parquet file")
    parser.add_argument("--space", type=Path, help="parameter-space JSON file; runs a sweep")
    parser.add_argument("--start", help="UTC start, inclusive")
    parser.add_argument("--end", help="UTC end, exclusive")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--out", type=Path, help="CSV path for sweep results (default: results/sweeps/...)")
    args = parser.parse_args()

    try:
        spec = StrategySpec.from_json(args.strategy.read_text())
    except SpecError as exc:
        print(f"Invalid strategy: {exc}", file=sys.stderr)
        return 1
    data = pd.read_parquet(args.data).sort_values("timestamp").reset_index(drop=True)
    dataset = str(data["symbol"].iloc[0]) if "symbol" in data else args.data.stem

    if args.space is None:
        result = evaluate_strategy(data, spec, args.cost_bps, start=args.start, end=args.end)
        print(f"{spec.name} on {dataset}, {result.backtest['timestamp'].iloc[0]} -> "
              f"{result.backtest['timestamp'].iloc[-1]}, cost {args.cost_bps:g} bps")
        for key, value in result.metrics.items():
            print(f"  {key:<22} {value:,.4f}" if isinstance(value, float) else f"  {key:<22} {value}")
        return 0

    space = json.loads(args.space.read_text())
    start_time = time.perf_counter()
    try:
        results = run_sweep(data, spec, space, args.cost_bps, start=args.start, end=args.end,
                            max_candidates=args.max_candidates, dataset=dataset)
    except SpecError as exc:
        print(f"Invalid sweep: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start_time
    out = args.out or Path("results/sweeps") / f"{spec.name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Evaluated {count_candidates(space)} candidates in {elapsed:.1f}s "
          f"({elapsed / len(results) * 1000:.1f} ms each); full table: {out}\n")
    print(json.dumps(summarize_sweep(results), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
