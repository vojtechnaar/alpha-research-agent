"""Speed and correctness benchmark: Python vs C++ CPU vs CUDA on the same sweep.

    python -m src.backends.benchmark --data data/raw/bitstamp_BTC-USD_1h.parquet \
        --strategy configs/strategies/momentum_low_volatility.json \
        --space configs/sweeps/momentum_low_volatility_dense.json --backends python cpp cuda

Python is slow (~25 ms/candidate), so by default it only runs the first --python-limit candidates
and its full-sweep time is extrapolated. Every backend's results on those candidates are compared
with Python's (max absolute metric differences, and trade-count mismatches).

Used by: nothing else: a CLI run by hand; it produced the speed table in docs/strategy_engine.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backends import BACKENDS, get_evaluator
from src.backtest.metrics import METRIC_NAMES
from src.strategies.schema import StrategySpec
from src.strategies.sweep import generate_candidates


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark backtest backends on one sweep.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--strategy", type=Path, required=True)
    p.add_argument("--space", type=Path, required=True)
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2023-01-01")
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--python-limit", type=int, default=200, help="candidates run by the Python backend")
    p.add_argument("--device", type=int, help="GPU index for cuda (default: $BACKTEST_DEVICE or 0)")
    p.add_argument("--max-candidates", type=int, default=1_000_000)
    args = p.parse_args()

    data = pd.read_parquet(args.data).sort_values("timestamp").reset_index(drop=True)
    spec = StrategySpec.from_json(args.strategy.read_text())
    candidates = generate_candidates(spec, json.loads(args.space.read_text()), args.max_candidates)
    cost_bps = args.transaction_cost * 10_000
    subset = candidates[: args.python_limit]
    print(f"{len(candidates)} candidates, {args.start} -> {args.end}, cost {args.transaction_cost:g}\n")

    reference = None
    rows = []
    for name in args.backends:
        evaluator = get_evaluator(name, args.device)
        todo = subset if name == "python" else candidates
        if name != "python":
            evaluator(data, candidates[:1], cost_bps, start=args.start, end=args.end)  # warm-up (CUDA context)
        started = time.perf_counter()
        table = evaluator(data, todo, cost_bps, start=args.start, end=args.end)
        seconds = time.perf_counter() - started
        ms = 1000 * seconds / len(todo)
        row = {"backend": getattr(evaluator, "info", "python (pandas)"), "candidates": len(todo),
               "seconds": round(seconds, 3), "ms_per_candidate": round(ms, 4),
               "full_sweep_seconds": round(ms * len(candidates) / 1000, 2)}
        if name == "python":
            reference = table
        elif reference is not None:
            same = table.iloc[: len(reference)]
            diffs = [np.nanmax(np.abs(same[m].to_numpy(float) - reference[m].to_numpy(float)), initial=0.0)
                     for m in METRIC_NAMES]
            row["max_abs_diff_vs_python"] = float(max(diffs))
            row["trade_count_mismatches"] = int((same["n_trades"].to_numpy() != reference["n_trades"].to_numpy()).sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    base = table["ms_per_candidate"].iloc[0]
    table["speedup_vs_first"] = (base / table["ms_per_candidate"]).round(1)
    print(table.to_string(index=False))
    print("\nfull_sweep_seconds is extrapolated for backends that ran fewer candidates than the sweep.")


if __name__ == "__main__":
    main()
