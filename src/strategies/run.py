"""Evaluate one StrategySpec, sweep its parameters, or run a train/validation experiment.

    # one strategy on one period (compared with the benchmarks on the same period)
    python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
        --data data/raw/bitstamp_BTC-USD_1h.parquet --start 2023-01-01 --end 2024-01-01

    # train-only sweep
    python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
        --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility_extended.json \
        --start 2017-01-01 --end 2023-01-01

    # train sweep -> top N frozen -> validation retest -> benchmarks -> experiment record
    python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
        --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility_extended.json \
        --train-start 2017-01-01 --train-end 2023-01-01 \
        --validation-start 2023-01-01 --validation-end 2025-01-01 --top 10

Costs: --transaction-cost is a fraction of notional per unit of position change (default 0.001 =
10 bps). --cost-bps is the same thing in basis points (kept for older commands).

Backend: --backend python (default) | cpp | cuda (see src/backends; build with make -C cuda).

Used by: nothing else: a CLI for manual runs without the LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.backends import BACKENDS, get_evaluator
from src.backtest.metrics import HOURS_PER_YEAR
from src.research.benchmarks import DEFAULT_BENCHMARKS_DIR, PROJECT_ROOT, evaluate_benchmarks, load_benchmarks
from src.research.experiment import SELECTION_METRICS, ExperimentSettings, Period, run_experiment
from src.research.records import add_results, append_record, new_record
from src.research.report import format_record
from src.strategies.schema import SpecError, StrategySpec
from src.strategies.sweep import (
    DEFAULT_MAX_CANDIDATES,
    CandidateEvaluator,
    count_candidates,
    generate_candidates,
    summarize_sweep,
)

DEFAULT_TRANSACTION_COST = 0.001
SHOWN_METRICS = ["cumulative_return", "annualized_return", "sharpe", "annualized_volatility",
                 "max_drawdown", "turnover", "n_trades", "exposure", "n_bars"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a StrategySpec, a sweep, or a train/validation experiment.")
    p.add_argument("strategy", type=Path, help="StrategySpec JSON file")
    p.add_argument("--data", type=Path, required=True, help="OHLCV Parquet file")
    p.add_argument("--space", type=Path, help="parameter-space JSON file")
    p.add_argument("--start", help="UTC start of the evaluation (or train) period, inclusive")
    p.add_argument("--end", help="UTC end of the evaluation (or train) period, exclusive")
    p.add_argument("--train-start", help="same as --start")
    p.add_argument("--train-end", help="same as --end")
    p.add_argument("--validation-start", help="with --space: retest the top candidates from here")
    p.add_argument("--validation-end")
    p.add_argument("--top", type=int, default=10, help="candidates retested on validation")
    p.add_argument("--selection-metric", default="sharpe", choices=SELECTION_METRICS)
    p.add_argument("--min-trades-per-year", type=float, default=10.0,
                   help="candidates trading less often are not selected; fewer validation trades are flagged")
    cost = p.add_mutually_exclusive_group()
    cost.add_argument("--transaction-cost", type=float, help="fraction per unit turnover (default 0.001)")
    cost.add_argument("--cost-bps", type=float, help="basis points per unit turnover")
    p.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p.add_argument("--periods-per-year", type=float, default=HOURS_PER_YEAR,
                   help="bars per year for annualising (8760 = hourly 24/7; e.g. 252 for daily stock bars)")
    p.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    p.add_argument("--backend", choices=BACKENDS, default="python")
    p.add_argument("--backtest-device", type=int, help="GPU index for --backend cuda")
    p.add_argument("--out", type=Path, help="CSV path for the (train) sweep table")
    args = p.parse_args(argv)
    for a, b in (("start", "train_start"), ("end", "train_end")):
        if getattr(args, a) and getattr(args, b):
            p.error(f"use either --{a} or --{b.replace('_', '-')}, not both")
        setattr(args, a, getattr(args, a) or getattr(args, b))
    args.transaction_cost = (args.cost_bps / 10_000 if args.cost_bps is not None
                             else args.transaction_cost if args.transaction_cost is not None
                             else DEFAULT_TRANSACTION_COST)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = StrategySpec.from_json(args.strategy.read_text())
        space = json.loads(args.space.read_text()) if args.space else None
    except SpecError as exc:
        print(f"Invalid strategy: {exc}", file=sys.stderr)
        return 1
    data = pd.read_parquet(args.data).sort_values("timestamp").reset_index(drop=True)
    dataset = str(data["symbol"].iloc[0]) if "symbol" in data else args.data.stem
    benchmarks = load_benchmarks(args.benchmarks_dir)
    evaluator = get_evaluator(args.backend, args.backtest_device)
    print(f"Backend: {getattr(evaluator, 'info', 'python (pandas reference)')}")
    cost_bps = args.transaction_cost * 10_000
    print(f"{dataset}: transaction cost {args.transaction_cost:g} per unit turnover ({cost_bps:g} bps)"
          + ("  [WARNING: zero costs]" if args.transaction_cost == 0 else ""))

    try:
        if space is not None and args.validation_start:
            return _experiment(args, spec, space, data, dataset, benchmarks, evaluator)
        if space is not None:
            return _sweep(args, spec, space, data, dataset, cost_bps, evaluator)
    except (SpecError, ValueError) as exc:
        print(f"Invalid experiment: {exc}", file=sys.stderr)
        return 1

    period = Period(args.start, args.end)
    table = evaluator(data, [({}, spec)], cost_bps, args.periods_per_year, start=period.start, end=period.end,
                      dataset=dataset)
    table.insert(1, "strategy", spec.name)
    bench = evaluate_benchmarks(data, benchmarks, period, evaluator=evaluator, cost_bps=cost_bps,
                                periods_per_year=args.periods_per_year, dataset=dataset)
    rows = pd.concat([table, bench.rename(columns={"benchmark": "strategy"})], ignore_index=True)
    print(f"{spec.describe()}\nPeriod {rows['start'].iloc[0]} -> {rows['end'].iloc[0]}\n")
    print(rows.set_index("strategy")[SHOWN_METRICS].T.to_string(float_format="{:,.4f}".format))
    return 0


def _sweep(args: argparse.Namespace, spec: StrategySpec, space: dict, data: pd.DataFrame, dataset: str,
           cost_bps: float, evaluator: CandidateEvaluator) -> int:
    candidates = generate_candidates(spec, space, args.max_candidates)
    started = time.perf_counter()
    results = evaluator(data, candidates, cost_bps, args.periods_per_year, start=args.start, end=args.end,
                        dataset=dataset)
    elapsed = time.perf_counter() - started
    out = args.out or PROJECT_ROOT / "results" / "sweeps" / f"{spec.name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Evaluated {count_candidates(space)} candidates in {elapsed:.1f}s "
          f"({elapsed / len(results) * 1000:.1f} ms each); full table: {out}\n")
    print(json.dumps(summarize_sweep(results), indent=2, default=str))
    return 0


def _experiment(args: argparse.Namespace, spec: StrategySpec, space: dict, data: pd.DataFrame, dataset: str,
                benchmarks: dict, evaluator: CandidateEvaluator) -> int:
    settings = ExperimentSettings(
        train=Period(args.start, args.end),
        validation=Period(args.validation_start, args.validation_end),
        transaction_cost=args.transaction_cost, top_n=args.top, selection_metric=args.selection_metric,
        max_candidates=args.max_candidates, min_trades_per_year=args.min_trades_per_year,
        periods_per_year=args.periods_per_year,
    )
    result = run_experiment(data, spec, space, settings, benchmarks, dataset, evaluator)
    record = new_record(settings, dataset=dataset, data_path=str(args.data), hypothesis=spec.description,
                        strategy_spec=spec.to_dict(), strategy_description=spec.describe(),
                        notes=f"manual experiment: {args.strategy} x {args.space}")
    add_results(record, result, space)
    out_dir = PROJECT_ROOT / "results" / "experiments" / "manual"
    csv_path = args.out or out_dir / "sweeps" / f"{record.experiment_id}_train.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.train.to_csv(csv_path, index=False)
    record.sweep_csv = str(csv_path)
    append_record(out_dir / "experiments.jsonl", record)
    print(format_record(record))
    print(f"\nRecord appended to {out_dir / 'experiments.jsonl'}; train table: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
