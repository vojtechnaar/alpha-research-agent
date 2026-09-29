"""Run the 24h momentum baseline on the downloaded Parquet files and print metrics.

Usage (from the repository root):
    python scripts/run_baseline.py
    python scripts/run_baseline.py --files data/raw/bitstamp_BTC-USD_1h.parquet --cost-bps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import DEFAULT_COST_BPS, run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.strategies import momentum_signal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="+", type=Path, default=sorted(Path("data/raw").glob("*.parquet")))
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()
    if not args.files:
        sys.exit("No Parquet files found in data/raw/; run python -m src.data.download first.")

    rows = {}
    for path in args.files:
        data = pd.read_parquet(path)
        result = run_backtest(data, momentum_signal(data, args.lookback), cost_bps=args.cost_bps)
        rows[path.stem] = compute_metrics(result)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"Momentum {args.lookback}h, cost {args.cost_bps} bps per unit turnover\n")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
