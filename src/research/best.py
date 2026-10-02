"""Rank finished experiments to choose which ideas go to the final test, by a fixed rule.

The rule uses only what the research loop already saw (train and validation), never the final test:
  - completed and "useful" (src.research.runs.experiment_flags: held up on validation, traded
    often enough, not buy-and-hold in disguise), and no unit problem
  - ranked by validation median Sharpe of the frozen top N
  - one experiment per idea family and market (the best one), so a family can't fill every slot

    python -m src.research.best                                  # top 3 per market, all runs
    python -m src.research.best --datasets BTC/USD SPY --top 1   # with the matching confirm commands

Picking the maximum out of hundreds of experiments overstates it (winner's curse): expect the
final test to be lower than the validation number. That decay is the point of the test.

Used by: nothing else: a CLI run by hand before src.research.confirm --final-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.records import load_records
from src.research.runs import DEFAULT_RUNS_DIR, experiment_flags
from src.strategies.schema import StrategySpec


def ranked_experiments(runs_dir: Path) -> pd.DataFrame:
    """One row per useful experiment (no unit problems) in every run under `runs_dir`."""
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(d for d in Path(runs_dir).iterdir() if (d / "experiments.jsonl").exists()):
        info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
        generator = info.get("generator", {})
        model = generator.get("model_id") or "replay"
        if generator.get("adapter"):
            model += f"+{Path(str(generator['adapter'])).name}"
        for record in load_records(run_dir / "experiments.jsonl"):
            flags = experiment_flags(record)
            if not flags["useful"] or flags["unit_problem"]:
                continue
            spec = StrategySpec.from_dict(record.strategy_spec)
            buy_and_hold = record.benchmarks.get("validation", {}).get("buy_and_hold", {}).get("sharpe")
            rows.append({
                "dataset": record.dataset,
                "validation_median": record.validation_summary.get("median"),
                "buy_and_hold": buy_and_hold,
                "train_median": record.train_summary.get("distribution", {}).get("median"),
                "idea": spec.describe_family(),
                "family": spec.family(),
                "model": model,
                "records": str(run_dir / "experiments.jsonl"),
                "experiment": (record.iteration or 0) + 1,
                "data_path": record.data_path,
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("validation_median", ascending=False)
    return table.drop_duplicates(subset=["dataset", "family"], keep="first").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rank useful experiments to choose final-test candidates.")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--datasets", nargs="+", help="only these markets (default: all)")
    p.add_argument("--top", type=int, default=3, help="best experiments per market")
    args = p.parse_args(argv)

    table = ranked_experiments(args.runs_dir)
    if args.datasets and not table.empty:
        table = table[table["dataset"].isin(args.datasets)]
    if table.empty:
        print("No useful experiments found.")
        return 0
    best = table.groupby("dataset", sort=True).head(args.top)
    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        print(best[["dataset", "validation_median", "buy_and_hold", "train_median", "idea", "model"]]
              .round(2).to_string(index=False))
    print("\nFinal-test commands for the best experiment per market (run each ONCE):")
    for _, row in best.groupby("dataset", sort=True).head(1).iterrows():
        print(f"python -m src.research.confirm {row['records']} --experiments {row['experiment']} --final-test "
              f"--backend python")  # 10 frozen candidates: no GPU needed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
