"""Re-test finished experiments without the LLM: on another asset, or once on the final test.

Qwen sees validation results, so after many runs the validation period is no longer blind. These
checks use data the research loop never showed it:

    # cross-asset: does the idea (and do the exact parameters) also work on ETH?
    python -m src.research.confirm data/research_runs/<run>/experiments.jsonl --experiments 3 6 \
        --data data/raw/bitstamp_ETH-USD_1h.parquet --backend cuda

    # final test: the frozen top N chosen on train, evaluated ONCE on the untouched period (2025+)
    python -m src.research.confirm data/research_runs/<run>/experiments.jsonl --experiments 3 6 \
        --final-test --backend cuda

Cross-asset reports two things:
  replicate  the same idea and parameter ranges, re-selected on the new asset's train period and
             retested on its validation period (does the idea generalise?)
  transfer   the exact top-N parameters chosen on the original asset, applied unchanged to the new
             asset (stricter: do the parameters generalise?)

Every final-test use is appended to results/final_test_log.jsonl and the command warns if the
test has been used before: each look at the test turns it a little more into validation data.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backends import BACKENDS, get_evaluator
from src.backtest.metrics import HOURS_PER_YEAR
from src.research.benchmarks import DEFAULT_BENCHMARKS_DIR, PROJECT_ROOT, evaluate_benchmarks, load_benchmarks
from src.research.experiment import ExperimentSettings, Period, run_experiment
from src.research.records import ExperimentRecord, load_records, to_json_safe
from src.research.summary import summarize_train, summarize_validation
from src.strategies.schema import StrategySpec
from src.strategies.sweep import Candidate, CandidateEvaluator

FINAL_TEST_LOG = PROJECT_ROOT / "results" / "final_test_log.jsonl"
DEFAULT_MIN_TRADES_PER_YEAR = 10.0


def select_records(records: list[ExperimentRecord], numbers: list[int] | None) -> list[ExperimentRecord]:
    """Completed experiments by their number in the run log ([3/10] -> 3); all completed ones if None."""
    completed = [r for r in records if r.status == "completed"]
    if not numbers:
        return completed
    by_number = {r.iteration + 1: r for r in completed if r.iteration is not None}
    missing = [n for n in numbers if n not in by_number]
    if missing:
        raise ValueError(f"no completed experiment with number(s) {missing}; completed: {sorted(by_number)}")
    return [by_number[n] for n in numbers]


def settings_from(record: ExperimentRecord) -> ExperimentSettings:
    """The exact settings an experiment ran with."""
    return ExperimentSettings(
        train=Period(**record.train_period),
        validation=Period(**record.validation_period),
        transaction_cost=record.transaction_cost if record.transaction_cost is not None else 0.001,
        top_n=record.top_n or 10,
        selection_metric=record.selection_metric,
        max_candidates=record.max_candidates or 1000,
        min_trades_per_year=record.min_trades_per_year if record.min_trades_per_year is not None
        else DEFAULT_MIN_TRADES_PER_YEAR,
        periods_per_year=record.periods_per_year or HOURS_PER_YEAR,
    )


def frozen_candidates(record: ExperimentRecord) -> list[tuple[int, Candidate]]:
    """(original candidate id, (params, spec)) of the top N selected on the original train period."""
    base = StrategySpec.from_dict(record.strategy_spec)
    out = []
    for row in record.comparison:
        params = {k: v for k, v in row.items() if "." in k}
        params = {k: int(v) if k.endswith(".lookback") else v for k, v in params.items()}
        out.append((int(row["candidate"]), (params, base.with_parameters(params))))
    return out


def sharpe_stats(table: pd.DataFrame) -> dict[str, Any]:
    """Median / best / worst Sharpe, share positive and median trades of a results table."""
    values = table["sharpe"].dropna()
    return {
        "median": _num(values.median()) if len(values) else None,
        "best": _num(values.max()) if len(values) else None,
        "worst": _num(values.min()) if len(values) else None,
        "share_positive": _num((values > 0).mean()) if len(values) else None,
        "median_trades": _num(table["n_trades"].median()) if len(table) else None,
        "n": int(len(table)),
    }


def evaluate_frozen(
    record: ExperimentRecord,
    data: pd.DataFrame,
    period: Period,
    settings: ExperimentSettings,
    evaluator: CandidateEvaluator,
    benchmarks: dict[str, StrategySpec],
    dataset: str,
) -> dict[str, Any]:
    """The frozen top-N (no re-selection) and the benchmarks on `period` of `data`."""
    frozen = frozen_candidates(record)
    table = evaluator(data, [c for _, c in frozen], cost_bps=settings.cost_bps,
                      periods_per_year=settings.periods_per_year, start=period.start, end=period.end,
                      dataset=dataset, ids=[i for i, _ in frozen])
    bench = evaluate_benchmarks(data, benchmarks, period, evaluator=evaluator, cost_bps=settings.cost_bps,
                                periods_per_year=settings.periods_per_year, dataset=dataset)
    first = table["start"].iloc[0] if len(table) else None
    last = table["end"].iloc[-1] if len(table) else None
    return {
        "period": {"start": period.start, "end": period.end, "first_bar": first, "last_bar": last},
        "candidates": sharpe_stats(table),
        "rows": table[["candidate", "sharpe", "annualized_return", "max_drawdown", "n_trades"]].to_dict("records"),
        "benchmarks": {row["benchmark"]: {"sharpe": _num(row["sharpe"]), "max_drawdown": _num(row["max_drawdown"])}
                       for _, row in bench.iterrows()} if not bench.empty else {},
    }


def cross_asset(
    record: ExperimentRecord,
    data: pd.DataFrame,
    dataset: str,
    evaluator: CandidateEvaluator,
    benchmarks: dict[str, StrategySpec],
) -> dict[str, Any]:
    """Replicate (re-select on the new asset) and transfer (frozen parameters) on another dataset."""
    settings = settings_from(record)
    base = StrategySpec.from_dict(record.strategy_spec)
    result = run_experiment(data, base, record.parameter_space, settings, benchmarks, dataset, evaluator)
    train_summary, _ = summarize_train(result)
    return {
        "replicate": {
            "train": train_summary["distribution"],
            "validation": summarize_validation(result),
            "buy_and_hold_validation_sharpe": _bench(result.benchmarks.get("validation"), "buy_and_hold"),
        },
        "transfer": {
            "train": evaluate_frozen(record, data, settings.train, settings, evaluator, benchmarks, dataset),
            "validation": evaluate_frozen(record, data, settings.validation, settings, evaluator, benchmarks, dataset),
        },
    }


def final_test(
    record: ExperimentRecord,
    data: pd.DataFrame,
    period: Period,
    dataset: str,
    evaluator: CandidateEvaluator,
    benchmarks: dict[str, StrategySpec],
) -> dict[str, Any]:
    """The frozen top-N on the untouched final-test period."""
    return evaluate_frozen(record, data, period, settings_from(record), evaluator, benchmarks, dataset)


def log_final_test_use(entry: dict[str, Any], log_path: Path | None = None) -> int:
    """Append a use of the final test; return how many uses were logged BEFORE this one."""
    log_path = log_path or FINAL_TEST_LOG  # looked up at call time (tests redirect it)
    previous = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(to_json_safe(entry)) + "\n")
    return previous


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-test finished experiments on another asset or the final test.")
    p.add_argument("records", type=Path, help="experiments.jsonl of a research run")
    p.add_argument("--experiments", type=int, nargs="+", help="experiment numbers as in the log ([3/10] -> 3)")
    p.add_argument("--data", type=Path, help="dataset to test on (default: the one the experiment used)")
    p.add_argument("--final-test", action="store_true", help="evaluate the frozen top N on the final-test period")
    p.add_argument("--test-start", help="default: the experiment's validation end")
    p.add_argument("--test-end", help="default: end of the data")
    p.add_argument("--backend", choices=BACKENDS, default="python")
    p.add_argument("--backtest-device", type=int)
    p.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "confirmations")
    args = p.parse_args(argv)

    try:
        records = select_records(load_records(args.records), args.experiments)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    evaluator = get_evaluator(args.backend, args.backtest_device)
    benchmarks = load_benchmarks(args.benchmarks_dir)
    data_path = args.data or Path(records[0].data_path)
    data = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    dataset = str(data["symbol"].iloc[0]) if "symbol" in data else data_path.stem
    mode = "final_test" if args.final_test else "cross_asset"
    if mode == "cross_asset" and args.data is None:
        print("Cross-asset confirmation needs --data (another dataset); use --final-test for the holdout.",
              file=sys.stderr)
        return 2

    if mode == "final_test":
        test = Period(args.test_start or records[0].validation_period["end"], args.test_end)
        previous = log_final_test_use({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": str(args.records), "experiments": [r.iteration + 1 for r in records],
            "dataset": dataset, "period": {"start": test.start, "end": test.end},
        })
        if previous:
            print(f"WARNING: the final test has already been used {previous} time(s) (results/final_test_log.jsonl). "
                  "Each look makes it less independent; decide what counts as success BEFORE looking.\n")

    results = []
    print(f"{mode.replace('_', ' ')} on {dataset} ({data_path}), backend {getattr(evaluator, 'info', 'python')}\n")
    for record in records:
        number = record.iteration + 1 if record.iteration is not None else "?"
        idea = StrategySpec.from_dict(record.strategy_spec).describe_family()
        original = record.validation_summary.get("median")
        entry: dict[str, Any] = {"experiment": number, "experiment_id": record.experiment_id, "idea": idea,
                                 "original_dataset": record.dataset, "original_validation_median": original}
        if mode == "cross_asset":
            entry.update(cross_asset(record, data, dataset, evaluator, benchmarks))
            rep, tra = entry["replicate"], entry["transfer"]
            print(f"#{number} {idea}\n"
                  f"  original ({record.dataset}) validation median Sharpe {_f(original)}\n"
                  f"  replicate on {dataset}: train median {_f(rep['train'].get('median'))} "
                  f"(share positive {_pct(rep['train'].get('share_positive'))}), validation median "
                  f"{_f(rep['validation'].get('median'))} vs buy-and-hold {_f(rep['buy_and_hold_validation_sharpe'])}\n"
                  f"  transfer to {dataset} (frozen {record.dataset} parameters): validation median "
                  f"{_f(tra['validation']['candidates']['median'])}, best {_f(tra['validation']['candidates']['best'])}, "
                  f"median trades {_f(tra['validation']['candidates']['median_trades'], '.0f')} vs buy-and-hold "
                  f"{_f(tra['validation']['benchmarks'].get('buy_and_hold', {}).get('sharpe'))}\n")
        else:
            entry["final_test"] = final_test(record, data, test, dataset, evaluator, benchmarks)
            t = entry["final_test"]
            c = t["candidates"]
            print(f"#{number} {idea}\n"
                  f"  validation median Sharpe {_f(original)} -> FINAL TEST ({t['period']['first_bar']} .. "
                  f"{t['period']['last_bar']}): median {_f(c['median'])}, best {_f(c['best'])}, worst {_f(c['worst'])}, "
                  f"share positive {_pct(c['share_positive'])}, median trades {_f(c['median_trades'], '.0f')}\n"
                  f"  buy-and-hold on the same period: Sharpe {_f(t['benchmarks'].get('buy_and_hold', {}).get('sharpe'))}\n")
        results.append(entry)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"  # unique even within a second
    out = args.out_dir / f"{mode}_{dataset.replace('/', '-')}_{stamp}.json"
    out.write_text(json.dumps(to_json_safe({"mode": mode, "records": str(args.records), "dataset": dataset,
                                            "results": results}), indent=1))
    print(f"Saved to {out}")
    return 0


def _bench(table: pd.DataFrame | None, name: str) -> float | None:
    if table is None or table.empty or name not in set(table["benchmark"]):
        return None
    return _num(table.set_index("benchmark").loc[name, "sharpe"])


def _num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)


def _f(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


if __name__ == "__main__":
    sys.exit(main())
