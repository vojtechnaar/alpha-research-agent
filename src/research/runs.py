"""Summarise saved research runs: how good was the researcher (LLM), run by run and per model.

    python -m src.research.runs                      # every run in data/research_runs/
    python -m src.research.runs data/research_runs --by model
    python -m src.research.runs --by model --datasets ETH/USD GLD   # base vs LoRA on held-out markets
    python -m src.research.runs --datasets ETH/USD GLD --models Qwen/Qwen3-8B Qwen/Qwen3-8B+v2

With exactly two models (e.g. Qwen/Qwen3-8B and Qwen/Qwen3-8B+v1), runs on the same market with the
same seed are also compared in pairs: wins, mean difference and a bootstrap 95% confidence interval.

These are the metrics for "is Qwen + LoRA a better researcher than base Qwen?" (see docs/progress.md).
Collecting base-model runs now builds the baseline that a LoRA adapter has to beat.

Per run:
  efficiency   LLM calls, first-try valid rate, calls per completed experiment, rejection codes
  quality      experiments whose frozen top-N kept a positive validation Sharpe ("holds up"), and
               "useful" ones: holds up AND trades often enough on validation AND is not
               buy-and-hold in disguise (long-only with >= 90% exposure)
  diversity    distinct idea families and features among completed experiments
  mistakes     automatic corrections, conditions (almost) never/always true
The headline number is useful experiments per 10 LLM calls: research value per unit of the
scarce resource (LLM time).

Used by: models/lora_data.py (experiment_flags decides which experiments are useful training
         examples); also the CLI that scores runs and compares base vs LoRA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.records import ExperimentRecord, load_records
from src.research.summary import NEAR_BUY_AND_HOLD_EXPOSURE
from src.strategies.schema import StrategySpec

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "research_runs"


def experiment_flags(record: ExperimentRecord) -> dict[str, bool]:
    """Quality flags of one completed experiment (all False for rejected/failed ones)."""
    flags = {"holds_up": False, "useful": False, "beats_buy_and_hold": False, "unit_problem": False}
    if record.status != "completed":
        return flags
    shares = [s for s in record.condition_activity.get("shares", {}).values() if s is not None]
    flags["unit_problem"] = any(s <= 0.01 or s >= 0.99 for s in shares)
    median = record.validation_summary.get("median")
    if median is None or not record.comparison:
        return flags
    rows = pd.DataFrame(record.comparison)
    flags["holds_up"] = median > 0
    spec = StrategySpec.from_dict(record.strategy_spec)
    long_only = -1 not in (spec.true_position, spec.false_position)
    near_buy_and_hold = long_only and rows["validation_exposure"].median() >= NEAR_BUY_AND_HOLD_EXPOSURE
    # annual turnover ~ position changes per year (a long -> short flip counts twice)
    trades_enough = rows["validation_annual_turnover"].median() >= (record.min_trades_per_year or 0)
    flags["useful"] = flags["holds_up"] and trades_enough and not near_buy_and_hold
    buy_and_hold = record.benchmarks.get("validation", {}).get("buy_and_hold", {}).get("sharpe")
    # Only meaningful when buy-and-hold itself made money (e.g. bonds in 2022 did not: flat would "beat" it).
    flags["beats_buy_and_hold"] = (flags["useful"] and buy_and_hold is not None and buy_and_hold > 0
                                   and median > buy_and_hold)
    return flags


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """One row of metrics for a run directory (run.json is optional)."""
    info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    generator = info.get("generator", {})
    model = generator.get("model_id") or ("replay" if "replay" in generator else "unknown")
    if generator.get("adapter"):
        model += f"+{Path(str(generator['adapter'])).name}"
    records = load_records(run_dir / "experiments.jsonl")
    attempts = [a for r in records for a in (r.llm or {}).get("attempts", [])]
    completed = [r for r in records if r.status == "completed"]
    flags = [experiment_flags(r) for r in completed]
    families = {StrategySpec.from_dict(r.strategy_spec).family() for r in completed}
    features = {c["feature"] for r in completed for c in r.strategy_spec["conditions"]}
    first_try = [bool((r.llm or {}).get("attempts")) and r.llm["attempts"][0].get("code") is None for r in records]
    codes = Counter(a["code"] for a in attempts if a.get("code"))
    calls = len(attempts)
    useful = sum(f["useful"] for f in flags)
    return {
        "run": run_dir.name,
        "model": model,
        "dataset": info.get("dataset") or (records[0].dataset if records else ""),
        "seed": info.get("loop", {}).get("seed"),
        "iterations": len(records),
        "completed": len(completed),
        "llm_calls": calls,
        "first_try_valid": round(sum(first_try) / len(records), 3) if records else None,
        "calls_per_completed": round(calls / len(completed), 2) if completed else None,
        "holds_up": sum(f["holds_up"] for f in flags),
        "useful": useful,
        "beats_buy_and_hold": sum(f["beats_buy_and_hold"] for f in flags),
        "useful_per_10_calls": round(10 * useful / calls, 2) if calls else None,
        "families": len(families),
        "features": len(features),
        "unit_problems": sum(f["unit_problem"] for f in flags),
        "auto_corrections": sum(bool(r.notes) and "rewritten" in r.notes for r in records),
        "rejection_codes": dict(codes),
    }


def summarize_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per run found under `runs_dir` (sorted by run id, i.e. time)."""
    dirs = sorted(d for d in Path(runs_dir).iterdir() if (d / "experiments.jsonl").exists())
    return pd.DataFrame([summarize_run(d) for d in dirs])


def totals(table: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Aggregate runs per model (or dataset): the base-vs-LoRA comparison table."""
    sums = table.groupby(by)[["iterations", "completed", "llm_calls", "holds_up", "useful", "beats_buy_and_hold",
                              "unit_problems", "auto_corrections"]].sum()
    sums["runs"] = table.groupby(by).size()
    sums["first_try_valid"] = table.groupby(by)["first_try_valid"].mean().round(3)
    sums["useful_per_10_calls"] = (10 * sums["useful"] / sums["llm_calls"]).round(2)
    sums["families_per_run"] = table.groupby(by)["families"].mean().round(1)
    return sums.reset_index()


def paired_comparison(table: pd.DataFrame, metric: str = "useful_per_10_calls", n_boot: int = 10_000,
                       seed: int = 0) -> dict[str, Any] | None:
    """Two models on the same (market, seed) pairs: wins and a bootstrap 95% CI of the mean difference.

    The base is the model whose name is shorter (Qwen/Qwen3-8B vs Qwen/Qwen3-8B+v1). None unless there
    are exactly two models and at least one pair.
    """
    models = sorted(table["model"].unique(), key=len)
    if len(models) != 2:
        return None
    base, other = models
    wide = table.pivot_table(index=["dataset", "seed"], columns="model", values=metric, aggfunc="mean").dropna()
    if wide.empty or not set(models) <= set(wide.columns):
        return None
    diff = wide[other] - wide[base]
    means = np.random.default_rng(seed).choice(diff.to_numpy(), size=(n_boot, len(diff))).mean(axis=1)
    per_dataset = {d: {"pairs": len(g), "wins": int((g > 0).sum()), "mean_difference": round(float(g.mean()), 2)}
                   for d, g in diff.groupby(level="dataset")}
    return {"base": base, "other": other, "metric": metric, "pairs": len(diff), "wins": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()), "mean_difference": round(float(diff.mean()), 2),
            "ci95": tuple(round(float(x), 2) for x in np.percentile(means, [2.5, 97.5])),
            "per_dataset": per_dataset}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarise saved research runs.")
    p.add_argument("runs_dir", type=Path, nargs="?", default=DEFAULT_RUNS_DIR)
    p.add_argument("--by", default="model", choices=["model", "dataset"])
    p.add_argument("--datasets", nargs="+", help="only runs on these markets (e.g. the held-out ETH/USD GLD)")
    p.add_argument("--models", nargs="+", help="only these models, e.g. Qwen/Qwen3-8B Qwen/Qwen3-8B+v2 "
                                              "(the paired comparison needs exactly two)")
    args = p.parse_args(argv)
    if not args.runs_dir.exists():
        print(f"No runs yet in {args.runs_dir}")
        return 0
    table = summarize_runs(args.runs_dir)
    if args.datasets and not table.empty:
        table = table[table["dataset"].isin(args.datasets)]
    if args.models and not table.empty:
        table = table[table["model"].isin(args.models)]
    if table.empty:
        print(f"No runs yet in {args.runs_dir}" + (f" on {', '.join(args.datasets)}" if args.datasets else ""))
        return 0
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.drop(columns=["rejection_codes"]).to_string(index=False))
        print(f"\nTotals per {args.by}:")
        print(totals(table, args.by).to_string(index=False))
    print()
    for name, group in table.groupby(args.by):
        codes = Counter()
        for c in group["rejection_codes"]:
            codes.update(c)
        print(f"Rejection codes, {name}: {dict(codes) or 'none'}")
    paired = paired_comparison(table)
    if paired:
        lo, hi = paired["ci95"]
        print(f"\nPaired by market and seed ({paired['metric']}): {paired['other']} vs {paired['base']}")
        print(f"  better in {paired['wins']} of {paired['pairs']} pairs, worse in {paired['losses']}; "
              f"mean difference {paired['mean_difference']:+.2f}, 95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]")
        for dataset, d in paired["per_dataset"].items():
            print(f"  {dataset}: better in {d['wins']} of {d['pairs']}, mean difference {d['mean_difference']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
