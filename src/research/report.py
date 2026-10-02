"""Human-readable reports from experiment records (fresh or loaded from JSONL).

    python -m src.research.report data/research_runs/<run_id>/experiments.jsonl

Used by: agents/research.py (format_record prints every experiment); also a
         CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.research.records import ExperimentRecord, load_records

TABLE_METRICS = ("sharpe", "annualized_return", "max_drawdown", "n_trades", "exposure")


def format_record(record: ExperimentRecord) -> str:
    """Multi-line report: settings, search, train distribution, train vs validation, benchmarks, warnings."""
    lines = [f"=== Experiment {record.experiment_id} [{record.status}] {record.dataset}"]
    if record.hypothesis:
        lines.append(f"Hypothesis: {record.hypothesis}")
    if record.strategy_description:
        lines.append(f"Strategy:   {record.strategy_description}")
    if record.error:
        lines.append(f"Error:      {record.error}")
    if record.status != "completed":
        return "\n".join(lines)

    tp, vp, metric = record.train_period, record.validation_period, record.selection_metric
    lines.append(f"Train {tp['start']} -> {tp['end']}  |  validation {vp['start']} -> {vp['end']}  |  "
                 f"cost {record.transaction_cost:g} ({record.cost_bps:g} bps) per unit turnover")
    d = record.train_summary.get("distribution", {})
    lines.append(f"TRAIN: {record.n_candidates} candidates, {metric} median {_f(d.get('median'))} "
                 f"(p10 {_f(d.get('p10'))}, p90 {_f(d.get('p90'))}), best {_f(d.get('max'))}, "
                 f"share positive {_pct(d.get('share_positive'))}")
    v = record.validation_summary
    if v.get("n_retested"):
        lines.append(f"VALIDATION: top {v['n_retested']} retested with frozen parameters, {metric} median "
                     f"{_f(v.get('median'))}, best {_f(v.get('best'))}, worst {_f(v.get('worst'))}, "
                     f"median change vs train {_f(v.get('median_change'))}")
        lines.append("\n" + _comparison_table(record))
    activity = record.condition_activity
    if activity.get("shares"):
        lines.append(f"CONDITIONS ({activity.get('scope')}, share of train bars true): "
                     + ", ".join(f"{key} {_pct(share)}" for key, share in activity["shares"].items()))
    c = record.costs
    if c:
        lines.append(f"COSTS ({c.get('scope')}, train): ~{_f(c.get('median_trades_per_year'), '.0f')} trades/year, "
                     f"position change every ~{_f(c.get('median_bars_between_trades'), '.0f')} bars, exposure "
                     f"{_pct(c.get('median_exposure'))}, fees ~{_pct(c.get('median_annual_cost'))} of capital/year "
                     f"vs net annualized return {_pct(c.get('median_annualized_return'))}")
    if record.benchmarks:
        lines.append("\nBENCHMARKS (same periods, same costs):\n" + _benchmark_table(record))
    if record.warnings:
        lines.append("\nWARNINGS:\n" + "\n".join(f"- {w}" for w in record.warnings))
    if record.timing:
        lines.append(f"\nTiming: {record.timing}")
    return "\n".join(lines)


def _comparison_table(record: ExperimentRecord) -> str:
    table = pd.DataFrame(record.comparison)
    params = [c for c in table.columns if "." in c]
    columns = ["train_rank", *params]
    for m in TABLE_METRICS:
        columns += [f"train_{m}", f"validation_{m}"]
    return table[[c for c in columns if c in table.columns]].to_string(index=False, float_format="{:.3f}".format)


def _benchmark_table(record: ExperimentRecord) -> str:
    rows = [
        {"period": period, "benchmark": name, **{m: metrics.get(m) for m in TABLE_METRICS}}
        for period, benchmarks in record.benchmarks.items()
        for name, metrics in benchmarks.items()
    ]
    return pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format)


def _f(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print experiment records.")
    parser.add_argument("path", type=Path, help="experiments.jsonl or a single record .json")
    for record in load_records(parser.parse_args().path):
        print(format_record(record) + "\n")


if __name__ == "__main__":
    main()
