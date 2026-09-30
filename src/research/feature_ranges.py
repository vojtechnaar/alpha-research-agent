"""Typical values of every registered feature on the TRAINING period, for the LLM prompt.

Thresholds only make sense in a feature's units: a price-level feature such as rolling_mean(close)
is never below 0, while volatility is a small per-bar fraction. Showing the model each feature's
5th percentile / median / 95th percentile keeps its thresholds in range. Only train-period values
are used, so nothing from validation or the final test leaks into the prompt.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.experiment import Period
from src.strategies.evaluator import compute_feature, period_slice
from src.strategies.features import FEATURE_REGISTRY
from src.strategies.schema import Condition

RANGE_LOOKBACK = 24


def feature_ranges(data: pd.DataFrame, period: Period, lookback: int = RANGE_LOOKBACK) -> list[dict[str, Any]]:
    """p5 / median / p95 of each feature (on close; volume_change on volume) over `period`."""
    rows = period_slice(data, period.start, period.end)
    ranges = []
    for name, definition in FEATURE_REGISTRY.items():
        field = "volume" if name == "volume_change" else "close"
        condition = Condition(feature=name, field=field, operator=">", threshold=0.0,
                              lookback=max(lookback, definition.min_lookback) if definition.uses_lookback else None)
        values = compute_feature(data, condition).iloc[rows].dropna()
        if values.empty:
            continue
        p5, median, p95 = values.quantile([0.05, 0.5, 0.95])
        ranges.append({"feature": name, "field": field, "lookback": condition.lookback,
                       "p5": float(p5), "median": float(median), "p95": float(p95)})
    return ranges


def format_ranges(ranges: list[dict[str, Any]]) -> str:
    """Prompt block, one line per feature."""
    lines = []
    for r in ranges:
        args = r["field"] if r["lookback"] is None else f"{r['field']}, {r['lookback']}"
        lines.append(f"- {r['feature']}({args}): {_number(r['p5'])} / {_number(r['median'])} / {_number(r['p95'])}")
    return ("TYPICAL FEATURE VALUES on the training data (5th percentile / median / 95th percentile). "
            "Thresholds outside these ranges are (almost) never or always true. momentum and rolling_std "
            "grow with the lookback:\n" + "\n".join(lines))


def _number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"
