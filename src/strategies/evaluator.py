"""Turn a StrategySpec into positions and backtest it with the existing engine.

Pipeline: validate -> compute features (registry) -> apply operators -> combine with AND/OR ->
map to positions -> src.backtest.engine.run_backtest (which applies the one-bar execution lag)
-> src.backtest.metrics.compute_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import DEFAULT_COST_BPS, run_backtest
from src.backtest.metrics import HOURS_PER_YEAR, compute_metrics
from src.strategies.features import FEATURE_REGISTRY
from src.strategies.operators import LOGIC_REGISTRY, OPERATOR_REGISTRY
from src.strategies.schema import Condition, StrategySpec, validate_strategy

# (feature, field, lookback) -> feature series. Lets a sweep compute each feature once and reuse it
# for every threshold; the CUDA engine will do the same with feature buffers.
FeatureCache = dict[tuple[str, str, int | None], pd.Series]


@dataclass
class StrategyResult:
    """Everything one evaluation produces. `backtest` has one row per bar of the evaluated period."""

    spec: StrategySpec
    backtest: pd.DataFrame
    metrics: dict[str, float]


def compute_feature(data: pd.DataFrame, condition: Condition, cache: FeatureCache | None = None) -> pd.Series:
    """Compute the condition's feature on its input field via the registry (cached if `cache` given)."""
    key = (condition.feature, condition.field, condition.lookback)
    if cache is not None and key in cache:
        return cache[key]
    definition = FEATURE_REGISTRY[condition.feature]
    series = data[condition.field]
    values = definition.compute(series, condition.lookback) if definition.uses_lookback else definition.compute(series)
    if cache is not None:
        cache[key] = values
    return values


def evaluate_conditions(data: pd.DataFrame, spec: StrategySpec, cache: FeatureCache | None = None) -> pd.DataFrame:
    """One column per condition (named by its id): 1.0 true, 0.0 false, NaN undefined."""
    return pd.DataFrame(
        {
            c.key: OPERATOR_REGISTRY[c.operator](compute_feature(data, c, cache), float(c.threshold))
            for c in spec.conditions
        },
        index=data.index,
    )


def generate_positions(data: pd.DataFrame, spec: StrategySpec, cache: FeatureCache | None = None) -> pd.Series:
    """Target position per bar (-1/0/+1), decided at that bar's close. Undefined conditions -> flat.

    A spec without conditions is unconditional (always `true_position`): buy-and-hold, cash.
    """
    validate_strategy(spec)
    if not spec.conditions:
        return pd.Series(float(spec.true_position), index=data.index, name="position")
    combined = LOGIC_REGISTRY[spec.logic](evaluate_conditions(data, spec, cache))
    positions = np.where(combined == 1.0, spec.true_position, spec.false_position).astype("float64")
    return pd.Series(np.where(combined.isna(), 0.0, positions), index=data.index, name="position")


def period_slice(data: pd.DataFrame, start: Any = None, end: Any = None) -> slice:
    """Row positions of [start, end) by UTC timestamp; None means open-ended."""
    timestamps = data["timestamp"]
    lo = 0 if start is None else int(timestamps.searchsorted(_utc(start)))
    hi = len(data) if end is None else int(timestamps.searchsorted(_utc(end)))
    return slice(lo, hi)


def evaluate_strategy(
    data: pd.DataFrame,
    spec: StrategySpec,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: float = HOURS_PER_YEAR,
    start: Any = None,
    end: Any = None,
    cache: FeatureCache | None = None,
) -> StrategyResult:
    """Backtest `spec` on the [start, end) period of `data`.

    Features are computed on the whole of `data`, so indicators are already warmed up at `start`.
    Every feature only uses past bars, so values inside the period never depend on later data.
    Returns and costs count only inside the period. This is where train/test splits and
    walk-forward windows plug in. A `cache` must only be reused with the same `data`.
    """
    rows = period_slice(data, start, end)
    positions = generate_positions(data, spec, cache).iloc[rows]
    backtest = run_backtest(data.iloc[rows], positions, cost_bps=cost_bps)
    return StrategyResult(spec, backtest, compute_metrics(backtest, periods_per_year))


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
