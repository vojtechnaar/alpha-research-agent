"""Parameter sweeps: one base StrategySpec + a parameter space -> many evaluated candidates.

The LLM proposes the hypothesis, base strategy and the value lists; this module generates the
Cartesian product itself. This is the interface the CUDA engine will later accelerate: same
inputs (base spec, parameter space, data), same output (one results row per candidate).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.backtest.engine import DEFAULT_COST_BPS
from src.backtest.metrics import HOURS_PER_YEAR
from src.strategies.evaluator import FeatureCache, evaluate_strategy
from src.strategies.schema import INVALID_SPEC, SpecError, StrategySpec

ParameterSpace = Mapping[str, Sequence[int | float]]
DEFAULT_MAX_CANDIDATES = 1000


def count_candidates(space: ParameterSpace) -> int:
    """Size of the Cartesian product."""
    return math.prod(len(values) for values in space.values())


def parameter_grid(space: ParameterSpace) -> list[dict[str, int | float]]:
    """All combinations, in a deterministic order (keys as given, values as listed)."""
    names = list(space)
    return [dict(zip(names, combo)) for combo in itertools.product(*(space[n] for n in names))]


def generate_candidates(
    base: StrategySpec, space: ParameterSpace, max_candidates: int = DEFAULT_MAX_CANDIDATES
) -> list[tuple[dict[str, int | float], StrategySpec]]:
    """Validated (parameters, spec) pairs for every combination.

    Raises SpecError if the space is empty, too large, or any combination is invalid, so nothing
    is backtested until the whole search is known to be valid.
    """
    if any(not isinstance(v, Sequence) or isinstance(v, str) or not v for v in space.values()):
        raise SpecError(INVALID_SPEC, "every parameter needs a non-empty list of values")
    n = count_candidates(space)
    if n > max_candidates:
        raise SpecError(INVALID_SPEC, f"parameter space has {n} candidates; the limit is {max_candidates}")
    return [(params, base.with_parameters(params)) for params in parameter_grid(space)]


def run_sweep(
    data: pd.DataFrame,
    base: StrategySpec,
    space: ParameterSpace,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: float = HOURS_PER_YEAR,
    start: Any = None,
    end: Any = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    dataset: str = "",
) -> pd.DataFrame:
    """Evaluate every candidate on one dataset/period; one row per candidate.

    Columns: candidate, dataset, start, end, one column per swept parameter ("<id>.<param>"),
    then the metrics. The long format means more assets, splits or cost levels are just more
    rows (pd.concat), ready for walk-forward, stability and significance analysis later.
    """
    cache: FeatureCache = {}
    rows = []
    for i, (params, spec) in enumerate(generate_candidates(base, space, max_candidates)):
        result = evaluate_strategy(data, spec, cost_bps, periods_per_year, start, end, cache)
        period = result.backtest["timestamp"]
        rows.append({
            "candidate": i,
            "dataset": dataset,
            "start": period.iloc[0] if len(period) else None,
            "end": period.iloc[-1] if len(period) else None,
            **params,
            **result.metrics,
        })
    return pd.DataFrame(rows)


def summarize_sweep(results: pd.DataFrame, metric: str = "sharpe", top: int = 5) -> dict[str, Any]:
    """Compact summary of a sweep, small enough to feed back to an LLM.

    Includes the metric's distribution, the best candidates and, per parameter, the mean metric
    for each value (a first look at parameter stability: a real effect should not hinge on one value).
    """
    params = [c for c in results.columns if "." in c]  # swept parameters are named "<id>.<param>"
    values = results[metric].dropna()
    best = results.sort_values(metric, ascending=False).head(top)
    return {
        "n_candidates": int(len(results)),
        "metric": metric,
        "distribution": {
            "min": _round(values.min()), "median": _round(values.median()), "max": _round(values.max()),
            "share_positive": _round((values > 0).mean()) if len(values) else None,
        },
        "top": [
            {**{p: row[p] for p in params}, metric: _round(row[metric]),
             "max_drawdown": _round(row["max_drawdown"]), "n_trades": int(row["n_trades"])}
            for _, row in best.iterrows()
        ],
        "parameter_sensitivity": {
            p: {str(k): _round(v) for k, v in results.groupby(p)[metric].mean().items()} for p in params
        },
    }


def _round(value: Any, digits: int = 4) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)
