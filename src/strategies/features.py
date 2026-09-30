"""Trusted, parameterised quantitative features.

Every feature maps a price/volume series to a series of the same length where the value at bar t
uses only bars <= t (no look-ahead). Windows are trailing and need `lookback` valid values, so the
first bars are NaN (warm-up) and a NaN anywhere in a window makes that window's value NaN.
Divisions by zero and infinities become NaN.

Parameters such as `lookback` are always runtime arguments, never baked into a function. The
native C++/CUDA engine (cuda/backtest.cu, feature_value) implements every entry of FEATURE_REGISTRY
with the same parameters and semantics (see docs/strategy_engine.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _clean(values: pd.Series) -> pd.Series:
    """Replace +/-inf (e.g. from division by zero) with NaN."""
    return values.replace([np.inf, -np.inf], np.nan).astype("float64")


def _rolling(series: pd.Series, lookback: int) -> pd.core.window.Rolling:
    return series.astype("float64").rolling(lookback, min_periods=lookback)


def compute_returns(series: pd.Series) -> pd.Series:
    """One-bar simple return: x[t] / x[t-1] - 1."""
    return compute_momentum(series, 1)


def compute_momentum(series: pd.Series, lookback: int) -> pd.Series:
    """Return over `lookback` bars: x[t] / x[t-lookback] - 1."""
    return _clean(series / series.shift(lookback) - 1)


def compute_rolling_mean(series: pd.Series, lookback: int) -> pd.Series:
    """Mean of the last `lookback` values."""
    return _clean(_rolling(series, lookback).mean())


def compute_rolling_std(series: pd.Series, lookback: int) -> pd.Series:
    """Sample standard deviation (ddof=1) of the last `lookback` values."""
    return _clean(_rolling(series, lookback).std(ddof=1))


def compute_volatility(series: pd.Series, lookback: int) -> pd.Series:
    """Standard deviation of one-bar returns over the last `lookback` bars (per bar, not annualised)."""
    return compute_rolling_std(compute_returns(series), lookback)


def compute_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """(x[t] - rolling mean) / rolling std over the last `lookback` values; NaN if std is 0."""
    std = compute_rolling_std(series, lookback)
    return _clean((series - compute_rolling_mean(series, lookback)) / std.where(std != 0))


def compute_rolling_min(series: pd.Series, lookback: int) -> pd.Series:
    """Minimum of the last `lookback` values."""
    return _clean(_rolling(series, lookback).min())


def compute_rolling_max(series: pd.Series, lookback: int) -> pd.Series:
    """Maximum of the last `lookback` values."""
    return _clean(_rolling(series, lookback).max())


def compute_volume_change(series: pd.Series, lookback: int) -> pd.Series:
    """Current value relative to the mean of the previous `lookback` values, minus 1.

    E.g. 0.5 means this bar's volume is 50% above its recent average (bar t itself is excluded
    from the average).
    """
    return _clean(series / compute_rolling_mean(series.shift(1), lookback) - 1)


def compute_distance_to_max(series: pd.Series, lookback: int) -> pd.Series:
    """x[t] / max of the last `lookback` values - 1. Always <= 0; 0 means a new `lookback`-bar high."""
    return _clean(series / compute_rolling_max(series, lookback) - 1)


def compute_distance_to_min(series: pd.Series, lookback: int) -> pd.Series:
    """x[t] / min of the last `lookback` values - 1. Always >= 0; 0 means a new `lookback`-bar low."""
    return _clean(series / compute_rolling_min(series, lookback) - 1)


def compute_distance_to_mean(series: pd.Series, lookback: int) -> pd.Series:
    """x[t] / mean of the last `lookback` values - 1, e.g. 0.02 = 2% above the trailing average."""
    return _clean(series / compute_rolling_mean(series, lookback) - 1)


@dataclass(frozen=True)
class FeatureDef:
    """A registered feature: its implementation and the parameters it accepts."""

    compute: Callable[..., pd.Series]
    uses_lookback: bool = True
    min_lookback: int = 1
    description: str = ""
    threshold_hint: str = ""  # units/typical range, shown to the LLM so thresholds are sensible


FEATURE_REGISTRY: dict[str, FeatureDef] = {
    "returns": FeatureDef(
        compute_returns, uses_lookback=False,
        description="one-bar return x[t]/x[t-1]-1 (for the return over N bars use momentum with lookback N)",
        threshold_hint="fraction per bar, e.g. -0.01 = a 1% drop within one bar",
    ),
    "momentum": FeatureDef(
        compute_momentum, description="return over lookback bars",
        threshold_hint="fraction, e.g. 0.02 = +2% over the lookback",
    ),
    "rolling_mean": FeatureDef(
        compute_rolling_mean, description="trailing mean",
        threshold_hint="units of the field (a price level); fixed price thresholds rarely generalise, "
                       "prefer distance_to_mean",
    ),
    "rolling_std": FeatureDef(
        compute_rolling_std, min_lookback=2, description="trailing sample std",
        threshold_hint="units of the field (price units); prefer volatility for a unit-free measure",
    ),
    "volatility": FeatureDef(
        compute_volatility, min_lookback=2, description="std of one-bar returns",
        threshold_hint="per-bar std of returns, NOT annualised (a small fraction, see the typical values)",
    ),
    "zscore": FeatureDef(
        compute_zscore, min_lookback=2, description="(x - trailing mean) / trailing std",
        threshold_hint="unitless, typically -3..3",
    ),
    "rolling_min": FeatureDef(
        compute_rolling_min, description="trailing minimum",
        threshold_hint="units of the field (a price level); prefer distance_to_min",
    ),
    "rolling_max": FeatureDef(
        compute_rolling_max, description="trailing maximum",
        threshold_hint="units of the field (a price level); prefer distance_to_max",
    ),
    "volume_change": FeatureDef(
        compute_volume_change, description="x / mean of previous lookback values - 1",
        threshold_hint="fraction vs the recent average, e.g. 0.5 = 50% above it",
    ),
    "distance_to_max": FeatureDef(
        compute_distance_to_max, description="x / trailing max - 1, always <= 0 (0 = at the lookback high)",
        threshold_hint="fraction, e.g. -0.05 = 5% below the recent high; >= 0 means a new high (breakout)",
    ),
    "distance_to_min": FeatureDef(
        compute_distance_to_min, description="x / trailing min - 1, always >= 0 (0 = at the lookback low)",
        threshold_hint="fraction, e.g. 0.05 = 5% above the recent low; <= 0 means a new low",
    ),
    "distance_to_mean": FeatureDef(
        compute_distance_to_mean, description="x / trailing mean - 1",
        threshold_hint="fraction, e.g. 0.02 = 2% above the trailing average, -0.02 = 2% below",
    ),
}
