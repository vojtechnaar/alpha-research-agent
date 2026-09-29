"""Feature library: values, warm-up NaNs, NaN handling and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.features import (
    FEATURE_REGISTRY,
    compute_momentum,
    compute_returns,
    compute_rolling_max,
    compute_rolling_mean,
    compute_rolling_min,
    compute_rolling_std,
    compute_volatility,
    compute_volume_change,
    compute_zscore,
)

S = pd.Series([1.0, 2.0, 4.0, 8.0, 4.0])


def values(series: pd.Series) -> list:
    return [None if np.isnan(v) else pytest.approx(v) for v in series]


def test_returns() -> None:
    assert values(compute_returns(S)) == [None, 1.0, 1.0, 1.0, -0.5]


def test_momentum_uses_lookback() -> None:
    assert values(compute_momentum(S, 2)) == [None, None, 3.0, 3.0, 0.0]
    assert values(compute_momentum(S, 1)) == values(compute_returns(S))


def test_rolling_mean_std_min_max() -> None:
    assert values(compute_rolling_mean(S, 2)) == [None, 1.5, 3.0, 6.0, 6.0]
    assert values(compute_rolling_std(S, 3))[2:] == [pytest.approx(np.std([1, 2, 4], ddof=1)),
                                                     pytest.approx(np.std([2, 4, 8], ddof=1)),
                                                     pytest.approx(np.std([4, 8, 4], ddof=1))]
    assert values(compute_rolling_min(S, 3)) == [None, None, 1.0, 2.0, 4.0]
    assert values(compute_rolling_max(S, 3)) == [None, None, 4.0, 8.0, 8.0]


def test_volatility_is_std_of_one_bar_returns() -> None:
    expected = compute_returns(S).rolling(2, min_periods=2).std(ddof=1)
    pd.testing.assert_series_equal(compute_volatility(S, 2), expected, check_names=False)
    assert np.isnan(compute_volatility(S, 2).iloc[:2]).all()  # needs 2 returns -> 3 prices


def test_zscore() -> None:
    z = compute_zscore(S, 3)
    window = np.array([2.0, 4.0, 8.0])
    assert z.iloc[3] == pytest.approx((8 - window.mean()) / window.std(ddof=1))
    assert np.isnan(compute_zscore(pd.Series([5.0, 5.0, 5.0]), 3).iloc[2])  # zero std -> NaN


def test_volume_change_compares_to_previous_bars() -> None:
    volume = pd.Series([10.0, 10.0, 30.0, 10.0])
    assert values(compute_volume_change(volume, 2)) == [None, None, 2.0, -0.5]


def test_division_by_zero_becomes_nan() -> None:
    assert np.isnan(compute_momentum(pd.Series([0.0, 1.0]), 1).iloc[1])


def test_nan_inside_window_gives_nan() -> None:
    s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    assert values(compute_rolling_mean(s, 2)) == [None, None, None, 3.5, 4.5]


@pytest.mark.parametrize("name", sorted(FEATURE_REGISTRY))
def test_no_feature_uses_future_data(name: str) -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(100 + rng.normal(0, 1, 80).cumsum())
    altered = series.copy()
    altered.iloc[50:] *= 3  # change only the "future"
    definition = FEATURE_REGISTRY[name]
    args = (5,) if definition.uses_lookback else ()
    before = definition.compute(series, *args).iloc[:50]
    after = definition.compute(altered, *args).iloc[:50]
    pd.testing.assert_series_equal(before, after)
