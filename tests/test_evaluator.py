"""StrategySpec evaluation: conditions, logic, positions, missing values and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.evaluator import evaluate_conditions, evaluate_strategy, generate_positions
from src.strategies.schema import StrategySpec


def make_data(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": volumes if volumes is not None else [1.0] * n,
    })


def condition(feature: str = "momentum", lookback: int | None = 1, op: str = ">", threshold: float = 0.0,
              field: str = "close", id: str | None = None) -> dict:
    c = {"feature": feature, "field": field, "operator": op, "threshold": threshold}
    if lookback is not None:
        c["lookback"] = lookback
    if id:
        c["id"] = id
    return c


def spec(*conditions: dict, logic: str = "AND", true: int = 1, false: int = 0) -> StrategySpec:
    return StrategySpec.from_dict({"name": "t", "conditions": list(conditions), "logic": logic,
                                   "true_position": true, "false_position": false})


CLOSES = [100, 101, 99, 100, 102, 101]  # 1-bar momentum: nan, +, -, +, +, -


def test_single_condition_long_or_flat() -> None:
    positions = generate_positions(make_data(CLOSES), spec(condition()))
    assert positions.tolist() == [0, 1, 0, 1, 1, 0]  # first bar undefined -> flat


def test_short_and_long_short_positions() -> None:
    data = make_data(CLOSES)
    assert generate_positions(data, spec(condition(), true=-1)).tolist() == [0, -1, 0, -1, -1, 0]
    assert generate_positions(data, spec(condition(), true=1, false=-1)).tolist() == [0, 1, -1, 1, 1, -1]


def test_and_or_logic() -> None:
    volumes = [1, 1, 5, 5, 1, 5]
    data = make_data(CLOSES, volumes)
    up = condition(id="up")
    busy = condition(feature="rolling_mean", field="volume", lookback=1, op=">=", threshold=5, id="busy")
    cond = evaluate_conditions(data, spec(up, busy))
    assert cond["busy"].tolist() == [0, 0, 1, 1, 0, 1]
    assert generate_positions(data, spec(up, busy, logic="AND")).tolist() == [0, 0, 0, 1, 0, 0]
    assert generate_positions(data, spec(up, busy, logic="OR")).tolist() == [0, 1, 1, 1, 1, 1]


def test_undefined_features_mean_flat_not_false_position() -> None:
    # With false_position=-1, warm-up and missing prices must be flat, not short.
    data = make_data([100, 101, np.nan, 103, 102, 101, 100])
    positions = generate_positions(data, spec(condition(lookback=2), true=1, false=-1))
    assert positions.iloc[:2].tolist() == [0, 0]  # warm-up
    assert positions.iloc[2] == 0 and positions.iloc[4] == 0  # NaN price at bar 2 (and 2 bars later)
    assert positions.iloc[6] == -1


def test_no_look_ahead_in_positions_or_returns() -> None:
    rng = np.random.default_rng(3)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    altered = closes[:120] + [c * 1.5 for c in closes[120:]]
    s = spec(condition(lookback=12, threshold=0.0, id="m"),
             condition(feature="volatility", lookback=24, op="<", threshold=0.02, id="v"))
    base = evaluate_strategy(make_data(closes), s, cost_bps=5).backtest
    changed = evaluate_strategy(make_data(altered), s, cost_bps=5).backtest
    pd.testing.assert_series_equal(base["position"].iloc[:120], changed["position"].iloc[:120])
    # Bar 120's jump is earned by the position set at bar 119 - never by a position set at 120.
    pd.testing.assert_series_equal(base["strategy_return"].iloc[:120], changed["strategy_return"].iloc[:120])


def test_evaluate_strategy_runs_backtest_with_lag_and_costs() -> None:
    data = make_data([100, 110, 121, 121])  # returns: 0, +10%, +10%, 0
    s = spec(condition(feature="momentum", lookback=1, op=">", threshold=0.05))
    result = evaluate_strategy(data, s, cost_bps=10)
    assert result.backtest["position"].tolist() == [0, 1, 1, 0]
    assert result.backtest["held_position"].tolist() == [0, 0, 1, 1]  # one-bar lag
    assert result.backtest["strategy_return"].tolist() == pytest.approx([0, 0, 0.1 - 0.001, 0])
    assert result.metrics["n_trades"] == 1 and result.metrics["n_bars"] == 4


def test_period_uses_warmed_up_features() -> None:
    data = make_data([100 + i for i in range(48)])
    s = spec(condition(lookback=24, threshold=0.0))
    result = evaluate_strategy(data, s, start="2024-01-02", cost_bps=0)
    assert result.backtest["timestamp"].iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert (result.backtest["position"] == 1).all()  # momentum(24) already defined at the period start
    assert result.backtest["held_position"].iloc[0] == 0  # but the period starts flat
