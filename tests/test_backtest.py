"""Unit tests for the backtest engine and metrics (small synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest, signal_to_position
from src.backtest.metrics import (
    compute_metrics,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)


def make_data(closes: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "close": closes})


def test_signal_to_position_uses_sign_and_flattens_invalid() -> None:
    signal = pd.Series([2.5, -0.1, 0.0, np.nan, np.inf])
    assert signal_to_position(signal).tolist() == [1.0, -1.0, 0.0, 0.0, 0.0]


def test_position_is_shifted_one_bar() -> None:
    data = make_data([100, 110, 99, 99])  # returns: 0, +10%, -10%, 0
    result = run_backtest(data, pd.Series([1, -1, 0, 0]), cost_bps=0)

    assert result["held_position"].tolist() == [0, 1, -1, 0]
    assert result["strategy_return"].tolist() == pytest.approx([0, 0.1, 0.1, 0])


def test_no_look_ahead_future_signal_changes_do_not_affect_past() -> None:
    rng = np.random.default_rng(0)
    data = make_data(list(100 * np.cumprod(1 + rng.normal(0, 0.01, 50))))
    signal = pd.Series(rng.choice([-1, 0, 1], 50))
    altered = signal.copy()
    altered.iloc[30:] = -altered.iloc[30:] + 1  # rewrite the "future" from bar 30 on

    base = run_backtest(data, signal)["strategy_return"]
    changed = run_backtest(data, altered)["strategy_return"]

    # Signal at bar 30 can first affect bar 31's return.
    pd.testing.assert_series_equal(base.iloc[:31], changed.iloc[:31])


def test_signal_equal_to_same_bar_return_cannot_profit_from_it() -> None:
    """A signal that 'knows' bar t's return must only trade bar t+1."""
    data = make_data([100, 110, 100, 110, 100])  # alternating up/down
    result = run_backtest(data, run_backtest(data, pd.Series(0, index=data.index))["return"], cost_bps=0)
    # Long after each up bar, short after each down bar -> always wrong on the next bar.
    assert (result["strategy_return"].iloc[2:] < 0).all()


def test_transaction_costs_charged_on_position_changes() -> None:
    data = make_data([100, 110, 99, 99])
    result = run_backtest(data, pd.Series([1, -1, 0, 0]), cost_bps=10)

    assert result["turnover"].tolist() == [0, 1, 2, 1]  # enter, flip, exit
    assert result["cost"].tolist() == pytest.approx([0, 0.001, 0.002, 0.001])
    assert result["strategy_return"].tolist() == pytest.approx([0, 0.099, 0.098, -0.001])


def test_holding_a_position_incurs_no_extra_cost() -> None:
    data = make_data([100, 101, 102, 103])
    result = run_backtest(data, pd.Series([1, 1, 1, 1]), cost_bps=10)
    assert result["cost"].tolist() == pytest.approx([0, 0.001, 0, 0])


def test_cumulative_return_compounds() -> None:
    data = make_data([100, 110, 99, 99])
    result = run_backtest(data, pd.Series([1, -1, 0, 0]), cost_bps=10)

    expected = 1.099 * 1.098 * 0.999 - 1
    assert result["cumulative_return"].iloc[-1] == pytest.approx(expected)
    assert cumulative_return(result["strategy_return"]) == pytest.approx(expected)


def test_buy_and_hold_matches_price_change() -> None:
    data = make_data([100, 120, 90, 150])
    result = run_backtest(data, pd.Series([1, 1, 1, 1]), cost_bps=0)
    assert result["cumulative_return"].iloc[-1] == pytest.approx(150 / 100 - 1)


def test_max_drawdown() -> None:
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)  # loss from the start
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_sharpe_ratio() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(4)
    assert sharpe_ratio(returns, periods_per_year=4) == pytest.approx(expected)
    assert sharpe_ratio(returns) == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(24 * 365))
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


def test_missing_values_are_handled() -> None:
    data = make_data([100, np.nan, 110, 121])
    result = run_backtest(data, pd.Series([1, np.nan, 1, 1]), cost_bps=0)

    cols = ["return", "position", "held_position", "strategy_return", "cumulative_return"]
    assert not result[cols].isna().any().any()
    assert result["position"].tolist() == [1, 0, 1, 1]  # NaN signal -> flat
    assert result["return"].tolist() == pytest.approx([0, 0, 0.1, 0.1])  # gap move lands on next valid bar
    assert result["strategy_return"].tolist() == pytest.approx([0, 0, 0, 0.1])  # flat during the gap move


def test_compute_metrics_counts_trades_and_turnover() -> None:
    data = make_data([100, 101, 102, 101, 100])
    result = run_backtest(data, pd.Series([1, 1, -1, 0, 0]), cost_bps=0)
    metrics = compute_metrics(result)

    assert metrics["n_trades"] == 3  # enter long, flip short, exit
    assert metrics["turnover"] == 4.0
    assert set(metrics) >= {"cumulative_return", "sharpe", "annualized_volatility", "max_drawdown"}


def test_rejects_misaligned_or_unsorted_input() -> None:
    data = make_data([100, 101, 102])
    with pytest.raises(ValueError, match="index"):
        run_backtest(data, pd.Series([1, 1, 1], index=[5, 6, 7]))
    with pytest.raises(ValueError, match="sorted"):
        run_backtest(data.iloc[::-1].reset_index(drop=True), pd.Series([1, 1, 1]))
