"""Vectorised single-asset backtester: signal -> position -> strategy returns.

Timing convention: the signal at bar t is computed from data up to and including the close of
bar t, the position is taken at that close, and it earns the return of bar t+1. The position is
therefore shifted by one bar before it touches returns, so there is no look-ahead.

Used by: strategies/evaluator.py (every Python-backend backtest goes through run_backtest),
         backends/native.py (period_returns, so the C++/CUDA engines get identical returns),
         strategies/sweep.py (DEFAULT_COST_BPS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_BPS = 10.0  # per unit of position change; a long -> short flip is 2 units


def signal_to_position(signal: pd.Series) -> pd.Series:
    """Map a numeric signal to +1 (long), 0 (flat) or -1 (short) by its sign; NaN/inf -> flat."""
    values = pd.to_numeric(signal, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return np.sign(values).fillna(0.0).astype("float64")


def period_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close returns; a missing close earns 0 and its move lands on the next valid bar.

    The first bar has no previous close in the period, so its return is 0. Native backends are
    given exactly these returns, so all engines share one definition.
    """
    filled = close.astype("float64").ffill()
    return (filled / filled.shift(1) - 1).fillna(0.0)


def run_backtest(
    data: pd.DataFrame,
    signal: pd.Series | str,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Backtest one asset.

    Args:
        data: bars in chronological order with a `close` column (and optionally `timestamp`).
        signal: numeric Series aligned with `data.index`, or the name of a column in `data`.
        cost_bps: transaction cost in basis points of notional per unit of position change.

    Returns:
        One row per bar with timestamp, close, return, signal, position (target set at this
        bar's close), held_position (position earning this bar's return), turnover, cost,
        strategy_return and cumulative_return. Returns are simple (percentage) returns, which
        compose correctly for short positions and costs.
    """
    if "close" not in data:
        raise KeyError("data must contain a 'close' column")
    if "timestamp" in data and not data["timestamp"].is_monotonic_increasing:
        raise ValueError("data must be sorted by timestamp")
    signal = data[signal] if isinstance(signal, str) else signal
    if not signal.index.equals(data.index):
        raise ValueError("signal index must match data index")

    close = data["close"].astype("float64")
    out = pd.DataFrame(index=data.index)
    out["timestamp"] = data["timestamp"] if "timestamp" in data else data.index
    out["close"] = close
    out["return"] = period_returns(close)
    out["signal"] = signal
    out["position"] = signal_to_position(signal)
    out["held_position"] = out["position"].shift(1, fill_value=0.0)  # decided one bar earlier: no look-ahead
    out["turnover"] = out["held_position"].diff().fillna(out["held_position"]).abs()  # first bar: entry from flat
    out["cost"] = out["turnover"] * cost_bps / 10_000
    out["strategy_return"] = out["held_position"] * out["return"] - out["cost"]
    out["cumulative_return"] = (1 + out["strategy_return"]).cumprod() - 1
    return out
