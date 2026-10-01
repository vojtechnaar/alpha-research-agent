"""Performance metrics for backtest results produced by src.backtest.engine.run_backtest.

Used by: strategies/evaluator.py (compute_metrics after every Python backtest), backends/native.py
         (METRIC_NAMES: the column order the C++/CUDA engine writes), backends/benchmark.py.
         HOURS_PER_YEAR is the default annualisation everywhere (CLIs, experiments, sweeps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365  # crypto trades 24/7

# Order of compute_metrics' output; native backends (cuda/backtest.cu, enum Metric) use the same order.
METRIC_NAMES = ("cumulative_return", "annualized_return", "sharpe", "annualized_volatility", "max_drawdown",
                "turnover", "annual_turnover", "n_trades", "exposure", "n_bars")


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the period."""
    return float((1 + returns.fillna(0.0)).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Geometric average return per year (CAGR) implied by the compounded return."""
    if len(returns) == 0:
        return float("nan")
    growth = 1 + cumulative_return(returns)
    return float(growth ** (periods_per_year / len(returns)) - 1) if growth > 0 else -1.0


def annualized_volatility(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Sample standard deviation of per-period returns, scaled to one year."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate; NaN if volatility is zero or undefined."""
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, as a negative fraction (e.g. -0.25).

    Equity starts at 1, so a loss on the very first bar counts as a drawdown.
    """
    equity = (1 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float(min(0.0, (equity / peak - 1).min())) if len(equity) else 0.0


def compute_metrics(result: pd.DataFrame, periods_per_year: float = HOURS_PER_YEAR) -> dict[str, float]:
    """Summary statistics for a run_backtest result.

    turnover is the total absolute position change (a long -> short flip counts 2);
    n_trades counts bars where the position changed (a flip counts 1); exposure is the share of
    bars with a non-zero position. All metrics are asset-agnostic; only periods_per_year depends
    on the bar frequency (default: hourly bars, trading 24/7).
    """
    returns = result["strategy_return"]
    turnover = result["turnover"]
    years = len(result) / periods_per_year
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "turnover": float(turnover.sum()),
        "annual_turnover": float(turnover.sum() / years) if years else float("nan"),
        "n_trades": int((turnover > 0).sum()),
        "exposure": float((result["held_position"] != 0).mean()) if len(result) else float("nan"),
        "n_bars": int(len(result)),
    }
