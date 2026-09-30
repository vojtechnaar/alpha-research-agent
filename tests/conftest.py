"""Shared synthetic fixtures (no network, no real market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.experiment import ExperimentSettings, Period


def make_hourly_data(n: int = 24 * 150, seed: int = 11, start: str = "2020-01-01") -> pd.DataFrame:
    """Random-walk OHLCV bars, hourly, UTC."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
        "volume": rng.lognormal(3, 0.5, n), "symbol": "SYN/USD",
    })


@pytest.fixture
def data() -> pd.DataFrame:
    return make_hourly_data()  # 2020-01-01 .. 2020-05-29


@pytest.fixture
def settings() -> ExperimentSettings:
    return ExperimentSettings(
        train=Period("2020-01-01", "2020-03-15"),
        validation=Period("2020-03-15", "2020-05-01"),
        transaction_cost=0.001, top_n=3, min_train_trades=1,
    )
