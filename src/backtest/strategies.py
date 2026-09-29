"""Signal functions: each takes an OHLCV DataFrame and returns a numeric signal aligned with it.

Signals at bar t may only use data up to and including bar t; the engine handles the shift.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

SignalFn = Callable[..., pd.Series]


def momentum_signal(data: pd.DataFrame, lookback: int = 24) -> pd.Series:
    """Sign of the trailing `lookback`-bar return: +1 after a rise, -1 after a fall, NaN during warm-up."""
    close = data["close"]
    return np.sign(close / close.shift(lookback) - 1)


# Name -> signal function; later, LLM-generated strategy specs will be resolved through this.
STRATEGIES: dict[str, SignalFn] = {"momentum": momentum_signal}
