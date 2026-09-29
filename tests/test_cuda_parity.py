"""The CUDA engine must reproduce the Python reference engine.

Uses cuda/build/backtest (GPU build, on the server) or cuda/build/backtest_cpu (`make -C cuda cpu`,
same code compiled for the CPU). Skipped if neither is built.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from src.backtest.evaluate import PROJECT_ROOT, Evaluator
from src.strategy.dsl import OPCODES, validate_spec

BUILD = PROJECT_ROOT / "cuda" / "build"
BINARY = next((p for p in (BUILD / "backtest", BUILD / "backtest_cpu") if p.exists()), None)

CLOSE = {"op": "field", "name": "close"}
VOLUME = {"op": "field", "name": "volume"}


def spec(signal: dict) -> dict:
    return validate_spec({"name": "t", "hypothesis": "t", "signal": signal})


# At least one spec per opcode.
SIGNALS = [
    {"op": "pct_change", "arg": CLOSE, "periods": 24},
    {"op": "sub", "left": {"op": "ema", "arg": CLOSE, "window": 12}, "right": {"op": "sma", "arg": CLOSE, "window": 48}},
    {"op": "neg", "arg": {"op": "zscore", "arg": CLOSE, "window": 72}},
    {"op": "sub", "left": CLOSE, "right": {"op": "lag", "arg": {"op": "max", "arg": {"op": "field", "name": "high"}, "window": 48}, "periods": 1}},
    {"op": "sub", "left": {"op": "lag", "arg": {"op": "min", "arg": {"op": "field", "name": "low"}, "window": 24}, "periods": 1}, "right": CLOSE},
    {"op": "mul", "left": {"op": "gt", "left": {"op": "std", "arg": {"op": "pct_change", "arg": CLOSE, "periods": 1}, "window": 24}, "right": {"op": "const", "value": 0.01}}, "right": {"op": "sign", "arg": {"op": "diff", "arg": CLOSE, "periods": 6}}},
    {"op": "div", "left": {"op": "log", "arg": VOLUME}, "right": {"op": "sma", "arg": {"op": "log", "arg": VOLUME}, "window": 24}},
    {"op": "add", "left": {"op": "lt", "left": CLOSE, "right": {"op": "sma", "arg": CLOSE, "window": 100}}, "right": {"op": "const", "value": -0.5}},
    {"op": "abs", "arg": {"op": "pct_change", "arg": {"op": "field", "name": "open"}, "periods": 3}},
    CLOSE,
]


def make_data(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    data = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.lognormal(3, 1, n),
    })
    data.loc[[500, 501, 1700], "close"] = np.nan  # gaps
    data.loc[900, "volume"] = 0.0  # log(0) -> NaN
    return data


def test_every_opcode_is_covered() -> None:
    used = {op for s in SIGNALS for op in re.findall(r"'op': '(\w+)'", str(s))}
    assert set(OPCODES) <= used


def test_opcodes_match_cuda_enum() -> None:
    source = (PROJECT_ROOT / "cuda" / "backtest.cu").read_text()
    enum = {name.lower(): int(value) for name, value in re.findall(r"OP_(\w+) = (\d+)", source)}
    assert enum == OPCODES


@pytest.mark.skipif(BINARY is None, reason="CUDA engine not built (make -C cuda, or make -C cuda cpu)")
@pytest.mark.parametrize("split", ["train", "test"])
def test_cuda_matches_python(split: str) -> None:
    datasets = {"SYN": make_data()}
    splits = {"train": ("2020-01-01", "2020-03-15"), "test": ("2020-03-15", None)}
    specs = [spec(s) for s in SIGNALS]

    python = Evaluator(datasets, splits, cost_bps=10, backend="python").evaluate(specs, split)
    cuda = Evaluator(datasets, splits, cost_bps=10, backend="cuda", cuda_binary=BINARY).evaluate(specs, split)

    for i, (py, cu) in enumerate(zip(python, cuda)):
        for key, expected in py["SYN"].items():
            actual = cu["SYN"][key]
            if math.isnan(expected):
                assert math.isnan(actual), (i, key)
            else:
                assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10), (i, key, SIGNALS[i])
