"""Tests for the strategy spec language: validation, Python evaluation semantics and compilation."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.strategy.dsl import FIELDS, OPCODES, SpecError, canonical_signal, compile_spec, evaluate_signal, validate_spec

CLOSE = {"op": "field", "name": "close"}


def spec(signal: dict) -> dict:
    return {"name": "s", "hypothesis": "h", "signal": signal}


def make_data(closes: list[float]) -> pd.DataFrame:
    close = np.asarray(closes, dtype="float64")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": np.ones(len(close))})


def run(signal: dict, closes: list[float]) -> list[float]:
    return evaluate_signal(validate_spec(spec(signal)), make_data(closes)).tolist()


@pytest.mark.parametrize(
    "bad, reason",
    [
        ({"op": "field", "name": "vwap"}, "field name"),
        ({"op": "sma", "arg": CLOSE, "window": 0}, "integer"),
        ({"op": "sma", "arg": CLOSE, "window": 5000}, "integer"),
        ({"op": "sma", "arg": CLOSE, "window": 2.5}, "integer"),
        ({"op": "std", "arg": CLOSE, "window": 1}, "integer"),
        ({"op": "lead", "arg": CLOSE, "periods": 1}, "unknown op"),
        ({"op": "lag", "arg": CLOSE, "periods": -1}, "integer"),
        ({"op": "add", "left": CLOSE}, "takes keys"),
        ({"op": "const", "value": float("inf")}, "finite"),
        ({"op": "sma", "arg": CLOSE, "window": 3, "extra": 1}, "takes keys"),
    ],
)
def test_validation_rejects_bad_nodes(bad: dict, reason: str) -> None:
    with pytest.raises(SpecError, match=reason):
        validate_spec(spec(bad))


def test_validation_rejects_missing_keys_and_deep_trees() -> None:
    with pytest.raises(SpecError, match="missing"):
        validate_spec({"signal": CLOSE})
    deep = CLOSE
    for _ in range(10):
        deep = {"op": "neg", "arg": deep}
    with pytest.raises(SpecError, match="deeper"):
        validate_spec(spec(deep))


def test_operator_semantics() -> None:
    closes = [1.0, 2.0, 4.0, 8.0]
    assert run({"op": "pct_change", "arg": CLOSE, "periods": 1}, closes)[1:] == [1.0, 1.0, 1.0]
    assert np.isnan(run({"op": "lag", "arg": CLOSE, "periods": 2}, closes)[:2]).all()
    assert run({"op": "sma", "arg": CLOSE, "window": 2}, closes)[1:] == [1.5, 3.0, 6.0]
    assert run({"op": "max", "arg": CLOSE, "window": 3}, closes)[2:] == [4.0, 8.0]
    assert run({"op": "ema", "arg": CLOSE, "window": 3}, closes) == [1.0, 1.5, 2.75, 5.375]
    assert run({"op": "std", "arg": CLOSE, "window": 2}, closes)[1] == pytest.approx(np.std([1, 2], ddof=1))
    assert run({"op": "gt", "left": CLOSE, "right": {"op": "const", "value": 3}}, closes) == [0, 0, 1, 1]


def test_nan_and_division_rules() -> None:
    zero = {"op": "const", "value": 0}
    assert np.isnan(run({"op": "div", "left": CLOSE, "right": zero}, [1.0, 2.0])).all()
    assert np.isnan(run({"op": "log", "arg": {"op": "neg", "arg": CLOSE}}, [1.0, 2.0])).all()
    # A NaN inside a rolling window makes the window NaN; gt with NaN is NaN, not 0.
    values = run({"op": "sma", "arg": CLOSE, "window": 2}, [1.0, np.nan, 3.0, 4.0])
    assert np.isnan(values[:3]).all() and values[3] == 3.5
    assert np.isnan(run({"op": "gt", "left": CLOSE, "right": zero}, [np.nan])[0])
    # EMA skips NaN inputs without resetting.
    assert run({"op": "ema", "arg": CLOSE, "window": 1}, [1.0, np.nan, 3.0])[2] == 3.0


def test_signals_never_use_future_data() -> None:
    signal = {"op": "zscore", "arg": {"op": "ema", "arg": {"op": "pct_change", "arg": CLOSE, "periods": 3}, "window": 5}, "window": 10}
    rng = np.random.default_rng(1)
    closes = list(100 + rng.normal(0, 1, 60).cumsum())
    altered = closes[:40] + [c * 2 for c in closes[40:]]
    np.testing.assert_array_equal(run(signal, closes)[:40], run(signal, altered)[:40])


def test_compile_spec_emits_register_program() -> None:
    s = validate_spec(spec({"op": "sub", "left": CLOSE, "right": {"op": "sma", "arg": CLOSE, "window": 24}}))
    instructions, output = compile_spec(s)
    close = FIELDS.index("close")
    assert instructions == [
        (OPCODES["sma"], 5, close, -1, 24, 0.0),
        (OPCODES["sub"], 6, close, 5, 0, 0.0),
    ]
    assert output == 6
    assert compile_spec(validate_spec(spec(CLOSE))) == ([], close)


def test_canonical_signal_ignores_key_order_and_name() -> None:
    a = spec({"op": "sma", "arg": CLOSE, "window": 3})
    b = copy.deepcopy(a)
    b["name"] = "other"
    b["signal"] = {"window": 3, "arg": CLOSE, "op": "sma"}
    assert canonical_signal(a) == canonical_signal(b)
