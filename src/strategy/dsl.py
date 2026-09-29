"""Strategy specification language shared by the LLM, the Python reference engine and the CUDA engine.

A strategy spec is JSON:

    {"name": "weekly_trend",
     "hypothesis": "BTC trends over one week",
     "signal": {"op": "pct_change", "arg": {"op": "field", "name": "close"}, "periods": 168}}

`signal` is an expression tree whose value at bar t is turned into a position by its sign
(+1 long, -1 short, 0 or NaN flat). Every operator only looks at bars <= t, so a valid spec
cannot contain look-ahead.

Semantics (the CUDA engine in cuda/backtest.cu must match these exactly):
- Any non-finite result (inf, NaN) becomes NaN, and NaN propagates through arithmetic.
- div by zero -> NaN; log of x <= 0 -> NaN; gt/lt with a NaN operand -> NaN (else 1.0/0.0).
- lag/diff/pct_change look back `periods` bars; NaN during warm-up.
- sma/std/zscore/max/min are trailing windows of `window` bars; NaN if any value in the window is NaN.
- std is the sample standard deviation (ddof=1); zscore = (x - sma) / std with std == 0 -> NaN.
- ema: alpha = 2 / (window + 1), seeded with the first non-NaN value; a NaN input outputs NaN
  and leaves the state unchanged.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")  # register order in compiled programs
MAX_LOOKBACK = 24 * 90  # 90 days of hourly bars
MAX_NODES = 40
MAX_DEPTH = 8

# Opcodes shared with cuda/backtest.cu (enum Op). tests/test_strategy.py checks they match.
OPCODES = {
    "const": 1, "add": 2, "sub": 3, "mul": 4, "div": 5,
    "neg": 6, "abs": 7, "sign": 8, "log": 9, "gt": 10, "lt": 11,
    "lag": 12, "diff": 13, "pct_change": 14,
    "sma": 15, "ema": 16, "std": 17, "zscore": 18, "max": 19, "min": 20,
}
BINARY_OPS = {"add", "sub", "mul", "div", "gt", "lt"}
UNARY_OPS = {"neg", "abs", "sign", "log"}
PERIOD_OPS = {"lag", "diff", "pct_change"}
WINDOW_OPS = {"sma", "ema", "std", "zscore", "max", "min"}

# Short reference used in the LLM prompt.
OPS_REFERENCE = """\
Leaves:
  {"op": "field", "name": "open"|"high"|"low"|"close"|"volume"}
  {"op": "const", "value": <number>}
Binary (keys "left", "right"): add, sub, mul, div, gt, lt   (gt/lt return 1 or 0)
Unary (key "arg"): neg, abs, sign, log
Lookback (keys "arg", "periods"): lag, diff, pct_change      (value `periods` bars ago)
Rolling window (keys "arg", "window"): sma, ema, std, zscore, max, min
periods/window are integers in hours, 1..2160 (std/zscore need window >= 2)."""


class SpecError(ValueError):
    """Raised when a strategy spec is malformed."""


# ---------------------------------------------------------------- validation


def validate_spec(spec: Any) -> dict[str, Any]:
    """Check a full strategy spec and return it; raise SpecError with a readable reason otherwise."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    missing = {"name", "hypothesis", "signal"} - set(spec)
    if missing:
        raise SpecError(f"spec is missing keys: {sorted(missing)}")
    if not isinstance(spec["name"], str) or not isinstance(spec["hypothesis"], str):
        raise SpecError("name and hypothesis must be strings")
    count = _validate_node(spec["signal"], depth=1, path="signal")
    if count > MAX_NODES:
        raise SpecError(f"signal has {count} nodes; the maximum is {MAX_NODES}")
    return spec


def _validate_node(node: Any, depth: int, path: str) -> int:
    """Validate one expression node recursively; return the number of nodes in its subtree."""
    if depth > MAX_DEPTH:
        raise SpecError(f"{path}: expression deeper than {MAX_DEPTH} levels")
    if not isinstance(node, dict) or "op" not in node:
        raise SpecError(f"{path}: every node must be an object with an 'op' key")
    op = node["op"]

    def expect_keys(*keys: str) -> None:
        extra = set(node) - {"op", *keys}
        absent = set(keys) - set(node)
        if absent or extra:
            raise SpecError(f"{path}: '{op}' takes keys {sorted(keys)}, got {sorted(set(node) - {'op'})}")

    def expect_int(key: str, minimum: int) -> None:
        value = node[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_LOOKBACK:
            raise SpecError(f"{path}.{key}: must be an integer in {minimum}..{MAX_LOOKBACK}, got {value!r}")

    if op == "field":
        expect_keys("name")
        if node["name"] not in FIELDS:
            raise SpecError(f"{path}: field name must be one of {list(FIELDS)}")
        return 1
    if op == "const":
        expect_keys("value")
        value = node["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise SpecError(f"{path}: const value must be a finite number")
        return 1
    if op in BINARY_OPS:
        expect_keys("left", "right")
        return 1 + _validate_node(node["left"], depth + 1, f"{path}.left") + _validate_node(
            node["right"], depth + 1, f"{path}.right"
        )
    if op in UNARY_OPS:
        expect_keys("arg")
        return 1 + _validate_node(node["arg"], depth + 1, f"{path}.arg")
    if op in PERIOD_OPS:
        expect_keys("arg", "periods")
        expect_int("periods", 1)
        return 1 + _validate_node(node["arg"], depth + 1, f"{path}.arg")
    if op in WINDOW_OPS:
        expect_keys("arg", "window")
        expect_int("window", 2 if op in {"std", "zscore"} else 1)
        return 1 + _validate_node(node["arg"], depth + 1, f"{path}.arg")
    raise SpecError(f"{path}: unknown op {op!r}")


def canonical_signal(spec: dict[str, Any]) -> str:
    """Stable string form of the signal expression, used to detect duplicate strategies."""
    return json.dumps(spec["signal"], sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------- Python reference evaluation


def evaluate_signal(spec: dict[str, Any], data: pd.DataFrame) -> pd.Series:
    """Evaluate a validated spec's signal on OHLCV data; returns a float Series aligned with `data`."""
    values = _eval(spec["signal"], {name: data[name].to_numpy(dtype="float64") for name in FIELDS})
    return pd.Series(values, index=data.index, name="signal")


def _finite(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nan)


def _shift(x: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if periods < len(x):
        out[periods:] = x[: len(x) - periods]
    return out


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _finite(np.where(b == 0, np.nan, a / np.where(b == 0, 1.0, b)))


def _ema(x: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1)
    out = np.full_like(x, np.nan)
    state = np.nan
    for t, value in enumerate(x):
        if np.isnan(value):
            continue
        state = value if np.isnan(state) else alpha * value + (1 - alpha) * state
        out[t] = state
    return out


def _eval(node: dict[str, Any], fields: dict[str, np.ndarray]) -> np.ndarray:
    op = node["op"]
    n = len(fields["close"])
    if op == "field":
        return fields[node["name"]]
    if op == "const":
        return np.full(n, float(node["value"]))
    if op in BINARY_OPS:
        a, b = _eval(node["left"], fields), _eval(node["right"], fields)
        if op == "div":
            return _div(a, b)
        if op in {"gt", "lt"}:
            result = (a > b) if op == "gt" else (a < b)
            return np.where(np.isnan(a) | np.isnan(b), np.nan, result.astype("float64"))
        return _finite({"add": np.add, "sub": np.subtract, "mul": np.multiply}[op](a, b))

    x = _eval(node["arg"], fields)
    if op == "neg":
        return -x
    if op == "abs":
        return np.abs(x)
    if op == "sign":
        return np.sign(x)
    if op == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return _finite(np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan))
    if op == "lag":
        return _shift(x, node["periods"])
    if op == "diff":
        return _finite(x - _shift(x, node["periods"]))
    if op == "pct_change":
        return _finite(_div(x, _shift(x, node["periods"])) - 1)

    window = node["window"]
    if op == "ema":
        return _ema(x, window)
    rolling = pd.Series(x).rolling(window, min_periods=window)
    if op == "sma":
        return _finite(rolling.mean().to_numpy())
    if op == "std":
        return _finite(rolling.std(ddof=1).to_numpy())
    if op == "max":
        return rolling.max().to_numpy()
    if op == "min":
        return rolling.min().to_numpy()
    if op == "zscore":
        return _div(x - rolling.mean().to_numpy(), rolling.std(ddof=1).to_numpy())
    raise SpecError(f"unknown op {op!r}")


# ---------------------------------------------------------------- compilation for the CUDA engine


def compile_spec(spec: dict[str, Any]) -> tuple[list[tuple[int, int, int, int, int, float]], int]:
    """Flatten a validated spec into register instructions for cuda/backtest.cu.

    Registers 0-4 hold the input fields (FIELDS order). Each instruction is
    (opcode, dst, a, b, int_param, float_param) and writes a new register. Returns
    (instructions, output_register).
    """
    instructions: list[tuple[int, int, int, int, int, float]] = []

    def emit(node: dict[str, Any]) -> int:
        op = node["op"]
        if op == "field":
            return FIELDS.index(node["name"])
        a = b = -1
        int_param, float_param = 0, 0.0
        if op == "const":
            float_param = float(node["value"])
        elif op in BINARY_OPS:
            a, b = emit(node["left"]), emit(node["right"])
        else:
            a = emit(node["arg"])
            int_param = node.get("periods", node.get("window", 0))
        dst = len(FIELDS) + len(instructions)
        instructions.append((OPCODES[op], dst, a, b, int_param, float_param))
        return dst

    output = emit(spec["signal"])
    return instructions, output
