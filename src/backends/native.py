"""Native (C++ CPU / CUDA GPU) candidate evaluator, loaded from cuda/build via ctypes.

NativeEvaluator has the same contract as src.strategies.sweep.evaluate_candidates, so it can be
passed anywhere an `evaluator=` is accepted (experiments, benchmarks, the research loop).

What runs where:
  Python  compiles the candidates into flat arrays (distinct feature buffers, conditions,
          positions) and computes the period returns with the same function as the Python engine.
  native  computes each feature buffer once, then backtests every candidate and returns only the
          metrics table (no per-bar series cross back).

Used by: backends/__init__.py (get_evaluator('cpp' | 'cuda')). Loads the libraries that
         cuda/Makefile builds from cuda/backtest.cu.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import DEFAULT_COST_BPS, period_returns
from src.backtest.metrics import HOURS_PER_YEAR, METRIC_NAMES
from src.strategies.evaluator import period_slice
from src.strategies.schema import StrategySpec, validate_strategy
from src.strategies.sweep import Candidate, evaluate_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "cuda" / "build"
LIBRARIES = {"cpp": "libbacktest_cpu.so", "cuda": "libbacktest_cuda.so"}

# Must match the enums in cuda/backtest.cu (checked by tests/test_native_backend.py).
FEATURE_CODES = {"returns": 1, "momentum": 2, "rolling_mean": 3, "rolling_std": 4, "volatility": 5,
                 "zscore": 6, "rolling_min": 7, "rolling_max": 8, "volume_change": 9,
                 "distance_to_max": 10, "distance_to_min": 11, "distance_to_mean": 12}
OPERATOR_CODES = {">": 1, ">=": 2, "<": 3, "<=": 4}
LOGIC_CODES = {"AND": 1, "OR": 2}

_F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class NativeEvaluator:
    """Callable backend backed by libbacktest_{cpu,cuda}.so."""

    def __init__(self, kind: str, device: int = 0, path: str | Path | None = None) -> None:
        if kind not in LIBRARIES:
            raise ValueError(f"kind must be one of {sorted(LIBRARIES)}")
        path = Path(path) if path else BUILD_DIR / LIBRARIES[kind]
        if not path.exists():
            target = "make -C cuda" if kind == "cuda" else "make -C cuda cpu"
            raise FileNotFoundError(f"{path} not found; build it with `{target}`")
        self.kind, self.device, self.path = kind, device, path
        self.lib = ctypes.CDLL(str(path))
        self.lib.bt_run.restype = ctypes.c_int
        self.lib.bt_run.argtypes = [
            _F64, ctypes.c_int, ctypes.c_longlong,              # fields, n_fields, n_rows
            _I32, _I32, _I32, ctypes.c_int,                     # buffers: feature, field, lookback, n
            _I32, _I32, _I32, _F64, _F64, ctypes.c_int,         # candidates: offset, ncond, logic, true, false, n
            _I32, _I32, _F64, ctypes.c_int,                     # conditions: buffer, op, threshold, n
            _F64, ctypes.c_longlong, ctypes.c_longlong,         # returns, start, end
            ctypes.c_double, ctypes.c_double, ctypes.c_int,     # cost_bps, periods_per_year, device
            _F64, ctypes.c_char_p, ctypes.c_int,                # out, error, error_len
        ]
        self.lib.bt_info.restype = ctypes.c_int
        self.lib.bt_info.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        info = ctypes.create_string_buffer(256)
        if self.lib.bt_info(device, info, len(info)) != 0:
            raise RuntimeError(info.value.decode())
        self.info = info.value.decode()
        self.__name__ = f"cuda:{device}" if kind == "cuda" else "cpp"  # label recorded in experiment timing

    def __call__(
        self,
        data: pd.DataFrame,
        candidates: Sequence[Candidate],
        cost_bps: float = DEFAULT_COST_BPS,
        periods_per_year: float = HOURS_PER_YEAR,
        start: Any = None,
        end: Any = None,
        dataset: str = "",
        ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Same inputs and output table as evaluate_candidates."""
        ids = list(range(len(candidates))) if ids is None else list(ids)
        rows = period_slice(data, start, end)
        if rows.stop - rows.start < 1 or not candidates:  # degenerate inputs: keep the reference behaviour
            return evaluate_candidates(data, candidates, cost_bps, periods_per_year, start, end, dataset, ids)

        history = data.iloc[: rows.stop]  # features only use bars <= t, so later bars are not needed
        compiled = compile_candidates(history, [spec for _, spec in candidates])
        returns = np.ascontiguousarray(period_returns(data["close"].iloc[rows]).to_numpy(), dtype=np.float64)
        out = np.empty((len(candidates), len(METRIC_NAMES)), dtype=np.float64)
        error = ctypes.create_string_buffer(512)
        status = self.lib.bt_run(
            compiled["fields"], compiled["n_fields"], len(history),
            compiled["buf_feature"], compiled["buf_field"], compiled["buf_lookback"], len(compiled["buf_feature"]),
            compiled["cand_offset"], compiled["cand_ncond"], compiled["cand_logic"],
            compiled["cand_true"], compiled["cand_false"], len(candidates),
            compiled["cond_buffer"], compiled["cond_op"], compiled["cond_threshold"], len(compiled["cond_buffer"]),
            returns, rows.start, rows.stop, float(cost_bps), float(periods_per_year), self.device,
            out, error, len(error),
        )
        if status != 0:
            raise RuntimeError(f"{self.__name__} backtest failed: {error.value.decode()}")

        timestamps = data["timestamp"].iloc[rows]
        table = pd.DataFrame({"candidate": ids, "dataset": dataset,
                              "start": timestamps.iloc[0], "end": timestamps.iloc[-1]})
        params = pd.DataFrame([params for params, _ in candidates], index=table.index)
        metrics = pd.DataFrame(out, columns=list(METRIC_NAMES), index=table.index)
        metrics[["n_trades", "n_bars"]] = metrics[["n_trades", "n_bars"]].astype("int64")
        return pd.concat([table, params, metrics], axis=1)


def compile_candidates(history: pd.DataFrame, specs: Sequence[StrategySpec]) -> dict[str, Any]:
    """Flatten specs into arrays: each distinct (feature, field, lookback) becomes one buffer."""
    fields: dict[str, int] = {}
    buffers: dict[tuple[str, str, int | None], int] = {}
    arrays: dict[str, list] = {name: [] for name in (
        "buf_feature", "buf_field", "buf_lookback", "cand_offset", "cand_ncond", "cand_logic",
        "cand_true", "cand_false", "cond_buffer", "cond_op", "cond_threshold")}
    for spec in specs:
        validate_strategy(spec)
        arrays["cand_offset"].append(len(arrays["cond_buffer"]))
        arrays["cand_ncond"].append(len(spec.conditions))
        arrays["cand_logic"].append(LOGIC_CODES[spec.logic])
        arrays["cand_true"].append(float(spec.true_position))
        arrays["cand_false"].append(float(spec.false_position))
        for c in spec.conditions:
            if c.feature not in FEATURE_CODES:
                raise NotImplementedError(f"feature {c.feature!r} has no native implementation yet; "
                                          "use backend 'python' or add it to cuda/backtest.cu")
            field_index = fields.setdefault(c.field, len(fields))
            key = (c.feature, c.field, c.lookback)
            if key not in buffers:
                buffers[key] = len(arrays["buf_feature"])
                arrays["buf_feature"].append(FEATURE_CODES[c.feature])
                arrays["buf_field"].append(field_index)
                arrays["buf_lookback"].append(c.lookback or 0)
            arrays["cond_buffer"].append(buffers[key])
            arrays["cond_op"].append(OPERATOR_CODES[c.operator])
            arrays["cond_threshold"].append(float(c.threshold))

    compiled: dict[str, Any] = {
        name: np.ascontiguousarray(values, dtype=np.float64 if name in ("cand_true", "cand_false", "cond_threshold")
                                   else np.int32)
        for name, values in arrays.items()
    }
    series = [history[name].to_numpy(dtype=np.float64) for name in fields] or [np.zeros(len(history))]
    compiled["fields"] = np.ascontiguousarray(np.stack(series))
    compiled["n_fields"] = len(fields)
    return compiled
