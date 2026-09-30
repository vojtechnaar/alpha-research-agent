"""Native (C++ / CUDA) backend: code tables in sync, and results identical to the Python engine.

Parity tests run for every native library that is built (`make -C cuda cpu` for C++, `make -C cuda`
for CUDA on a GPU machine) and are skipped otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backends import get_evaluator
from src.backends.native import FEATURE_CODES, LOGIC_CODES, OPERATOR_CODES, NativeEvaluator
from src.backtest.engine import run_backtest
from src.backtest.metrics import METRIC_NAMES, compute_metrics
from src.research.benchmarks import load_benchmarks
from src.research.experiment import ExperimentSettings, run_experiment
from src.strategies.features import FEATURE_REGISTRY, FeatureDef, compute_momentum
from src.strategies.operators import LOGIC_REGISTRY, OPERATOR_REGISTRY
from src.strategies.schema import StrategySpec
from src.strategies.sweep import evaluate_candidates

from conftest import make_hourly_data

SOURCE = (Path(__file__).resolve().parents[1] / "cuda" / "backtest.cu").read_text()


def available_backends() -> list[str]:
    kinds = []
    for kind in ("cpp", "cuda"):
        try:
            get_evaluator(kind)
            kinds.append(kind)
        except (FileNotFoundError, RuntimeError, OSError):
            pass
    return kinds


NATIVE = available_backends()
needs_native = pytest.mark.skipif(not NATIVE, reason="no native library built (make -C cuda cpu / make -C cuda)")


# ---------------------------------------------------------------- tables stay in sync


def enum(prefix: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(rf"\b{prefix}_(\w+) = (\d+)", SOURCE)}


def test_code_tables_match_the_cuda_source() -> None:
    assert enum("F") == {name.upper(): code for name, code in FEATURE_CODES.items()}
    assert enum("LOGIC") == {name: code for name, code in LOGIC_CODES.items()}
    op_names = {">": "GT", ">=": "GE", "<": "LT", "<=": "LE"}
    assert enum("OP") == {op_names[op]: code for op, code in OPERATOR_CODES.items()}
    metric_order = re.findall(r"\bM_(\w+),", SOURCE)
    assert [m.lower() for m in metric_order] == list(METRIC_NAMES)


def test_every_registry_primitive_has_a_native_code() -> None:
    assert set(FEATURE_CODES) == set(FEATURE_REGISTRY)
    assert set(OPERATOR_CODES) == set(OPERATOR_REGISTRY)
    assert set(LOGIC_CODES) == set(LOGIC_REGISTRY)


def test_metric_names_match_python_metrics() -> None:
    data = make_hourly_data(200)
    assert tuple(compute_metrics(run_backtest(data, pd.Series(1.0, index=data.index)))) == METRIC_NAMES


def test_missing_library_gives_build_instructions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make -C cuda"):
        NativeEvaluator("cuda", path=tmp_path / "missing.so")


# ---------------------------------------------------------------- parity with Python


def parity_data() -> pd.DataFrame:
    data = make_hourly_data(24 * 60, seed=5)
    data.loc[100:103, "close"] = np.nan  # price gap
    data.loc[400:407, "close"] = data.loc[400, "close"]  # flat prices: std exactly 0, zscore undefined
    data.loc[600:605, "volume"] = 0.0  # zero volume: division by zero in volume_change
    data.loc[900, "volume"] = np.nan
    return data


THRESHOLDS = {  # feature -> (field, lookbacks, thresholds)
    "returns": ("close", [None], [0.0, 0.004]),
    "momentum": ("close", [1, 24, 100], [0.0, 0.02]),
    "rolling_mean": ("volume", [2, 24], [20.0]),
    "rolling_std": ("close", [2, 48], [0.5]),
    "volatility": ("close", [2, 24], [0.008, 0.01]),
    "zscore": ("close", [3, 72], [-1.0, 0.5]),
    "rolling_min": ("close", [1, 24], [95.0]),
    "rolling_max": ("close", [12], [105.0]),
    "volume_change": ("volume", [1, 24], [0.0, 0.3]),
    "distance_to_max": ("close", [1, 24], [-0.02, 0.0]),
    "distance_to_min": ("close", [24, 100], [0.0, 0.02]),
    "distance_to_mean": ("close", [2, 48], [-0.01, 0.01]),
}


def parity_specs() -> list[StrategySpec]:
    specs = []
    for feature, (field, lookbacks, thresholds) in THRESHOLDS.items():
        for lookback in lookbacks:
            for threshold in thresholds:
                for op in OPERATOR_CODES:
                    cond = {"feature": feature, "field": field, "operator": op, "threshold": threshold}
                    if lookback is not None:
                        cond["lookback"] = lookback
                    specs.append(StrategySpec.from_dict({"name": "s", "conditions": [cond]}))
    two = [
        {"id": "a", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
        {"id": "b", "feature": "volatility", "field": "close", "lookback": 24, "operator": "<", "threshold": 0.01},
        {"id": "c", "feature": "volume_change", "field": "volume", "lookback": 12, "operator": ">=", "threshold": 0.1},
    ]
    for logic in LOGIC_CODES:
        for true, false in ((1, 0), (1, -1), (-1, 0), (-1, 1), (0, 1)):
            specs.append(StrategySpec.from_dict({"name": "combo", "conditions": two, "logic": logic,
                                                 "true_position": true, "false_position": false}))
    specs.extend(load_benchmarks().values())  # no conditions: buy-and-hold, flat
    return specs


def assert_same_results(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    assert list(actual.columns) == list(expected.columns)
    for column in ("candidate", "dataset", "n_trades", "n_bars"):
        assert actual[column].tolist() == expected[column].tolist(), column
    assert (actual["start"] == expected["start"]).all() and (actual["end"] == expected["end"]).all()
    for metric in METRIC_NAMES:
        e, a = expected[metric].to_numpy(float), actual[metric].to_numpy(float)
        assert np.array_equal(np.isnan(e), np.isnan(a)), metric
        np.testing.assert_allclose(a[~np.isnan(a)], e[~np.isnan(e)], rtol=1e-9, atol=1e-12, err_msg=metric)


@needs_native
@pytest.mark.parametrize("kind", NATIVE)
@pytest.mark.parametrize("period", [(None, None), ("2020-01-10", "2020-02-20"), (None, "2020-01-05")])
def test_native_matches_python(kind: str, period: tuple) -> None:
    data = parity_data()
    candidates = [({}, spec) for spec in parity_specs()]
    ids = [100 + i for i in range(len(candidates))]
    expected = evaluate_candidates(data, candidates, 10.0, start=period[0], end=period[1], dataset="SYN", ids=ids)
    actual = get_evaluator(kind)(data, candidates, 10.0, start=period[0], end=period[1], dataset="SYN", ids=ids)
    assert_same_results(expected, actual)


@needs_native
@pytest.mark.parametrize("kind", NATIVE)
def test_native_experiment_matches_python(kind: str, data: pd.DataFrame, settings: ExperimentSettings) -> None:
    base = StrategySpec.from_dict({"name": "m", "conditions": [
        {"id": "m", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
        {"id": "v", "feature": "volatility", "field": "close", "lookback": 24, "operator": "<", "threshold": 0.02}]})
    space = {"m.lookback": [6, 24, 72], "m.threshold": [0.0, 0.01], "v.threshold": [0.008, 0.012]}
    python = run_experiment(data, base, space, settings, load_benchmarks())
    native = run_experiment(data, base, space, settings, load_benchmarks(), evaluator=get_evaluator(kind))
    assert native.selected == python.selected
    assert_same_results(python.train, native.train)
    assert_same_results(python.validation, native.validation)
    assert native.timing["backend"] == get_evaluator(kind).__name__


@needs_native
def test_unimplemented_feature_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(FEATURE_REGISTRY, "new_feature", FeatureDef(compute_momentum))
    spec = StrategySpec.from_dict({"name": "n", "conditions": [
        {"feature": "new_feature", "field": "close", "lookback": 5, "operator": ">", "threshold": 0}]})
    with pytest.raises(NotImplementedError, match="new_feature"):
        get_evaluator(NATIVE[0])(make_hourly_data(200), [({}, spec)])
