"""Parameter sweeps: Cartesian product, parameter insertion, determinism, limits, summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.schema import SpecError, StrategySpec
from src.strategies.sweep import count_candidates, generate_candidates, parameter_grid, run_sweep, summarize_sweep

BASE = StrategySpec.from_dict({
    "name": "mom_lowvol",
    "conditions": [
        {"id": "momentum", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
        {"id": "vol", "feature": "volatility", "field": "close", "lookback": 12, "operator": "<", "threshold": 0.05},
    ],
})
SPACE = {"momentum.lookback": [6, 12, 24], "momentum.threshold": [0.0, 0.01], "vol.lookback": [12, 48]}


def make_data(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close, "volume": rng.lognormal(3, 1, n),
    })


def test_cartesian_product() -> None:
    grid = parameter_grid(SPACE)
    assert count_candidates(SPACE) == len(grid) == 12
    assert len({tuple(p.items()) for p in grid}) == 12
    assert grid[0] == {"momentum.lookback": 6, "momentum.threshold": 0.0, "vol.lookback": 12}
    assert grid[-1] == {"momentum.lookback": 24, "momentum.threshold": 0.01, "vol.lookback": 48}


def test_parameters_are_inserted_into_specs() -> None:
    for params, spec in generate_candidates(BASE, SPACE):
        current = spec.parameters()
        assert all(current[name] == value for name, value in params.items())
        assert current["vol.threshold"] == 0.05  # unswept parameters keep the base value


def test_candidate_limit_and_invalid_values() -> None:
    with pytest.raises(SpecError, match="limit is 10"):
        generate_candidates(BASE, SPACE, max_candidates=10)
    with pytest.raises(SpecError, match="non-empty"):
        generate_candidates(BASE, {"momentum.lookback": []})
    with pytest.raises(SpecError):
        generate_candidates(BASE, {"vol.lookback": [12, 1]})  # volatility needs lookback >= 2
    with pytest.raises(SpecError, match="unknown parameter"):
        generate_candidates(BASE, {"rsi.lookback": [14]})


def test_sweep_results_are_deterministic_and_complete() -> None:
    data = make_data()
    first = run_sweep(data, BASE, SPACE, cost_bps=10, dataset="SYN")
    second = run_sweep(data, BASE, SPACE, cost_bps=10, dataset="SYN")
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12 and first["candidate"].tolist() == list(range(12))
    assert set(SPACE) <= set(first.columns) and {"sharpe", "max_drawdown", "turnover", "n_trades"} <= set(first.columns)
    assert (first["dataset"] == "SYN").all()


def test_sweep_row_matches_single_evaluation() -> None:
    from src.strategies.evaluator import evaluate_strategy

    data = make_data()
    results = run_sweep(data, BASE, SPACE, cost_bps=10)
    params, spec = generate_candidates(BASE, SPACE)[5]
    direct = evaluate_strategy(data, spec, cost_bps=10).metrics
    assert results.loc[5, "sharpe"] == pytest.approx(direct["sharpe"])


def test_summary_is_compact() -> None:
    results = run_sweep(make_data(), BASE, SPACE, cost_bps=10)
    summary = summarize_sweep(results, top=3)
    assert summary["n_candidates"] == 12 and len(summary["top"]) == 3
    assert set(summary["parameter_sensitivity"]) == set(SPACE)
    assert set(summary["parameter_sensitivity"]["momentum.lookback"]) == {"6", "12", "24"}
    tops = [row["sharpe"] for row in summary["top"]]
    assert tops == sorted(tops, reverse=True)
