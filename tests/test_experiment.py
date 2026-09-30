"""Train/validation experiments and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.benchmarks import evaluate_benchmarks, load_benchmarks
from src.research.experiment import ExperimentSettings, Period, run_experiment, select_top
from src.research.summary import robustness_warnings, summarize_costs
from src.strategies.evaluator import evaluate_strategy
from src.strategies.schema import StrategySpec
from src.strategies.sweep import evaluate_candidates

BASE = StrategySpec.from_dict({
    "name": "mom_lowvol",
    "conditions": [
        {"id": "mom", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
        {"id": "vol", "feature": "volatility", "field": "close", "lookback": 24, "operator": "<", "threshold": 0.02},
    ],
})
SPACE = {"mom.lookback": [6, 24, 72], "mom.threshold": [0.0, 0.01]}


def test_benchmarks_load_and_evaluate(data: pd.DataFrame) -> None:
    benchmarks = load_benchmarks()
    assert {"buy_and_hold", "flat", "naive_momentum_24h"} <= set(benchmarks)
    table = evaluate_benchmarks(data, benchmarks, Period("2020-02-01", "2020-03-01"), cost_bps=10).set_index("benchmark")

    window = data[(data.timestamp >= "2020-02-01") & (data.timestamp < "2020-03-01")]
    returns = window["close"].pct_change().fillna(0.0).to_numpy(copy=True)
    returns[1] -= 0.001  # buy-and-hold pays one entry cost when the position is first held
    assert table.loc["buy_and_hold", "cumulative_return"] == pytest.approx(np.prod(1 + returns) - 1)
    assert table.loc["buy_and_hold", "n_trades"] == 1
    assert table.loc["flat", "cumulative_return"] == 0 and table.loc["flat", "n_trades"] == 0
    assert np.isnan(table.loc["flat", "sharpe"])


def test_periods_are_respected(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    result = run_experiment(data, BASE, SPACE, settings, load_benchmarks())
    assert (result.train["start"] == pd.Timestamp("2020-01-01", tz="UTC")).all()
    assert (result.train["end"] == pd.Timestamp("2020-03-14 23:00", tz="UTC")).all()
    assert (result.validation["start"] == pd.Timestamp("2020-03-15", tz="UTC")).all()
    assert (result.validation["end"] == pd.Timestamp("2020-04-30 23:00", tz="UTC")).all()
    assert result.n_candidates == len(result.train) == 6 and len(result.validation) == 3


def test_select_top_ranks_on_train_only() -> None:
    train = pd.DataFrame({
        "candidate": [0, 1, 2, 3, 4],
        "sharpe": [0.5, 1.2, np.nan, 1.2, 2.0],
        "n_trades": [50, 50, 50, 60, 3],
        "cumulative_return": [0.1, 0.3, 0.2, 0.4, 0.9],
    })
    assert select_top(train, "sharpe", 3) == [4, 1, 3]
    assert select_top(train, "sharpe", 3, min_trades=10) == [1, 3, 0]  # tie -> lower id first; NaN excluded


def test_select_top_skips_candidates_that_traded_identically() -> None:
    train = pd.DataFrame({
        "candidate": [0, 1, 2, 3],
        "sharpe": [1.0, 1.0, 0.8, 0.5],
        "n_trades": [40, 40, 30, 20],
        "turnover": [40.0, 40.0, 30.0, 20.0],
        "exposure": [0.3, 0.3, 0.2, 0.1],
        "cumulative_return": [0.5, 0.5, 0.4, 0.2],
    })
    assert select_top(train, "sharpe", 3) == [0, 2, 3]  # 1 is a copy of 0
    assert select_top(train, "sharpe", 3, distinct=False) == [0, 1, 2]


def test_validation_does_not_influence_selection(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    altered = data.copy()
    in_validation = (altered.timestamp >= "2020-03-15") & (altered.timestamp < "2020-05-01")
    altered.loc[in_validation, "close"] = altered.loc[in_validation, "close"].to_numpy()[::-1]  # different future

    a = run_experiment(data, BASE, SPACE, settings)
    b = run_experiment(altered, BASE, SPACE, settings)
    assert a.selected == b.selected
    pd.testing.assert_frame_equal(a.train, b.train)
    assert not a.validation["sharpe"].equals(b.validation["sharpe"])


def test_selected_parameters_are_frozen(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    result = run_experiment(data, BASE, SPACE, settings)
    first = result.comparison.iloc[0]
    params = {name: first[name] for name in SPACE}
    spec = BASE.with_parameters({"mom.lookback": int(params["mom.lookback"]), "mom.threshold": params["mom.threshold"]})
    direct = evaluate_strategy(data, spec, settings.cost_bps, start="2020-03-15", end="2020-05-01").metrics
    assert first["validation_sharpe"] == pytest.approx(direct["sharpe"])
    assert first["train_sharpe"] == result.train["sharpe"].max()


def test_transaction_cost_reaches_every_evaluation(data: pd.DataFrame) -> None:
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs["cost_bps"])
        return evaluate_candidates(*args, **kwargs)

    settings = ExperimentSettings(Period("2020-01-01", "2020-03-15"), Period("2020-03-15", "2020-05-01"),
                                  transaction_cost=0.002, top_n=2, min_train_trades=1)
    result = run_experiment(data, BASE, SPACE, settings, load_benchmarks(), evaluator=spy)
    assert calls and set(calls) == {20.0}  # train, validation and both benchmark periods
    free = run_experiment(data, BASE, SPACE, ExperimentSettings(settings.train, settings.validation, 0.0,
                                                                top_n=2, min_train_trades=1))
    assert (result.train["cumulative_return"] < free.train["cumulative_return"]).all()


def test_benchmarks_use_exactly_the_same_periods(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    result = run_experiment(data, BASE, SPACE, settings, load_benchmarks())
    for period, candidates in (("train", result.train), ("validation", result.validation)):
        bench = result.benchmarks[period]
        for column in ("start", "end", "n_bars"):
            assert set(bench[column]) == set(candidates[column])


def test_data_after_validation_end_is_never_used(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    corrupted = data.copy()
    corrupted.loc[corrupted.timestamp >= "2020-05-01", ["close", "volume"]] = [np.nan, -1.0]
    seen_ends = []

    def spy(data, *args, **kwargs):
        seen_ends.append(data["timestamp"].max())
        return evaluate_candidates(data, *args, **kwargs)

    clean = run_experiment(data, BASE, SPACE, settings)
    dirty = run_experiment(corrupted, BASE, SPACE, settings, evaluator=spy)
    pd.testing.assert_frame_equal(clean.validation, dirty.validation)
    assert max(seen_ends) < pd.Timestamp("2020-05-01", tz="UTC")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(train=Period("2020-01-01", "2020-04-01"), validation=Period("2020-03-01", "2020-05-01")),  # overlap
        dict(train=Period("2020-01-01", "2020-03-01"), validation=Period("2020-03-01", "2020-05-01"), transaction_cost=-0.001),
        dict(train=Period("2020-01-01", "2020-03-01"), validation=Period("2020-03-01", "2020-05-01"), selection_metric="max_drawdown"),
        dict(train=Period("2020-01-01", None), validation=Period("2020-03-01", "2020-05-01")),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExperimentSettings(**kwargs)


def test_warnings_flag_multiple_testing_and_identical_candidates(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    # vol.threshold values far above any hourly volatility: the filter never binds.
    space = {"mom.lookback": [24, 48], "vol.threshold": [0.5, 0.6, 0.7]}
    result = run_experiment(data, BASE, space, settings, load_benchmarks())
    warnings = robustness_warnings(result, space)
    assert any("multiple testing" in w for w in warnings)
    assert any("traded exactly like another combination" in w for w in warnings)
    assert len(result.selected) == 2  # 6 combinations but only 2 distinct strategies (one per mom.lookback)
    assert any("buy-and-hold" in w for w in warnings)


def test_cli_transaction_cost_arguments() -> None:
    from src.strategies.run import parse_args

    base = ["spec.json", "--data", "data.parquet"]
    assert parse_args(base).transaction_cost == 0.001  # explicit non-zero default
    assert parse_args([*base, "--transaction-cost", "0.0005"]).transaction_cost == 0.0005
    assert parse_args([*base, "--cost-bps", "20"]).transaction_cost == pytest.approx(0.002)
    with pytest.raises(SystemExit):
        parse_args([*base, "--cost-bps", "20", "--transaction-cost", "0.002"])


def test_cost_summary_and_warning_for_a_churning_strategy(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    churn = StrategySpec.from_dict({"name": "churn", "conditions": [
        {"id": "r", "feature": "returns", "field": "close", "operator": ">", "threshold": 0.0}]})
    result = run_experiment(data, churn, {"r.threshold": [0.0, 0.001]}, settings)
    costs = summarize_costs(result)
    assert costs["median_bars_between_trades"] < 5  # flips roughly every other bar
    assert costs["median_trades_per_year"] > 1000
    assert costs["median_annual_cost"] > 1.0  # more than 100% of capital per year in fees
    assert any(w.startswith("Trading costs") and "costs exceed the net return" in w
               for w in robustness_warnings(result, {"r.threshold": [0.0, 0.001]}))
