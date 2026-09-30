"""One train/validation experiment: sweep on TRAIN, freeze the top N, retest them on VALIDATION.

    all candidates --(TRAIN metrics only)--> top N ids --(frozen parameters)--> VALIDATION metrics

Selection uses train results only; validation results are computed afterwards and never feed
back into which parameters are chosen. Benchmarks are evaluated on exactly the same periods,
with the same costs, through the same backend.

Data after the validation end is cut off before anything runs, so a later final-test period is
never touched by an experiment.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.backtest.metrics import HOURS_PER_YEAR
from src.research.benchmarks import evaluate_benchmarks
from src.strategies.evaluator import period_slice
from src.strategies.schema import StrategySpec
from src.strategies.sweep import (
    DEFAULT_MAX_CANDIDATES,
    CandidateEvaluator,
    ParameterSpace,
    evaluate_candidates,
    generate_candidates,
)

SELECTION_METRICS = ("sharpe", "annualized_return", "cumulative_return")  # higher is better
COMPARISON_METRICS = ("sharpe", "annualized_return", "max_drawdown", "n_trades", "exposure", "annual_turnover")
MAX_TRANSACTION_COST = 0.1


@dataclass(frozen=True)
class Period:
    """[start, end) in UTC; None means open-ended."""

    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError(f"period start {self.start} must be before end {self.end}")


@dataclass(frozen=True)
class ExperimentSettings:
    """Everything that defines how an experiment is run (recorded with every result).

    transaction_cost is a fraction of traded notional per unit of position change
    (0.001 = 0.1% = 10 bps; a long -> short flip pays it twice).
    """

    train: Period
    validation: Period
    transaction_cost: float = 0.001
    top_n: int = 10
    selection_metric: str = "sharpe"
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    min_train_trades: int = 10
    periods_per_year: float = HOURS_PER_YEAR

    def __post_init__(self) -> None:
        if not 0 <= self.transaction_cost < MAX_TRANSACTION_COST:
            raise ValueError(f"transaction_cost must be in [0, {MAX_TRANSACTION_COST}), got {self.transaction_cost}")
        if self.top_n < 1 or self.max_candidates < 1 or self.min_train_trades < 0:
            raise ValueError("top_n and max_candidates must be >= 1, min_train_trades >= 0")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        if self.train.end is None or self.validation.start is None:
            raise ValueError("train end and validation start are required")
        if pd.Timestamp(self.validation.start) < pd.Timestamp(self.train.end):
            raise ValueError("validation must start at or after the end of the train period")

    @property
    def cost_bps(self) -> float:
        return self.transaction_cost * 10_000

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cost_bps": self.cost_bps}


@dataclass
class ExperimentResult:
    """Tables produced by run_experiment (candidates are identified by their `candidate` id)."""

    settings: ExperimentSettings
    n_candidates: int
    train: pd.DataFrame                  # every candidate, train period
    selected: list[int]                  # top-N ids, chosen from `train` only
    validation: pd.DataFrame             # the selected candidates, validation period
    comparison: pd.DataFrame             # selected candidates, train vs validation side by side
    benchmarks: dict[str, pd.DataFrame]  # "train"/"validation" -> one row per benchmark
    timing: dict[str, Any]


def select_top(train: pd.DataFrame, metric: str, n: int, min_trades: int = 0) -> list[int]:
    """Ids of the `n` best candidates by `metric` on TRAIN; ties go to the lower id.

    Candidates with an undefined metric or fewer than `min_trades` trades are not eligible.
    """
    eligible = train[(train["n_trades"] >= min_trades) & train[metric].notna()]
    ranked = eligible.sort_values([metric, "candidate"], ascending=[False, True])
    return [int(i) for i in ranked["candidate"].head(n)]


def compare(train: pd.DataFrame, validation: pd.DataFrame, selected: list[int], metric: str) -> pd.DataFrame:
    """Side-by-side table for the selected candidates, in train-rank order."""
    if not selected:
        return pd.DataFrame()
    params = [c for c in train.columns if "." in c]
    t = train.set_index("candidate").loc[selected]
    v = validation.set_index("candidate").loc[selected]
    table = pd.DataFrame({"train_rank": range(1, len(selected) + 1)}, index=pd.Index(selected, name="candidate"))
    for p in params:
        table[p] = t[p]
    for m in dict.fromkeys((metric, *COMPARISON_METRICS)):
        table[f"train_{m}"] = t[m]
        table[f"validation_{m}"] = v[m]
    table[f"{metric}_change"] = table[f"validation_{metric}"] - table[f"train_{metric}"]
    return table.reset_index()


def truncate_after(data: pd.DataFrame, end: str | None) -> pd.DataFrame:
    """Drop every bar at or after `end`, so later (final-test) data cannot influence anything."""
    return data if end is None else data.iloc[: period_slice(data, None, end).stop]


def run_experiment(
    data: pd.DataFrame,
    base: StrategySpec,
    space: ParameterSpace,
    settings: ExperimentSettings,
    benchmarks: Mapping[str, StrategySpec] | None = None,
    dataset: str = "",
    evaluator: CandidateEvaluator = evaluate_candidates,
) -> ExperimentResult:
    """Train sweep -> top-N selection -> validation retest -> benchmarks on both periods."""
    data = truncate_after(data, settings.validation.end)
    candidates = generate_candidates(base, space, settings.max_candidates)
    common = dict(cost_bps=settings.cost_bps, periods_per_year=settings.periods_per_year, dataset=dataset)

    started = time.perf_counter()
    train = evaluator(data, candidates, start=settings.train.start, end=settings.train.end, **common)
    train_seconds = time.perf_counter() - started

    # Selection sees only the train table. Parameters are frozen from here on.
    selected = select_top(train, settings.selection_metric, settings.top_n, settings.min_train_trades)
    frozen = [candidates[i] for i in selected]

    started = time.perf_counter()
    if frozen:
        validation = evaluator(data, frozen, start=settings.validation.start, end=settings.validation.end,
                               ids=selected, **common)
    else:
        validation = train.iloc[0:0]
    validation_seconds = time.perf_counter() - started

    benchmark_tables = {
        name: evaluate_benchmarks(data, benchmarks or {}, period, evaluator=evaluator, **common)
        for name, period in (("train", settings.train), ("validation", settings.validation))
    }
    return ExperimentResult(
        settings=settings,
        n_candidates=len(candidates),
        train=train,
        selected=selected,
        validation=validation,
        comparison=compare(train, validation, selected, settings.selection_metric),
        benchmarks=benchmark_tables,
        timing={
            "backend": getattr(evaluator, "__name__", type(evaluator).__name__),
            "train_seconds": round(train_seconds, 3),
            "validation_seconds": round(validation_seconds, 3),
            "ms_per_train_candidate": round(1000 * train_seconds / max(len(candidates), 1), 2),
        },
    )
