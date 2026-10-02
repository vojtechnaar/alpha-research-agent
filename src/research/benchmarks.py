"""Benchmark strategies (buy-and-hold, cash, naive momentum) evaluated like any candidate.

Benchmarks are ordinary StrategySpec files in configs/benchmarks/. Unconditional ones have no
conditions (buy-and-hold: always long; flat: always in cash). They run through the same
backend, costs and periods as the candidates, so the comparison is like-for-like.

Used by: research/experiment.py (benchmarks on the same periods and costs as every experiment),
         agents/research.py and research/confirm.py (load_benchmarks).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.strategies.schema import StrategySpec
from src.strategies.sweep import CandidateEvaluator, evaluate_candidates

if TYPE_CHECKING:
    from src.research.experiment import Period

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARKS_DIR = PROJECT_ROOT / "configs" / "benchmarks"


def load_benchmarks(directory: str | Path = DEFAULT_BENCHMARKS_DIR) -> dict[str, StrategySpec]:
    """Every *.json spec in `directory`, keyed by strategy name (sorted for a stable order)."""
    benchmarks: dict[str, StrategySpec] = {}
    for path in sorted(Path(directory).glob("*.json")):
        spec = StrategySpec.from_json(path.read_text())
        if spec.name in benchmarks:
            raise ValueError(f"duplicate benchmark name {spec.name!r} in {directory}")
        benchmarks[spec.name] = spec
    return benchmarks


def evaluate_benchmarks(
    data: pd.DataFrame,
    benchmarks: Mapping[str, StrategySpec],
    period: Period,
    evaluator: CandidateEvaluator = evaluate_candidates,
    **kwargs: Any,
) -> pd.DataFrame:
    """One metrics row per benchmark on `period` (same columns as candidate results + 'benchmark')."""
    if not benchmarks:
        return pd.DataFrame()
    table = evaluator(data, [({}, spec) for spec in benchmarks.values()], start=period.start, end=period.end, **kwargs)
    table.insert(1, "benchmark", list(benchmarks))
    return table
