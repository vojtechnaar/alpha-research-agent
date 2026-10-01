"""Trusted comparison operators and logical combinators for strategy conditions.

A condition's value per bar is 1.0 (true), 0.0 (false) or NaN (unknown: the feature is NaN, e.g.
during warm-up). Combinators propagate NaN, and the evaluator maps NaN to a flat position.

Used by: strategies/evaluator.py (applies operators and AND/OR), strategies/schema.py (validates
         them), agents/prompts.py (lists them for Qwen). Mirrored in cuda/backtest.cu.
"""

from __future__ import annotations

import operator
from collections.abc import Callable

import pandas as pd


def _comparison(compare: Callable[[pd.Series, float], pd.Series]) -> Callable[[pd.Series, float], pd.Series]:
    """Wrap a comparison so it returns 1.0/0.0, and NaN where the feature value is NaN."""

    def apply(values: pd.Series, threshold: float) -> pd.Series:
        return compare(values, threshold).astype("float64").where(values.notna())

    return apply


OPERATOR_REGISTRY: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">": _comparison(operator.gt),
    ">=": _comparison(operator.ge),
    "<": _comparison(operator.lt),
    "<=": _comparison(operator.le),
}

# Conditions are 1.0/0.0 columns, so AND is the row-wise minimum and OR the row-wise maximum.
# skipna=False: a NaN condition makes the combined value NaN.
LOGIC_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "AND": lambda conditions: conditions.min(axis=1, skipna=False),
    "OR": lambda conditions: conditions.max(axis=1, skipna=False),
}
