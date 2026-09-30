"""Compact, JSON-safe summaries of an experiment, and heuristic overfitting warnings.

The warnings are descriptive red flags, not statistical tests. Deflated/probabilistic Sharpe
ratios, bootstrap intervals and multiple-testing corrections are future work (docs/architecture.md).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.experiment import ExperimentResult
from src.strategies.sweep import summarize_sweep

BENCHMARK_METRICS = ("cumulative_return", "annualized_return", "sharpe", "annualized_volatility",
                     "max_drawdown", "turnover", "n_trades", "exposure", "n_bars")


def summarize_train(result: ExperimentResult, top: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    """(train summary, parameter sensitivity) over ALL candidates."""
    summary = summarize_sweep(result.train, result.settings.selection_metric, top)
    sensitivity = summary.pop("parameter_sensitivity")
    return summary, sensitivity


def summarize_validation(result: ExperimentResult) -> dict[str, Any]:
    """How the frozen top-N candidates did on validation, and how much they degraded."""
    metric = result.settings.selection_metric
    table = result.comparison
    if table.empty:
        return {"n_retested": 0, "metric": metric}
    train, validation = table[f"train_{metric}"], table[f"validation_{metric}"]
    params = [c for c in table.columns if "." in c]
    first = table.iloc[0]
    return {
        "n_retested": int(len(table)),
        "metric": metric,
        "median": _num(validation.median()),
        "best": _num(validation.max()),
        "worst": _num(validation.min()),
        "share_positive": _num((validation > 0).mean()),
        "train_median_of_selected": _num(train.median()),
        "median_change": _num((validation - train).median()),
        "train_best_candidate": {  # what naive "pick the in-sample winner" would have produced
            **{p: first[p] for p in params},
            f"train_{metric}": _num(first[f"train_{metric}"]),
            f"validation_{metric}": _num(first[f"validation_{metric}"]),
        },
    }


def summarize_benchmarks(result: ExperimentResult) -> dict[str, dict[str, dict[str, Any]]]:
    """{period: {benchmark: metrics}} including each period's actual first/last bar."""
    return {
        period: {
            row["benchmark"]: {"start": row["start"], "end": row["end"], **{m: row[m] for m in BENCHMARK_METRICS}}
            for _, row in table.iterrows()
        }
        for period, table in result.benchmarks.items()
        if not table.empty
    }


def robustness_warnings(result: ExperimentResult, space: dict[str, list]) -> list[str]:
    """Plain-language red flags for over-fitting and weak evidence."""
    s = result.settings
    metric, train, table = s.selection_metric, result.train, result.comparison
    warnings = []
    if result.n_candidates > 1:
        warnings.append(f"{result.n_candidates} parameter combinations were tested; the best train {metric} "
                        "is inflated by selection (multiple testing).")
    median_train = train[metric].median()
    if pd.notna(median_train) and median_train <= 0:
        warnings.append(f"Median train {metric} is {median_train:.2f}: the idea fails for most parameter values, "
                        "so the top results may be luck.")
    if not result.selected:
        warnings.append(f"No candidate had >= {s.min_train_trades} train trades and a defined {metric}; "
                        "nothing was validated.")
        return warnings

    best = train.set_index("candidate").loc[result.selected[0]]
    for name, values in space.items():
        if len(set(values)) >= 3 and best[name] in (min(values), max(values)):
            warnings.append(f"Best train value of {name} ({best[name]}) is at the edge of the tested range; "
                            "the optimum may lie outside it.")

    # Different parameters can give identical positions (e.g. a filter that never binds).
    identical = len(table) - len(table.drop_duplicates([f"train_{metric}", "train_n_trades", "train_exposure"]))
    if identical:
        warnings.append(f"{identical} of the top {len(table)} have exactly the same train results as a higher-ranked "
                        "candidate: some parameters do not change the positions (e.g. a filter that never binds).")

    t, v = table[f"train_{metric}"].median(), table[f"validation_{metric}"].median()
    if pd.notna(v) and v <= 0:
        warnings.append(f"Median validation {metric} of the top {len(table)} is {v:.2f}: "
                        "the train results did not carry over.")
    elif pd.notna(t) and pd.notna(v) and t > 0 and v < 0.5 * t:
        warnings.append(f"Top candidates kept less than half their train {metric} on validation "
                        f"(median {t:.2f} -> {v:.2f}): likely over-fit.")
    few = int((table["validation_n_trades"] < s.min_train_trades).sum())
    if few:
        warnings.append(f"{few} of {len(table)} retested candidates traded fewer than {s.min_train_trades} "
                        "times on validation; their metrics are unreliable.")

    bh = _benchmark_metric(result, "validation", "buy_and_hold", metric)
    if bh is not None and table[f"validation_{metric}"].notna().any():
        best_v = table[f"validation_{metric}"].max()
        if best_v < bh:
            warnings.append(f"No retested candidate beat buy-and-hold on validation {metric} ({bh:.2f}).")
        else:
            warnings.append(f"Some candidates beat buy-and-hold on validation {metric} ({bh:.2f}); after "
                            f"{result.n_candidates} trials this alone is not evidence of alpha.")
    return warnings


def _benchmark_metric(result: ExperimentResult, period: str, name: str, metric: str) -> float | None:
    table = result.benchmarks.get(period)
    if table is None or table.empty or name not in set(table["benchmark"]):
        return None
    value = table.set_index("benchmark").loc[name, metric]
    return None if pd.isna(value) else float(value)


def _num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)
