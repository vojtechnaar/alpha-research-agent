"""Compact, JSON-safe summaries of an experiment, and heuristic overfitting warnings.

The warnings are descriptive red flags, not statistical tests. Deflated/probabilistic Sharpe
ratios, bootstrap intervals and multiple-testing corrections are future work (docs/architecture.md).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.experiment import TRADE_SIGNATURE, ExperimentResult
from src.strategies.sweep import summarize_sweep

COST_WARNING_ANNUAL_COST = 0.05  # warn when fees exceed 5% of capital per year

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


def summarize_costs(result: ExperimentResult) -> dict[str, Any]:
    """How much the selected top-N candidates trade on TRAIN and what that costs them.

    annual_cost = annual turnover x transaction cost: the fees paid per year as a fraction of
    capital (a long -> short flip counts twice). Compare it with the net annualized return:
    gross return ~= net return + fees.
    """
    s = result.settings
    ids = result.selected or list(result.train["candidate"])
    table = result.train.set_index("candidate").loc[ids]
    if table.empty:
        return {}
    years = table["n_bars"] / s.periods_per_year
    trades = table["n_trades"]
    return {
        "scope": f"top {len(ids)} by train {s.selection_metric}" if result.selected else "all candidates",
        "transaction_cost": s.transaction_cost,
        "median_trades_per_year": _num((trades / years).median()),
        "median_bars_between_trades": _num((table["n_bars"] / trades.where(trades > 0)).median()),
        "median_exposure": _num(table["exposure"].median()),
        "median_annual_cost": _num((table["annual_turnover"] * s.transaction_cost).median()),
        "median_annualized_return": _num(table["annualized_return"].median()),
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
    """Plain-language red flags for over-fitting, weak evidence and cost drag."""
    s = result.settings
    metric, train, table = s.selection_metric, result.train, result.comparison
    warnings = []
    if result.n_candidates > 1:
        warnings.append(f"{result.n_candidates} parameter combinations were tested; the best train {metric} "
                        "is inflated by selection (multiple testing).")
    redundant = len(train) - len(train.drop_duplicates(list(TRADE_SIGNATURE)))
    if redundant:
        warnings.append(f"{redundant} of {len(train)} combinations traded exactly like another combination "
                        "(some parameter values do not change the positions, e.g. a filter that never binds); "
                        "only distinct candidates were retested.")
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

    costs = summarize_costs(result)
    annual_cost, net = costs.get("median_annual_cost"), costs.get("median_annualized_return")
    if annual_cost is not None and annual_cost > COST_WARNING_ANNUAL_COST:
        verdict = "costs exceed the net return" if net is not None and annual_cost > abs(net) else "costs are large"
        warnings.append(
            f"Trading costs: the top candidates change position every ~{_fmt(costs['median_bars_between_trades'], '.0f')} "
            f"bars (~{_fmt(costs['median_trades_per_year'], '.0f')} trades/year) and pay ~{100 * annual_cost:.0f}% of "
            f"capital per year in fees vs {_fmt(None if net is None else 100 * net, '.0f')}% net annualized return "
            f"({verdict}). Slower signals or fewer position changes keep more of the gross return."
        )

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


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)
