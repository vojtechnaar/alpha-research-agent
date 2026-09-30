"""Prompts for the research loop, generated from the trusted registries.

The system prompt lists exactly what FEATURE_REGISTRY / OPERATOR_REGISTRY / LOGIC_REGISTRY
contain, so adding a primitive to a registry automatically makes it available to the LLM.
Feedback is a compact text summary of recent experiments (no CSVs, no price history).
"""

from __future__ import annotations

import json

from src.research.records import ExperimentRecord
from src.strategies.features import FEATURE_REGISTRY
from src.strategies.operators import LOGIC_REGISTRY, OPERATOR_REGISTRY
from src.strategies.schema import FIELDS, POSITIONS, SpecError, StrategySpec

# Shows the JSON structure only. Kept deliberately plain (one condition): a small model tends to
# copy whatever strategy the example contains, so the example must not be a promising idea.
EXAMPLE_PROPOSAL = {
    "hypothesis": "Prices far below their weekly average tend to revert upward.",
    "rationale": "One sentence on why the effect could exist.",
    "strategy": {
        "name": "weekly_mean_reversion",
        "conditions": [
            {"id": "z", "feature": "zscore", "field": "close", "lookback": 168, "operator": "<", "threshold": -2.0},
        ],
        "logic": "AND", "true_position": 1, "false_position": 0,
    },
    "parameter_space": {"z.lookback": [72, 168, 336], "z.threshold": [-2.5, -2.0, -1.5]},
}


def available_primitives() -> str:
    """The allowed features, operators and logic, straight from the registries."""
    features = []
    for name, d in FEATURE_REGISTRY.items():
        lookback = f"lookback >= {d.min_lookback}" if d.uses_lookback else "no lookback"
        features.append(f"- {name}: {d.description}; {lookback}; threshold: {d.threshold_hint}")
    return (
        "AVAILABLE FEATURES (feature(field, lookback)):\n" + "\n".join(features) + "\n"
        f"AVAILABLE FIELDS: {', '.join(FIELDS)}\n"
        f"AVAILABLE OPERATORS: {' '.join(OPERATOR_REGISTRY)}\n"
        f"AVAILABLE LOGIC: {' '.join(LOGIC_REGISTRY)}\n"
        f"POSITIONS: {', '.join(str(p) for p in POSITIONS)} (1 long, 0 flat, -1 short)"
    )


def system_prompt(max_candidates: int, transaction_cost: float, bar: str = "1-hour") -> str:
    """Role, rules, primitives and output format."""
    return f"""You are a quantitative researcher. You test hypotheses about {bar} crypto OHLCV bars by \
proposing rule-based strategies. A numerical engine runs the parameter search and backtests; \
you only decide WHAT to test.

How strategies work: each condition is feature(field, lookback) <operator> threshold. Conditions \
are combined with AND/OR. When the combination is true the position is true_position, otherwise \
false_position (flat while a feature is undefined). Positions are set at a bar's close and earn \
the next bar's return. Each unit of position change costs {transaction_cost:g} of notional.

{available_primitives()}

Output format: exactly one JSON object with keys "hypothesis", "rationale", "strategy", \
"parameter_space". Parameter names are "<condition id>.lookback" or "<condition id>.threshold". \
Format example (structure only; do NOT reuse its idea or values):
{json.dumps(EXAMPLE_PROPOSAL, separators=(",", ":"))}

Rules:
- Output JSON only. No Python, no CUDA, no markdown, no explanations outside the JSON.
- Use only the features, fields, operators and logic listed above. Do not invent new ones.
- Features only use current and past bars; do not try to use future information.
- Propose ONE testable hypothesis; hypothesis and rationale are one short sentence each.
- 1 to 3 conditions. Do not propose buy-and-hold or cash; they are benchmarks.
- Costs matter: every position change pays the cost, so a rule that flips every few bars pays it \
thousands of times. Use the COSTS line of the feedback to see how often strategies traded.
- You may refine an earlier idea with different parameter ranges, but never resubmit an identical experiment.
- Parameter ranges must be sensible for each feature's threshold units, and the product of \
the list lengths must be at most {max_candidates}. 3-6 values per parameter is usually enough."""


def initial_request() -> str:
    return "No experiments have been run yet. Propose the first research proposal as JSON."


def repair_message(error: SpecError) -> str:
    """Ask the model to fix its previous reply, given the validator's error."""
    return (f"Your reply was rejected by the validator: {error}\n"
            "Reply with the corrected JSON object only, following the rules and format exactly.")


def describe_experiment(record: ExperimentRecord, include_validation: bool = True) -> str:
    """Compact text block for one experiment (a few hundred tokens at most)."""
    lines = [f"HYPOTHESIS: {record.hypothesis or '(none)'}"]
    if record.status != "completed":
        return "\n".join(lines + [f"STATUS: {record.status} ({record.error})"])

    metric = record.selection_metric
    if record.strategy_spec:
        lines.append(f"STRATEGY: {StrategySpec.from_dict(record.strategy_spec).describe()}")
    lines.append(f"SEARCH: {record.n_candidates} candidates, "
                 + "; ".join(f"{k} {v}" for k, v in (record.parameter_space or {}).items()))
    d, tp = record.train_summary.get("distribution", {}), record.train_period
    lines.append(f"TRAIN ({tp['start']} to {tp['end']}): {metric} median {_f(d.get('median'))}, "
                 f"p10 {_f(d.get('p10'))}, p90 {_f(d.get('p90'))}, best {_f(d.get('max'))}, "
                 f"share positive {_pct(d.get('share_positive'))}")
    period = "train"
    v, vp = record.validation_summary, record.validation_period
    if include_validation and v.get("n_retested"):
        period = "validation"
        lines.append(f"VALIDATION ({vp['start']} to {vp['end']}): top {v['n_retested']} retested with frozen "
                     f"parameters, {metric} median {_f(v.get('median'))}, best {_f(v.get('best'))}, "
                     f"worst {_f(v.get('worst'))}")
        lines.append(f"GENERALIZATION: median {metric} of the selected went {_f(v.get('train_median_of_selected'))} "
                     f"(train) -> {_f(v.get('median'))} (validation), change {_f(v.get('median_change'))}")
    c = record.costs
    if c:
        lines.append(f"COSTS ({c.get('scope')}, train): ~{_f(c.get('median_trades_per_year'), '.0f')} trades/year "
                     f"(a position change every ~{_f(c.get('median_bars_between_trades'), '.0f')} bars), "
                     f"exposure {_pct(c.get('median_exposure'))}, fees ~{_pct(c.get('median_annual_cost'))} of capital "
                     f"per year vs net annualized return {_pct(c.get('median_annualized_return'))}")
    bench = record.benchmarks.get(period, {})
    if bench:
        lines.append(f"BENCHMARKS ({period} {metric}): "
                     + ", ".join(f"{name} {_f(m.get(metric))}" for name, m in bench.items()))
    if record.parameter_sensitivity:
        lines.append(f"PARAMETER SENSITIVITY (mean train {metric} per value):")
        for name, values in record.parameter_sensitivity.items():
            lines.append(f"  {name}: " + ", ".join(f"{k}: {_f(x)}" for k, x in values.items()))
    warnings = [w for w in record.warnings if include_validation or "validation" not in w.lower()]
    if warnings:
        lines.append("ROBUSTNESS WARNINGS:\n" + "\n".join(f"- {w}" for w in warnings))
    return "\n".join(lines)


def feedback_message(records: list[ExperimentRecord], recent: int = 2, include_validation: bool = True) -> str:
    """Next user message: one line per earlier hypothesis, details for the most recent ones."""
    history = "\n".join(
        f"{i}. [{r.status}] {r.hypothesis or '(invalid proposal)'}" for i, r in enumerate(records, 1)
    )
    details = "\n\n".join(describe_experiment(r, include_validation) for r in records[-recent:])
    return (f"PREVIOUS HYPOTHESES (do not repeat):\n{history}\n\nMOST RECENT RESULTS:\n{details}\n\n"
            "Propose ONE next research proposal as JSON: refine what the evidence supports or test a "
            "different idea. Do not simply chase the highest train result.")


def _f(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"
