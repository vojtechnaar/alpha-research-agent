"""Prompts for the research agent and parsing of its replies."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from src.strategy.dsl import OPS_REFERENCE, SpecError, validate_spec

EXAMPLE_SPEC = {
    "name": "weekly_trend_low_vol",
    "hypothesis": "BTC trends over one week, but only when hourly volatility is calm.",
    "signal": {
        "op": "mul",
        "left": {"op": "pct_change", "arg": {"op": "field", "name": "close"}, "periods": 168},
        "right": {"op": "lt",
                  "left": {"op": "std", "arg": {"op": "pct_change", "arg": {"op": "field", "name": "close"}, "periods": 1}, "window": 24},
                  "right": {"op": "const", "value": 0.01}},
    },
}

BASE_REQUEST = "Propose one new trading strategy spec."


def system_prompt(cost_bps: float) -> str:
    """Instructions describing the task and the strategy language."""
    return f"""You are a quantitative crypto researcher. You propose trading strategies for hourly \
BTC/USD and ETH/USD bars as JSON specs, which are backtested automatically.

A spec has three keys:
- "name": short snake_case name
- "hypothesis": one sentence on which market behaviour the strategy exploits and why
- "signal": an expression tree. Every hour the position is the sign of the signal: long if > 0, \
short if < 0, flat if 0 or undefined. Positions are set at the close and earn the next hour's \
return. Every unit of position change costs {cost_bps:g} basis points.

Expression operators:
{OPS_REFERENCE}

Example spec:
{json.dumps(EXAMPLE_SPEC)}

Guidelines: prefer simple, economically motivated ideas. Signals that flip often lose money to \
costs; slower signals (days to weeks) or filters that keep the strategy flat in noisy regimes \
usually trade less. Reply with exactly one JSON object and nothing else."""


def format_attempt(attempt: dict[str, Any]) -> str:
    """One line summarising a past attempt for the LLM (train results only)."""
    if "error" in attempt:
        return f"- INVALID: {attempt['error']}"
    spec, train = attempt["spec"], attempt["train"]
    stats = "; ".join(
        f"{asset}: Sharpe {_fmt(m['sharpe'])}, return {_fmt(100 * m['cumulative_return'], '.0f')}%, "
        f"max DD {_fmt(100 * m['max_drawdown'], '.0f')}%, trades {m['n_trades']}"
        for asset, m in train.items()
    )
    note = "" if attempt.get("score") is not None else " [not scored: too few trades]"
    return f"- {spec['name']}: {stats}{note}\n  signal: {json.dumps(spec['signal'], separators=(',', ':'))}"


def build_messages(
    history: list[dict[str, Any]], cost_bps: float, n_best: int = 5, n_recent: int = 5
) -> list[dict[str, str]]:
    """Chat messages for the next proposal: the system prompt plus the best and most recent attempts."""
    messages = [{"role": "system", "content": system_prompt(cost_bps)}]
    if not history:
        return messages + [{"role": "user", "content": BASE_REQUEST}]

    scored = sorted((a for a in history if a.get("score") is not None), key=lambda a: a["score"], reverse=True)
    best = scored[:n_best]
    recent = [a for a in history[-n_recent:] if a not in best]
    parts = []
    if best:
        parts.append("Best strategies so far (training period):\n" + "\n".join(map(format_attempt, best)))
    if recent:
        parts.append("Most recent attempts:\n" + "\n".join(map(format_attempt, recent)))
    parts.append(
        "Propose one new strategy that differs from all of the above and could beat the best one. "
        "Reply with one JSON object."
    )
    return messages + [{"role": "user", "content": "\n\n".join(parts)}]


def parse_response(text: str) -> dict[str, Any]:
    """Extract and validate the JSON spec in an LLM reply; raise SpecError if there is none."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise SpecError("reply contained no JSON object")
    try:
        spec = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SpecError(f"reply was not valid JSON ({exc.msg})") from None
    return validate_spec(spec)


def score(train: dict[str, dict[str, float]], min_trades: int) -> float | None:
    """Mean train Sharpe across assets; None if any asset trades too little or Sharpe is undefined."""
    if any(m["n_trades"] < min_trades for m in train.values()):
        return None
    sharpes = [m["sharpe"] for m in train.values()]
    if any(math.isnan(s) for s in sharpes):
        return None
    return sum(sharpes) / len(sharpes)


def _fmt(value: float, spec: str = ".2f") -> str:
    return "nan" if math.isnan(value) else format(value, spec)
