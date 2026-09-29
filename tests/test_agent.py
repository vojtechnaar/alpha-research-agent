"""Tests for prompts, reply parsing, the agent loop (with a fake LLM) and LoRA data selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.loop import run_agent, summarize
from src.agents.prompts import build_messages, parse_response, score
from src.backtest.evaluate import Evaluator
from src.models.sft_data import select_attempts, to_example
from src.strategy.dsl import SpecError

CLOSE = {"op": "field", "name": "close"}


def spec_json(signal: dict, name: str = "s") -> str:
    return json.dumps({"name": name, "hypothesis": "h", "signal": signal})


def make_evaluator() -> Evaluator:
    rng = np.random.default_rng(0)
    n = 24 * 120
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    data = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close, "volume": np.ones(n),
    })
    splits = {"train": ("2020-01-01", "2020-03-01"), "validation": ("2020-03-01", "2020-04-01"), "test": ("2020-04-01", None)}
    return Evaluator({"A": data, "B": data.copy()}, splits)


class FakeGenerator:
    """Returns scripted replies in order and records the prompts it was given."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], n: int = 1) -> list[str]:
        self.prompts.append(messages)
        return [self.replies.pop(0) for _ in range(n)]


def test_parse_response_handles_thinking_and_fences() -> None:
    text = "<think>{not json}</think>\nSure:\n```json\n" + spec_json(CLOSE) + "\n```"
    assert parse_response(text)["signal"] == CLOSE
    with pytest.raises(SpecError, match="no JSON"):
        parse_response("I think momentum works.")
    with pytest.raises(SpecError, match="not valid JSON"):
        parse_response("{'name': 'single quotes'}")
    with pytest.raises(SpecError, match="unknown op"):
        parse_response(spec_json({"op": "future_close"}))


def test_score_requires_trades_and_finite_sharpe() -> None:
    good = {"A": {"sharpe": 1.0, "n_trades": 50}, "B": {"sharpe": 0.0, "n_trades": 50}}
    assert score(good, min_trades=20) == 0.5
    assert score({**good, "B": {"sharpe": 2.0, "n_trades": 5}}, min_trades=20) is None
    assert score({"A": {"sharpe": float("nan"), "n_trades": 50}}, min_trades=20) is None


def test_agent_loop_logs_attempts_and_feeds_back_train_results(tmp_path) -> None:
    trend = {"op": "pct_change", "arg": CLOSE, "periods": 24}
    reversal = {"op": "neg", "arg": {"op": "zscore", "arg": CLOSE, "window": 48}}
    generator = FakeGenerator([
        spec_json(trend, "trend"), "no idea",                     # iteration 1: one valid, one invalid
        spec_json(trend, "trend_again"), spec_json(reversal, "reversal"),  # iteration 2: duplicate + new
    ])

    attempts = run_agent(generator, make_evaluator(), tmp_path, iterations=2, candidates_per_iteration=2, min_trades=1)

    assert [("spec" in a, "error" in a) for a in attempts] == [(True, False), (False, True), (False, True), (True, False)]
    assert "duplicate" in attempts[2]["error"]
    assert set(attempts[0]["train"]) == {"A", "B"} and set(attempts[0]["validation"]) == {"A", "B"}
    logged = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [a["id"] for a in logged] == [0, 1, 2, 3]

    feedback = generator.prompts[1][-1]["content"]
    assert "trend" in feedback and "INVALID" in feedback
    assert "validation" not in feedback.lower()  # the LLM never sees validation results
    assert [row["name"] for row in summarize(attempts)] == sorted(
        ["trend", "reversal"], key=lambda name: -next(a["score"] for a in attempts if a.get("spec", {}).get("name") == name)
    )


def test_first_prompt_has_no_history() -> None:
    messages = build_messages([], cost_bps=10)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "10 basis points" in messages[0]["content"]


def _attempt(name: str, train: float, validation: float, trades: int = 50) -> dict:
    return {
        "spec": {"name": name, "hypothesis": "h", "signal": {"op": "sma", "arg": CLOSE, "window": len(name) + 1}},
        "score": train,
        "validation": {"A": {"sharpe": validation, "n_trades": trades}},
    }


def test_sft_selection_uses_train_and_validation_thresholds() -> None:
    attempts = [
        _attempt("good", 1.0, 0.8),
        _attempt("overfit", 2.0, -0.5),
        _attempt("weak", 0.1, 0.9),
        _attempt("sparse", 1.0, 1.0, trades=3),
        _attempt("dupe", 1.0, 0.7),  # same signal as "good" (same window) -> dropped
        {"error": "bad json"},
    ]
    attempts[4]["spec"]["signal"] = attempts[0]["spec"]["signal"]
    selected = select_attempts(attempts, min_train_sharpe=0.5, min_validation_sharpe=0.3, min_trades=20)
    assert [a["spec"]["name"] for a in selected] == ["good"]

    example = to_example(selected[0], cost_bps=10)
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert json.loads(example["messages"][-1]["content"])["name"] == "good"
