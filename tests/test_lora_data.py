"""LoRA training-set builder: filters, held-out markets, targets, split, token masking (no GPU/model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.lora_data import build_examples, direction_mismatch, main, split, tokenize_example
from src.research.experiment import ExperimentSettings, Period
from src.research.records import append_record, new_record
from src.strategies.schema import ResearchProposal, StrategySpec

SETTINGS = ExperimentSettings(Period("2017-01-01", "2023-01-01"), Period("2023-01-01", "2025-01-01"))
LONG = {"name": "calm_trend", "conditions": [
    {"id": "d", "feature": "distance_to_mean", "field": "close", "lookback": 168, "operator": ">", "threshold": 0.02},
    {"id": "v", "feature": "volatility", "field": "close", "lookback": 168, "operator": "<", "threshold": 0.008}],
    "logic": "AND", "true_position": 1, "false_position": 0}


def record(dataset: str = "BTC/USD", hypothesis: str = "Calm uptrends above the weekly mean tend to continue higher.",
           median: float = 1.2, exposure: float = 0.4, turnover: float = 50.0, activity: float = 0.3,
           space: dict | None = None, spec: dict | None = None):
    spec = spec or LONG
    rec = new_record(SETTINGS, "completed", dataset=dataset, hypothesis=hypothesis, rationale="r",
                     strategy_spec=spec, parameter_space={"d.lookback": [48, 168]},
                     requested_parameter_space=space or {"d.lookback": {"min": 48, "max": 168}},
                     llm={"context": f"results so far for {dataset}", "attempts": [{"code": None}]})
    rec.validation_summary = {"median": median, "n_retested": 2}
    rec.comparison = [{"candidate": 0, "validation_exposure": exposure, "validation_annual_turnover": turnover}] * 2
    rec.condition_activity = {"scope": "best train candidate", "shares": {"d": activity, "v": 0.5}}
    rec.min_trades_per_year = 10.0
    return rec


def write_run(runs_dir: Path, name: str, records: list, system: str | None = "SYSTEM PROMPT", replay: bool = False):
    run = runs_dir / name
    run.mkdir(parents=True)
    generator = {"replay": "x.json"} if replay else {"model_id": "Qwen/Qwen3-8B"}
    (run / "run.json").write_text(json.dumps({"system_prompt": system, "generator": generator}))
    for r in records:
        append_record(run / "experiments.jsonl", r)


def test_direction_mismatch_heuristic() -> None:
    short = StrategySpec.from_dict({**LONG, "true_position": -1})
    long = StrategySpec.from_dict(LONG)
    assert direction_mismatch("Prices near recent lows may reverse upward.", short)
    assert direction_mismatch("A breakdown leads to a further decline.", long)
    assert not direction_mismatch("A sharp drop may signal a rebound.", long)
    assert not direction_mismatch("Volatility regimes matter.", short)  # no direction words: can't tell


def test_only_good_research_steps_become_examples(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs, "20260101-000000", [
        record(),                                                    # good
        record(median=-0.3),                                         # did not hold up on validation
        record(exposure=0.99),                                       # buy-and-hold in disguise
        record(turnover=2.0),                                        # trades too rarely
        record(activity=1.0),                                        # a condition is always true
        record(hypothesis="Prices will decline further.", space={"d.lookback": [24]}),  # text contradicts long rule
        record(),                                                    # exact repeat of the first
        record(space={"d.lookback": {"min": 24, "max": 336}}),        # same rule, new ranges: kept
    ])
    write_run(runs, "20260101-000100", [record(dataset="GLD")])     # held-out market
    write_run(runs, "20260101-000200", [record()], replay=True)     # not an LLM run
    examples, reasons = build_examples(runs)
    assert len(examples) == 2 and reasons["kept"] == 2
    assert reasons["not useful"] == 3 and reasons["unit problem"] == 1
    assert reasons["hypothesis contradicts rule"] == 1 and reasons["duplicate example"] == 1
    assert reasons["held-out market"] == 1 and reasons["no system prompt / replay run"] == 1

    messages = examples[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == "SYSTEM PROMPT" and "BTC/USD" in messages[1]["content"]
    # the target is itself a valid proposal, written as compact single-line JSON
    ResearchProposal.from_dict(json.loads(messages[2]["content"]), max_candidates=20000)
    assert "\n" not in messages[2]["content"]


def test_split_is_by_run_and_deterministic() -> None:
    examples = [{"meta": {"run": f"run{i // 3}"}} for i in range(300)]
    train, val = split(examples, 0.2)
    assert len(train) + len(val) == 300 and 0 < len(val) < 150
    assert split(examples, 0.2) == (train, val)
    runs_train, runs_val = {e["meta"]["run"] for e in train}, {e["meta"]["run"] for e in val}
    assert not runs_train & runs_val  # a run is never on both sides


class CharTokenizer:
    """One token per character; the chat template just tags roles."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_loss_is_only_on_the_assistant_reply() -> None:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                {"role": "assistant", "content": '{"a":1}'}]
    example = tokenize_example(CharTokenizer(), messages)
    prompt_length = len("<system>S<user>U<assistant>")
    assert example["labels"][:prompt_length] == [-100] * prompt_length
    reply = "".join(chr(t) for t in example["labels"][prompt_length:])
    assert reply == '{"a":1}<|im_end|>'
    assert tokenize_example(CharTokenizer(), messages, max_length=10) is None  # too long -> skipped


def test_cli_writes_train_and_val(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    runs = tmp_path / "runs"
    for i in range(6):
        write_run(runs, f"2026010{i}-000000", [record(space={"d.lookback": {"min": 24 + i, "max": 336}})])
    assert main(["--runs-dir", str(runs), "--out-dir", str(tmp_path / "lora")]) == 0
    train = (tmp_path / "lora" / "train.jsonl").read_text().splitlines()
    val = (tmp_path / "lora" / "val.jsonl").read_text().splitlines()
    assert len(train) + len(val) == 6
    assert "Examples: 6 kept" in capsys.readouterr().out
