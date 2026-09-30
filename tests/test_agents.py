"""Proposal parsing, prompts, feedback and the bounded research loop (with a fake LLM)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.agents import prompts
from src.agents.proposals import extract_json, parse_proposal
from src.agents.research import LoopSettings, ReplayGenerator, main, run_research
from src.research.benchmarks import load_benchmarks
from src.research.experiment import ExperimentSettings
from src.research.records import load_records
from src.strategies.features import FEATURE_REGISTRY, FeatureDef, compute_momentum
from src.strategies.schema import (
    DUPLICATE_PROPOSAL,
    INVALID_SPEC,
    MALFORMED_JSON,
    SEARCH_SPACE_TOO_LARGE,
    UNSUPPORTED_FEATURE,
    UNSUPPORTED_OPERATOR,
    SpecError,
)

from conftest import make_hourly_data

PROPOSAL = {
    "hypothesis": "Momentum persists when volatility is low.",
    "rationale": "Calm trends continue.",
    "strategy": {
        "name": "mom_lowvol",
        "conditions": [
            {"id": "mom", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
            {"id": "vol", "feature": "volatility", "field": "close", "lookback": 24, "operator": "<", "threshold": 0.02},
        ],
    },
    "parameter_space": {"mom.lookback": [6, 24, 72], "mom.threshold": [0.0, 0.01]},
}


def proposal(**changes: object) -> dict:
    p = copy.deepcopy(PROPOSAL)
    p.update(changes)
    return p


def variant(lookbacks: list[int]) -> str:
    return json.dumps(proposal(hypothesis=f"Momentum over {lookbacks}.", parameter_space={"mom.lookback": lookbacks}))


class FakeGenerator:
    """Scripted replies; records every conversation and seed it was given."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.seeds: list[int | None] = []

    def generate(self, messages, n=1, seed=None):
        self.calls.append(copy.deepcopy(messages))
        self.seeds.append(seed)
        return [self.replies.pop(0) if self.replies else "no json here"]


def run(generator, settings, tmp_path, **loop_kwargs):
    loop = LoopSettings(**{"hypotheses": 3, **loop_kwargs})
    return run_research(generator, make_hourly_data(), settings, loop, tmp_path, load_benchmarks(), "SYN/USD",
                        log=lambda _: None)


# ---------------------------------------------------------------- parsing and validation


def test_parse_proposal_tolerates_wrapping() -> None:
    text = "<think>{draft}</think>\nHere it is:\n```json\n" + json.dumps(PROPOSAL) + "\n```"
    parsed = parse_proposal(text, max_candidates=100)
    assert parsed.hypothesis == PROPOSAL["hypothesis"]
    assert parsed.parameter_space == PROPOSAL["parameter_space"]


@pytest.mark.parametrize(
    "text, code",
    [
        ("Momentum is a great idea.", MALFORMED_JSON),
        ('{"hypothesis": "cut off", "strategy": {"name": ', MALFORMED_JSON),
        ("{'hypothesis': 'single quotes'}", MALFORMED_JSON),
        ("[1, 2, 3]", MALFORMED_JSON),
        (json.dumps(proposal(strategy={**PROPOSAL["strategy"], "conditions": [
            {"feature": "rsi", "field": "close", "lookback": 14, "operator": ">", "threshold": 70}]})), UNSUPPORTED_FEATURE),
        (json.dumps(proposal(strategy={**PROPOSAL["strategy"], "logic": "XOR"})), UNSUPPORTED_OPERATOR),
        (json.dumps(proposal(parameter_space={"mom.window": [5]})), INVALID_SPEC),
        (json.dumps(proposal(parameter_space={"vol.lookback": [1, 24]})), INVALID_SPEC),
        (json.dumps(proposal(parameter_space={"mom.lookback": list(range(1, 11)), "mom.threshold": [0.0] * 11})),
         SEARCH_SPACE_TOO_LARGE),
        (json.dumps(proposal(strategy={"name": "bh", "conditions": [], "true_position": 1})), INVALID_SPEC),
        (json.dumps(proposal(code="import os; os.system('rm -rf /')")), INVALID_SPEC),
    ],
)
def test_invalid_proposals_are_rejected_with_codes(text: str, code: str) -> None:
    with pytest.raises(SpecError) as info:
        parse_proposal(text, max_candidates=100)
    assert info.value.code == code


def test_duplicate_proposals_are_rejected() -> None:
    first = parse_proposal(json.dumps(PROPOSAL), 100)
    renamed = proposal(hypothesis="Same thing, new words.", strategy={**PROPOSAL["strategy"], "name": "other"})
    with pytest.raises(SpecError) as info:
        parse_proposal(json.dumps(renamed), 100, seen={first.key()})
    assert info.value.code == DUPLICATE_PROPOSAL


def test_duplicate_detection_ignores_names_ids_and_order_but_allows_refinements() -> None:
    first = parse_proposal(json.dumps(PROPOSAL), 100)
    mom, vol = PROPOSAL["strategy"]["conditions"]
    renamed = proposal(
        hypothesis="Reworded.",
        strategy={"name": "other", "conditions": [{**vol, "id": "volatility"}, {**mom, "id": "momentum"}]},
        parameter_space={"momentum.threshold": [0.01, 0.0], "momentum.lookback": [72, 24, 6]},
    )
    assert parse_proposal(json.dumps(renamed), 100).key() == first.key()  # same experiment in disguise

    refined = proposal(parameter_space={"mom.lookback": [48, 72, 96], "mom.threshold": [0.0, 0.01]})
    assert parse_proposal(json.dumps(refined), 100, seen={first.key()}).key() != first.key()  # allowed


def test_extract_json_reports_truncation() -> None:
    with pytest.raises(SpecError, match="cut off at the token limit"):
        extract_json('{"hypothesis": "x", "strategy": {"name": "a", "conditions": [{"feature": "momentum"}')


def test_extract_json_ignores_text_after_the_first_object() -> None:
    text = json.dumps(PROPOSAL) + '\nNote: I could also try {"feature": "zscore"} next.\n' + json.dumps(PROPOSAL)
    assert extract_json(text) == PROPOSAL  # previously failed with "Extra data"


def test_extract_json_error_is_actionable() -> None:
    with pytest.raises(SpecError, match="Check commas") as info:
        extract_json('{"hypothesis": "x" "rationale": "y"}')
    assert "near:" in str(info.value) and "position" not in str(info.value)


# ---------------------------------------------------------------- prompts and feedback


def test_system_prompt_is_generated_from_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(FEATURE_REGISTRY, "test_feature", FeatureDef(compute_momentum, description="added in a test"))
    text = prompts.system_prompt(max_candidates=321, transaction_cost=0.001)
    for name in FEATURE_REGISTRY:
        assert name in text
    for op in (">", ">=", "<", "<=", "AND", "OR"):
        assert op in text
    assert "321" in text and "exactly ONE compact JSON object" in text and "No Python, no CUDA" in text
    assert "returns: one-bar return" in text and "NO lookback" in text and "use momentum" in text
    json.loads(json.dumps(prompts.EXAMPLE_PROPOSAL))
    parse_proposal(json.dumps(prompts.EXAMPLE_PROPOSAL), 321)  # the example itself must be valid


def test_feedback_is_compact_and_complete(settings: ExperimentSettings, tmp_path: Path) -> None:
    records = run(FakeGenerator([json.dumps(PROPOSAL)]), settings, tmp_path, hypotheses=1)
    text = prompts.feedback_message(records)
    for section in ("PREVIOUS HYPOTHESES", "HYPOTHESIS:", "STRATEGY:", "TRAIN (", "VALIDATION (", "GENERALIZATION",
                    "COSTS (", "trades/year", "BENCHMARKS (validation", "PARAMETER SENSITIVITY", "mom.lookback",
                    "ROBUSTNESS WARNINGS"):
        assert section in text
    assert len(text) < 4000  # a few hundred tokens, never the CSV or price history

    blind = prompts.feedback_message(records, include_validation=False)
    assert "VALIDATION" not in blind and "BENCHMARKS (train" in blind


# ---------------------------------------------------------------- the loop


def test_loop_stops_after_n_hypotheses(settings: ExperimentSettings, tmp_path: Path) -> None:
    generator = FakeGenerator([variant([6, 24]), variant([24, 72]), variant([72, 168]), variant([168, 336])])
    records = run(generator, settings, tmp_path, hypotheses=3)
    assert len(records) == 3 and len(generator.calls) == 3
    assert [r.status for r in records] == ["completed"] * 3
    assert len(load_records(tmp_path / "experiments.jsonl")) == 3
    assert all(Path(r.sweep_csv).exists() for r in records)
    assert generator.seeds == [42, 1042, 2042]


def test_feedback_reaches_the_next_iteration(settings: ExperimentSettings, tmp_path: Path) -> None:
    generator = FakeGenerator([variant([6, 24]), variant([24, 72])])
    run(generator, settings, tmp_path, hypotheses=2)
    first_context, second_context = generator.calls[0][-1]["content"], generator.calls[1][-1]["content"]
    assert "No experiments" in first_context
    assert "Momentum over [6, 24]." in second_context and "TRAIN (" in second_context


def test_invalid_reply_is_repaired_within_retry_limit(settings: ExperimentSettings, tmp_path: Path) -> None:
    generator = FakeGenerator(['{"hypothesis": "x", "strategy": {"name": "x", "conditions": [{"feature": "rsi"}]}}',
                               json.dumps(PROPOSAL)])
    (record,) = run(generator, settings, tmp_path, hypotheses=1)
    assert record.status == "completed" and len(generator.calls) == 2
    repair = generator.calls[1][-1]["content"]
    assert "rejected by the validator" in repair and "INVALID_SPEC" in repair
    assert [a["code"] for a in record.llm["attempts"]] == ["INVALID_SPEC", None]


def test_retry_limit_and_early_stop_on_repeated_rejections(settings: ExperimentSettings, tmp_path: Path) -> None:
    generator = FakeGenerator([])  # never returns JSON
    records = run(generator, settings, tmp_path, hypotheses=5, max_proposal_retries=2, max_consecutive_rejections=2)
    assert [r.status for r in records] == ["rejected", "rejected"]  # stopped early, never 5
    assert len(generator.calls) == 2 * 3  # 1 + 2 retries per hypothesis
    assert records[0].error.startswith(MALFORMED_JSON)


def test_duplicate_proposal_triggers_a_repair(settings: ExperimentSettings, tmp_path: Path) -> None:
    generator = FakeGenerator([variant([6, 24]), variant([6, 24]), variant([24, 72])])
    records = run(generator, settings, tmp_path, hypotheses=2)
    assert [r.status for r in records] == ["completed", "completed"]
    assert records[1].llm["attempts"][0]["code"] == DUPLICATE_PROPOSAL
    retry = generator.calls[2]  # the retry after the duplicate names what it repeated...
    assert "repeats experiment 1 ('Momentum over [6, 24].')" in retry[-1]["content"]
    assert "different features" in retry[-1]["content"]
    # ...and does not show the model its own copied reply (small models tend to repeat it)
    assert [m["role"] for m in retry] == ["system", "user"]
    assert retry[-1]["content"].startswith(generator.calls[1][-1]["content"])


def test_evaluation_errors_are_recorded_not_fatal(settings: ExperimentSettings, tmp_path: Path) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("backend exploded")

    loop = LoopSettings(hypotheses=2)
    records = run_research(FakeGenerator([variant([6, 24]), variant([24, 72])]), make_hourly_data(), settings,
                           loop, tmp_path, evaluator=broken, log=lambda _: None)
    assert [r.status for r in records] == ["failed", "failed"]
    assert "backend exploded" in records[0].error


def test_loop_settings_have_hard_limits() -> None:
    for bad in (0, 101):
        with pytest.raises(ValueError):
            LoopSettings(hypotheses=bad)


def test_cli_replay_runs_without_llm(tmp_path: Path) -> None:
    data_path = tmp_path / "syn.parquet"
    make_hourly_data().to_parquet(data_path)
    replay = tmp_path / "proposals.jsonl"
    replay.write_text(variant([6, 24]) + "\n")
    code = main(["--data", str(data_path), "--hypotheses", "1", "--replay", str(replay), "--output-dir", str(tmp_path / "out"),
                 "--train-start", "2020-01-01", "--train-end", "2020-03-15", "--validation-start", "2020-03-15",
                 "--validation-end", "2020-05-01", "--min-train-trades", "1", "--transaction-cost", "0.002"])
    assert code == 0
    (run_dir,) = (tmp_path / "out").iterdir()
    run_info = json.loads((run_dir / "run.json").read_text())
    assert run_info["settings"]["transaction_cost"] == 0.002 and run_info["settings"]["cost_bps"] == 20
    (record,) = load_records(run_dir / "experiments.jsonl")
    assert record.status == "completed" and record.cost_bps == 20
    assert ReplayGenerator(replay).generate([])[0] == variant([6, 24])


def test_feedback_is_identical_for_fresh_and_reloaded_records(settings: ExperimentSettings, tmp_path: Path) -> None:
    records = run(FakeGenerator([json.dumps(PROPOSAL)]), settings, tmp_path, hypotheses=1)
    reloaded = load_records(tmp_path / "experiments.jsonl")
    assert prompts.feedback_message(records) == prompts.feedback_message(reloaded)
    assert "flat n/a" in prompts.feedback_message(records)  # undefined Sharpe shown as n/a, not nan


def test_cli_runs_with_native_backend_when_built(tmp_path: Path) -> None:
    from src.backends import get_evaluator

    try:
        get_evaluator("cpp")
    except (FileNotFoundError, OSError):
        pytest.skip("C++ backend not built (make -C cuda cpu)")
    data_path = tmp_path / "syn.parquet"
    make_hourly_data().to_parquet(data_path)
    replay = tmp_path / "proposals.jsonl"
    replay.write_text(variant([6, 24]) + "\n")
    assert main(["--data", str(data_path), "--hypotheses", "1", "--replay", str(replay), "--backend", "cpp",
                 "--output-dir", str(tmp_path / "out"), "--train-start", "2020-01-01", "--train-end", "2020-03-15",
                 "--validation-start", "2020-03-15", "--validation-end", "2020-05-01", "--min-train-trades", "1"]) == 0
    (run_dir,) = (tmp_path / "out").iterdir()
    assert json.loads((run_dir / "run.json").read_text())["backend"]["name"] == "cpp"
    (record,) = load_records(run_dir / "experiments.jsonl")
    assert record.status == "completed" and record.timing["backend"] == "cpp"


def test_returns_with_lookback_is_repaired_using_the_error_hint(settings: ExperimentSettings, tmp_path: Path) -> None:
    bad = proposal(strategy={"name": "r", "conditions": [
        {"id": "r", "feature": "returns", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0}]},
        parameter_space={"r.lookback": [12, 24]})
    good = proposal(strategy={"name": "r", "conditions": [
        {"id": "r", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0}]},
        parameter_space={"r.lookback": [12, 24]})
    generator = FakeGenerator([json.dumps(bad), json.dumps(good)])
    (record,) = run(generator, settings, tmp_path, hypotheses=1)
    assert record.status == "completed"
    assert "use momentum with lookback N" in generator.calls[1][-1]["content"]  # the repair prompt carries the hint


def test_prompt_asks_for_ranges_and_shows_training_feature_ranges(settings: ExperimentSettings) -> None:
    from src.agents.research import build_system_prompt

    text = build_system_prompt(make_hourly_data(), settings)
    assert '{"min": a, "max": b}' in text and "budget of 1000 combinations" in text
    assert "TYPICAL FEATURE VALUES on the training data" in text
    for name in FEATURE_REGISTRY:
        assert f"- {name}(" in text  # one line of typical values per registered feature


def test_range_proposal_is_expanded_to_the_budget() -> None:
    ranged = proposal(parameter_space={"mom.lookback": {"min": 6, "max": 720}, "mom.threshold": {"min": 0.0, "max": 0.02}})
    parsed = parse_proposal(json.dumps(ranged), max_candidates=100)
    lookbacks, thresholds = parsed.parameter_space["mom.lookback"], parsed.parameter_space["mom.threshold"]
    assert len(lookbacks) * len(thresholds) <= 100 and len(lookbacks) == len(thresholds) == 10
    assert lookbacks[0] == 6 and lookbacks[-1] == 720 and all(isinstance(v, int) for v in lookbacks)
    assert thresholds[0] == 0.0 and thresholds[-1] == 0.02
    assert parsed.requested_space == ranged["parameter_space"]


def test_already_tested_grid_is_rejected_but_partial_overlap_is_allowed(settings: ExperimentSettings, tmp_path: Path) -> None:
    subset = proposal(hypothesis="Same grid, fewer values.", parameter_space={"mom.lookback": [24, 6], "mom.threshold": [0.0]})
    overlap = proposal(hypothesis="Longer lookbacks.", parameter_space={"mom.lookback": [24, 168], "mom.threshold": [0.0]})
    generator = FakeGenerator([json.dumps(PROPOSAL), json.dumps(subset), json.dumps(overlap)])
    records = run(generator, settings, tmp_path, hypotheses=2)
    assert [r.status for r in records] == ["completed", "completed"]
    rejected = records[1].llm["attempts"][0]
    assert rejected["code"] == DUPLICATE_PROPOSAL
    assert "all 2 parameter combinations were already tested in experiment 1" in rejected["error"]
    assert records[1].hypothesis == "Longer lookbacks."  # 168 is new, so the refinement ran


def test_feedback_shows_exploration_coverage(settings: ExperimentSettings, tmp_path: Path) -> None:
    short = proposal(hypothesis="Short sharp drops.", strategy={"name": "s", "true_position": -1, "conditions": [
        {"id": "r", "feature": "returns", "field": "close", "operator": "<", "threshold": -0.01}]},
        parameter_space={"r.threshold": [-0.02, -0.01]})
    records = run(FakeGenerator([json.dumps(PROPOSAL), json.dumps(short), "not json"]), settings, tmp_path,
                  hypotheses=3, max_proposal_retries=0)
    text = prompts.exploration_summary(records)
    assert "features used: " in text and "momentum x1" in text and "returns x1" in text
    untried = text.split("features not yet tried: ")[1].splitlines()[0]
    assert "zscore" in untried and "momentum" not in untried and "returns" not in untried
    assert "long/flat x1" in text and "short/flat x1" in text  # the rejected 3rd proposal is not counted
    assert "EXPLORATION" in prompts.feedback_message(records)
    assert "(budget 1000)" in prompts.describe_experiment(records[0])


def test_truncated_reply_is_reported_with_how_to_shorten_it(settings: ExperimentSettings, tmp_path: Path) -> None:
    class Truncating(FakeGenerator):
        def generate(self, messages, n=1, seed=None):
            reply = super().generate(messages, n, seed)
            self.last_stats = {"new_tokens": 512, "hit_max_new_tokens": len(self.calls) == 1}
            return reply

    cut = json.dumps(PROPOSAL)[:400] + '}]}'  # braces balanced by luck, but the JSON is broken
    generator = Truncating([cut, json.dumps(PROPOSAL)])
    (record,) = run(generator, settings, tmp_path, hypotheses=1)
    assert record.status == "completed"
    first = record.llm["attempts"][0]
    assert first["code"] == MALFORMED_JSON and "cut off at the token limit" in first["error"]
    assert first["hit_max_new_tokens"] is True
    assert "compact single-line JSON" in generator.calls[1][-1]["content"]


def test_feedback_shows_condition_activity_and_compact_search(settings: ExperimentSettings, tmp_path: Path) -> None:
    ranged = proposal(parameter_space={"mom.lookback": {"min": 6, "max": 168}, "mom.threshold": [0.0, 0.01]})
    (record,) = run(FakeGenerator([json.dumps(ranged)]), settings, tmp_path, hypotheses=1)
    text = prompts.describe_experiment(record)
    assert "mom.lookback 6..168 (" in text  # long value lists are summarised
    assert "CONDITIONS (best train candidate, share of train bars where true): mom " in text
    assert record.requested_parameter_space == ranged["parameter_space"]
