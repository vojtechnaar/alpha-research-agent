"""StrategySpec parsing, serialisation and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.strategies.schema import (
    INVALID_SPEC,
    UNSUPPORTED_FEATURE,
    UNSUPPORTED_OPERATOR,
    ResearchProposal,
    SpecError,
    StrategySpec,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SPEC = {
    "name": "momentum_low_volatility",
    "conditions": [
        {"id": "momentum", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.02},
        {"id": "vol", "feature": "volatility", "field": "close", "lookback": 12, "operator": "<", "threshold": 0.03},
    ],
    "logic": "AND",
    "true_position": 1,
    "false_position": 0,
}


def with_change(path: tuple, value: object) -> dict:
    """Copy of SPEC with one nested value replaced, e.g. (("conditions", 0, "operator"), "==")."""
    spec = copy.deepcopy(SPEC)
    target = spec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return spec


def test_valid_spec_parses() -> None:
    spec = StrategySpec.from_dict(SPEC)
    assert spec.name == "momentum_low_volatility"
    assert [c.key for c in spec.conditions] == ["momentum", "vol"]
    assert spec.conditions[0].lookback == 24 and spec.conditions[1].threshold == 0.03


def test_json_round_trip() -> None:
    spec = StrategySpec.from_dict(SPEC)
    assert StrategySpec.from_json(spec.to_json()) == spec
    assert json.loads(spec.to_json()) == SPEC


@pytest.mark.parametrize("path", sorted((CONFIGS / "strategies").glob("*.json")), ids=lambda p: p.name)
def test_example_configs_are_valid(path: Path) -> None:
    StrategySpec.from_json(path.read_text())


@pytest.mark.parametrize(
    "path, value, code",
    [
        (("conditions", 0, "feature"), "rsi", UNSUPPORTED_FEATURE),
        (("conditions", 0, "feature"), ["momentum"], UNSUPPORTED_FEATURE),
        (("conditions", 0, "operator"), "==", UNSUPPORTED_OPERATOR),
        (("logic", ), "XOR", UNSUPPORTED_OPERATOR),
        (("true_position",), 2, INVALID_SPEC),
        (("false_position",), True, INVALID_SPEC),
        (("conditions", 0, "lookback"), 0, INVALID_SPEC),
        (("conditions", 0, "lookback"), 24.5, INVALID_SPEC),
        (("conditions", 0, "lookback"), 10**6, INVALID_SPEC),
        (("conditions", 1, "lookback"), 1, INVALID_SPEC),  # volatility needs >= 2
        (("conditions", 0, "threshold"), float("nan"), INVALID_SPEC),
        (("conditions", 0, "threshold"), "0.02", INVALID_SPEC),
        (("conditions", 0, "field"), "vwap", INVALID_SPEC),
        (("conditions", 1, "id"), "momentum", INVALID_SPEC),  # duplicate id
        (("conditions", 0, "code"), "import os", INVALID_SPEC),  # unknown key
        (("conditions",), "momentum > 0", INVALID_SPEC),
    ],
)
def test_invalid_specs_are_rejected_with_code(path: tuple, value: object, code: str) -> None:
    with pytest.raises(SpecError) as info:
        StrategySpec.from_dict(with_change(path, value))
    assert info.value.code == code


def test_returns_feature_takes_no_lookback() -> None:
    cond = {"feature": "returns", "field": "close", "operator": ">", "threshold": 0}
    StrategySpec.from_dict({"name": "r", "conditions": [cond]})
    with pytest.raises(SpecError, match="no lookback") as info:
        StrategySpec.from_dict({"name": "r", "conditions": [{**cond, "lookback": 3}]})
    assert "momentum with lookback N" in str(info.value)  # the error says how to fix it


def test_malformed_json_is_invalid_spec() -> None:
    with pytest.raises(SpecError) as info:
        StrategySpec.from_json("{not json")
    assert info.value.code == INVALID_SPEC


def test_with_parameters_replaces_values_and_validates() -> None:
    spec = StrategySpec.from_dict(SPEC)
    changed = spec.with_parameters({"momentum.lookback": 48, "vol.threshold": 0.01})
    assert changed.parameters()["momentum.lookback"] == 48
    assert changed.parameters()["vol.threshold"] == 0.01
    assert spec.conditions[0].lookback == 24  # original untouched
    with pytest.raises(SpecError, match="unknown parameter"):
        spec.with_parameters({"momentum.window": 5})
    with pytest.raises(SpecError):
        spec.with_parameters({"vol.lookback": 1})


def test_research_proposal_matches_future_llm_output() -> None:
    proposal = ResearchProposal.from_dict({
        "hypothesis": "Momentum may be stronger in low-volatility regimes.",
        "strategy": SPEC,
        "parameter_space": {"momentum.lookback": [6, 12, 24], "vol.threshold": [0.01, 0.02]},
        "rationale": "Calm markets trend more cleanly.",
    })
    assert proposal.strategy.name == SPEC["name"]
    with pytest.raises(SpecError, match="unknown parameter"):
        ResearchProposal.from_dict({"hypothesis": "h", "strategy": SPEC, "parameter_space": {"rsi.lookback": [5]}})
    with pytest.raises(SpecError):
        ResearchProposal.from_dict({"hypothesis": "h", "strategy": SPEC, "parameter_space": {"vol.lookback": [1, 12]}})


def test_unconditional_spec_is_valid_for_benchmarks() -> None:
    spec = StrategySpec.from_dict({"name": "buy_and_hold", "conditions": [], "true_position": 1})
    assert spec.conditions == () and spec.describe() == "always long"
    assert StrategySpec.from_json(spec.to_json()) == spec


def test_describe_is_readable() -> None:
    assert StrategySpec.from_dict(SPEC).describe() == (
        "long if momentum(close, 24) > 0.02 AND volatility(close, 12) < 0.03, else flat"
    )
