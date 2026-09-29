"""StrategySpec: the validated, JSON-serialisable strategy format shared by the LLM and the engines.

    {"name": "momentum_low_volatility",
     "conditions": [
       {"id": "momentum", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.02},
       {"id": "vol", "feature": "volatility", "field": "close", "lookback": 12, "operator": "<", "threshold": 0.03}],
     "logic": "AND",
     "true_position": 1,
     "false_position": 0}

At each bar the conditions are combined with `logic`; the position is `true_position` when the
result is true, `false_position` when false, and flat (0) when any feature is still undefined.

Each condition has an `id` (default: its feature name, must be unique). Sweepable parameters are
addressed as "<id>.<param>", e.g. "momentum.lookback" or "vol.threshold".

Only features/operators in the trusted registries are accepted. Anything else raises SpecError
with code UNSUPPORTED_FEATURE / UNSUPPORTED_OPERATOR / INVALID_SPEC; nothing is ever executed
from the spec itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.strategies.features import FEATURE_REGISTRY
from src.strategies.operators import LOGIC_REGISTRY, OPERATOR_REGISTRY

UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"
UNSUPPORTED_OPERATOR = "UNSUPPORTED_OPERATOR"
INVALID_SPEC = "INVALID_SPEC"

FIELDS = ("open", "high", "low", "close", "volume")
POSITIONS = (-1, 0, 1)
SWEEPABLE_PARAMS = ("lookback", "threshold")
MAX_LOOKBACK = 10_000  # bars
MAX_CONDITIONS = 8


class SpecError(ValueError):
    """A spec was rejected. `code` is machine-readable feedback for the LLM."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Condition:
    """feature(field, lookback) <operator> threshold."""

    feature: str
    field: str
    operator: str
    threshold: float
    lookback: int | None = None
    id: str | None = None

    @property
    def key(self) -> str:
        """Identifier used in parameter names and result columns."""
        return self.id or self.feature


@dataclass(frozen=True)
class StrategySpec:
    """A rule-based strategy built only from registered features and operators."""

    name: str
    conditions: tuple[Condition, ...]
    logic: str = "AND"
    true_position: int = 1
    false_position: int = 0
    description: str = ""

    # ---------------------------------------------------------------- serialisation

    @classmethod
    def from_dict(cls, data: Any) -> StrategySpec:
        """Build and validate a spec from parsed JSON."""
        if not isinstance(data, dict):
            raise SpecError(INVALID_SPEC, "strategy must be a JSON object")
        _reject_unknown_keys(data, {f for f in cls.__dataclass_fields__}, "strategy")
        if "name" not in data or "conditions" not in data:
            raise SpecError(INVALID_SPEC, "strategy needs 'name' and 'conditions'")
        conditions = data["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise SpecError(INVALID_SPEC, "'conditions' must be a non-empty list")
        spec = cls(
            name=data["name"],
            conditions=tuple(_condition_from_dict(c, i) for i, c in enumerate(conditions)),
            **{k: data[k] for k in ("logic", "true_position", "false_position", "description") if k in data},
        )
        return validate_strategy(spec)

    @classmethod
    def from_json(cls, text: str) -> StrategySpec:
        """Parse and validate a JSON string."""
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise SpecError(INVALID_SPEC, f"not valid JSON ({exc.msg})") from None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (None fields and empty description omitted)."""
        data = asdict(self)
        data["conditions"] = [{k: v for k, v in c.items() if v is not None} for c in data["conditions"]]
        if not data["description"]:
            del data["description"]
        return data

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ---------------------------------------------------------------- parameters

    def parameters(self) -> dict[str, int | float | None]:
        """Current values of all sweepable parameters, keyed "<condition id>.<param>"."""
        return {f"{c.key}.{p}": getattr(c, p) for c in self.conditions for p in SWEEPABLE_PARAMS}

    def with_parameters(self, params: dict[str, Any]) -> StrategySpec:
        """Copy of the spec with some parameters replaced, validated."""
        by_key = {c.key: i for i, c in enumerate(self.conditions)}
        conditions = list(self.conditions)
        for name, value in params.items():
            cond_key, _, param = name.rpartition(".")
            if cond_key not in by_key or param not in SWEEPABLE_PARAMS:
                raise SpecError(INVALID_SPEC, f"unknown parameter {name!r}; available: {sorted(self.parameters())}")
            i = by_key[cond_key]
            conditions[i] = replace(conditions[i], **{param: value})
        return validate_strategy(replace(self, conditions=tuple(conditions)))


@dataclass(frozen=True)
class ResearchProposal:
    """Future LLM output: a hypothesis, a base strategy and the parameter space to sweep."""

    hypothesis: str
    strategy: StrategySpec
    parameter_space: dict[str, list[int | float]] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> ResearchProposal:
        if not isinstance(data, dict):
            raise SpecError(INVALID_SPEC, "proposal must be a JSON object")
        _reject_unknown_keys(data, {"hypothesis", "strategy", "parameter_space", "rationale"}, "proposal")
        if not isinstance(data.get("hypothesis"), str) or "strategy" not in data:
            raise SpecError(INVALID_SPEC, "proposal needs a 'hypothesis' string and a 'strategy'")
        strategy = StrategySpec.from_dict(data["strategy"])
        space = data.get("parameter_space", {})
        if not isinstance(space, dict) or not all(isinstance(v, list) and v for v in space.values()):
            raise SpecError(INVALID_SPEC, "'parameter_space' must map parameter names to non-empty lists")
        for name, values in space.items():  # every value must produce a valid strategy
            for value in values:
                strategy.with_parameters({name: value})
        return cls(data["hypothesis"], strategy, space, str(data.get("rationale", "")))


# -------------------------------------------------------------------- validation


def validate_strategy(spec: StrategySpec) -> StrategySpec:
    """Check a spec against the registries and limits; return it or raise SpecError."""
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise SpecError(INVALID_SPEC, "'name' must be a non-empty string")
    if not isinstance(spec.description, str):
        raise SpecError(INVALID_SPEC, "'description' must be a string")
    if not spec.conditions or len(spec.conditions) > MAX_CONDITIONS:
        raise SpecError(INVALID_SPEC, f"need 1..{MAX_CONDITIONS} conditions")
    if not isinstance(spec.logic, str) or spec.logic not in LOGIC_REGISTRY:
        raise SpecError(UNSUPPORTED_OPERATOR, f"logic {spec.logic!r} not in {sorted(LOGIC_REGISTRY)}")
    for name in ("true_position", "false_position"):
        value = getattr(spec, name)
        if isinstance(value, bool) or value not in POSITIONS:
            raise SpecError(INVALID_SPEC, f"'{name}' must be one of {POSITIONS}, got {value!r}")
    keys = [c.key for c in spec.conditions]
    if len(set(keys)) != len(keys):
        raise SpecError(INVALID_SPEC, f"condition ids must be unique, got {keys}; set 'id' explicitly")
    for condition in spec.conditions:
        _validate_condition(condition)
    return spec


def _validate_condition(c: Condition) -> None:
    where = f"condition {c.key!r}"
    if not isinstance(c.feature, str) or c.feature not in FEATURE_REGISTRY:
        raise SpecError(UNSUPPORTED_FEATURE, f"{where}: feature {c.feature!r} not in {sorted(FEATURE_REGISTRY)}")
    if not isinstance(c.operator, str) or c.operator not in OPERATOR_REGISTRY:
        raise SpecError(UNSUPPORTED_OPERATOR, f"{where}: operator {c.operator!r} not in {sorted(OPERATOR_REGISTRY)}")
    if not isinstance(c.field, str) or c.field not in FIELDS:
        raise SpecError(INVALID_SPEC, f"{where}: field {c.field!r} not in {list(FIELDS)}")
    if c.id is not None and (not isinstance(c.id, str) or not c.id.isidentifier()):
        raise SpecError(INVALID_SPEC, f"{where}: id must be a simple name like 'mom_fast'")
    if isinstance(c.threshold, bool) or not isinstance(c.threshold, (int, float)) or not math.isfinite(c.threshold):
        raise SpecError(INVALID_SPEC, f"{where}: threshold must be a finite number, got {c.threshold!r}")

    definition = FEATURE_REGISTRY[c.feature]
    if not definition.uses_lookback:
        if c.lookback is not None:
            raise SpecError(INVALID_SPEC, f"{where}: feature {c.feature!r} takes no lookback")
        return
    lb = c.lookback
    if isinstance(lb, bool) or not isinstance(lb, int) or not definition.min_lookback <= lb <= MAX_LOOKBACK:
        raise SpecError(
            INVALID_SPEC,
            f"{where}: lookback must be an integer in {definition.min_lookback}..{MAX_LOOKBACK}, got {lb!r}",
        )


def _condition_from_dict(data: Any, index: int) -> Condition:
    if not isinstance(data, dict):
        raise SpecError(INVALID_SPEC, f"condition {index} must be a JSON object")
    _reject_unknown_keys(data, set(Condition.__dataclass_fields__), f"condition {index}")
    missing = {"feature", "field", "operator", "threshold"} - set(data)
    if missing:
        raise SpecError(INVALID_SPEC, f"condition {index} is missing {sorted(missing)}")
    return Condition(**data)


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise SpecError(INVALID_SPEC, f"{where} has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
