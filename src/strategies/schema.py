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
A spec with no conditions is unconditional (always `true_position`); that is how benchmarks such
as buy-and-hold ({"conditions": [], "true_position": 1}) and cash are expressed. Research
proposals from the LLM must have at least one condition.

Each condition has an `id` (default: its feature name, must be unique). Sweepable parameters are
addressed as "<id>.<param>", e.g. "momentum.lookback" or "vol.threshold".

Only features/operators in the trusted registries are accepted. Anything else raises SpecError
with code UNSUPPORTED_FEATURE / UNSUPPORTED_OPERATOR / INVALID_SPEC; nothing is ever executed
from the spec itself.

Used by: almost every module: the evaluators (Python and native) and benchmarks run StrategySpecs;
         agents/proposals.py and agents/research.py use ResearchProposal and the error codes;
         identity()/family() drive the duplicate and family checks; records store specs via
         to_dict().
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
MALFORMED_JSON = "MALFORMED_JSON"
SEARCH_SPACE_TOO_LARGE = "SEARCH_SPACE_TOO_LARGE"
DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
FAMILY_EXHAUSTED = "FAMILY_EXHAUSTED"  # the same idea (rules without parameter values) was tested too often

POSITION_NAMES = {1: "long", 0: "flat", -1: "short"}

FIELDS = ("open", "high", "low", "close", "volume")
POSITIONS = (-1, 0, 1)
SWEEPABLE_PARAMS = ("lookback", "threshold")
MAX_LOOKBACK = 10_000  # bars
MAX_CONDITIONS = 8
MAX_VALUES_PER_RANGE = 12  # a {"min", "max"} range is expanded into at most this many values


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

    def describe_family(self) -> str:
        """Readable form without parameter values, e.g. 'momentum(close) > x'."""
        return f"{self.feature}({self.field}) {self.operator} x"

    def describe(self) -> str:
        """Readable form, e.g. 'momentum(close, 24) > 0.02'."""
        args = self.field if self.lookback is None else f"{self.field}, {self.lookback}"
        return f"{self.feature}({args}) {self.operator} {self.threshold:g}"


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
        if not isinstance(conditions, list):
            raise SpecError(INVALID_SPEC, "'conditions' must be a list")
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

    def identity(self) -> str:
        """Canonical form of the concrete trading rules, ignoring name, description, condition ids and
        condition order. Two specs with the same identity always produce the same positions."""
        conditions = sorted(
            json.dumps({"feature": c.feature, "field": c.field, "operator": c.operator,
                        "lookback": c.lookback, "threshold": float(c.threshold)}, sort_keys=True)
            for c in self.conditions
        )
        return json.dumps({"conditions": conditions, "logic": self.logic if len(conditions) > 1 else "AND",
                           "true_position": self.true_position, "false_position": self.false_position},
                          sort_keys=True)

    def family(self) -> str:
        """The idea without its parameter values: which features, fields and comparison directions,
        combined how, with which long/short rule. Refining lookbacks or thresholds keeps the family;
        changing a feature, a direction or the position rule makes a new one."""
        conditions = sorted(json.dumps({"feature": c.feature, "field": c.field, "operator": c.operator},
                                       sort_keys=True) for c in self.conditions)
        return json.dumps({"conditions": conditions, "logic": self.logic if len(conditions) > 1 else "AND",
                           "true_position": self.true_position, "false_position": self.false_position},
                          sort_keys=True)

    def describe_family(self) -> str:
        """Readable family, e.g. 'short if volatility(close) > x AND returns(close) < x, else flat'."""
        rule = f" {self.logic} ".join(c.describe_family() for c in self.conditions) or "always"
        return f"{POSITION_NAMES[self.true_position]} if {rule}, else {POSITION_NAMES[self.false_position]}"

    def describe(self) -> str:
        """Readable rule, e.g. 'long if momentum(close, 24) > 0.02 AND ..., else flat'."""
        if not self.conditions:
            return f"always {POSITION_NAMES[self.true_position]}"
        rule = f" {self.logic} ".join(c.describe() for c in self.conditions)
        return f"{POSITION_NAMES[self.true_position]} if {rule}, else {POSITION_NAMES[self.false_position]}"

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
    """LLM output: a hypothesis, a base strategy and the parameter space to sweep.

        {"hypothesis": "...", "rationale": "...",
         "strategy": {<StrategySpec>},
         "parameter_space": {"momentum.lookback": [24, 48, 72], "momentum.threshold": [0.01, 0.02]}}
    """

    hypothesis: str
    strategy: StrategySpec
    parameter_space: dict[str, list[int | float]] = field(default_factory=dict)  # explicit values
    rationale: str = ""
    requested_space: dict[str, Any] = field(default_factory=dict)  # as written, ranges not yet expanded
    notes: tuple[str, ...] = ()  # automatic corrections applied before validation (see agents/proposals.py)

    @classmethod
    def from_dict(cls, data: Any, max_candidates: int | None = None) -> ResearchProposal:
        """Validate everything before anything is evaluated: spec, parameter names, every value,
        and the size of the search (SEARCH_SPACE_TOO_LARGE above `max_candidates`).

        A parameter is either an explicit list of values or a range {"min": a, "max": b} that is
        expanded to fill the candidate budget (see expand_parameter_space)."""
        if not isinstance(data, dict):
            raise SpecError(INVALID_SPEC, "proposal must be a JSON object")
        _reject_unknown_keys(data, {"hypothesis", "strategy", "parameter_space", "rationale"}, "proposal")
        hypothesis = data.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip() or "strategy" not in data:
            raise SpecError(INVALID_SPEC, "proposal needs a non-empty 'hypothesis' string and a 'strategy'")
        if not isinstance(data.get("rationale", ""), str):
            raise SpecError(INVALID_SPEC, "'rationale' must be a string")
        strategy = StrategySpec.from_dict(data["strategy"])
        if not strategy.conditions:
            raise SpecError(INVALID_SPEC, "a research strategy needs at least one condition "
                                          "(unconditional strategies are benchmarks)")
        requested = data.get("parameter_space", {})
        if not isinstance(requested, dict) or not all(
            (isinstance(v, list) and v) or isinstance(v, dict) for v in requested.values()
        ):
            raise SpecError(INVALID_SPEC, "'parameter_space' must map parameter names to non-empty lists "
                                          'or to ranges {"min": number, "max": number}')
        space = expand_parameter_space(requested, max_candidates)
        n_candidates = math.prod(len(v) for v in space.values())
        if max_candidates is not None and n_candidates > max_candidates:
            raise SpecError(SEARCH_SPACE_TOO_LARGE,
                            f"parameter_space has {n_candidates} combinations; the budget is {max_candidates}")
        for name, values in space.items():  # every value must produce a valid strategy
            for value in values:
                strategy.with_parameters({name: value})
        return cls(hypothesis.strip(), strategy, space, data.get("rationale", "").strip(), dict(requested))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "strategy": self.strategy.to_dict(),
            "parameter_space": self.parameter_space,
            "requested_space": self.requested_space,
        }

    def key(self) -> str:
        """Identity of the experiment, for rejecting exact repeats.

        Ignores names, prose, condition ids and condition order. A swept parameter is represented by
        its sorted value list (its base value is irrelevant). So renaming or reordering conditions
        is still a duplicate, while the same rules with different parameter ranges are a new
        experiment (a refinement).
        """
        conditions = []
        for c in self.strategy.conditions:
            item: dict[str, Any] = {"feature": c.feature, "field": c.field, "operator": c.operator}
            for param in SWEEPABLE_PARAMS:
                values = self.parameter_space.get(f"{c.key}.{param}")
                base = getattr(c, param)
                item[param] = (sorted(float(v) for v in values) if values is not None
                               else None if base is None else float(base))
            conditions.append(json.dumps(item, sort_keys=True))
        s = self.strategy
        return json.dumps({"conditions": sorted(conditions), "logic": s.logic if len(conditions) > 1 else "AND",
                           "true_position": s.true_position, "false_position": s.false_position}, sort_keys=True)


# -------------------------------------------------------------------- parameter ranges


def expand_parameter_space(space: dict[str, Any], max_candidates: int | None = None) -> dict[str, list]:
    """Turn {"min": a, "max": b} ranges into value lists that fit the candidate budget.

    Explicit lists are kept as given. All ranges get the same number of points k: the largest k
    (between 2 and MAX_VALUES_PER_RANGE) for which the whole grid fits `max_candidates`. Lookbacks
    are spaced geometrically and rounded to whole bars (so hours-to-weeks ranges are covered evenly
    in relative terms); thresholds are spaced evenly. Deterministic for the same inputs.
    """
    ranges = [name for name, value in space.items() if isinstance(value, dict)]
    if not ranges:
        return dict(space)
    fixed = math.prod(len(value) for value in space.values() if isinstance(value, list))
    k = MAX_VALUES_PER_RANGE
    if max_candidates is not None:
        k = int((max_candidates / fixed) ** (1 / len(ranges)) + 1e-9)
        k = max(2, min(MAX_VALUES_PER_RANGE, k))
    return {name: _range_values(name, value, k) if isinstance(value, dict) else value
            for name, value in space.items()}


def _range_values(name: str, bounds: dict[str, Any], k: int) -> list[int | float]:
    if set(bounds) != {"min", "max"} or not all(_is_number(bounds[b]) for b in ("min", "max")):
        raise SpecError(INVALID_SPEC, f'range for {name!r} must be {{"min": number, "max": number}}')
    lo, hi = sorted((float(bounds["min"]), float(bounds["max"])))
    if name.rpartition(".")[2] == "lookback":
        lo_bars, hi_bars = max(1, round(lo)), max(1, round(hi))
        steps = [lo_bars * (hi_bars / lo_bars) ** (i / (k - 1)) for i in range(k)]
        return sorted({round(v) for v in steps})
    values = [float(f"{lo + (hi - lo) * i / (k - 1):.4g}") for i in range(k)]
    return list(dict.fromkeys(values))  # deduplicate, keep order


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# -------------------------------------------------------------------- validation


def validate_strategy(spec: StrategySpec) -> StrategySpec:
    """Check a spec against the registries and limits; return it or raise SpecError."""
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise SpecError(INVALID_SPEC, "'name' must be a non-empty string")
    if not isinstance(spec.description, str):
        raise SpecError(INVALID_SPEC, "'description' must be a string")
    if len(spec.conditions) > MAX_CONDITIONS:
        raise SpecError(INVALID_SPEC, f"at most {MAX_CONDITIONS} conditions are allowed")
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
            raise SpecError(INVALID_SPEC, f"{where}: feature {c.feature!r} takes no lookback: remove the "
                                          f"'lookback' key and any '{c.key}.lookback' parameter "
                                          f"({c.feature} = {definition.description})")
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
