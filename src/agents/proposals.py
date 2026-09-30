"""Turn raw LLM text into a validated ResearchProposal, or reject it with a machine-readable code.

Nothing in the reply is executed: it is parsed as JSON and validated against the registries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Set

from src.strategies.features import FEATURE_REGISTRY
from src.strategies.schema import DUPLICATE_PROPOSAL, FAMILY_EXHAUSTED, MALFORMED_JSON, ResearchProposal, SpecError
from src.strategies.sweep import parameter_grid


TRUNCATED_HINT = ("the reply was cut off at the token limit before the JSON was complete. Make it shorter: "
                  "compact single-line JSON, one short sentence each for hypothesis and rationale, at most 8 values "
                  "per parameter")


def extract_json(text: str) -> dict:
    """The first complete JSON object in a reply.

    Tolerates <think> blocks, markdown fences, prose before the object and anything after it
    (a second object, an explanation containing braces).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start == -1:
        raise SpecError(MALFORMED_JSON, "reply contains no JSON object")
    try:
        data, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        if text.count("{") > text.count("}"):  # braces never close: the reply stopped mid-object
            raise SpecError(MALFORMED_JSON, TRUNCATED_HINT) from None
        near = text[max(start, exc.pos - 40) : exc.pos + 20].replace("\n", " ")
        raise SpecError(MALFORMED_JSON, f"invalid JSON ({exc.msg}) near: ...{near}... "
                                        "Check commas between items and that every quote, [ and { is closed.") from None
    if not isinstance(data, dict):
        raise SpecError(MALFORMED_JSON, "the JSON value must be an object")
    return data


def candidate_identities(proposal: ResearchProposal) -> list[str]:
    """StrategySpec.identity() of every combination the proposal would test."""
    return [proposal.strategy.with_parameters(params).identity() for params in parameter_grid(proposal.parameter_space)]


def parse_proposal(
    text: str,
    max_candidates: int,
    seen: Mapping[str, str] | Set[str] | None = None,
    tested: Mapping[str, str] | None = None,
    families: Mapping[str, list[str]] | None = None,
    max_per_family: int | None = None,
) -> ResearchProposal:
    """Parse and fully validate a reply; reject experiments that would add nothing new.

    `seen` maps ResearchProposal.key() to a label of the earlier experiment (e.g. "experiment 1
    ('...')"): an exact repeat is rejected, naming that experiment. `tested` maps the identity of
    every combination already backtested to its experiment: a proposal whose combinations were
    ALL tested before is rejected too. Partial overlap is allowed (a refinement adds new values).
    `families` maps StrategySpec.family() to the experiments that tested it: once a family has
    `max_per_family` experiments, further proposals of it are rejected (FAMILY_EXHAUSTED).
    """
    proposal = ResearchProposal.from_dict(extract_json(text), max_candidates=max_candidates)
    if seen is not None and proposal.key() in seen:
        earlier = seen[proposal.key()] if isinstance(seen, Mapping) else "an earlier experiment"
        raise SpecError(DUPLICATE_PROPOSAL, f"this exact strategy and parameter space repeats {earlier}. Do not "
                                            "resubmit it: use different features or conditions, or clearly "
                                            "different parameter ranges.")
    family = proposal.strategy.family()
    if families and max_per_family and len(families.get(family, [])) >= max_per_family:
        used = {json.loads(c)["feature"] for key in families for c in json.loads(key)["conditions"]}
        untried = [name for name in FEATURE_REGISTRY if name not in used]
        raise SpecError(FAMILY_EXHAUSTED, f"the idea '{proposal.strategy.describe_family()}' was already tested "
                                          f"{len(families[family])} times ({', '.join(families[family])}); the limit "
                                          f"is {max_per_family}. Propose a different idea: change the features, a "
                                          "comparison direction or the long/short rule. Features not tried yet: "
                                          f"{', '.join(untried) or 'none'}.")
    if tested:
        identities = candidate_identities(proposal)
        if all(identity in tested for identity in identities):
            earlier = ", ".join(sorted({tested[identity] for identity in identities})[:3])
            raise SpecError(DUPLICATE_PROPOSAL, f"all {len(identities)} parameter combinations were already tested "
                                                f"in {earlier}. Test a different idea, or values outside the "
                                                "ranges already tested.")
    return proposal
