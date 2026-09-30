"""Turn raw LLM text into a validated ResearchProposal, or reject it with a machine-readable code.

Nothing in the reply is executed: it is parsed as JSON and validated against the registries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Set

from src.strategies.schema import DUPLICATE_PROPOSAL, MALFORMED_JSON, ResearchProposal, SpecError


def extract_json(text: str) -> dict:
    """The JSON object in a reply, tolerating <think> blocks, markdown fences and stray prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise SpecError(MALFORMED_JSON, "reply contains no JSON object")
    if end < start:
        raise SpecError(MALFORMED_JSON, "reply ends before the JSON object is complete (too many tokens?)")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SpecError(MALFORMED_JSON, f"invalid JSON: {exc.msg} at position {exc.pos}") from None
    if not isinstance(data, dict):
        raise SpecError(MALFORMED_JSON, "the JSON value must be an object")
    return data


def parse_proposal(
    text: str, max_candidates: int, seen: Mapping[str, str] | Set[str] | None = None
) -> ResearchProposal:
    """Parse and fully validate a reply; reject repeats of already-run experiments.

    `seen` maps ResearchProposal.key() to a label of the earlier experiment (e.g. "experiment 1
    ('...')"), so the rejection tells the model exactly which experiment it repeated.
    """
    proposal = ResearchProposal.from_dict(extract_json(text), max_candidates=max_candidates)
    if seen is not None and proposal.key() in seen:
        earlier = seen[proposal.key()] if isinstance(seen, Mapping) else "an earlier experiment"
        raise SpecError(DUPLICATE_PROPOSAL, f"this exact strategy and parameter space repeats {earlier}. Do not "
                                            "resubmit it: use different features or conditions, or clearly "
                                            "different parameter ranges.")
    return proposal
