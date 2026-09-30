"""Turn raw LLM text into a validated ResearchProposal, or reject it with a machine-readable code.

Nothing in the reply is executed: it is parsed as JSON and validated against the registries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Set

from src.strategies.schema import DUPLICATE_PROPOSAL, MALFORMED_JSON, ResearchProposal, SpecError


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
