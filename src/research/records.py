"""Experiment records: one compact, JSON-serialisable entry per research iteration.

Records hold settings, the hypothesis, the StrategySpec, the parameter space, summaries,
benchmarks, warnings and (for LLM runs) the prompt context and raw replies. They never hold
time series; the full per-candidate train table is written to a separate CSV and referenced by
path. Records are appended to JSONL files, one line per experiment.

They are the raw material for agent context, reproducibility and a future LoRA dataset, which
is why validation, benchmark and warning information is kept next to every proposal.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.research.experiment import ExperimentResult, ExperimentSettings
from src.research.summary import (
    robustness_warnings,
    summarize_benchmarks,
    summarize_costs,
    summarize_train,
    summarize_validation,
)

SCHEMA_VERSION = 2  # v2 added `costs`; v1 records still load (missing fields get defaults)
STATUSES = ("completed", "rejected", "failed")  # rejected: invalid LLM proposal; failed: evaluation error


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    status: str
    dataset: str = ""
    data_path: str = ""
    train_period: dict[str, Any] = field(default_factory=dict)
    validation_period: dict[str, Any] = field(default_factory=dict)
    transaction_cost: float | None = None
    cost_bps: float | None = None
    selection_metric: str = "sharpe"
    top_n: int | None = None
    max_candidates: int | None = None
    min_train_trades: int | None = None
    hypothesis: str = ""
    rationale: str = ""
    strategy_spec: dict[str, Any] | None = None
    strategy_description: str = ""
    parameter_space: dict[str, Any] | None = None
    n_candidates: int = 0
    train_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    costs: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    parameter_sensitivity: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    sweep_csv: str | None = None
    run_id: str | None = None
    iteration: int | None = None
    llm: dict[str, Any] | None = None
    error: str | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Strict-JSON-safe dict (NaN/inf -> None, timestamps -> ISO strings)."""
        return to_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Inverse of to_dict; unknown keys (from newer versions) are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def new_experiment_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def new_record(settings: ExperimentSettings, status: str = "completed", **values: Any) -> ExperimentRecord:
    """Record pre-filled with the experiment settings; `values` sets any other field."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    return ExperimentRecord(
        experiment_id=values.pop("experiment_id", None) or new_experiment_id(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=status,
        train_period=asdict(settings.train),
        validation_period=asdict(settings.validation),
        transaction_cost=settings.transaction_cost,
        cost_bps=settings.cost_bps,
        selection_metric=settings.selection_metric,
        top_n=settings.top_n,
        max_candidates=settings.max_candidates,
        min_train_trades=settings.min_train_trades,
        **values,
    )


def add_results(record: ExperimentRecord, result: ExperimentResult, space: dict[str, list]) -> ExperimentRecord:
    """Fill a record's result fields from an ExperimentResult (compact summaries only).

    Values are made JSON-safe immediately, so a fresh record and one reloaded from disk are
    identical (and produce identical LLM feedback).
    """
    train_summary, sensitivity = summarize_train(result)
    record.status = "completed"
    record.parameter_space = to_json_safe(dict(space))
    record.n_candidates = result.n_candidates
    record.train_summary = to_json_safe(train_summary)
    record.parameter_sensitivity = to_json_safe(sensitivity)
    record.validation_summary = to_json_safe(summarize_validation(result))
    record.comparison = to_json_safe(result.comparison.to_dict("records"))
    record.costs = to_json_safe(summarize_costs(result))
    record.benchmarks = to_json_safe(summarize_benchmarks(result))
    record.warnings = robustness_warnings(result, space)
    record.timing = to_json_safe(result.timing)
    return record


def append_record(path: str | Path, record: ExperimentRecord) -> None:
    """Append one record as a JSON line (creating the file and folders if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")


def load_records(path: str | Path) -> list[ExperimentRecord]:
    """Read a JSONL file (or a single-record .json file) of experiment records."""
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        return [ExperimentRecord.from_dict(json.loads(text))]
    return [ExperimentRecord.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def to_json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas values to plain JSON types; NaN and inf become None."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, datetime):  # includes pandas.Timestamp
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
