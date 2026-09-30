"""Experiment records: serialisation, loading, compactness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.benchmarks import load_benchmarks
from src.research.experiment import ExperimentSettings, run_experiment
from src.research.records import ExperimentRecord, add_results, append_record, load_records, new_record
from src.research.report import format_record
from src.strategies.schema import StrategySpec

FIXTURE = Path(__file__).parent / "fixtures" / "experiment_record_v1.json"
BASE = StrategySpec.from_dict({"name": "mom", "conditions": [
    {"id": "mom", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0}]})
SPACE = {"mom.lookback": [6, 24, 72], "mom.threshold": [0.0, 0.01]}


def make_record(data: pd.DataFrame, settings: ExperimentSettings) -> ExperimentRecord:
    result = run_experiment(data, BASE, SPACE, settings, load_benchmarks(), "SYN/USD")
    record = new_record(settings, dataset="SYN/USD", hypothesis="h", strategy_spec=BASE.to_dict())
    return add_results(record, result, SPACE)


def test_record_serialises_to_strict_json_and_back(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    record = make_record(data, settings)
    text = json.dumps(record.to_dict(), allow_nan=False)  # raises if any NaN/inf slipped through
    loaded = ExperimentRecord.from_dict(json.loads(text))
    assert loaded.to_dict() == record.to_dict()
    assert loaded.benchmarks["validation"]["flat"]["sharpe"] is None  # NaN -> null
    assert loaded.transaction_cost == 0.001 and loaded.cost_bps == 10
    assert loaded.train_period == {"start": "2020-01-01", "end": "2020-03-15"}


def test_record_is_compact(data: pd.DataFrame, settings: ExperimentSettings) -> None:
    record = make_record(data, settings)
    assert len(record.comparison) == settings.top_n  # only the retested candidates, not the whole sweep
    assert len(json.dumps(record.to_dict())) < 15_000  # no time series


def test_append_and_load_jsonl(tmp_path: Path, data: pd.DataFrame, settings: ExperimentSettings) -> None:
    path = tmp_path / "run" / "experiments.jsonl"
    records = [make_record(data, settings), new_record(settings, "rejected", error="MALFORMED_JSON: ...")]
    for record in records:
        append_record(path, record)
    loaded = load_records(path)
    assert [r.experiment_id for r in loaded] == [r.experiment_id for r in records]
    assert [r.status for r in loaded] == ["completed", "rejected"]


def test_fixture_from_schema_v1_still_loads() -> None:
    (record,) = load_records(FIXTURE)  # also ignores the unknown 'future_field'
    assert record.schema_version == 1 and record.status == "completed"
    assert record.strategy_spec["name"] == "mom_lowvol"
    StrategySpec.from_dict(record.strategy_spec)
    assert set(record.benchmarks) == {"train", "validation"}
    assert "VALIDATION" in format_record(record)
