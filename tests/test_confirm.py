"""Confirmation of finished experiments: cross-asset replicate/transfer and the logged final test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.research import LoopSettings, run_research
from src.research import confirm
from src.research.benchmarks import load_benchmarks
from src.research.experiment import ExperimentSettings
from src.research.records import load_records

from conftest import make_hourly_data

PROPOSAL = {
    "hypothesis": "Momentum persists when volatility is low.",
    "strategy": {"name": "mom_lowvol", "conditions": [
        {"id": "mom", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0},
        {"id": "vol", "feature": "volatility", "field": "close", "lookback": 24, "operator": "<", "threshold": 0.02}]},
    "parameter_space": {"mom.lookback": {"min": 6, "max": 168}, "mom.threshold": [0.0, 0.01]},
}


class OneReply:
    def generate(self, messages, n=1, seed=None, temperature=None):
        return [json.dumps(PROPOSAL)]


@pytest.fixture
def run(tmp_path: Path, settings: ExperimentSettings) -> tuple[Path, Path]:
    """A finished one-experiment run: (experiments.jsonl, the parquet it used)."""
    data_path = tmp_path / "btc.parquet"
    make_hourly_data().to_parquet(data_path)
    run_research(OneReply(), make_hourly_data(), settings, LoopSettings(hypotheses=1), tmp_path / "run",
                 load_benchmarks(), "SYN/USD", str(data_path), log=lambda _: None)
    return tmp_path / "run" / "experiments.jsonl", data_path


def test_select_records_by_log_number(run: tuple[Path, Path]) -> None:
    records = load_records(run[0])
    assert [r.iteration for r in confirm.select_records(records, [1])] == [0]
    with pytest.raises(ValueError, match="no completed experiment"):
        confirm.select_records(records, [7])


def test_frozen_parameters_reproduce_the_recorded_validation(run: tuple[Path, Path]) -> None:
    from src.strategies.sweep import evaluate_candidates

    (record,) = load_records(run[0])
    settings = confirm.settings_from(record)
    again = confirm.evaluate_frozen(record, make_hourly_data(), settings.validation, settings,
                                    evaluate_candidates, load_benchmarks(), "SYN/USD")
    recorded = {row["candidate"]: row["validation_sharpe"] for row in record.comparison}
    for row in again["rows"]:
        assert row["sharpe"] == pytest.approx(recorded[row["candidate"]])
    assert again["candidates"]["n"] == len(record.comparison)


def test_cross_asset_replicates_and_transfers(run: tuple[Path, Path]) -> None:
    from src.strategies.sweep import evaluate_candidates

    (record,) = load_records(run[0])
    other = make_hourly_data(seed=99)  # a different "asset"
    result = confirm.cross_asset(record, other, "OTHER/USD", evaluate_candidates, load_benchmarks())
    assert result["replicate"]["validation"]["n_retested"] >= 1
    assert result["transfer"]["validation"]["candidates"]["n"] == len(record.comparison)
    assert "buy_and_hold" in result["transfer"]["validation"]["benchmarks"]


def test_final_test_uses_only_the_holdout_and_logs_every_use(run: tuple[Path, Path], tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "final_test_log.jsonl"
    monkeypatch.setattr(confirm, "FINAL_TEST_LOG", log)
    records, data_path = run
    args = [str(records), "--final-test", "--out-dir", str(tmp_path / "out")]
    assert confirm.main(args) == 0
    assert confirm.main(args) == 0
    assert len(log.read_text().splitlines()) == 2  # every use is logged
    saved, other = sorted((tmp_path / "out").iterdir())  # two runs, two files (no overwrite)
    first_bar = json.loads(saved.read_text())["results"][0]["final_test"]["period"]["first_bar"]
    assert first_bar >= "2020-05-01"  # starts at the validation end: never overlaps train/validation


def test_cli_cross_asset_requires_another_dataset(run: tuple[Path, Path], tmp_path: Path) -> None:
    records, _ = run
    assert confirm.main([str(records), "--out-dir", str(tmp_path / "out")]) == 2
    other = tmp_path / "eth.parquet"
    make_hourly_data(seed=99).assign(symbol="SYN2/USD").to_parquet(other)
    assert confirm.main([str(records), "--data", str(other), "--experiments", "1",
                         "--out-dir", str(tmp_path / "out")]) == 0
    (saved,) = (tmp_path / "out").iterdir()
    content = json.loads(saved.read_text())
    assert content["mode"] == "cross_asset" and content["dataset"] == "SYN2/USD"
    assert {"replicate", "transfer"} <= set(content["results"][0])
