"""Yahoo daily-bar conversion and research-run metrics (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.research import LoopSettings, run_research
from src.data.download_yahoo import output_path, to_ohlcv
from src.research.benchmarks import load_benchmarks
from src.research.experiment import ExperimentSettings
from src.research.runs import main as runs_main, paired_comparison, summarize_run, totals

from conftest import make_hourly_data


def yahoo_history(volume: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-04"], tz="America/New_York")
    return pd.DataFrame({"Open": [2.0, 1.0, 3.0, 3.1], "High": [2.5, 1.5, 3.5, 3.6], "Low": [1.5, 0.5, 2.5, 2.6],
                         "Close": [2.2, 1.2, np.nan, 3.2], "Volume": volume, "Dividends": 0.0}, index=index)


def test_yahoo_history_is_converted_to_the_project_format() -> None:
    df = to_ohlcv(yahoo_history([10, 20, 30, 40]), "SPY")
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert df["timestamp"].tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"], utc=True))
    assert df["close"].tolist() == [1.2, 2.2, 3.2]  # sorted; missing close dropped; duplicate date keeps the last
    assert (df["symbol"] == "SPY").all()


def test_markets_without_volume_get_nan_volume() -> None:
    df = to_ohlcv(yahoo_history([0, 0, 0, 0]), "EURUSD=X")
    assert df["volume"].isna().all()
    assert output_path("EURUSD=X").name == "yahoo_EURUSDX_1d.parquet"


class Replies:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = [json.dumps(r) for r in replies]

    def generate(self, messages, n=1, seed=None, temperature=None):
        return [self.replies.pop(0) if self.replies else "no json"]


def test_run_metrics(tmp_path: Path, settings: ExperimentSettings) -> None:
    proposal = {"hypothesis": "h", "strategy": {"name": "m", "conditions": [
        {"id": "m", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.0}]},
        "parameter_space": {"m.lookback": [24, 72]}}
    run_dir = tmp_path / "20260101-000000"
    run_research(Replies([proposal]), make_hourly_data(), settings,
                 LoopSettings(hypotheses=2, max_proposal_retries=0), run_dir, load_benchmarks(), "SYN/USD",
                 log=lambda _: None)
    (run_dir / "run.json").write_text(json.dumps({"dataset": "SYN/USD", "generator": {"model_id": "Qwen/Qwen3-8B"}}))
    row = summarize_run(run_dir)
    assert row["model"] == "Qwen/Qwen3-8B" and row["iterations"] == 2 and row["completed"] == 1
    assert row["llm_calls"] == 2 and row["first_try_valid"] == 0.5
    assert row["rejection_codes"] == {"MALFORMED_JSON": 1}
    assert row["families"] == 1 and row["features"] == 1
    assert 0 <= row["useful"] <= row["holds_up"] <= 1
    table = totals(pd.DataFrame([row, {**row, "run": "b"}]))
    assert table.loc[0, "runs"] == 2 and table.loc[0, "llm_calls"] == 4
    assert runs_main([str(tmp_path)]) == 0
    assert runs_main([str(tmp_path), "--datasets", "GLD"]) == 0  # no matching runs: handled
    assert runs_main([str(tmp_path), "--models", "Qwen/Qwen3-8B+v9"]) == 0  # no such model: handled


def test_paired_comparison_matches_runs_by_market_and_seed() -> None:
    rows = [("Qwen/Qwen3-8B", "ETH/USD", 1, 1.0), ("Qwen/Qwen3-8B+v1", "ETH/USD", 1, 3.0),
            ("Qwen/Qwen3-8B", "ETH/USD", 2, 2.0), ("Qwen/Qwen3-8B+v1", "ETH/USD", 2, 1.0),
            ("Qwen/Qwen3-8B", "GLD", 1, 0.0), ("Qwen/Qwen3-8B+v1", "GLD", 1, 2.0),
            ("Qwen/Qwen3-8B+v1", "GLD", 9, 5.0)]  # no base run with seed 9: not a pair
    table = pd.DataFrame(rows, columns=["model", "dataset", "seed", "useful_per_10_calls"])
    result = paired_comparison(table)
    assert result["base"] == "Qwen/Qwen3-8B" and result["other"] == "Qwen/Qwen3-8B+v1"
    assert (result["pairs"], result["wins"], result["losses"]) == (3, 2, 1)
    assert result["mean_difference"] == 1.0
    assert result["ci95"][0] <= 1.0 <= result["ci95"][1]
    assert result["per_dataset"]["GLD"] == {"pairs": 1, "wins": 1, "mean_difference": 2.0}
    assert paired_comparison(table[table["model"] == "Qwen/Qwen3-8B"]) is None  # one model: nothing to pair
