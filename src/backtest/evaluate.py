"""Evaluate strategy specs on a date split with the Python reference engine or the CUDA engine."""

from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import DEFAULT_COST_BPS, run_backtest
from src.backtest.metrics import HOURS_PER_YEAR, compute_metrics
from src.strategy.dsl import FIELDS, compile_spec, evaluate_signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_BINARY = PROJECT_ROOT / "cuda" / "build" / "backtest"

# asset -> metric name -> value
AssetMetrics = dict[str, dict[str, float]]


def load_datasets(files: list[str | Path]) -> dict[str, pd.DataFrame]:
    """Load OHLCV Parquet files into {asset name: DataFrame}, sorted by timestamp."""
    datasets = {}
    for file in files:
        path = Path(file) if Path(file).is_absolute() else PROJECT_ROOT / file
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        name = str(df["symbol"].iloc[0]) if "symbol" in df and len(df) else path.stem
        datasets[name] = df
    return datasets


@dataclass
class Evaluator:
    """Backtests specs on named date splits of every dataset.

    splits maps a name to (start, end) UTC dates; start inclusive, end exclusive, None = open.
    Signals are computed from the first bar of the data so indicators are warmed up at the split
    start; positions and returns are only counted inside the split.
    """

    datasets: dict[str, pd.DataFrame]
    splits: dict[str, tuple[str | None, str | None]]
    cost_bps: float = DEFAULT_COST_BPS
    periods_per_year: float = HOURS_PER_YEAR
    backend: str = "python"
    cuda_binary: Path = DEFAULT_CUDA_BINARY
    _data_files: dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend not in {"python", "cuda"}:
            raise ValueError(f"backend must be 'python' or 'cuda', got {self.backend!r}")
        if self.backend == "cuda" and not Path(self.cuda_binary).exists():
            raise FileNotFoundError(f"CUDA binary not found at {self.cuda_binary}; run `make -C cuda`")

    def split_range(self, asset: str, split: str) -> tuple[int, int]:
        """Row positions [start, end) of `split` in the asset's data."""
        start, end = self.splits[split]
        timestamps = self.datasets[asset]["timestamp"]
        lo = 0 if start is None else int(timestamps.searchsorted(pd.Timestamp(start, tz="UTC")))
        hi = len(timestamps) if end is None else int(timestamps.searchsorted(pd.Timestamp(end, tz="UTC")))
        return lo, hi

    def evaluate(self, specs: list[dict[str, Any]], split: str) -> list[AssetMetrics]:
        """Metrics per asset for each (validated) spec, in the same order as `specs`."""
        results: list[AssetMetrics] = [{} for _ in specs]
        for asset in self.datasets:
            start, end = self.split_range(asset, split)
            if end - start < 2:
                raise ValueError(f"split {split!r} has fewer than 2 bars for {asset}")
            run = self._run_cuda if self.backend == "cuda" else self._run_python
            for result, metrics in zip(results, run(specs, asset, start, end)):
                result[asset] = metrics
        return results

    def _run_python(self, specs: list[dict[str, Any]], asset: str, start: int, end: int) -> list[dict[str, float]]:
        data = self.datasets[asset].iloc[:end]
        window = data.iloc[start:end]
        out = []
        for spec in specs:
            signal = evaluate_signal(spec, data).iloc[start:end]
            result = run_backtest(window, signal, cost_bps=self.cost_bps)
            out.append(compute_metrics(result, self.periods_per_year))
        return out

    def _run_cuda(self, specs: list[dict[str, Any]], asset: str, start: int, end: int) -> list[dict[str, float]]:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.bin"
            programs_path = Path(tmp) / "programs.txt"
            write_data_file(self.datasets[asset], data_path)
            write_programs_file(specs, programs_path)
            proc = subprocess.run(
                [str(self.cuda_binary), str(data_path), str(programs_path), str(start), str(end),
                 repr(float(self.cost_bps)), repr(float(self.periods_per_year))],
                capture_output=True, text=True,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"CUDA backtest failed: {proc.stderr.strip()}")
        table = pd.read_csv(io.StringIO(proc.stdout)).drop(columns="index")
        rows = table.to_dict("records")
        for row in rows:
            row["n_trades"] = int(row["n_trades"])
        return rows


def write_data_file(data: pd.DataFrame, path: Path) -> None:
    """Binary input for the CUDA engine: int64 row count, then each field as float64 (field-major)."""
    values = np.stack([data[name].to_numpy(dtype="float64") for name in FIELDS])
    with open(path, "wb") as f:
        f.write(np.int64(len(data)).tobytes())
        f.write(np.ascontiguousarray(values).tobytes())


def write_programs_file(specs: list[dict[str, Any]], path: Path) -> None:
    """Text input for the CUDA engine: compiled register programs, one block per spec."""
    lines = [str(len(specs))]
    for spec in specs:
        instructions, output = compile_spec(spec)
        lines.append(f"{len(instructions)} {output}")
        lines.extend(f"{op} {dst} {a} {b} {ip} {fp!r}" for op, dst, a, b, ip, fp in instructions)
    path.write_text("\n".join(lines) + "\n")
