"""Load the agent config and build the evaluator it describes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.backtest.evaluate import PROJECT_ROOT, Evaluator, load_datasets

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "agent.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config."""
    return yaml.safe_load(Path(path).read_text())


def build_evaluator(config: dict[str, Any], backend: str | None = None) -> Evaluator:
    """Evaluator over the configured datasets and splits; `backend` overrides the config."""
    bt = config["backtest"]
    return Evaluator(
        datasets=load_datasets(config["data"]["files"]),
        splits={name: tuple(bounds) for name, bounds in config["splits"].items()},
        cost_bps=bt["cost_bps"],
        periods_per_year=bt["periods_per_year"],
        backend=backend or bt["backend"],
        cuda_binary=PROJECT_ROOT / bt["cuda_binary"],
    )
