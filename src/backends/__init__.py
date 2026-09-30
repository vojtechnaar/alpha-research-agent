"""Backtest backends behind the evaluate_candidates contract.

    python  pandas reference implementation (src.strategies.sweep.evaluate_candidates)
    cpp     C++ on the CPU      (cuda/build/libbacktest_cpu.so,  `make -C cuda cpu`)
    cuda    CUDA on one GPU     (cuda/build/libbacktest_cuda.so, `make -C cuda`)
"""

from __future__ import annotations

import os

from src.strategies.sweep import CandidateEvaluator, evaluate_candidates

BACKENDS = ("python", "cpp", "cuda")


def get_evaluator(name: str = "python", device: int | None = None) -> CandidateEvaluator:
    """Evaluator for `name`. CUDA runs on `device` (default: $BACKTEST_DEVICE, else 0)."""
    if name == "python":
        return evaluate_candidates
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {name!r}")
    from src.backends.native import NativeEvaluator

    return NativeEvaluator(name, device=int(os.environ.get("BACKTEST_DEVICE", 0)) if device is None else device)
