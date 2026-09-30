"""Controlled research loop: Qwen proposes a hypothesis, the engine tests it, a summary goes back.

    python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 5 --device cuda:0
    python -m src.agents.research ... --backend cuda --max-candidates 20000   # backtests on the GPU

--hypotheses N is a hard upper bound on research iterations (LLM proposals), NOT on backtests;
each iteration may backtest up to --max-candidates parameter combinations.

One iteration:
  context -> Qwen proposal (at most 1 + max_proposal_retries LLM calls) -> validation ->
  train sweep -> top N (train only) -> frozen retest on validation -> benchmarks ->
  experiment record -> compact feedback for the next iteration.

Stopping: after N iterations, or earlier when `max_consecutive_rejections` proposals in a row
were invalid even after retries. A rejected proposal uses up its iteration. There is no other loop.

Against repetition: an idea family (same rules, any parameter values) may be tested at most
`max_per_family` times, and retries after a repeat are sampled at a higher temperature.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import pandas as pd

from src.agents.proposals import TRUNCATED_HINT, candidate_identities, parse_proposal
from src.backends import BACKENDS, get_evaluator
from src.agents.prompts import feedback_message, initial_request, repair_message, system_prompt
from src.research.benchmarks import DEFAULT_BENCHMARKS_DIR, PROJECT_ROOT, load_benchmarks
from src.research.experiment import ExperimentSettings, Period, run_experiment
from src.research.feature_ranges import feature_ranges
from src.research.records import ExperimentRecord, add_results, append_record, new_record
from src.research.report import format_record
from src.strategies.schema import (
    DUPLICATE_PROPOSAL,
    FAMILY_EXHAUSTED,
    MALFORMED_JSON,
    ResearchProposal,
    SpecError,
    StrategySpec,
)
from src.strategies.sweep import CandidateEvaluator, evaluate_candidates

MAX_HYPOTHESES = 100  # guard against typos like --hypotheses 5000
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"


class Generator(Protocol):
    """QwenGenerator, a replay of saved proposals, or a fake in tests."""

    def generate(
        self, messages: list[dict[str, str]], n: int = 1, seed: int | None = None, temperature: float | None = None
    ) -> list[str]: ...


@dataclass(frozen=True)
class LoopSettings:
    hypotheses: int
    max_proposal_retries: int = 2
    max_consecutive_rejections: int = 3
    recent_experiments: int = 2
    include_validation_in_feedback: bool = True
    seed: int | None = 42
    max_per_family: int = 2  # experiments per idea family (same rules, any parameter values)
    retry_temperatures: tuple[float, ...] = (1.0, 1.3)  # sampling temperature for retries after a repeat

    def __post_init__(self) -> None:
        if not 1 <= self.hypotheses <= MAX_HYPOTHESES:
            raise ValueError(f"hypotheses must be in 1..{MAX_HYPOTHESES}")
        if not 0 <= self.max_proposal_retries <= 5:
            raise ValueError("max_proposal_retries must be in 0..5")
        if self.max_consecutive_rejections < 1 or self.recent_experiments < 1 or self.max_per_family < 1:
            raise ValueError("max_consecutive_rejections, recent_experiments and max_per_family must be >= 1")
        if not self.retry_temperatures or any(t <= 0 for t in self.retry_temperatures):
            raise ValueError("retry_temperatures must be positive")


def request_proposal(
    generator: Generator,
    messages: list[dict[str, str]],
    max_candidates: int,
    seen: Mapping[str, str],
    max_retries: int,
    seed: int | None,
    tested: Mapping[str, str] | None = None,
    families: Mapping[str, list[str]] | None = None,
    max_per_family: int | None = None,
    retry_temperatures: tuple[float, ...] = (1.0, 1.3),
) -> tuple[ResearchProposal | None, list[dict[str, Any]], SpecError | None]:
    """Ask for a proposal; on a validation error, show the error and ask for a repair.

    Makes at most 1 + max_retries LLM calls. Returns (proposal or None, attempts, last error).
    A reply that hit the token limit is reported as truncated (with how to shorten it). After a
    repeat (duplicate, already-tested grid, exhausted idea family) the model's copied reply is NOT
    kept in the conversation - a small model tends to repeat the last JSON it wrote - and the next
    attempt is sampled at the next of `retry_temperatures` to break out of the repetition. Format
    errors are retried at the normal temperature.
    """
    conversation = list(messages)
    attempts: list[dict[str, Any]] = []
    error: SpecError | None = None
    duplicate_notes: list[str] = []
    temperature: float | None = None  # None = the generator's own setting
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        text = generator.generate(conversation, n=1, seed=None if seed is None else seed + attempt,
                                  temperature=temperature)[0]
        info = {"attempt": attempt, "response": text, "seconds": round(time.perf_counter() - started, 2),
                "temperature": temperature, **getattr(generator, "last_stats", {})}
        try:
            proposal = parse_proposal(text, max_candidates, seen, tested, families, max_per_family)
        except SpecError as exc:
            if exc.code == MALFORMED_JSON and info.get("hit_max_new_tokens"):
                exc = SpecError(MALFORMED_JSON, TRUNCATED_HINT)
            error = exc
            attempts.append({**info, "error": str(exc), "code": exc.code})
            if exc.code in (DUPLICATE_PROPOSAL, FAMILY_EXHAUSTED):
                temperature = retry_temperatures[min(len(duplicate_notes), len(retry_temperatures) - 1)]
                duplicate_notes.append(f"NOTE: a previous reply was rejected: {exc}")
                request = {"role": "user", "content": "\n\n".join([messages[-1]["content"], *duplicate_notes])}
                conversation = [*messages[:-1], request]
            else:
                conversation += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": repair_message(exc)}]
            continue
        attempts.append({**info, "error": None, "code": None})
        return proposal, attempts, None
    return None, attempts, error


def run_research(
    generator: Generator,
    data: pd.DataFrame,
    settings: ExperimentSettings,
    loop: LoopSettings,
    run_dir: Path,
    benchmarks: dict[str, StrategySpec] | None = None,
    dataset: str = "",
    data_path: str = "",
    run_id: str = "",
    evaluator: CandidateEvaluator = evaluate_candidates,
    log: Callable[[str], None] = print,
    system: str | None = None,
) -> list[ExperimentRecord]:
    """Run at most `loop.hypotheses` research iterations; returns their records (also saved as JSONL).

    `system` defaults to build_system_prompt(data, settings) (rules, primitives, train feature ranges).
    """
    system = system or build_system_prompt(data, settings)
    records: list[ExperimentRecord] = []
    seen: dict[str, str] = {}  # proposal key -> label of the experiment that ran it
    tested: dict[str, str] = {}  # identity of every backtested combination -> its experiment
    families: dict[str, list[str]] = {}  # idea family -> experiments that tested it
    consecutive_rejections = 0

    for iteration in range(loop.hypotheses):
        context = (feedback_message(records, loop.recent_experiments, loop.include_validation_in_feedback,
                                    loop.max_per_family) if records else initial_request())
        messages = [{"role": "system", "content": system}, {"role": "user", "content": context}]
        seed = None if loop.seed is None else loop.seed + 1000 * iteration
        proposal, attempts, error = request_proposal(
            generator, messages, settings.max_candidates, seen, loop.max_proposal_retries, seed, tested,
            families, loop.max_per_family, loop.retry_temperatures,
        )
        common = dict(dataset=dataset, data_path=data_path, run_id=run_id, iteration=iteration,
                      llm={"context": context, "seed": seed, "attempts": attempts})

        if proposal is None:
            consecutive_rejections += 1
            record = new_record(settings, "rejected", error=str(error), **common)
        else:
            consecutive_rejections = 0
            label = f"experiment {iteration + 1} ('{proposal.hypothesis}')"
            seen[proposal.key()] = label
            record = new_record(
                settings, "completed", hypothesis=proposal.hypothesis, rationale=proposal.rationale,
                strategy_spec=proposal.strategy.to_dict(), strategy_description=proposal.strategy.describe(),
                parameter_space=proposal.parameter_space, requested_parameter_space=proposal.requested_space,
                **common,
            )
            try:
                result = run_experiment(data, proposal.strategy, proposal.parameter_space, settings,
                                        benchmarks, dataset, evaluator)
                add_results(record, result, proposal.parameter_space)
                csv_path = run_dir / "sweeps" / f"{record.experiment_id}_train.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                result.train.to_csv(csv_path, index=False)
                record.sweep_csv = str(csv_path)
                for identity in candidate_identities(proposal):
                    tested.setdefault(identity, label)
                families.setdefault(proposal.strategy.family(), []).append(f"experiment {iteration + 1}")
            except Exception as exc:  # recorded and reported; one bad experiment should not end the run
                record.status, record.error = "failed", f"{type(exc).__name__}: {exc}"

        append_record(run_dir / "experiments.jsonl", record)
        records.append(record)
        log(f"\n[{iteration + 1}/{loop.hypotheses}] LLM calls: {len(attempts)}\n{format_record(record)}")
        if consecutive_rejections >= loop.max_consecutive_rejections:
            log(f"Stopping early: {consecutive_rejections} consecutive proposals were rejected.")
            break
    return records


def build_system_prompt(data: pd.DataFrame, settings: ExperimentSettings) -> str:
    """System prompt including the typical feature values on the TRAIN period only."""
    return system_prompt(settings.max_candidates, settings.transaction_cost,
                         ranges=feature_ranges(data, settings.train))


class ReplayGenerator:
    """Returns saved proposals in order instead of calling the LLM (debugging / reproduction)."""

    def __init__(self, path: Path) -> None:
        text = path.read_text()
        items = json.loads(text) if path.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
        items = items if isinstance(items, list) else [items]
        self.replies = [item if isinstance(item, str) else json.dumps(item) for item in items]
        self.settings = {"replay": str(path)}

    def generate(self, messages: list[dict[str, str]], n: int = 1, seed: int | None = None,
                 temperature: float | None = None) -> list[str]:
        return [self.replies.pop(0) if self.replies else "" for _ in range(n)]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bounded LLM research loop (see module docstring).")
    p.add_argument("--data", type=Path, required=True, help="OHLCV Parquet file")
    p.add_argument("--hypotheses", type=int, required=True, help=f"max research iterations (1..{MAX_HYPOTHESES})")
    p.add_argument("--train-start", default="2017-01-01")
    p.add_argument("--train-end", default="2023-01-01")
    p.add_argument("--validation-start", default="2023-01-01")
    p.add_argument("--validation-end", default="2025-01-01", help="data from here on is never loaded (final test)")
    p.add_argument("--top", type=int, default=10, help="candidates retested on validation")
    p.add_argument("--selection-metric", default="sharpe")
    p.add_argument("--min-trades-per-year", type=float, default=10.0,
                   help="candidates trading less often are not selected; fewer validation trades are flagged")
    p.add_argument("--max-candidates", type=int, default=1000, help="max parameter combinations per hypothesis")
    p.add_argument("--transaction-cost", type=float, default=0.001, help="fraction per unit turnover (0.001 = 10 bps)")
    p.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    p.add_argument("--backend", choices=BACKENDS, default="python", help="backtest engine (cpp/cuda need make -C cuda)")
    p.add_argument("--backtest-device", type=int, help="GPU index for --backend cuda (default: $BACKTEST_DEVICE or 0)")
    p.add_argument("--max-proposal-retries", type=int, default=2)
    p.add_argument("--max-consecutive-rejections", type=int, default=3)
    p.add_argument("--recent", type=int, default=2, help="experiments shown in detail in the feedback")
    p.add_argument("--max-per-family", type=int, default=2,
                   help="experiments allowed per idea family (same rules, any parameter values)")
    p.add_argument("--no-validation-feedback", action="store_true",
                   help="show the LLM train results only (keeps validation blind)")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--device", help="e.g. cuda:2 (default: $QWEN_DEVICE or cuda:0)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="generation stops at the end of the JSON; this only caps runaway replies")
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy decoding")
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay", type=Path, help="use proposals from a .json/.jsonl file instead of Qwen")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = ExperimentSettings(
            train=Period(args.train_start, args.train_end),
            validation=Period(args.validation_start, args.validation_end),
            transaction_cost=args.transaction_cost, top_n=args.top, selection_metric=args.selection_metric,
            max_candidates=args.max_candidates, min_trades_per_year=args.min_trades_per_year,
        )
        loop = LoopSettings(args.hypotheses, args.max_proposal_retries, args.max_consecutive_rejections,
                            args.recent, not args.no_validation_feedback, args.seed, args.max_per_family)
    except ValueError as exc:
        print(f"Invalid settings: {exc}", file=sys.stderr)
        return 2

    data = pd.read_parquet(args.data).sort_values("timestamp").reset_index(drop=True)
    dataset = str(data["symbol"].iloc[0]) if "symbol" in data else args.data.stem
    benchmarks = load_benchmarks(args.benchmarks_dir)
    evaluator = get_evaluator(args.backend, args.backtest_device)  # fails fast if the library is not built
    backend_info = getattr(evaluator, "info", "python (pandas reference)")
    print(f"Backtest backend: {backend_info}")

    if args.replay:
        generator: Generator = ReplayGenerator(args.replay)
    else:
        from src.models.llm import QwenGenerator  # imported lazily: tests and replays need no torch

        generator = QwenGenerator(args.model_id, device=args.device, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    system = build_system_prompt(data, settings)
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "command": sys.argv,
        "git_commit": _git_commit(),
        "data_path": str(args.data),
        "dataset": dataset,
        "settings": settings.to_dict(),
        "backend": {"name": getattr(evaluator, "__name__", args.backend), "info": backend_info},
        "loop": asdict(loop),
        "generator": getattr(generator, "settings", {}),
        "benchmarks": {name: spec.to_dict() for name, spec in benchmarks.items()},
        "system_prompt": system,
    }, indent=2))
    print(f"Run {run_id}: at most {loop.hypotheses} hypotheses, <= {settings.max_candidates} candidates each, "
          f"cost {settings.transaction_cost:g}. Records: {run_dir / 'experiments.jsonl'}")

    records = run_research(generator, data, settings, loop, run_dir, benchmarks, dataset, str(args.data), run_id,
                           evaluator=evaluator, system=system)
    print(f"\nDone: {len(records)} iterations "
          f"({sum(r.status == 'completed' for r in records)} completed, "
          f"{sum(r.status == 'rejected' for r in records)} rejected, {sum(r.status == 'failed' for r in records)} failed).")
    print(f"Review: python -m src.research.report {run_dir / 'experiments.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
