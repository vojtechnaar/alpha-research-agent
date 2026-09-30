# Architecture

The goal is an automated quantitative research system:
- **Qwen3-8B is the researcher.** It decides *what* to test: a hypothesis, a strategy and a parameter space.
- **A numerical engine is the experiment engine.** It runs the parameter search and backtests, in Python, C++ or CUDA (`--backend`).
- **Only a compact summary goes back to the LLM,** which then proposes the next hypothesis.

```
Market data (CCXT → Parquet)                                   src/data/                  NOW
        ↓
Qwen3-8B (one selected GPU)  (+ LoRA later)                    src/models/llm.py          NOW (LoRA: FUTURE)
        ↓
ResearchProposal JSON: hypothesis + StrategySpec + parameter space   src/agents/proposals.py    NOW
        ↓
Validation against the trusted registries (rejections → repair)      src/strategies/schema.py   NOW
        ↓
Parameter sweep (Cartesian product, candidate budget)           src/strategies/sweep.py    NOW
        ↓
Candidate evaluator backend:  Python | C++ CPU | CUDA GPU       src/backends/, cuda/       NOW
        ↓
TRAIN backtests (all candidates) → top N by train metric → frozen
        ↓
VALIDATION backtests (the frozen top N only)                    src/research/experiment.py NOW
        ↓
Benchmarks on the same periods (buy-and-hold, cash, naive momentum)  src/research/benchmarks.py NOW
        ↓
Summaries + robustness warnings                                 src/research/summary.py    NOW
        ↓
Experiment record (JSONL)                                       src/research/records.py    NOW
        ↓
Compact feedback (~500 tokens) → Qwen → next hypothesis         src/agents/prompts.py      NOW
        ↑_________________ bounded loop: src/agents/research.py (--hypotheses N) _______|
```

## Components

| Component | Path | Role |
|---|---|---|
| Data | `src/data/download.py` | Hourly OHLCV from Bitstamp → `data/raw/*.parquet` |
| Spec | `src/strategies/schema.py` | `StrategySpec`, `ResearchProposal`, validation, rejection codes |
| Primitives | `src/strategies/features.py`, `operators.py` | Trusted feature, operator and logic registries |
| Evaluation | `src/strategies/evaluator.py`, `sweep.py` | Spec → positions → backtest; candidate generation; `evaluate_candidates` (Python reference backend) |
| Native backends | `cuda/backtest.cu`, `src/backends/` | The same evaluation in C++ (CPU, OpenMP) and CUDA (GPU), called via ctypes; parity-tested against Python |
| Backtest | `src/backtest/` | Execution lag, costs, asset-agnostic metrics |
| Experiments | `src/research/experiment.py` | Train sweep → top N → validation retest |
| Benchmarks | `src/research/benchmarks.py`, `configs/benchmarks/` | Buy-and-hold, flat (cash), naive 24h momentum |
| Summaries | `src/research/summary.py` | Distributions, parameter sensitivity, degradation, warnings |
| Records | `src/research/records.py`, `report.py` | JSONL experiment records and readable reports |
| LLM | `src/models/llm.py` | Qwen3-8B on one explicitly chosen GPU |
| Agent | `src/agents/prompts.py`, `proposals.py`, `research.py` | Prompts from registries, parsing, bounded loop |

## The research loop

```bash
python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 5 --device cuda:0
```

`--hypotheses 5` means **at most five research iterations**, which is five LLM proposals. It does not mean five backtests. Each iteration can backtest up to `--max-candidates` parameter combinations (default 1,000). One iteration runs these steps:

1. **Build the context.** On the first iteration it's "no experiments yet". After that it's the compact feedback: one line per earlier hypothesis, an EXPLORATION block (features used so far, features not yet tried, position rules used), and details of the most recent `--recent` experiments. The system prompt's search guidance scales with the candidate budget: with a large budget, Qwen is asked for 5–10 values per parameter over wide ranges.
2. **Ask Qwen for one proposal:** JSON with `hypothesis`, `rationale`, `strategy` and `parameter_space`.
3. **Parse and validate it before anything runs.** Validation checks the JSON syntax, the features and operators against the registries, every parameter value, the size of the search (`SEARCH_SPACE_TOO_LARGE`), and exact repeats of experiments already run (`DUPLICATE_PROPOSAL`). The repeat check ignores names, condition ids and condition order, so renaming a condition doesn't get a repeat through. The same rules with *different* parameter ranges count as a refinement and are allowed. A rejection names the experiment that was repeated and says what to change.
4. **Repair invalid replies (bounded).** The validator's error is shown to Qwen with a request to fix only the JSON. That's at most `--max-proposal-retries` extra calls (default 2), so at most 3 LLM calls per iteration. The parser takes the first complete JSON object and ignores anything after it. A reply that hit the token limit is reported as cut off, with instructions to shorten it. After a duplicate, the retry re-sends the original request with a note naming the repeated experiment, *without* Qwen's copied reply, because a small model tends to repeat the last JSON it wrote.
5. **Generate the Cartesian product** of the parameter space and enforce the candidate budget.
6. **Train sweep:** backtest every candidate on TRAIN.
7. **Select the top N** (`--top`) by the selection metric (default Sharpe), using train results only. Candidates with fewer than `--min-train-trades` trades aren't eligible, and a candidate that traded identically to a better-ranked one is skipped, so the N slots go to N different strategies.
8. **Freeze** the selected parameters.
9. **Validation retest:** backtest only those frozen candidates on VALIDATION.
10. **Benchmarks:** evaluate buy-and-hold, cash and naive momentum on both periods, with the same costs and backend.
11. **Summarise:** train distribution, validation results, train → validation degradation, trading costs (trades per year, bars between position changes, exposure, fees as % of capital per year), parameter sensitivity and warnings.
12. **Save** the experiment record, plus the train table as a separate CSV.
13. **Feed the compact summary** into the next iteration.

**Stopping rules:**
- The loop is a `for` over `range(hypotheses)`, capped at 100. There is no other loop.
- A proposal still invalid after its retries is recorded as `rejected`, and that iteration is used up.
- After `--max-consecutive-rejections` rejected iterations in a row (default 3), the run stops early. A model that keeps breaking the format is wasting GPU time.
- An evaluation error is recorded as `failed` and the loop continues.

So the worst-case LLM cost of a run is `hypotheses × (1 + max_proposal_retries)` calls.

## Train, validation and final test

| Split | Default | Used for |
|---|---|---|
| TRAIN | 2017-01-01 → 2023-01-01 | The parameter search. All candidates are ranked here. |
| VALIDATION | 2023-01-01 → 2025-01-01 | Retesting the frozen top N. **Never used to select parameters.** |
| FINAL TEST | 2025-01-01 → now | Nothing yet. Reserved for a final, untouched evaluation. |

- **The final test is enforced in code.** `run_experiment` drops every bar at or after the validation end before doing anything, so no experiment can see final-test data.
- **Features stay causal.** They're computed over the history up to each bar, so indicators are warmed up at the start of each split without using later data.

**Important: validation doesn't stay out-of-sample forever.** Within one experiment, validation is unseen. The parameters are frozen before it's evaluated. But the loop shows validation results to Qwen, and Qwen's next hypothesis is shaped by them. Over many iterations the *choice of ideas* adapts to the validation period, so validation gradually becomes training information, just as it would for a human researcher who keeps checking the same holdout. The mitigations:

- **Blind validation:** `--no-validation-feedback` shows Qwen train results only. Validation is still computed and recorded, but it no longer guides the search.
- **An untouched final test period:** used only once the research is finished. Treat it as spent after you look at it.
- **Future options:** walk-forward evaluation (rolling train/validation windows), cross-asset validation (discover on BTC, confirm on ETH), and a fresh final-test period when new data arrives.

## Benchmarks

`configs/benchmarks/*.json` are ordinary StrategySpecs:
- **`buy_and_hold`** and **`flat`** have no conditions, so they're always long or always in cash.
- **`naive_momentum_24h`** is the simplest trend rule.

They run through the same evaluator, costs and periods as the candidates, and a test checks that their periods match exactly.

A candidate beating buy-and-hold's Sharpe is **not** a conclusion that it has alpha. After many trials the best result is inflated by selection, and the report says so explicitly.

## Experiment records

Records are stored per run:

```
results/experiments/<run_id>/run.json             settings, loop limits, model + generation settings, seed,
                                                  git commit, data path, benchmarks, system prompt
results/experiments/<run_id>/experiments.jsonl    one record per iteration (completed / rejected / failed)
results/experiments/<run_id>/sweeps/*.csv         full train table per experiment (referenced by path)
results/experiments/manual/                       records from `python -m src.strategies.run ... --validation-start`
```

A record holds the following, and never any time series (a typical record is a few KB):
- ids and the timestamp
- dataset and data path
- the train and validation periods
- the transaction cost, as a fraction and in bps
- the selection metric, top N, candidate budget and minimum trades
- the hypothesis and rationale
- the StrategySpec and its readable description
- the parameter space and number of candidates
- the train summary (quantiles, share positive, best candidates)
- the validation summary (median, best, worst, degradation)
- the top-N comparison rows
- the benchmarks for both periods
- the parameter sensitivity
- the warnings and timing
- the LLM context and every raw attempt, with error code, seed and token statistics

`results/` is git-ignored. A small schema fixture lives in `tests/fixtures/`. Print any run with `python -m src.research.report <experiments.jsonl>`.

## Reproducibility

- **`run.json` stores everything needed to repeat a run:** the command line, git commit, settings, generation settings and the full system prompt.
- **Seeds:** each LLM call is seeded (`--seed`, default 42, offset by iteration and attempt). `--temperature 0` switches to greedy decoding. GPU inference is still not bit-exact, so identical text across runs isn't guaranteed.
- **Replay:** `--replay proposals.json` feeds saved proposals through the full pipeline without the LLM. Given the same data and code, it reproduces the numerical results exactly. It's also the quickest check that the pipeline works on the server.

## Why these design choices

- **Generated code is never executed.** Qwen produces JSON only, and it's validated against registries that humans wrote and tested. Unsupported requests are rejected with `UNSUPPORTED_FEATURE`, `UNSUPPORTED_OPERATOR`, `INVALID_SPEC`, `MALFORMED_JSON`, `SEARCH_SPACE_TOO_LARGE` or `DUPLICATE_PROPOSAL`, and the code goes back to Qwen as feedback. There is no `eval` or `exec`. New primitives go through human review.
- **The LLM decides; the engine explores.** Qwen generates about 10 tokens per second on DeepDish, so one proposal (about 250 tokens of JSON) costs about 25 s. One Python backtest costs about 25 ms. So each hypothesis gets one LLM call and hundreds of backtests, and Qwen reads a summary of about 500 tokens, never CSVs or prices.
- **`--max-new-tokens` defaults to 512.** Generation stops at the end of the JSON, so the limit only caps runaway replies. A proposal with wide grids (several parameters × 5–8 values) can exceed 300 tokens, and the earlier limit of 320 cut such replies off. A cut-off reply is wasted entirely, which costs more than the extra tokens.
- **Correctness first, then CUDA.** Everything above the backend depends only on the `evaluate_candidates` contract (`CandidateEvaluator` in `sweep.py`): data, candidates, costs and a period in; one metrics row per candidate out. The C++ and CUDA backends (`src/backends/`) implement the same contract, so choosing one is just `--backend cuda`. Nothing else changes: not the StrategySpec, the LLM, the records, the validation logic or the reporting. `tests/test_native_backend.py` requires them to match the Python engine (every feature × operator, AND/OR, all position combinations, benchmarks, price gaps, flat prices, zero volume), and `python -m src.backends.benchmark` compares speed and results on real data. Every record includes `timing.backend` and `ms_per_train_candidate`.
- **The loop has explicit limits.** See the stopping rules above.

**Measured speed-up** (DeepDish4, 30 September 2026, BTC/USD hourly bars from 2017 to 2022, a 5,600-candidate sweep): pandas takes 28.5 ms per candidate, C++ with 32 OpenMP threads 0.175 ms (163×), and CUDA on an RTX A6000 0.045 ms (635×). The results are identical: 0 trade-count mismatches and at most 1.8e-12 metric difference. Details are in `docs/strategy_engine.md`.

## Avoiding false discoveries

Implemented now. These are descriptive, not formal tests:
- number of candidates tested
- the train metric distribution (min, p10, p25, median, p75, p90, max) and share positive
- the top candidates
- parameter sensitivity (mean metric per parameter value)
- validation results of the frozen top N and their train → validation change
- benchmarks on both periods
- warnings:
  - multiple testing
  - median train Sharpe ≤ 0
  - best value at the edge of the grid
  - top candidates with identical results (a parameter with no effect)
  - weak or negative validation
  - too few validation trades
  - buy-and-hold comparison

**Future work, to be implemented carefully rather than approximated:**
- **Probabilistic Sharpe Ratio** and **Deflated Sharpe Ratio.** The DSR needs the number of *effective* independent trials and the variance of Sharpe across trials. The candidates here are highly correlated, so the raw candidate count would over-deflate.
- **Bootstrap confidence intervals** that respect autocorrelation (block bootstrap), using the per-bar returns the engine already produces.
- **Multiple-testing corrections** across the whole research history, not just one sweep.
- **Walk-forward** analysis and **cross-asset** validation.
- **Cost sensitivity** (re-running the frozen top N at several cost levels) and **regime breakdowns**.

## LoRA: later, and trained on research behaviour

LoRA isn't implemented yet. The experiment records are designed to become its training data:
- **Input:** the context the model saw (`llm.context`, together with the run's `system_prompt`), meaning the available primitives, earlier hypotheses and experiment summaries.
- **Target:** the proposal it produced (hypothesis, StrategySpec, parameter space, rationale).
- **Labels and metadata:** validation results, benchmark comparison, robustness warnings, degradation, rejection codes.

Examples won't be labelled "good" just for a high in-sample Sharpe. Label quality will come from validation and final-test behaviour, parameter stability and the warnings. Bad and rejected proposals stay in the data, so the model can learn what not to do.

The planned experiment compares **base Qwen3-8B** against **Qwen3-8B + quant LoRA** on identical research tasks with the same budget. Measures:
- valid-proposal rate
- rate of unsupported features or operators
- retries needed
- hypothesis diversity
- validation and final-test performance
- parameter stability
- useful hypotheses per GPU-hour

## GPU usage

DeepDish4 has several shared RTX A6000s, and Qwen3-8B (bf16, about 16 GB) fits on one. `QwenGenerator` puts the whole model on a single device and never spreads it across GPUs.

To pick a free GPU:

```bash
nvidia-smi                 # look at Memory-Usage and the Processes table
python -m src.agents.research ... --device cuda:2      # or: QWEN_DEVICE=cuda:2 python -m ...
```

Choose a GPU with at least about 20 GB free and no heavy processes. Never kill other users' processes.

With `--backend cuda`, the backtests run on `--backtest-device` (default `$BACKTEST_DEVICE` or 0). They need little memory: the feature buffers and a metrics table, typically well under 1 GB. So they can share the GPU with Qwen, or use another free one. The LLM and the backtests take turns within an iteration, so they never compete for the GPU at the same time.
