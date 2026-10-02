# Alpha Research Agent

An automated quantitative research system in which a local LLM acts as the researcher and a GPU
backtester tests its ideas, built end to end and evaluated with an emphasis on avoiding false
discoveries.

```
Qwen3-8B (+ LoRA) ──► hypothesis + strategy as validated JSON + parameter ranges
       ▲                                  │
       │                    CUDA engine: up to 20,000 variants on TRAIN
       │                                  │
 compact feedback ◄── frozen top 10 retested on VALIDATION, vs benchmarks, overfitting warnings
```

## What was built

- **LLM researcher with a safe strategy language.** Qwen3-8B proposes one hypothesis at a time as JSON
  over a whitelist of 12 features and 4 operators. Nothing it writes is executed; every reply is
  validated, and errors go back to the model as precise, machine-readable feedback.
- **Bounded research loop.** Hard limits on iterations and retries, duplicate and near-duplicate detection
  (canonical strategy identities, "idea families"), higher sampling temperature after repeats, and a
  compact feedback message with costs, condition activity and untried ideas.
- **Research methodology against overfitting.** Parameter sweeps on a train period, the best 10 frozen and
  retested on a separate validation period, like-for-like benchmarks (buy-and-hold, cash, naive momentum),
  minimum-trade filters, overfitting warnings, held-out markets, and a final test period that the loop
  never loads, used once per idea.
- **C++/CUDA backtester, 635× faster than Python.** One GPU thread per strategy variant, the same
  results as the pandas reference engine (max difference 1.8e-12, parity-tested). A hypothesis is
  tested in about a second, so the bottleneck moved from computation to the LLM.
- **Multi-asset data.** Hourly crypto (BTC, ETH) and daily stocks, bonds, gold and FX, in one format.
- **LoRA fine-tuning pipeline.** About 1,900 research experiments collected, filtered into ~480
  good research steps, and used to fine-tune Qwen3-8B with LoRA (loss on the proposal only, memory-efficient
  for 8k-token prompts).
- **Pre-registered evaluation of the researcher itself.** Base vs fine-tuned models compared on held-out
  markets in 30 paired runs, with the primary metric, minimum effect and decision rule fixed in advance,
  reported with bootstrap confidence intervals.
- **About 4,700 lines of Python and CUDA, ~170 offline tests.**

## What the project showed

- **Engineering made the LLM a usable researcher:** first-try valid proposals went from 0% to about 60–70%,
  useful experiments per 10 LLM calls from 0 to about 2.5–3, and idea diversity per run from 1–3 to 7–8 families.
- **Fine-tuning improved the mechanics, not the judgement:** LoRA reduced format and unit mistakes, but its
  gain in useful ideas per LLM call was small and not statistically significant, even with 3× more data.
  Imitating good past proposals doesn't teach a model to find new ones or to stop repeating itself.
- **The selection bias is real:** the best of hundreds of experiments looks much better on validation than
  on untouched data, which is exactly what the frozen retests and the one-time final test are there to expose.

**What it would take to find market-beating strategies:** a stronger (larger) LLM with better research
judgement; preference training (e.g. DPO) against repeated and failed ideas; much more and richer data
(lower timeframes such as minute bars, order-book and alternative data, more assets and longer histories);
and formal statistics for multiple testing (deflated Sharpe ratio, walk-forward validation).

## Documentation

- [docs/architecture.md](docs/architecture.md): the pipeline, the research loop, the splits, records and design decisions.
- [docs/strategy_engine.md](docs/strategy_engine.md): the spec format, features, operators, experiments, the backend interface and the CUDA mapping.
- [docs/progress.md](docs/progress.md): every problem found and how it was fixed, the measurements, and the open issues.

| Path | What it does |
|---|---|
| `src/data/download.py` | Downloads hourly OHLCV to `data/raw/` (config: `configs/data.yaml`) |
| `src/strategies/` | StrategySpec schema, feature and operator registries, evaluator, sweeps, CLI |
| `src/backtest/` | Backtest engine (execution lag, costs) and metrics |
| `src/research/` | Train/validation experiments, benchmarks, summaries, experiment records, reports |
| `src/agents/` | Prompts generated from the registries, proposal parsing, the bounded research loop |
| `src/backends/`, `cuda/` | Native backtest engine: C++ (CPU) and CUDA (GPU), same results as Python |
| `src/models/llm.py` | Qwen3-8B on one chosen GPU |
| `configs/` | Example strategies, parameter spaces, benchmarks and an example LLM proposal |

## Usage

```bash
pip install -r requirements.txt
python -m pytest                                   # offline tests: no network, no GPU, no market data

python -m src.data.download                        # hourly BTC/ETH history -> data/raw/ (server)
python -m src.data.download_yahoo                  # daily SPY, QQQ, TLT, GLD, EURUSD -> data/raw/yahoo_*_1d.parquet

# Train sweep -> top 10 frozen -> validation retest -> benchmarks -> experiment record
python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
    --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility_extended.json \
    --train-start 2017-01-01 --train-end 2023-01-01 --validation-start 2023-01-01 --validation-end 2025-01-01 \
    --top 10 --transaction-cost 0.001

# Native engines (server): GPU + multi-core C++; then compare all three
make -C cuda && make -C cuda cpu OMP=1
python -m src.backends.benchmark --data data/raw/bitstamp_BTC-USD_1h.parquet \
    --strategy configs/strategies/momentum_low_volatility.json --space configs/sweeps/momentum_low_volatility_dense.json

# Research loop: at most 5 Qwen proposals, backtests on the GPU
nvidia-smi                                         # pick a GPU with >= 20 GB free
python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 5 --device cuda:2 \
    --backend cuda --backtest-device 2 --max-candidates 20000

# Same pipeline without the LLM, replaying a saved proposal
python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 1 \
    --replay configs/proposals/example_momentum_low_volatility.json

python -m src.research.report data/research_runs/<run_id>/experiments.jsonl   # review a run
python -m src.research.runs                                                    # metrics of all saved runs

# LoRA: build the training set from saved runs (ETH and GLD held out), train, compare
bash scripts/collect.sh 2 5 [--adapter checkpoints/lora/v1]                    # collect runs on GPU 2 for 5 hours (BTC SPY QQQ TLT)
python -m src.models.lora_data
python -m src.models.train_lora --device cuda:0 --name v1
bash scripts/compare_lora.sh 2                                                 # base Qwen on ETH + GLD, seeds 1-5
bash scripts/compare_lora.sh 4 --adapter checkpoints/lora/v1                  # the same with LoRA
python -m src.research.runs --by model --datasets ETH/USD GLD                  # base vs LoRA

# Confirm chosen experiments without the LLM: on another asset, then ONCE on the final test (2025+)
python -m src.research.confirm data/research_runs/<run_id>/experiments.jsonl --experiments 3 6 \
    --data data/raw/bitstamp_ETH-USD_1h.parquet --backend cuda
python -m src.research.confirm data/research_runs/<run_id>/experiments.jsonl --experiments 3 6 --final-test --backend cuda
```

Data after `--validation-end` (default 2025-01-01) is never loaded by experiments. It's reserved as the final test.
