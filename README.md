# Alpha Research Agent

An automated quantitative crypto research system:
- **Qwen3-8B is the researcher.** It proposes a hypothesis, a validated JSON strategy spec and a parameter space.
- **A numerical engine runs the experiment.** It backtests every combination on a training period, retests the frozen best few on a validation period, and compares them with benchmarks. It's Python now, with CUDA later.
- **A compact summary goes back to Qwen** for the next hypothesis, within a strict iteration limit.

- [docs/architecture.md](docs/architecture.md): the pipeline, the research loop, the splits, records and design decisions.
- [docs/strategy_engine.md](docs/strategy_engine.md): the spec format, features, operators, experiments, the backend interface and the CUDA mapping.

**Current milestone:** the first controlled, reproducible, out-of-sample-aware research loop. CUDA kernels and LoRA training come later.

| Path | What it does |
|---|---|
| `src/data/download.py` | Downloads hourly OHLCV to `data/raw/` (config: `configs/data.yaml`) |
| `src/strategies/` | StrategySpec schema, feature and operator registries, evaluator, sweeps, CLI |
| `src/backtest/` | Backtest engine (execution lag, costs) and metrics |
| `src/research/` | Train/validation experiments, benchmarks, summaries, experiment records, reports |
| `src/agents/` | Prompts generated from the registries, proposal parsing, the bounded research loop |
| `src/models/llm.py` | Qwen3-8B on one chosen GPU |
| `configs/` | Example strategies, parameter spaces, benchmarks and an example LLM proposal |

## Usage

```bash
pip install -r requirements.txt
python -m pytest                                   # offline tests: no network, no GPU, no market data

python -m src.data.download                        # hourly BTC/ETH history -> data/raw/ (server)

# Train sweep -> top 10 frozen -> validation retest -> benchmarks -> experiment record
python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
    --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility_extended.json \
    --train-start 2017-01-01 --train-end 2023-01-01 --validation-start 2023-01-01 --validation-end 2025-01-01 \
    --top 10 --transaction-cost 0.001

# Research loop: at most 5 Qwen proposals (each = up to 1000 backtests) on one GPU
nvidia-smi                                         # pick a GPU with >= 20 GB free
python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 5 --device cuda:2

# Same pipeline without the LLM, replaying a saved proposal
python -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet --hypotheses 1 \
    --replay configs/proposals/example_momentum_low_volatility.json

python -m src.research.report results/experiments/<run_id>/experiments.jsonl   # review a run
```

Data after `--validation-end` (default 2025-01-01) is never loaded by experiments. It's reserved as the final test.
