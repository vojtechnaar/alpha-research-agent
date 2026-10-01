# Alpha Research Agent

An automated quantitative crypto research system:
- **Qwen3-8B is the researcher.** It proposes a hypothesis, a validated JSON strategy spec and a parameter space.
- **A numerical engine runs the experiment.** It backtests every combination on a training period, retests the frozen best few on a validation period, and compares them with benchmarks. It runs in Python, C++ or CUDA.
- **A compact summary goes back to Qwen** for the next hypothesis, within a strict iteration limit.

- [docs/architecture.md](docs/architecture.md): the pipeline, the research loop, the splits, records and design decisions.
- [docs/strategy_engine.md](docs/strategy_engine.md): the spec format, features, operators, experiments, the backend interface and the CUDA mapping.
- [docs/progress.md](docs/progress.md): what had to be fixed and how, what the experiments showed, open issues and the LoRA plan.

**Current milestone:** LoRA adapter v1 is trained on 164 good research steps; base vs LoRA is being compared on the held-out markets ETH and GLD.

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
