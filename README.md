# Alpha Research Agent

An automated quantitative crypto research system. Qwen3-8B (with LoRA later) will act as the researcher: it proposes hypotheses as validated JSON strategy specs plus parameter ranges. A numerical engine runs the backtests, in Python now and CUDA later, and sweeps each hypothesis over many parameter combinations.

- [docs/architecture.md](docs/architecture.md): the full pipeline and design decisions.
- [docs/strategy_engine.md](docs/strategy_engine.md): the spec format, features, operators, sweeps, and how they map to CUDA.

**Current milestone:** data, StrategySpec, feature and operator registries, evaluator, parameter sweeps and backtester. The LLM loop, LoRA training and CUDA kernels come in later milestones.

| Path | What it does |
|---|---|
| `src/data/download.py` | Downloads hourly OHLCV candles to `data/raw/` (config: `configs/data.yaml`) |
| `src/strategies/` | StrategySpec schema, feature and operator registries, evaluator, sweeps, CLI |
| `src/backtest/` | Backtest engine (execution lag, costs) and metrics |
| `src/models/llm.py` | Qwen3-8B inference on one chosen GPU |
| `configs/strategies/`, `configs/sweeps/` | Example strategies and parameter spaces (JSON) |

## Data Pipeline

```bash
pip install -r requirements.txt
python -m src.data.download --symbols BTC/USD --start 2024-01-01 --end 2024-01-02   # small smoke test
python -m src.data.download                                                          # full history (server)
```

Output: `data/raw/<exchange>_<BASE-QUOTE>_<timeframe>.parquet` with columns `timestamp` (UTC), `open`, `high`, `low`, `close`, `volume`, `symbol`.

## Strategy engine

```bash
python -m pytest                                              # offline tests, no network/GPU

# One strategy on one period
python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
    --data data/raw/bitstamp_BTC-USD_1h.parquet --start 2023-01-01 --end 2024-01-01

# Parameter sweep (300 candidates); table saved to results/sweeps/
python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
    --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility.json \
    --start 2017-01-01 --end 2023-01-01

# LLM smoke test (GPU server; QWEN_DEVICE=cuda:N picks the GPU)
python -m src.models.llm
```
