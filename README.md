# Alpha Research Agent

An LLM (Qwen3-8B, improved with LoRA) proposes crypto trading strategies. A CUDA C++ backtester evaluates them, and the results feed back into the LLM.

```
data (CCXT) ──> OHLCV Parquet ──┐
                                ▼
 Qwen3-8B (+LoRA) ──> strategy spec (JSON) ──> backtest (CUDA, Python reference) ──> metrics
        ▲                                                                              │
        └──────── train results as feedback; good train+validation specs → LoRA data ───┘
```

| Path | What it does |
|---|---|
| `src/data/download.py` | Downloads hourly OHLCV candles to `data/raw/` (config: `configs/data.yaml`) |
| `src/strategy/dsl.py` | Strategy spec language: validation, Python evaluation, compilation for CUDA |
| `src/backtest/` | Python reference engine, metrics, and `Evaluator` (Python or CUDA backend) |
| `cuda/backtest.cu` | CUDA batch backtester, one thread block per strategy |
| `src/agents/` | Agent loop, prompts, holdout report (config: `configs/agent.yaml`) |
| `src/models/` | Qwen generation, LoRA dataset building and training |

## Strategy specs

The LLM writes JSON like this:

```json
{"name": "weekly_trend",
 "hypothesis": "BTC trends over one week",
 "signal": {"op": "pct_change", "arg": {"op": "field", "name": "close"}, "periods": 168}}
```

The position each hour is the sign of `signal`. The engine applies it one bar later, and costs are charged per unit of position change. Operators only look backwards, so a valid spec can't use future data. The full operator list and exact semantics are in `src/strategy/dsl.py`.

## Splits

Defined in `configs/agent.yaml`:
- **Train (2017–2022):** the agent sees these results.
- **Validation (2023–2024):** logged, and used to pick strategies and LoRA data. Never shown to the LLM.
- **Test (2025 onwards):** final holdout. Check it rarely.

## Running on the GPU server

```bash
pip install -r requirements.txt
python -m src.data.download                        # 1. data
make -C cuda && python -m pytest                   # 2. build CUDA engine; tests include CUDA-vs-Python parity
python -m src.models.llm                           # 3. LLM smoke test
python -m src.agents.loop --backend cuda           # 4. agent run -> results/<run>/
python -m src.agents.report results/<run> --split validation
python -m src.models.sft_data results/*/ --out results/sft.jsonl       # 5. LoRA data
python -m src.models.train_lora --data results/sft.jsonl --output checkpoints/lora/v1
python -m src.agents.loop --backend cuda --adapter checkpoints/lora/v1  # 6. agent with LoRA
python -m src.agents.report results/<run> --split test                 # final holdout
```

Without a GPU, `make -C cuda cpu` builds the same backtester for the CPU, so the parity tests can run anywhere.
