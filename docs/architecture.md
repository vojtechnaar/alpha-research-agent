# Architecture

The goal is an automated quantitative research system. **Qwen3-8B acts as the researcher**: it proposes hypotheses and the parameter space to test. **A numerical engine is the experiment engine**: it runs the backtests, first in Python and later in CUDA. A compact summary of the results goes back to the LLM, which proposes the next hypothesis.

```
Market data (CCXT → Parquet)                              src/data/            NOW
        ↓
Qwen3-8B  (+ LoRA later)                                  src/models/llm.py    inference only
        ↓
Hypothesis + StrategySpec + parameter space (JSON)        ResearchProposal     schema NOW, LLM later
        ↓
Validation against trusted registries                     schema.py            NOW
        ↓
Feature / operator registry                               features.py, operators.py   NOW
        ↓
Parameter sweep (Cartesian product)                       sweep.py             NOW
        ↓
Evaluator:  Python (NOW) → C++ CPU (FUTURE) → CUDA (FUTURE)
        ↓
Backtest (1-bar execution lag, costs)                     src/backtest/        NOW
        ↓
Metrics → robustness metrics (splits, stability, significance)   partly NOW
        ↓
Compact experiment summary                                summarize_sweep()    first version NOW
        ↓
Qwen feedback → next hypothesis                           FUTURE (bounded loop)
```

## What exists now

| Component | Path | Role |
|---|---|---|
| Data | `src/data/download.py` | Hourly OHLCV from Bitstamp to `data/raw/*.parquet` |
| Spec | `src/strategies/schema.py` | `StrategySpec`, `Condition`, `ResearchProposal`, validation, error codes |
| Features | `src/strategies/features.py` | `FEATURE_REGISTRY` of parameterised, causal features |
| Operators | `src/strategies/operators.py` | `OPERATOR_REGISTRY` (`> >= < <=`) and `LOGIC_REGISTRY` (`AND`, `OR`) |
| Evaluator | `src/strategies/evaluator.py` | spec → features → conditions → positions → backtest → metrics |
| Sweeps | `src/strategies/sweep.py` | base spec + parameter space → candidate specs → results table + summary |
| CLI | `src/strategies/run.py` | Run one spec or one sweep on a Parquet file |
| Backtest | `src/backtest/engine.py`, `metrics.py` | Execution lag, costs, returns; asset-agnostic metrics |
| LLM | `src/models/llm.py` | Qwen3-8B on one chosen GPU; not connected to the engine yet |

## Design decisions

### Generated code is never executed

The LLM only ever produces JSON. That JSON is validated against fixed registries of features and operators that humans wrote and tested. Anything outside the registries is rejected with a machine-readable code (`UNSUPPORTED_FEATURE`, `UNSUPPORTED_OPERATOR`, `INVALID_SPEC`), and that code becomes feedback for the LLM. There is no `eval`, no `exec`, and no generated Python or CUDA.

This gives three things:
- **Safety:** nothing the model writes can touch the machine.
- **Correctness:** every primitive has unit tests, including a no-look-ahead test.
- **Portability:** the same small registry can be implemented in C++ and CUDA, and checked against Python.

Letting the LLM propose new operators is a possible later step. Those would go to human review before being added to the registry.

### Parameter search is separate from LLM generation

Qwen3-8B generates about 10 tokens per second on DeepDish, so one LLM call costs seconds. One backtest costs milliseconds in Python, and will cost far less in CUDA. The LLM is therefore used once per hypothesis, to say what to test and over which ranges:

```json
{"hypothesis": "Momentum may be stronger in low-volatility regimes.",
 "strategy": { ...StrategySpec... },
 "parameter_space": {"momentum.lookback": [6, 12, 24, 48], "momentum.threshold": [0.005, 0.01, 0.02]},
 "rationale": "..."}
```

The engine generates the Cartesian product itself and evaluates every candidate. Only a compact summary goes back to the LLM:
- the distribution of the metric
- the best candidates
- the mean metric for each value of each parameter

So one hypothesis becomes hundreds or thousands of backtests but only one LLM call. `ResearchProposal` already parses and validates this format, including checking that every value in the parameter space produces a valid strategy.

### Correctness first, then CUDA

CUDA is there to make large searches over parameters, strategies, assets and periods fast. It isn't there for its own sake, and a fast wrong answer is worthless. So the order is:

1. **Python reference (now).** Readable pandas and numpy, with unit tests.
2. **C++ CPU implementation.** Same registry, same parameters. A parity test requires its metrics to match Python on the same sweep.
3. **CUDA implementation.** Same parity test.
4. **Benchmark** on identical sweeps: Python vs C++ vs CUDA, measured in candidates per second.

`run_sweep` is the interface all three implement. The inputs are data, a base spec, a parameter space, costs and a period. The output is one row per candidate with the same columns. That makes the benchmark a matter of swapping the implementation behind this one function.

### Out-of-sample evaluation will decide what counts as a discovery

A system that tests thousands of variants will always find some that look excellent in-sample purely by chance. The results design keeps rigorous evaluation cheap to add:
- **Periods:** `evaluate_strategy` and `run_sweep` take `start`/`end`. Features are computed on the full history, so indicators are warmed up at the start of a period, but returns only count inside it. Train/test splits and walk-forward windows are just repeated calls with different periods.
- **Long-format results:** `run_sweep` returns one row per candidate with `dataset`, `start` and `end` columns. More assets, periods or cost levels are just more rows added with `pd.concat`. Parameter stability, cost sensitivity, regime breakdowns and multiple-testing corrections are then group-bys over that table.
- **Bar-level returns:** `StrategyResult.backtest` keeps the returns for every bar. That's what bootstrap tests and information-coefficient calculations need.

Planned next: fixed train/validation/test splits, walk-forward evaluation, a deflated Sharpe ratio or a similar multiple-testing correction, bootstrap confidence intervals, and cost sensitivity.

### Loops always have explicit limits

The future loop (`python -m src.agents.research`) will be bounded by configuration, for example:

```yaml
num_hypotheses: 10                    # LLM research hypotheses, NOT backtests
max_candidates_per_hypothesis: 1000   # enforced by generate_candidates(max_candidates=...)
max_llm_tokens: 150
random_seed: 42
```

Each run stops after `num_hypotheses` LLM calls. It can also stop earlier on a stopping criterion, such as no improvement in out-of-sample results for K hypotheses. There is never an open-ended loop. The candidate limit already exists: `generate_candidates` refuses a parameter space larger than `max_candidates` rather than silently truncating it.

Each iteration will look like this:
1. The LLM returns a `ResearchProposal`. If it's invalid, the error code is fed back, and that still counts against `num_hypotheses`.
2. `run_sweep` runs on the train period.
3. `summarize_sweep` produces the summary for the LLM.
4. The best candidates are checked on validation and logged. The validation results are never shown to the LLM.
5. The test period is used only for the final report.

### LoRA comes later, trained on research behaviour, not just winners

LoRA fine-tuning needs data that only a working research system produces. Each logged research step will become a training record:

- **Input:** the available features and operators, the market and research context, summaries of previous experiments, and the hypotheses already tested.
- **Target:** the next hypothesis, the StrategySpec, the parameter space and the rationale.
- **Result metadata:** out-of-sample Sharpe, drawdown, turnover, robustness and parameter stability, an ACCEPT or REJECT verdict, and an explanation.

The dataset will contain both good and bad research steps, labelled. Training only on historical winners would teach the model to memorise lucky strategies, not how to do research.

The evaluation compares **base Qwen3-8B** against **Qwen3-8B + quant LoRA** on the same tasks with the same compute budget. Measures:
- rate of valid StrategySpecs
- rate of unsupported features or operators
- hypothesis diversity
- out-of-sample performance and robustness of what each finds
- parameter stability
- number of useful hypotheses per compute budget

### GPU usage

DeepDish4 has several shared RTX A6000s. Qwen3-8B in bf16 (about 16 GB) fits on one card, so `QwenGenerator` puts the whole model on one explicitly chosen device. The default is `cuda:0`; override it with `QWEN_DEVICE=cuda:2` or `CUDA_VISIBLE_DEVICES`. Nothing assumes a particular number of GPUs. The CUDA backtester can later run on a different GPU, or in sequence with the LLM, depending on what's free.
