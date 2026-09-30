# Strategy engine

## StrategySpec

```json
{
  "name": "momentum_low_volatility",
  "description": "optional free text",
  "conditions": [
    {"id": "momentum", "feature": "momentum", "field": "close", "lookback": 24, "operator": ">", "threshold": 0.02},
    {"id": "volatility", "feature": "volatility", "field": "close", "lookback": 12, "operator": "<", "threshold": 0.03}
  ],
  "logic": "AND",
  "true_position": 1,
  "false_position": 0
}
```

- Each condition reads `feature(field, lookback) <operator> threshold`.
- `logic` (`AND` or `OR`) combines the conditions.
- **Position rule:** if the combination is true, the position is `true_position`; if false, `false_position`. Positions are -1, 0 or 1.
- **Undefined features mean flat.** If any feature is undefined at a bar (during warm-up, or where a price is missing), the position is 0. This keeps a strategy with `false_position: -1` from going short just because it has no data yet.
- **`id`:** each condition gets one. It defaults to the feature name and must be unique within the strategy. Parameters are addressed as `<id>.<param>`, where the sweepable params are `lookback` and `threshold`.
- **No conditions means unconditional:** the position is always `true_position`. This is how benchmarks are written: buy-and-hold is `{"conditions": [], "true_position": 1}` and cash is `true_position: 0`. LLM research proposals must have at least one condition.
- **`describe()`** gives a readable rule, for example `long if momentum(close, 24) > 0.02 AND volatility(close, 12) < 0.03, else flat`.
- **Rejected input** (the `SpecError.code` is fed back to the LLM):

| Code | Meaning |
|---|---|
| `UNSUPPORTED_FEATURE` | A feature not in `FEATURE_REGISTRY` |
| `UNSUPPORTED_OPERATOR` | An operator or logic not in the registries |
| `INVALID_SPEC` | A bad position, a lookback out of range, a non-finite threshold, unknown keys, duplicate ids, an unknown parameter name, or an empty research strategy |
| `MALFORMED_JSON` | The reply isn't a parseable JSON object, or was cut off |
| `SEARCH_SPACE_TOO_LARGE` | The parameter space's Cartesian product is over the candidate budget (explicit lists only; ranges are sized to fit) |
| `DUPLICATE_PROPOSAL` | The same rules and parameter space were already tested, or every combination was already tested, in this run |

Examples are in `configs/strategies/` (research), `configs/benchmarks/` (benchmarks) and `configs/proposals/` (an LLM-format proposal).

## Features (`FEATURE_REGISTRY`)

| Feature | Definition at bar t | Lookback |
|---|---|---|
| `returns` | x[t]/x[t-1] − 1 | none |
| `momentum` | x[t]/x[t-L] − 1 | ≥ 1 |
| `rolling_mean` | mean(x[t-L+1..t]) | ≥ 1 |
| `rolling_std` | sample std (ddof=1) of x[t-L+1..t] | ≥ 2 |
| `volatility` | rolling_std (ddof=1) of one-bar simple `returns`, **per bar, not annualised** | ≥ 2 |
| `zscore` | (x[t] − rolling_mean) / rolling_std; NaN if std = 0 | ≥ 2 |
| `rolling_min`, `rolling_max` | min/max of x[t-L+1..t] | ≥ 1 |
| `volume_change` | x[t] / mean(x[t-L..t-1]) − 1 (bar t excluded from the mean) | ≥ 1 |

Rules shared by every feature, which the CUDA version must match:
- **Past data only:** each feature uses only bars ≤ t. `tests/test_features.py` checks this for every feature in the registry.
- **Warm-up:** a window needs L valid values, so the first bars are NaN.
- **NaN in a window:** a NaN anywhere in the window makes the result NaN.
- **Division:** division by zero and ±inf become NaN.

Adding a feature takes three steps:
1. Write `compute_x(series, lookback)` in `features.py`.
2. Register it in `FEATURE_REGISTRY` with its minimum lookback, a description and a `threshold_hint` (its units and typical range).
3. Add tests.

Specs, sweeps, the CLI and the LLM prompt pick it up without other changes. The prompt is generated from the registry.

### Units of `volatility` and threshold choice

`volatility(close, L)` is the sample standard deviation of hourly simple returns over the last L bars. It is **not** annualised: multiply by √8760 ≈ 93.6 for an annualised figure. For crypto, 40–110% annualised volatility is about 0.004–0.012 per hour.

That's why the first sweep's volatility thresholds (0.01, 0.02, 0.03) barely mattered: hourly volatility is rarely above 0.02, so a `< 0.02` filter almost never binds. `configs/sweeps/momentum_low_volatility_extended.json` uses `[0.004, 0.006, 0.008, 0.012]` instead, and longer momentum lookbacks (24–336 h), giving 600 candidates. The original `momentum_low_volatility.json` stays unchanged so the first experiment can be reproduced.

Check the actual distribution on your data before trusting any threshold:

```bash
python -c "import pandas as pd; from src.strategies.features import compute_volatility as v; d = pd.read_parquet('data/raw/bitstamp_BTC-USD_1h.parquet'); print(v(d.close, 24).quantile([.1, .25, .5, .75, .9]))"
```

## Operators

- **`OPERATOR_REGISTRY`:** `>`, `>=`, `<`, `<=`. Each returns 1.0/0.0 per bar, or NaN where the feature is NaN.
- **`LOGIC_REGISTRY`:** `AND` is the row-wise minimum of the conditions and `OR` the row-wise maximum. A NaN in either makes the result NaN.
- **Adding an operator** is one registry entry. A crossing operator, for example, would compare the current and previous bar.

## Evaluation

`evaluate_strategy(data, spec, cost_bps, periods_per_year, start, end, cache)` runs these steps:
1. **Validate** the spec.
2. **Compute features** through the registry. Each `(feature, field, lookback)` is computed once and cached.
3. **Apply** the operators.
4. **Combine** the conditions with AND/OR.
5. **Convert to positions:** -1, 0 or +1.
6. **Backtest** with `run_backtest`. The position set at bar t's close earns bar t+1's return, which removes look-ahead. Costs are `cost_bps` per unit of position change.
7. **Compute metrics** with `compute_metrics`:
   - cumulative and annualised return
   - Sharpe
   - annualised volatility
   - max drawdown
   - turnover
   - number of trades
   - exposure
   - number of bars

Annualisation uses `periods_per_year`, which defaults to 8,760 for hourly bars trading 24/7. Nothing in the metrics is specific to BTC or ETH.

## Parameter sweeps

```python
space = {"momentum.lookback": [6, 12, 24, 48, 72], "momentum.threshold": [0.005, 0.01, 0.015, 0.02, 0.03],
         "volatility.lookback": [6, 12, 24, 48], "volatility.threshold": [0.01, 0.02, 0.03]}
results = run_sweep(data, base_spec, space, cost_bps=10, start="2017-01-01", end="2023-01-01", dataset="BTC/USD")
summary = summarize_sweep(results)
```

- **`generate_candidates`:** builds the full Cartesian product (300 candidates here) in a fixed order and validates every candidate before anything runs. It refuses spaces larger than `max_candidates`.
- **`run_sweep`:** returns one row per candidate with `candidate`, `dataset`, `start`, `end`, one column per swept parameter, and all the metrics.
- **Feature cache:** momentum(close, 24) is computed once and reused for all 5 × 4 × 3 = 60 candidates that use it.

## Train/validation experiments

`run_experiment(data, base, space, settings, benchmarks)` in `src/research/experiment.py` runs these steps:
1. Evaluate **all** candidates on TRAIN.
2. Pick the top N by `selection_metric` (default Sharpe), using the train table only. Ties go to the lower candidate id, and candidates with fewer than `min_train_trades` trades aren't eligible.
3. Freeze those parameters and evaluate **only** them on VALIDATION.
4. Evaluate the benchmarks on both periods with the same costs and the same backend.
5. Return the train table, the selected ids, the validation table, a side-by-side comparison and the benchmark tables.

Data at or after the validation end is dropped before anything runs.

From the command line (`--transaction-cost` is a fraction per unit of position change; the default is 0.001 = 10 bps):

```bash
python -m src.strategies.run configs/strategies/momentum_low_volatility.json \
    --data data/raw/bitstamp_BTC-USD_1h.parquet --space configs/sweeps/momentum_low_volatility_extended.json \
    --train-start 2017-01-01 --train-end 2023-01-01 --validation-start 2023-01-01 --validation-end 2025-01-01 \
    --top 10 --transaction-cost 0.001
```

This prints the report and appends an experiment record to `results/experiments/manual/experiments.jsonl`. Without `--validation-*` you get the older train-only sweep. Without `--space` you get a single strategy compared with the benchmarks over the same period.

## Backend interface

`evaluate_candidates(data, candidates, cost_bps, periods_per_year, start, end, dataset, ids)` in `sweep.py` is the only function that runs backtests. `CandidateEvaluator` describes its contract:
- **In:** OHLCV data, a list of `(params, StrategySpec)`, costs and a period.
- **Out:** one row per candidate with `candidate`, `dataset`, `start`, `end`, the parameter columns and the metrics.

`run_experiment`, the benchmarks and the research loop all take `evaluator=`. A C++ or CUDA backend with the same contract plugs in without other changes, and a parity test can run both backends on the same candidates and compare the tables.

## Native backends (C++ and CUDA)

`cuda/backtest.cu` implements the same evaluation natively. It's built two ways:

```bash
make -C cuda              # CUDA GPU backend -> cuda/build/libbacktest_cuda.so (A6000: ARCH=sm_86 if -arch=native fails)
make -C cuda cpu OMP=1    # C++ CPU backend (OpenMP, all cores) -> cuda/build/libbacktest_cpu.so
```

`src/backends/native.py` loads the library with `ctypes`, so there are no Python build dependencies. `NativeEvaluator` has the `evaluate_candidates` signature, and `get_evaluator("python" | "cpp" | "cuda")` returns the chosen backend.

**Per call** (one dataset, one period):
1. **Python prepares the inputs.** It compiles the candidate specs into flat arrays: one entry per *distinct* `(feature, field, lookback)`, and per candidate its conditions (buffer index, operator, threshold), logic and positions. It computes the period returns with `period_returns`, the exact function the Python engine uses.
2. **The feature kernel** computes each feature buffer once over bars `[0, end)`, with one GPU thread per (buffer, bar). Each registry entry is one `case` of `feature_value(feature, series, lookback, t)`. The lookback is a runtime argument: there's never a `momentum_24h_kernel`.
3. **The candidate kernel** runs one GPU thread per candidate. It combines its buffers with its thresholds into positions and backtests over `[start, end)` in time order. That's the same arithmetic order as pandas, so results match to rounding (about 1e-13). It writes one row of `METRIC_NAMES`.
4. **Only the metrics table** goes back to Python, never per-bar series.

**Semantics that the native code mirrors on purpose:**
- NaN/inf handling.
- Warm-up and NaN-in-window rules.
- pandas' constant-window behaviour: a window of identical values has mean exactly that value and standard deviation exactly 0, so zscore is undefined.
- "Undefined condition means flat."
- The one-bar execution lag.
- Costs per unit of position change.
- Two-pass sample variance.

**Adding a feature to the registry** also needs a `case` in `feature_value` and an entry in `FEATURE_CODES`. Until then the native backends raise `NotImplementedError` for it, and a test checks the code tables stay in sync.

**Measured on DeepDish4** (30 September 2026). BTC/USD hourly bars from 2017-01-01 to 2023-01-01 (about 52k bars), the 5,600-candidate `momentum_low_volatility_dense.json` sweep, cost 10 bps:

| Backend | ms / candidate | Full sweep | Speed-up vs pandas | Max metric diff vs Python | Trade-count mismatches |
|---|---|---|---|---|---|
| Python (pandas) | 28.5 | ~160 s (extrapolated from 200) | 1× | reference | reference |
| C++, OpenMP 32 threads | 0.175 | 0.98 s | 163× | 1.8e-12 | 0 |
| CUDA, RTX A6000 | 0.045 | 0.25 s | 635× | 1.8e-12 | 0 |

The GPU time includes the Python-side compilation of the candidates, the host↔device transfers and building the result table. At 5,600 candidates the GPU is far from saturated, so larger sweeps should widen the gap over the CPU.

Reproduce with:

```bash
python -m src.backends.benchmark --data data/raw/bitstamp_BTC-USD_1h.parquet \
    --strategy configs/strategies/momentum_low_volatility.json \
    --space configs/sweeps/momentum_low_volatility_dense.json --device 0
```

The benchmark also reports the maximum metric difference and trade-count mismatches against Python on the same candidates.

**Possible future optimisations**, once correctness is established on real data:
- Parallelise inside a candidate (prefix products for the drawdown, block reductions).
- Use float32 feature buffers.
- Share feature buffers across periods and assets.
