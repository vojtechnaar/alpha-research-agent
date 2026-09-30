# Development progress

A dated log of what was built, what the experiments showed, and what changed because of it. Newest first. The design itself is described in [architecture.md](architecture.md) and [strategy_engine.md](strategy_engine.md).

## Current status (2026-09-30)

Working end to end on DeepDish4:
- **Data:** hourly BTC/USD and ETH/USD from Bitstamp, 2017 to now.
- **Qwen3-8B:** runs on one selected A6000.
- **Validation:** proposals are checked against the trusted feature and operator registries.
- **Backtesting:** CUDA backtests, parity-tested against the Python reference.
- **Experiments:** train sweep → frozen top 10 retested on validation → benchmarks → experiment record → compact feedback to Qwen.
- **Stopping:** the loop is bounded by `--hypotheses` and stops early after repeated rejections.

Not built yet: LoRA fine-tuning, formal significance tests (PSR/DSR, bootstrap), walk-forward and cross-asset evaluation.

## How the project developed (summary)

**1. Data and a correct backtester**
- Downloaded hourly BTC/ETH candles (2017 to now) from Bitstamp through CCXT and stored them as Parquet.
- Built a backtester with a one-bar execution lag (no look-ahead) and costs charged on every position change.
- First lesson: at hourly frequency, trading costs decide almost everything. The 24h momentum baseline went from +470% (BTC, zero costs) to about -100% with 10 bps costs.

**2. From a prototype to a safe, explainable strategy format**
- The first prototype had the LLM write one expression-tree strategy per call. It worked, but it spent one slow LLM call per backtest.
- It was replaced by a `StrategySpec` built only from a registry of 9 trusted, parameterised features and fixed operators. The LLM writes JSON; nothing it writes is ever executed.
- The LLM now proposes a *hypothesis plus a parameter space*, and the engine generates and tests every combination. One LLM call becomes hundreds or thousands of backtests.

**3. Scientific guard rails**
- **Train/validation split:** parameters are chosen on 2017–2022, frozen, then retested on 2023–2024. Data from 2025 onwards is never loaded; it's reserved as the final test.
- **Benchmarks** (buy-and-hold, cash, naive momentum) run on exactly the same periods and costs.
- **Warnings** flag multiple testing, parameters at the edge of the grid, parameters with no effect, train → validation degradation, and cost drag.
- **Experiment records** in JSONL keep every proposal, including rejected ones, with its results. That's the future LoRA training data.

**4. A controlled LLM research loop**
- Qwen3-8B runs on one chosen GPU. Every proposal is validated first; invalid ones get a bounded number of repair attempts.
- The loop is capped by `--hypotheses` and stops early after repeated rejections, so it can never run away.
- The feedback to Qwen is a compact summary of about 700 tokens (results, costs, sensitivity, exploration coverage), never raw data.
- It copes with a small model's format slips: the parser reads the first JSON object and ignores trailing text, truncation is detected and explained, and duplicate retries don't show the model its own copy.

**5. CUDA: backtesting stopped being the bottleneck**
- One C++/CUDA source file runs the same evaluation as the Python engine, parity-tested to 1e-12 with 0 trade mismatches.
- Speed on 5,600 strategies over 52k hourly bars:

  | Backend | Full sweep | vs Python |
  |---|---|---|
  | Python | ~160 s | 1× |
  | C++, 32 threads | 0.98 s | 163× |
  | CUDA, RTX A6000 | 0.25 s | **635×** |

- **The bottleneck is now the LLM.** Qwen3-8B generates about 10 tokens/s, so one proposal takes 20–30 s. Testing 20,000 parameter combinations for that proposal takes about 1 s on the GPU. Over 95% of each iteration is spent waiting for the LLM.
- This changes the research design:
  - Each LLM call should buy as much evidence as possible, which means wide parameter grids of thousands of combinations instead of 27.
  - Further speed-ups must come from better proposals (fewer invalid or duplicate replies, more diverse ideas), not faster backtests.

**6. Using the runs to improve the researcher**
- Each real run exposed a failure, and each was fixed and re-checked:
  - Qwen copied the prompt example.
  - It re-sent renamed duplicates.
  - It proposed high-churn ideas it couldn't see were failing on costs.
  - It gave `returns` a lookback.
  - It repeated duplicates and used tiny searches.
- The fixes:
  - a format-only prompt example
  - duplicate detection that ignores names and order
  - COSTS lines in the feedback
  - error messages that say how to repair the proposal
  - budget-scaled search guidance and an EXPLORATION block
- What's left is mostly the 8B model's research judgement. That's the motivation for LoRA fine-tuning on the collected experiment records.

---

## 2026-09-30: JSON robustness after the 10-hypothesis run

**Observed** (run `20260930-012429`, 10 hypotheses planned):
- **Better:** the search grew from 27 to 80 combinations, and Qwen tried a new position rule (long after a sharp drop, i.e. a rebound idea). Validation median Sharpe was 0.18, still far below buy-and-hold at 2.04.
- **Stopped early after 5 iterations:** 4 were rejected, which triggered the 3-in-a-row safety stop.
- **"Extra data" (2 iterations):** Qwen wrote a valid JSON object followed by more text containing braces. The parser took everything from the first `{` to the last `}` and failed.
- **"Expecting ',' delimiter at position 707":** a reply that was most likely cut off at the 320-token limit. The new wide-grid guidance made proposals longer. The message ("position 707") gave the model nothing it could act on.
- **Duplicate repeated on all 3 attempts again:** the repair conversation showed Qwen its own duplicate reply, and it copied it.

**Changed:**
- The parser now reads the **first complete JSON object** and ignores anything after it.
- **Truncation is detected** from the generator's token statistics, or from unclosed braces. The error then says how to shorten the reply: compact single-line JSON, short sentences, at most 8 values per parameter.
- **Other JSON errors show the text near the problem** and say what to check (commas, closed quotes and brackets) instead of a character position.
- **The default `--max-new-tokens` went from 320 to 512.** Generation still stops at the end of the JSON, so this only caps runaway replies.
- **The prompt asks for exactly one compact, single-line JSON object.** Search guidance is now 5–8 values per parameter (was 5–10), to keep replies shorter.
- **After a duplicate,** the retry re-sends the original request with a note naming the repeated experiment, without showing Qwen its copied reply.

## 2026-09-30: Research-quality fixes after the first CUDA runs

**Observed** (Qwen run `20260930-011458`, `--backend cuda --max-candidates 20000`):
- **Repeats:** Qwen resubmitted an identical experiment three times in a row, even after `DUPLICATE_PROPOSAL`. The generic "propose something different" message didn't get through to an 8B model.
- **Tiny searches:** every hypothesis used 27 combinations (3×3×3), against a budget of 20,000 that the GPU tests in 0.16 s. The narrow grids put almost every best value "at the edge of the tested range". The prompt still said "3–6 values per parameter", advice written for the Python backend.
- **Narrow ideas:** all three completed hypotheses were "short after a 1% hourly drop, plus one filter". Every variant lost money in a period when BTC rose about 15×. One positive sign: after two "fees ~55% / ~34% of capital per year" warnings, the third idea traded about 20× less.

**Changed:**
- **Duplicate rejections name the experiment repeated,** for example `repeats experiment 1 ('...')`, and tell the model what to change (different features or conditions, or clearly different ranges).
- **Search guidance scales with the candidate budget:**

  | Budget | Values per parameter |
  |---|---|
  | ≥ 5,000 | 5–10 |
  | ≥ 500 | 4–8 |
  | smaller | 3–5 |

  The guidance asks for wide ranges, such as lookbacks from 6 to 720 hours. The feedback's SEARCH line now shows the budget.
- **An EXPLORATION block in the feedback,** built from the registry and the records: features used so far with counts, features not yet tried, and position rules used (for example `short/flat x3`). The prompt asks Qwen to prefer untried features or position rules unless the evidence clearly supports refining.

## 2026-09-30: `returns` with a lookback blocked a whole run

**Observed** (run `20260930-010929`): all three hypotheses gave `returns` a lookback, for example "returns over 24 bars". That's `momentum` in our registry. The validator rejected it correctly, but its message only said what was wrong, so all 3 repair attempts repeated the mistake. The run stopped after 3 consecutive rejections, which is the safety stop working as designed.

**Changed:**
- The error now says how to fix the problem: *remove the 'lookback' key … for the return over N bars use momentum with lookback N*.
- The prompt's feature list says the same (`NO lookback: omit the lookback key`).
- Both texts are generated from the registry.

**Result:** in the next run the repair succeeded (`LLM calls: 2`).

## 2026-09-30: Native C++/CUDA backtest engine

**Built:**
- `cuda/backtest.cu`: one source file, compiled with nvcc for the GPU or as C++ with OpenMP for the CPU.
- It's called from Python through `ctypes` (`src/backends/`), behind the same `evaluate_candidates` contract, so the loop, records and reports didn't change. Pass `--backend cuda` to use it.
- A parity test suite covers every feature × operator, AND/OR, all position combinations, benchmarks, price gaps, flat prices and zero volume.
- A benchmark CLI: `python -m src.backends.benchmark`.

**Measured** (BTC 2017–2022, about 52k hourly bars, 5,600-candidate sweep):

| Backend | ms / candidate | Speed-up | Max diff vs Python | Trade mismatches |
|---|---|---|---|---|
| Python (pandas) | 28.5 | 1× | reference | reference |
| C++, 32 OpenMP threads | 0.175 | 163× | 1.8e-12 | 0 |
| CUDA, RTX A6000 | 0.045 | 635× | 1.8e-12 | 0 |

Backtesting is no longer the bottleneck. A 20,000-combination hypothesis takes about 1 s, while Qwen takes 20–30 s to write one.

**Also changed in this milestone:**
- **Cost diagnostics:** trades per year, bars between position changes, exposure, and fees as % of capital per year against the net return. They appear in the records, the report and Qwen's feedback, with a warning when fees are large.
- **Duplicate detection** ignores names, condition ids and condition order, so renaming a condition no longer sneaks a repeat through. The same rules with different parameter ranges are still allowed as refinements.
- **The top-N validation slots skip candidates that traded identically** to a better-ranked one.
- **The prompt's example strategy is now format-only.** Qwen had copied the previous example, momentum + low volatility, which was the only success of that run.

## 2026-09-30: First Qwen research runs (Python backend)

**Run `20260930-003542`** (5 hypotheses, all completed, 1–2 LLM calls each):
- **Volume-spike ideas all failed.** Hypotheses #1–#4 had 1,300–6,000 trades at 2–10% exposure. Fees ate everything; median train Sharpe was -2.6 to -4.8.
- **One apparent duplicate.** #4 was #3 with the conditions renamed, giving identical results. The duplicate check has since been fixed.
- **The only success wasn't Qwen's idea.** #5 (train median Sharpe 0.70, validation 0.66) was the prompt's example strategy. The example has since been replaced.
- **Why the failures repeated:** Qwen couldn't see trade counts or costs in its feedback, so it kept proposing churning ideas. That's why the cost diagnostics were added.

**Replay of the example proposal** (momentum + low volatility, 36 candidates):
- **Holds up on validation:** all variants profitable on train (median Sharpe 0.66), and the frozen top 10 held on validation (median 0.63).
- **Still below buy-and-hold:** its validation Sharpe was 2.04 in the 2023–24 bull market.
- **Lower drawdowns:** -18% for the top candidate vs -84% for buy-and-hold on train.

## 2026-09-29: Controlled research loop

**Built:**
- Benchmarks (buy-and-hold, flat/cash, naive 24h momentum) as zero-condition StrategySpecs.
- Train/validation experiments: the top N are chosen on train only, their parameters frozen, then retested on validation.
- Data after the validation end is never loaded; it's the final test.
- JSONL experiment records.
- Qwen proposals validated before anything runs.
- Bounded retries and early stop.
- `--replay` for reproducing runs without the LLM.
- `--transaction-cost` everywhere.

**Real-data sweep** (`momentum_low_volatility`, 300 candidates, BTC 2017–2022, 7.5 s in Python):
- **Looks like noise overall:** best Sharpe 0.92, median 0.01, 50% positive.
- **Lookback is the one clear pattern:** longer momentum lookbacks did much better (6 h: -2.2 mean Sharpe; 72 h: +0.63), and it was still improving at the grid edge.
- **The volatility thresholds did nothing.** Thresholds of 0.01–0.03 are far above typical hourly volatility (about 0.004–0.012), so the filter never bound. That led to `momentum_low_volatility_extended.json`, which uses 0.004–0.012.

## 2026-09-29: StrategySpec engine (replaced the first prototype)

**Built:**
- The condition-based `StrategySpec`.
- Nine trusted, parameterised features and the operator and logic registries.
- An evaluator on top of the existing backtester.
- Cartesian-product parameter sweeps with a candidate limit and compact summaries.

**Removed:** the earlier prototype, which had an expression-tree strategy format, an agent loop that proposed one spec per LLM call, a CUDA interpreter, and LoRA scripts trained on winners only. It's kept in git history.

**Why it was replaced:** the LLM should propose *hypotheses and search spaces*, not individual backtests, and LoRA data must include failures.

**First prototype agent run:** 7 of 12 replies valid, Sharpe -2 to -12. That showed costs dominating and a model that was loaded across all 5 GPUs.

## 2026-09-28: Data, backtester, first baseline

**Built:**
- The CCXT downloader (Bitstamp, hourly, 2017 onwards, Parquet).
- A vectorised backtester with a one-bar execution lag and costs per unit of position change.
- Metrics.
- A Qwen3-8B inference smoke test on DeepDish.

**24h momentum baseline, 2017 to now:**
- **With 10 bps costs, both BTC and ETH lost about 100%.** That's about 17,000 units of turnover.
- **With zero costs, BTC made +470%** (Sharpe 0.61, max drawdown -92%), while ETH lost 68%.

This was the first lesson of the project: at hourly frequency, trading costs decide almost everything.

---

## Known issues and limitations

- **Validation leaks through feedback.** Qwen sees validation results, so over many iterations its choice of ideas adapts to 2023–2024. `--no-validation-feedback` keeps validation blind. 2025 onwards is untouched.
- **No formal significance testing yet.** The warnings are descriptive. PSR/DSR, block bootstrap and multiple-testing corrections are still to do.
- **The benchmarks are hard to beat in a bull market.** Long-only strategies are compared with buy-and-hold over 2023–24. An exposure-matched benchmark would be a fairer comparison.
- **Qwen3-8B's research quality is the current bottleneck:** repeats, narrow ideas, format slips. LoRA is intended to help here.
- **Model loading is slow from NFS.** The first load after a while takes about 2.5 min, then about 10 s once cached.

## Next steps

1. **Run 10 hypotheses with the new feedback.** Check whether duplicates disappear, searches get wider, and ideas diversify.
2. **Collect enough research history** for a first LoRA dataset, including failed and rejected proposals.
3. **LoRA fine-tuning (PEFT)** and a base vs LoRA comparison on identical research tasks.
4. **Cross-asset validation** (discover on BTC, confirm on ETH) and walk-forward evaluation.
5. **Formal statistics:** PSR/DSR with an effective number of trials, and block-bootstrap confidence intervals.
