# Development progress

What had to be fixed along the way, and how. The design itself is in [architecture.md](architecture.md) and [strategy_engine.md](strategy_engine.md).

## Where it stands

- **The full loop runs on DeepDish:** Qwen3-8B proposes a hypothesis → it's validated → CUDA backtests up to 20,000 parameter combinations → the top 10 are retested on unseen data → compared with benchmarks → a compact summary goes back to Qwen.
- **Speed:** Python 28.5 ms per backtest, C++ 0.175 ms, **CUDA 0.045 ms (635× faster)**, with identical results. A hypothesis is now tested in about 1 s, while Qwen needs 20–30 s to write one. **The LLM is the bottleneck, not the backtests.**
- **Best result so far:** "long when multi-day momentum is positive in calm markets" was positive on training and validation three separate times. It's robust across parameters, but it hasn't beaten buy-and-hold.

## Problems and fixes

**Engine and research design**
1. **Hourly trading costs wiped out simple strategies.** 24h momentum made +470% before costs and lost about 100% after them.
   → Costs are charged on every position change, `--transaction-cost` is explicit everywhere, and trading costs are reported for every experiment.
2. **The first prototype made one LLM call per backtest,** which is slow and wasteful.
   → The LLM proposes a hypothesis plus a parameter search, and the engine tests every combination.
3. **LLM output could be anything, including code.**
   → The output must be JSON using only a fixed list of trusted features and operators. Nothing it writes is executed, and invalid proposals get a clear error code.
4. **The best in-sample result is usually luck.**
   → The best 10 are picked on 2017–2022, their parameters frozen, then retested on 2023–2024. They're compared with buy-and-hold, cash and naive momentum, and flagged by overfitting warnings. Data from 2025 onwards is never loaded; it's the final test.
5. **Validation leaks into Qwen's ideas over time,** because Qwen sees validation results.
   → This is documented, and `--no-validation-feedback` keeps validation blind.
6. **Volatility thresholds made no sense.** 0.01–0.03 never binds, because hourly volatility is about 0.004–0.012.
   → The units are documented and a corrected sweep was added.
7. **Python backtests were slow,** at 28.5 ms each.
   → A C++/CUDA engine gives identical results 635× faster. The same code also builds for the CPU, which is how it's tested without a GPU.
8. **Qwen was spread across all 5 GPUs.**
   → It's now loaded onto one chosen GPU.

**The LLM as a researcher**

9. **Qwen copied the example strategy from the prompt.**
   → The example is now format-only.
10. **Qwen couldn't see why its ideas failed.** Most of them traded thousands of times and fees ate everything.
    → The feedback now includes trades per year, exposure and fees against the return, plus a cost warning.
11. **Renamed or reordered duplicates got through.**
    → Duplicate detection ignores names, ids and condition order.
12. **Qwen kept resubmitting the same duplicate.**
    → The rejection names the experiment it repeats, and the retry doesn't show Qwen its own copied reply.
13. **Qwen re-tested a subset of an earlier grid,** so nothing new was learned.
    → A proposal is rejected if every combination was already tested. Partial overlap is allowed as a refinement.
14. **Qwen gave `returns` a lookback,** which blocked a whole run.
    → The error now says how to fix it ("use momentum with lookback N").
15. **JSON broke on trailing text or truncated replies.**
    → The parser reads the first JSON object, truncation is detected and explained, the token limit is 512, and replies are compact.
16. **Searches were tiny:** 27 combinations out of a 20,000 budget.
    → Qwen gives each parameter a `{"min", "max"}` range and the engine fills in up to 12 values within the budget. That's more coverage *and* shorter LLM replies.
17. **Thresholds were in the wrong units,** e.g. `rolling_mean(close) < 0` (never true) and `rolling_std(close) > 0.012` (always true).
    → The prompt shows each feature's typical values on the training data. There are warnings and a CONDITIONS line when a condition is almost never or always true.
18. **Ideas were narrow** (mostly "short after a drop").
    → The feedback lists the features and position rules not yet tried.
19. **The top 10 held identical copies** (parameters with no effect).
    → Only distinct strategies are retested.
20. **Noisy "Mean of empty slice" warnings** appeared when nothing traded.
    → Guarded.

## Still open

- **Qwen's research judgement:** a short bias in a bull market, and repeating families of ideas. This is the target for LoRA fine-tuning on the collected experiment records.
- **Formal statistics:** Probabilistic/Deflated Sharpe, block bootstrap, multiple-testing corrections.
- **Cross-asset and walk-forward validation** (discover on BTC, confirm on ETH), and an exposure-matched benchmark.
