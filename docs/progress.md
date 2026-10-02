# Development progress

What had to be fixed along the way, and how. The design itself is in [architecture.md](architecture.md) and [strategy_engine.md](strategy_engine.md).

## Where it stands

- **The full loop runs on DeepDish:** Qwen3-8B proposes a hypothesis → it's validated → CUDA backtests up to 20,000 parameter combinations → the top 10 are retested on unseen data → compared with benchmarks → a compact summary goes back to Qwen.
- **Speed:** Python 28.5 ms per backtest, C++ 0.175 ms, **CUDA 0.045 ms (635× faster)**, with identical results. A hypothesis is now tested in about 1 s, while Qwen needs 20–30 s to write one. **The LLM is the bottleneck, not the backtests.**
- **Best result so far:** "long when the price is above its weekly average and volatility is low" (`distance_to_mean` + `volatility`). About 80–88% of 14,641 variants were profitable on train. The frozen top 10 scored a validation Sharpe of about 1.5–2.0 with almost no degradation, and roughly half buy-and-hold's drawdown on train (-33% vs -84%). It still doesn't beat buy-and-hold's validation Sharpe (2.00 vs 2.04). It's the same family as the earlier "momentum in calm markets" result, expressed better.

- **The fixes measurably improved the researcher** (`python -m src.research.runs` over the first 11 Qwen runs on BTC):
  - first-try valid replies: 0% → 60%
  - useful experiments per 10 LLM calls: 0 → about 2.9
  - idea families per run: 1–3 → 7–8

  About 2.9 useful experiments per 10 calls is the base-model baseline a LoRA has to beat.
- **Research data collected:** 632 completed experiments on BTC (hourly) and SPY, QQQ and TLT (daily), 0 crashes. First-try valid rate is 54–65% per market, and useful experiments per 10 LLM calls range from 1.5 (BTC) to 2.7 (QQQ). About a third of all LLM calls were rejected repeats (289 duplicates, 99 exhausted idea families), the clearest thing for LoRA to improve.
- **LoRA v1 vs base Qwen on the held-out ETH and GLD** (30 paired runs: seeds 1–15 on each market, 10 hypotheses each, about 300 experiments per model):
  - useful experiments per 10 LLM calls: 2.39 → 2.84
  - first-try valid: 67% → 71%
  - experiments holding up on validation: 48% → 55%
  - automatic corrections: 43 → 8 (it learned to write `momentum` instead of `returns` with a lookback)
  - unit mistakes: 34% → 31% of experiments
  - rejected repeats (duplicates + exhausted families): 35% → 31% of LLM calls
  - idea families per run: 7.1 → 7.0 (no loss of diversity)

  **Verdict: not proven.** LoRA was better in 17 of 30 pairs, with a mean difference of +0.39 useful experiments per 10 calls, but the 95% bootstrap CI (−0.43 to +1.19) includes 0. It learned the mechanics (format, corrections) but not to stop repeating ideas, the biggest waste for both models. The first 10 pairs looked better (+0.63, 7 of 10): small samples overstate effects, which is why the decision rule and the number of seeds were fixed in advance.
- **LoRA v2 vs base Qwen** (trained on 479 examples, 3× v1; validation loss 0.429 → 0.063, best at epoch 2; the same 30 pairs, evaluated as fixed in advance):
  - better in 15 of 30 pairs; mean difference **+0.28** useful experiments per 10 calls, 95% bootstrap CI (−0.46 to +1.02)
  - GLD +0.92 (9 of 15 better), ETH −0.36 (6 of 15)
  - rejected repeats: 138 vs 149 for base (still about a third of LLM calls)

  **Verdict: no improvement**, below the +0.5 fixed in advance. Three times more data lowered the imitation loss but did not make Qwen a better researcher: supervised fine-tuning teaches the *style* of good proposals, not how to find new ones or to stop repeating. Both adapters help on daily GLD and hurt on hourly ETH, matching the training data (80% daily markets).

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
21. **Qwen got stuck on one idea.** 7 of 10 hypotheses were "short when volatility is high and price drops". The duplicates were resubmitted on every retry, so 12 of 28 LLM calls were wasted.
    → Each idea family (same features, directions and long/short rule, any parameter values) may be tested at most twice (`--max-per-family`). The rejection lists untried features, and the feedback lists ideas at the limit. Result in the next run: 0 duplicate rejections, 7 different idea families, 4 long and 4 short.
22. **Retries repeated the same JSON,** because Qwen's output was too concentrated on one answer.
    → Retries after a repeat are sampled at a higher temperature (0.7 → 1.0 → 1.3). Format errors are still retried at the normal temperature.
23. **The "best" candidates barely traded.** For a losing idea, the variants that almost never trade look best, with Sharpe near 0 instead of negative. The top 10 then made 0 trades on validation. The old filter (10 trades in 6 years) was far too weak.
    → Candidates must trade at least 10 times per year (`--min-trades-per-year`), i.e. 60 on train. Validation uses the same per-year rate to flag unreliable results.
24. **Qwen kept giving `returns` a lookback,** even with the error explaining the fix. Two hypotheses were lost after 3 attempts each.
    → The mistake is unambiguous (the return over N bars *is* `momentum` with lookback N), so it's now rewritten automatically before validation. The correction is recorded and shown to Qwen.
25. **Qwen couldn't express "price near its recent high, low or average".** It tried fixed thresholds on price levels, e.g. `rolling_min(close) >= 0.95` (always true) and `rolling_max(close) > 9800` (meaningless when the price went from $1,000 to $90,000).
    → Added three unit-free features, `distance_to_max`, `distance_to_min` and `distance_to_mean`, in Python and CUDA, parity-tested. Qwen used them in 7 of 9 hypotheses in the next run, and they produced the best results so far. The `returns` rewrite (24) also removed all rejections of that kind.
26. **The prompt and hints were crypto-specific.**
    → The wording is now asset-neutral. The market and bar length come from the data, typical values from the training period, and `--periods-per-year` sets the annualisation for other markets.
27. **Validation stopped being blind.** Qwen had seen 2023–24 results across many runs, so the best ideas were partly fitted to that period.
    → `python -m src.research.confirm` re-tests chosen experiments without the LLM: *replicate* (the same idea re-selected on another asset), *transfer* (the frozen parameters applied unchanged to another asset), and a one-time *final test* on the untouched 2025+ period. Every final-test use is logged and repeat use triggers a warning.
28. **Runs repeated each other.** A fixed default seed (42) made every run start with the same proposals, so a new run re-tested yesterday's ideas.
    → Each run gets a fresh random seed, printed and saved in `run.json`. `--seed N` reproduces a run. (Two saved runs with seed 42 had identical metrics, confirming the problem.)
29. **Buy-and-hold in disguise looked like the best result.** One candidate was long 99.8% of the validation period, so its Sharpe simply equalled buy-and-hold's.
    → A warning fires when a long-only strategy is in the market at least 90% of the validation period, and the run metrics don't count it as useful.
30. **Only one market (BTC, hourly).** Ideas tuned on one asset may be luck.
    → `python -m src.data.download_yahoo` adds daily, split- and dividend-adjusted bars for stocks (SPY, QQQ), bonds (TLT), gold (GLD) and FX (EURUSD) in the same format. The momentum benchmark was renamed `naive_momentum_24` (bars, not hours).
31. **Research runs were treated as disposable,** but they're the LoRA training data.
    → Every run is kept in `data/research_runs/` (git-ignored; full train tables only with `--save-sweeps`, to save disk). `python -m src.research.runs` measures every run (efficiency, research quality, diversity, mistakes), which is the baseline a LoRA must beat.
32. **"Beats buy-and-hold" was misleading on bonds.** TLT's buy-and-hold lost money over 2019–2022, so almost anything "beat" it.
    → Beating buy-and-hold only counts when buy-and-hold itself was profitable.
33. **The LoRA must not learn from bad examples.** The records also contain failed ideas, unit mistakes and hypotheses that contradict their rules.
    → `python -m src.models.lora_data` keeps only good research steps: useful, no unit problems, text consistent with the long/short rule, no repeats. ETH and GLD are always held out. The target is the validated proposal as compact JSON. `python -m src.models.train_lora` trains a LoRA adapter (frozen bf16 base, loss on the proposal only, best validation loss kept), and `--adapter` runs the research loop with it.
34. **LoRA training dropped over half the examples and nearly ran out of GPU memory.** Prompts (rules, typical values, feedback) are 3–6k tokens: 57% were over the 4,096-token limit. The model's output over the whole sequence (tokens × 152k vocabulary, about 2.4 GB) was allocated just to compute a loss on the ~300-token proposal.
    → The loss is computed from outputs for the proposal tokens only (`logits_to_keep`). A test checks it equals the standard loss exactly, and that only the LoRA weights get gradients. The limit is raised to 8,192 tokens, so all examples fit, and the log prints token lengths.
35. **Training looked stuck for 20 minutes.** Output redirected to a log file is buffered, so nothing showed up until the end, and there was no progress inside an epoch.
    → Line-buffered output plus a line after every step (`epoch 1/3, step 5/51 ... ~38 min left`). The first run (v1, 129 examples) lowered validation loss from 0.419 (base Qwen) to 0.087, 0.075 and 0.071 over 3 epochs (no overfitting yet; epoch 3 kept).
36. **Base and LoRA runs in parallel could crash each other.** Run ids were the start time to the second, so two runs starting in the same second would collide.
    → Run ids get a short random suffix. `scripts/compare_lora.sh <gpu> [--adapter ...]` runs the held-out comparison (ETH + GLD, seeds 1–5) on one GPU.
37. **Our runs held ~0.3 GB on every GPU, including other users' GPUs.** The memory log queried all 5 GPUs, which opens a CUDA context on each.
    → The log only reports the GPU the process uses. That wasn't the only cause: loading the LoRA adapter still opened contexts on every GPU and put the adapter on GPU 0. So runs now make only their own GPU visible (`CUDA_VISIBLE_DEVICES`), which `scripts/compare_lora.sh` does automatically.
38. **Base vs LoRA was judged by eye from a long table,** with rejection codes mixed across models.
    → `python -m src.research.runs` pairs runs by market and seed and prints wins, the mean difference and a bootstrap 95% CI, plus rejection codes per model.

## Final test: fixed before looking

- **Which ideas:** for each training market (BTC/USD, SPY, QQQ, TLT), the single best *useful* experiment (no unit problems) by validation median Sharpe, one per idea family (`python -m src.research.best --top 1`). Chosen from train and validation results only.
- **Test periods:** everything after each experiment's validation end, which the research loop never loaded: BTC from 2025-01-01, stocks and bonds from 2023-01-01, until the end of the downloaded data.
- **What counts:** the idea *holds up* if the frozen top 10's median Sharpe on the test period is above 0; it *beats buy-and-hold* only if that median is above buy-and-hold's Sharpe on the same period. Each test is run once (`results/final_test_log.jsonl`), and the result is reported whatever it shows. Expect it to be lower than validation: the best of hundreds of experiments is inflated by selection.

## Still open

- **The comparison ran with a 1,000-candidate budget** (`scripts/compare_lora.sh`), while the training data was collected with 20,000. That's the same for both models, so the comparison is fair, but LoRA saw a slightly different prompt than in training. Keep 1,000 for comparisons with the existing 30 base runs, or rerun base when changing it.
- **The hypothesis text often contradicts the rule,** e.g. "may reverse upward" with a *short* rule, or "near recent lows" with "5% above the average". The engine tests the rule correctly, but these records would teach LoRA sloppy reasoning. Possible fix: have Qwen state the direction explicitly and check it against `true_position`, or filter such records out of the LoRA data.
- **Confirm the best idea:** run `src.research.confirm` for the best experiments on ETH, then once on the 2025+ final test, and report the result honestly, whatever it is.
- **Qwen's research judgement:** short ideas keep failing in this mostly bull-market sample, but Qwen keeps proposing them. The family limit forces variety but not *good* ideas. This is the target for LoRA fine-tuning on the collected experiment records.
- **Formal statistics:** Probabilistic/Deflated Sharpe, block bootstrap, multiple-testing corrections.
- **Cross-asset and walk-forward validation** (discover on BTC, confirm on ETH), and an exposure-matched benchmark.

## Plan: LoRA

1. **Collect research data: done.** 632 experiments on BTC, SPY, QQQ and TLT are saved in `data/research_runs/`. **ETH and GLD are held out**: never used for LoRA training, so the comparison below is fair.
2. **Build the training set.** Pairs of (the context Qwen saw → the proposal it wrote), keeping only good research steps:
   - valid on the first try
   - a hypothesis that matches the rule
   - no unit problems
   - held up on validation
   - not buy-and-hold in disguise, not a repeat

   Failed and rejected attempts stay in the records for analysis, and could later be used for preference training.
   Built: `python -m src.models.lora_data`.
3. **Train** a LoRA adapter (PEFT) for Qwen3-8B on one A6000. Built: `python -m src.models.train_lora`.
4. **Measure base vs LoRA:**
   - **Setup:** the same held-out markets (ETH, GLD), the same settings, and the same number of runs with the same seeds for both models (e.g. 5 runs × 10 hypotheses each). Compare runs in pairs by seed. Run: `bash scripts/compare_lora.sh <gpu> [--adapter checkpoints/lora/v1]`.
   - **Primary metric,** fixed in advance: *useful experiments per 10 LLM calls* (from `python -m src.research.runs`).
   - **Secondary metrics:** first-try valid rate, LLM calls per completed experiment, rejection codes, unit problems, idea families per run, share holding up on validation, train → validation degradation.
   - **Decision rule:** LoRA is better if the primary metric improves in most paired runs (e.g. at least 4 of 5) with a bootstrap confidence interval above 0, and no secondary quality metric gets worse.
   - **Final check:** the best idea from each model is evaluated once on the 2025+ final test.
   - **Pitfall:** LoRA is trained on examples labelled with validation results, so comparing on the *training* markets' validation period would favour it. That's why the comparison uses held-out markets and the final test.
5. **v2, fixed before running:** trained like v1 on the larger collection (base and v1 runs on BTC, SPY, QQQ, TLT; ETH and GLD still held out). Evaluated on exactly the v1 setup: seeds 1–15 on ETH and GLD, 10 hypotheses, 1,000-candidate budget, paired against the same 30 base runs. Primary metric: useful experiments per 10 LLM calls. **Minimum meaningful effect: +0.5** (about +20%). The result is reported with its 95% bootstrap CI whatever it shows, and no seeds are added afterwards.
