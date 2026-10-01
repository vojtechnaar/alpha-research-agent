#!/usr/bin/env bash
# Collect research runs (LoRA training data) on the TRAINING markets; ETH/USD and GLD stay held out.
#
#   nohup bash scripts/collect.sh 2 5 > collect_base.log 2>&1 &                                  # base Qwen, 5 hours
#   nohup bash scripts/collect.sh 4 5 --adapter checkpoints/lora/v1 > collect_v1.log 2>&1 &      # Qwen + LoRA v1
#
# Arguments: GPU index, hours (hard limit: the run in progress is stopped, its finished hypotheses
# are kept), then extra src.agents.research arguments. MARKETS="BTC SPY" picks the markets
# (default BTC SPY QQQ TLT; daily Yahoo symbols by file name, e.g. EURUSDX). Same settings as the
# earlier collection: up to 20,000 candidates, daily markets with 2 bps costs.
set -u
GPU=$1
HOURS=$2
shift 2
export CUDA_VISIBLE_DEVICES=$GPU  # only this GPU is visible (cuda:0 inside the process)
COMMON=(--hypotheses 10 --backend cuda --device cuda:0 --backtest-device 0 --max-candidates 20000)
DAILY=(--train-start 2005-01-01 --train-end 2019-01-01 --validation-start 2019-01-01 --validation-end 2023-01-01
       --periods-per-year 252 --transaction-cost 0.0002)
deadline=$(( $(date +%s) + HOURS * 3600 ))
for round in $(seq 1 100); do  # bounded twice: at most 100 rounds, and the deadline
  for m in ${MARKETS:-BTC SPY QQQ TLT}; do
    case $m in ETH|GLD) echo "$m is held out for the base vs LoRA comparison"; exit 1 ;; esac
    left=$(( deadline - $(date +%s) ))
    [ "$left" -gt 0 ] || { echo "Time limit reached after $((round - 1)) full rounds."; exit 0; }
    echo "=== round $round, $m, $((left / 60)) min left"
    if [ "$m" = BTC ]; then
      timeout "$left" python -u -m src.agents.research --data data/raw/bitstamp_BTC-USD_1h.parquet "${COMMON[@]}" "$@"
    else
      timeout "$left" python -u -m src.agents.research --data "data/raw/yahoo_${m}_1d.parquet" "${COMMON[@]}" "${DAILY[@]}" "$@"
    fi
  done
done
