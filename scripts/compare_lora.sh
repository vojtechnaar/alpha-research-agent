#!/usr/bin/env bash
# Base vs LoRA researcher on the held-out markets (ETH/USD hourly, GLD daily), same seeds for both.
#
#   nohup bash scripts/compare_lora.sh 2 > compare_base.log 2>&1 &
#   nohup bash scripts/compare_lora.sh 3 --adapter checkpoints/lora/v1 > compare_lora.log 2>&1 &
#   python -m src.research.runs --by model --datasets ETH/USD GLD
#
# First argument: GPU index (Qwen and the CUDA backtester). The rest goes to src.agents.research.
# SEEDS="1 2 3" overrides the seeds (default 1-5): 10 runs x 10 hypotheses, ~45 min per model.
set -u
GPU=$1
shift
ETH=data/raw/bitstamp_ETH-USD_1h.parquet
GLD=data/raw/yahoo_GLD_1d.parquet
for f in "$ETH" "$GLD"; do
  [ -f "$f" ] || { echo "Missing $f"; exit 1; }
done
# Only this GPU is visible to the process (inside it, it is cuda:0), so no library can open
# contexts or put tensors on other users' GPUs (loading a LoRA adapter did).
export CUDA_VISIBLE_DEVICES=$GPU
COMMON=(--hypotheses 10 --backend cuda --device cuda:0 --backtest-device 0)
DAILY=(--train-start 2005-01-01 --train-end 2019-01-01 --validation-start 2019-01-01 --validation-end 2023-01-01
       --periods-per-year 252 --transaction-cost 0.0002)
for seed in ${SEEDS:-1 2 3 4 5}; do
  python -u -m src.agents.research --data "$ETH" "${COMMON[@]}" --seed "$seed" "$@"
  python -u -m src.agents.research --data "$GLD" "${COMMON[@]}" "${DAILY[@]}" --seed "$seed" "$@"
done
echo "Done. Compare with: python -m src.research.runs --by model --datasets ETH/USD GLD"
