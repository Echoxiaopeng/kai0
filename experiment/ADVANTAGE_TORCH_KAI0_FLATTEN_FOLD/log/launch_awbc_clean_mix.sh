#!/usr/bin/env bash
# Launch after discretize check OK. Uses free GPUs 0,1 by default (6,7 busy).
set -euo pipefail
cd /data3/kian_workspace/kai0
# Force 2 GPUs for fsdp_devices=2 (do not inherit a single-GPU CUDA_VISIBLE_DEVICES).
export CUDA_VISIBLE_DEVICES="${AWBC_GPUS:-0,1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export WANDB_MODE="${WANDB_MODE:-offline}"
LOGDIR=checkpoints/pi05_flatten_fold_awbc_clean_mix/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/awbc_clean_from_mix1110_10k.log"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "logging to $LOG"
nohup .venv/bin/python scripts/train.py pi05_flatten_fold_awbc_clean_mix \
  --exp_name=awbc_clean_from_mix1110_10k \
  >"$LOG" 2>&1 &
echo "PID=$!"
echo "PID=$!" > "$LOGDIR/awbc_clean_from_mix1110_10k.pid"
