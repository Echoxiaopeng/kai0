#!/usr/bin/env bash
set -euo pipefail
cd /data3/kian_workspace/kai0
# caches must live under kai0 workspace (sandbox / home may be read-only)
mkdir -p /data3/kian_workspace/kai0/.cache/{uv,hf/datasets,hf/hub,xdg}
export UV_CACHE_DIR=/data3/kian_workspace/kai0/.cache/uv
export HF_HOME=/data3/kian_workspace/kai0/.cache/hf
export HF_DATASETS_CACHE=/data3/kian_workspace/kai0/.cache/hf/datasets
export HF_HUB_CACHE=/data3/kian_workspace/kai0/.cache/hf/hub
export XDG_CACHE_HOME=/data3/kian_workspace/kai0/.cache/xdg
export WANDB_DIR=/data3/kian_workspace/kai0/wandb
export WANDB_CACHE_DIR=/data3/kian_workspace/kai0/.cache/wandb
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD
RUNTIME=base_cont_from30k
mkdir -p "./experiment/${RUNNAME}/log"
rm -rf "./checkpoints/${RUNNAME}/${RUNTIME}"
LOG="./experiment/${RUNNAME}/log/${RUNTIME}.log"
if [ -f "${LOG}" ]; then mv "${LOG}" "${LOG}.bak_$(date +%H%M%S)"; fi
echo "launching at $(date) -> $LOG"
nohup .venv/bin/python -u scripts/train_pytorch.py "${RUNNAME}" \
  --exp_name="${RUNTIME}" --save_interval 5000 --batch_size 16 \
  > "${LOG}" 2>&1 &
echo "PID=$!"
echo $! > "./experiment/${RUNNAME}/log/${RUNTIME}.pid"
sleep 45
echo '==== log head ===='
head -n 100 "${LOG}" || true
echo '==== gpu ===='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true
ps -p "$(cat ./experiment/${RUNNAME}/log/${RUNTIME}.pid)" -o pid,etime,stat,cmd || true
