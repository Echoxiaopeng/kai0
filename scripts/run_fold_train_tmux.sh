#!/usr/bin/env bash
set -euo pipefail
cd /home/kian/kai0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export PATH="/home/kian/kai0/.venv/bin:/home/kian/.local/bin:$PATH"
unset PYTHONPATH

EXP_NAME="${EXP_NAME:-my_fold_mix1026}"
# Log outside ckpt dir so train.py can create the checkpoint directory itself.
LOG_DIR="checkpoints/pi05_flatten_fold_normal/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${EXP_NAME}.log"
EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
else
  EXTRA_ARGS+=(--overwrite)
fi
echo "==== $(date -Is) start train exp=${EXP_NAME} gpus=${CUDA_VISIBLE_DEVICES} args=${EXTRA_ARGS[*]} on $(hostname) ====" | tee -a "$LOG"
exec /home/kian/kai0/.venv/bin/python scripts/train.py \
  pi05_flatten_fold_normal \
  --exp-name="${EXP_NAME}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
