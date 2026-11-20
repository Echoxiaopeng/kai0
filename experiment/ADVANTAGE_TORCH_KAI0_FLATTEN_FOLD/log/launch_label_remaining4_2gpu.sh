#!/usr/bin/env bash
# Label the 4 remaining FlattenFold datasets on only 2 GPUs (sequential queue per GPU).
# Already-written data_KAI0_10000 episodes are skipped by eval.py.
set -euo pipefail
ROOT=/data3/kian_workspace/kai0
DATA_ROOT=/data3/kian_workspace/RLinf/FlattenFold
LOGDIR=$ROOT/experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/log
CKPT_TAG=10000
mkdir -p "$LOGDIR" "$ROOT/.cache/hf/datasets" "$ROOT/.cache/xdg"

export HF_HOME=$ROOT/.cache/hf
export HF_DATASETS_CACHE=$ROOT/.cache/hf/datasets
export HF_HUB_CACHE=$ROOT/.cache/hf/hub
export XDG_CACHE_HOME=$ROOT/.cache/xdg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Balanced by episode count: GPU0 ~109, GPU5 ~99
declare -A QUEUE
QUEUE[0]="flatten_fold_clothes_bw_1116_lerobot kai0_dagger_human_clothes_b_1117_lerobot"
QUEUE[5]="kai0_dagger_human_clothes_1106_lerobot kai0_dagger_human_clothes_mix1110_lerobot"

run_queue() {
  local gpu=$1
  local names=$2
  for name in $names; do
    local ds=$DATA_ROOT/$name
    local n_src n_out
    n_src=$(find "$ds/data" -name 'episode_*.parquet' | wc -l)
    n_out=0
    [ -d "$ds/data_KAI0_${CKPT_TAG}" ] && n_out=$(find "$ds/data_KAI0_${CKPT_TAG}" -name 'episode_*.parquet' | wc -l)
    if [ "$n_src" -gt 0 ] && [ "$n_out" -ge "$n_src" ]; then
      echo "[$(date +%H:%M:%S)] already done gpu=$gpu $name ($n_out/$n_src)"
      continue
    fi
    local log=$LOGDIR/label_${name}_cont${CKPT_TAG}_gpu${gpu}.log
    echo "[$(date +%H:%M:%S)] START gpu=$gpu $name ($n_out/$n_src) -> $log"
    (
      cd "$ROOT"
      export CUDA_VISIBLE_DEVICES=$gpu
      .venv/bin/python -u stage_advantage/annotation/eval.py Flatten-Fold KAI0 "$ds" >"$log" 2>&1
      echo "[$(date +%H:%M:%S)] DONE gpu=$gpu $name exit=$?" | tee -a "$log"
    )
  done
  echo "[$(date +%H:%M:%S)] QUEUE DONE gpu=$gpu"
}

declare -a PIDS=()
for gpu in 0 5; do
  (
    run_queue "$gpu" "${QUEUE[$gpu]}"
  ) >"$LOGDIR/label_remaining4_2gpu_worker${gpu}.log" 2>&1 &
  echo $! >"$LOGDIR/label_remaining4_2gpu_worker${gpu}.pid"
  PIDS+=($!)
  echo "worker gpu=$gpu pid=${PIDS[-1]} queue=${QUEUE[$gpu]}"
done

echo "waiting on ${#PIDS[@]} workers: ${PIDS[*]}"
fail=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    echo "worker $p failed"
    fail=1
  fi
done
echo "ALL DONE $(date) fail=$fail" | tee "$LOGDIR/label_remaining4_2gpu_done.txt"
exit $fail
