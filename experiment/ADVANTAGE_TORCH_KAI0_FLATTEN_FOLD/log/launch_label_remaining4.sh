#!/usr/bin/env bash
# Label the 4 remaining FlattenFold datasets with base_cont_from30k/10000 on GPUs 0-3.
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
# Avoid OOM from CUDA caching allocator fragmentation when co-tenant jobs leave ~60GB free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(
  flatten_fold_clothes_bw_1116_lerobot
  kai0_dagger_human_clothes_1106_lerobot
  kai0_dagger_human_clothes_b_1117_lerobot
  kai0_dagger_human_clothes_mix1110_lerobot
)

# Freest GPUs after 0-3 OOM'd (~19GB co-tenants); 0/5 have the most free (~62GB).
GPUS=(0 5 1 2)
declare -a PIDS=()

for i in "${!DATASETS[@]}"; do
  name=${DATASETS[$i]}
  gpu=${GPUS[$i]}
  ds=$DATA_ROOT/$name
  n_src=$(find "$ds/data" -name 'episode_*.parquet' | wc -l)
  n_out=0
  [ -d "$ds/data_KAI0_${CKPT_TAG}" ] && n_out=$(find "$ds/data_KAI0_${CKPT_TAG}" -name 'episode_*.parquet' | wc -l)
  if [ "$n_src" -gt 0 ] && [ "$n_out" -ge "$n_src" ]; then
    echo "already done: $name ($n_out/$n_src)"
    continue
  fi
  log=$LOGDIR/label_${name}_cont${CKPT_TAG}_gpu${gpu}.log
  echo "[$(date +%H:%M:%S)] START gpu=$gpu $name ($n_out/$n_src) -> $log"
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES=$gpu
    .venv/bin/python -u stage_advantage/annotation/eval.py Flatten-Fold KAI0 "$ds" >"$log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE gpu=$gpu $name exit=$?" | tee -a "$log"
  ) &
  echo $! >"$LOGDIR/label_${name}_cont${CKPT_TAG}_gpu${gpu}.pid"
  PIDS+=($!)
done

echo "waiting on ${#PIDS[@]} jobs: ${PIDS[*]}"
fail=0
for p in "${PIDS[@]:-}"; do
  if ! wait "$p"; then
    echo "job $p failed"
    fail=1
  fi
done
echo "ALL DONE $(date) fail=$fail" | tee "$LOGDIR/label_remaining4_done.txt"
exit $fail
