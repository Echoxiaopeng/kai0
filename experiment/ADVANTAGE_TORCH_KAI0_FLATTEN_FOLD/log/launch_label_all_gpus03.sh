#!/usr/bin/env bash
# Label all FlattenFold datasets with base_cont_from30k/10000 on GPUs 0-3.
# One dataset at a time per GPU; remaining jobs queued FIFO.
set -euo pipefail
ROOT=/data3/kian_workspace/kai0
DATA_ROOT=/data3/kian_workspace/RLinf/FlattenFold
LOGDIR=$ROOT/experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/log
CKPT_TAG=10000
mkdir -p "$LOGDIR" \
  "$ROOT/.cache/hf/datasets" "$ROOT/.cache/hf/hub" "$ROOT/.cache/xdg"

export HF_HOME=$ROOT/.cache/hf
export HF_DATASETS_CACHE=$ROOT/.cache/hf/datasets
export HF_HUB_CACHE=$ROOT/.cache/hf/hub
export XDG_CACHE_HOME=$ROOT/.cache/xdg
export PYTHONUNBUFFERED=1

# Skip collect_1026 (already fully labeled). Include everything else under FlattenFold.
DATASETS=(
  base
  collect_1027
  collect_1029_dagger
  collect_1029_human
  collect_1030
  collect_1106_dagger
  collect_1106_human
  collect_1110
)

# Drop datasets that are already fully labeled for this ckpt.
QUEUE=()
for name in "${DATASETS[@]}"; do
  ds=$DATA_ROOT/$name
  if [ ! -d "$ds/data" ]; then
    echo "skip missing: $name"
    continue
  fi
  n_src=$(find "$ds/data" -name 'episode_*.parquet' | wc -l)
  n_out=0
  if [ -d "$ds/data_KAI0_${CKPT_TAG}" ]; then
    n_out=$(find "$ds/data_KAI0_${CKPT_TAG}" -name 'episode_*.parquet' | wc -l)
  fi
  if [ "$n_src" -gt 0 ] && [ "$n_out" -ge "$n_src" ]; then
    echo "already done: $name ($n_out/$n_src)"
    continue
  fi
  echo "queue: $name ($n_out/$n_src labeled)"
  QUEUE+=("$name")
done

if [ ${#QUEUE[@]} -eq 0 ]; then
  echo "nothing to do"
  exit 0
fi

GPUS=(0 1 2 3)
N_GPU=${#GPUS[@]}
declare -a PIDS=()
declare -a GPU_BUSY=()
for g in "${GPUS[@]}"; do GPU_BUSY[$g]=0; done

run_one() {
  local gpu=$1
  local name=$2
  local ds=$DATA_ROOT/$name
  local log=$LOGDIR/label_${name}_cont${CKPT_TAG}_gpu${gpu}.log
  echo "[$(date +%H:%M:%S)] START gpu=$gpu dataset=$name -> $log"
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES=$gpu
    .venv/bin/python -u stage_advantage/annotation/eval.py Flatten-Fold KAI0 "$ds" \
      >"$log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE gpu=$gpu dataset=$name exit=$?" | tee -a "$log"
  ) &
  local pid=$!
  echo "$pid" >"$LOGDIR/label_${name}_cont${CKPT_TAG}_gpu${gpu}.pid"
  PIDS+=("$pid:$gpu:$name")
  GPU_BUSY[$gpu]=1
}

idx=0
# fill GPUs
while [ $idx -lt ${#QUEUE[@]} ] && [ $idx -lt $N_GPU ]; do
  run_one "${GPUS[$idx]}" "${QUEUE[$idx]}"
  idx=$((idx + 1))
done

# as each finishes, launch next
while [ $idx -lt ${#QUEUE[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  sleep 20
  NEW_PIDS=()
  for entry in "${PIDS[@]:-}"; do
    [ -z "${entry:-}" ] && continue
    pid=${entry%%:*}
    rest=${entry#*:}
    gpu=${rest%%:*}
    name=${rest#*:}
    if kill -0 "$pid" 2>/dev/null; then
      NEW_PIDS+=("$entry")
    else
      wait "$pid" || true
      echo "[$(date +%H:%M:%S)] FREE gpu=$gpu after $name"
      GPU_BUSY[$gpu]=0
      if [ $idx -lt ${#QUEUE[@]} ]; then
        run_one "$gpu" "${QUEUE[$idx]}"
        idx=$((idx + 1))
      fi
    fi
  done
  PIDS=("${NEW_PIDS[@]:-}")
  # progress summary
  {
    echo "---- $(date) ----"
    for name in "${QUEUE[@]}"; do
      ds=$DATA_ROOT/$name
      n_src=$(find "$ds/data" -name 'episode_*.parquet' | wc -l)
      n_out=0
      [ -d "$ds/data_KAI0_${CKPT_TAG}" ] && n_out=$(find "$ds/data_KAI0_${CKPT_TAG}" -name 'episode_*.parquet' | wc -l)
      echo "$name $n_out/$n_src"
    done
  } >"$LOGDIR/label_all_gpus03_status.txt"
done

echo "ALL DONE $(date)" | tee "$LOGDIR/label_all_gpus03_done.txt"
