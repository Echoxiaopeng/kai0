#!/usr/bin/env bash
# Copy 5 clean FlattenFold datasets to *_awbc and run discretize_advantage.
set -euo pipefail

ROOT=/data3/kian_workspace/RLinf/FlattenFold
KAI0=/data3/kian_workspace/kai0
PY="$KAI0/.venv/bin/python"
DISC="$KAI0/stage_advantage/annotation/discretize_advantage.py"
LOGDIR="$KAI0/experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/log"
mkdir -p "$LOGDIR"

CLEANS=(
  flatten_fold_clothes_base_lerobot_adv_KAI0_10000_clean
  flatten_fold_clothes_bw_1116_lerobot__adv_KAI0_10000_clean
  kai0_dagger_human_clothes_1106_lerobot_adv_KAI0_10000_clean
  kai0_dagger_human_clothes_b_1117_lerobot__adv_KAI0_10000_clean
  kai0_dagger_human_clothes_mix1110_lerobot_adv_KAI0_10000_clean
)

copy_one() {
  local src_name="$1"
  local src="$ROOT/$src_name"
  local dst="$ROOT/${src_name}_awbc"
  echo "==== COPY $src_name -> ${src_name}_awbc ===="
  if [[ -d "$dst" ]]; then
    echo "Removing existing $dst"
    rm -rf "$dst"
  fi
  mkdir -p "$dst"
  # Copy data/ and meta/ (and export_manifest if present); symlink videos
  cp -a "$src/data" "$dst/data"
  cp -a "$src/meta" "$dst/meta"
  if [[ -f "$src/export_manifest.json" ]]; then
    cp -a "$src/export_manifest.json" "$dst/export_manifest.json"
  fi
  if [[ -e "$src/videos" ]]; then
    ln -sfn "$src/videos" "$dst/videos"
  fi
  echo "Copied. parquet count: $(find "$dst/data" -name 'episode_*.parquet' | wc -l)"
}

disc_one() {
  local src_name="$1"
  local dst="$ROOT/${src_name}_awbc"
  local log="$LOGDIR/discretize_${src_name}_awbc.log"
  local extra=()
  if [[ "$src_name" == *base* ]]; then
    extra+=(--stage-nums 2)
  fi
  echo "==== DISCRETIZE ${src_name}_awbc ${extra[*]:-} ===="
  "$PY" "$DISC" "$dst" \
    --threshold 30 --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage \
    "${extra[@]}" \
    2>&1 | tee "$log"
}

for c in "${CLEANS[@]}"; do
  copy_one "$c"
  disc_one "$c"
done

echo "ALL DONE" | tee "$LOGDIR/awbc_copy_discretize_DONE.txt"
