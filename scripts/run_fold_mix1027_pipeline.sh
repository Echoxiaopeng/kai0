#!/usr/bin/env bash
# Unattended: wait for Baidu download → extract → LeRobot convert → patch mixture →
# compute norm stats → 4-GPU train. Retries / OOM fallback included.
set -euo pipefail

ROOT="/home/kian/kai0"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:/home/kian/.local/bin:$PATH"
unset PYTHONPATH

TAR="$ROOT/data/collect_bimanual_10_27.tar.gz"
TAR_MARK="${TAR}.BaiduPCS-Go-downloading"
EXTRACT_SHM="/dev/shm/kai0_zarr_1027"
EXTRACT_DISK="$ROOT/data/_extract_1027"
OUT_REPO="FlattenFold/collect_1027"
OUT_DIR="$ROOT/data/$OUT_REPO"
CONFIG_PY="$ROOT/src/openpi/training/config.py"
NORM_JSON="$ROOT/data/FlattenFold/base/norm_stats.json"
LOG_DIR="$ROOT/checkpoints/pi05_flatten_fold_normal/logs"
mkdir -p "$LOG_DIR"
PIPE_LOG="$LOG_DIR/pipeline_mix1027.log"
EXP_NAME="${EXP_NAME:-my_fold_mix1027}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
PROMPT="Flatten and fold the cloth."

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }
fail() { log "FATAL: $*"; exit 1; }

retry() {
  local n="$1" sleep_s="$2"; shift 2
  local i=1
  until "$@"; do
    if (( i >= n )); then
      return 1
    fi
    log "retry $i/$n failed: $* — sleep ${sleep_s}s"
    sleep "$sleep_s"
    ((i++)) || true
  done
}

########################################
# 1) Wait for download
########################################
wait_download() {
  log "Waiting for download: $TAR"
  local prev=-1 stable=0
  while true; do
    if [[ -e "$TAR_MARK" ]]; then
      local sz=0
      [[ -f "$TAR" ]] && sz=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
      log "downloading… size=${sz} mark present"
      stable=0
      sleep 60
      continue
    fi
    [[ -f "$TAR" ]] || { log "tar missing, waiting…"; sleep 30; continue; }
    local sz
    sz=$(stat -c%s "$TAR")
    if [[ "$sz" -eq "$prev" && "$sz" -gt 1000000000 ]]; then
      ((stable++)) || true
    else
      stable=0
    fi
    prev=$sz
    log "no mark; size=$sz stable_checks=$stable"
    if (( stable >= 2 )); then
      break
    fi
    sleep 30
  done
  log "Download looks complete ($(numfmt --to=iec "$sz" 2>/dev/null || echo "$sz")). Validating archive…"
  if ! gzip -t "$TAR" 2>>"$PIPE_LOG"; then
    fail "gzip -t failed — archive corrupt?"
  fi
  if ! tar -tzf "$TAR" >/dev/null 2>>"$PIPE_LOG"; then
    fail "tar -tzf failed — archive corrupt?"
  fi
  log "Archive OK"
}

########################################
# 2) Extract
########################################
do_extract() {
  local dest="$EXTRACT_SHM"
  local avail_shm
  avail_shm=$(df -B1 --output=avail /dev/shm | tail -1 | tr -d ' ')
  # Prefer shm if > 80G free
  if [[ "${avail_shm:-0}" -lt $((80 * 1024 * 1024 * 1024)) ]]; then
    log "shm free too small ($avail_shm); using disk extract"
    dest="$EXTRACT_DISK"
  fi
  rm -rf "$dest"
  mkdir -p "$dest"
  log "Extracting to $dest"
  tar -xzf "$TAR" -C "$dest"
  local zarr
  zarr=$(find "$dest" -type d -name 'replay_buffer.zarr' | head -1 || true)
  [[ -n "$zarr" ]] || fail "no replay_buffer.zarr under $dest"
  log "Found zarr: $zarr"
  echo "$dest" > /tmp/kai0_extract_1027_path.txt
}

########################################
# 3) Convert
########################################
do_convert() {
  local dest
  dest=$(cat /tmp/kai0_extract_1027_path.txt)
  log "Converting zarr → $OUT_DIR"
  rm -rf "$OUT_DIR"
  "$ROOT/.venv/bin/python" scripts/convert_zarr_bimanual_to_lerobot.py \
    --zarr "$dest" \
    --out-root "$ROOT/data" \
    --repo-id "$OUT_REPO" \
    --fps 20 \
    --prompt "$PROMPT" \
    2>&1 | tee -a "$PIPE_LOG"
  [[ -f "$OUT_DIR/meta/info.json" ]] || fail "missing meta/info.json"
  python3 - <<'PY' "$OUT_DIR" "$PIPE_LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
info = json.loads((out / "meta/info.json").read_text())
eps, frames = info.get("total_episodes", 0), info.get("total_frames", 0)
msg = f"converted episodes={eps} frames={frames}"
print(msg)
open(sys.argv[2], "a").write(msg + "\n")
assert eps > 0 and frames > 0, msg
assert (out / "data").is_dir() and (out / "videos").is_dir()
PY
  log "Convert OK — cleaning extract"
  rm -rf "$EXTRACT_SHM" "$EXTRACT_DISK" || true
}

########################################
# 4) Patch config mixtures
########################################
do_patch_config() {
  log "Patching mixtures in config.py → base@1 + collect_1026@2 + collect_1027@2"
  python3 - <<'PY' "$CONFIG_PY"
from pathlib import Path
import re, sys
path = Path(sys.argv[1])
text = path.read_text()
pattern = re.compile(
    r'(name="pi05_flatten_fold_normal".*?mixtures=\()(.*?)(\n\s*\),)',
    re.S,
)
m = pattern.search(text)
if not m:
    raise SystemExit("could not find pi05_flatten_fold_normal mixtures block")
new_body = """
                LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/base", weight=1.0),
                LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/collect_1026", weight=2.0),
                LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/collect_1027", weight=2.0),
            """
text2 = text[: m.start(2)] + new_body + text[m.end(2) :]
path.write_text(text2)
print("config patched OK")
PY
}

########################################
# 5) Norm stats
########################################
do_norm() {
  if [[ -f "$NORM_JSON" ]]; then
    cp -a "$NORM_JSON" "${NORM_JSON}.bak_before_mix1027"
    log "Backed up norm_stats → ${NORM_JSON}.bak_before_mix1027"
  fi
  log "Computing norm stats…"
  "$ROOT/.venv/bin/python" scripts/compute_norm_states_fast.py \
    --config-name pi05_flatten_fold_normal \
    2>&1 | tee -a "$PIPE_LOG"
  [[ -f "$NORM_JSON" ]] || fail "norm_stats.json not written"
  log "Norm stats OK"
}

########################################
# 6) Train (4 GPUs) with OOM fallback
########################################
set_batch_size() {
  local bs="$1"
  python3 - <<PY "$CONFIG_PY" "$bs"
from pathlib import Path
import re, sys
path = Path(sys.argv[1]); bs = sys.argv[2]
text = path.read_text()

def repl(m):
    block = m.group(0)
    block2, n = re.subn(r'batch_size=\d+', f'batch_size={bs}', block, count=1)
    if n != 1:
        raise SystemExit('batch_size not found in config block')
    return block2

text2, n = re.subn(
    r'name="pi05_flatten_fold_normal".*?wandb_enabled=False,',
    repl,
    text,
    count=1,
    flags=re.S,
)
if n != 1:
    raise SystemExit('pi05_flatten_fold_normal block not found for batch_size')
path.write_text(text2)
print(f'batch_size set to {bs}')
PY
}

start_train() {
  log "Starting train exp=$EXP_NAME gpus=$TRAIN_GPUS"
  if tmux has-session -t kai0_fold 2>/dev/null; then
    log "killing existing tmux kai0_fold"
    tmux kill-session -t kai0_fold || true
    sleep 2
  fi
  pkill -f 'scripts/train.py pi05_flatten_fold_normal' 2>/dev/null || true
  sleep 2

  tmux new-session -d -s kai0_fold -c "$ROOT" \
    "export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS; \
     export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9; \
     export WANDB_MODE=offline; \
     export PATH=$ROOT/.venv/bin:/home/kian/.local/bin:\$PATH; \
     unset PYTHONPATH; \
     EXP_NAME=$EXP_NAME; \
     LOG=$LOG_DIR/\${EXP_NAME}.log; \
     echo \"==== \$(date -Is) start train exp=\${EXP_NAME} gpus=\$CUDA_VISIBLE_DEVICES args=--overwrite on \$(hostname) ====\" | tee -a \$LOG; \
     $ROOT/.venv/bin/python scripts/train.py pi05_flatten_fold_normal --exp-name=\${EXP_NAME} --overwrite 2>&1 | tee -a \$LOG"
}

train_healthy() {
  pgrep -f 'scripts/train.py pi05_flatten_fold_normal' >/dev/null || return 1
  return 0
}

watch_train_and_fix_oom() {
  local logf="$LOG_DIR/${EXP_NAME}.log"
  local start_ts
  start_ts=$(date +%s)
  local saw_progress=0
  local oom_hit=0
  while true; do
    sleep 60
    if ! train_healthy; then
      log "train process dead"
      if grep -qiE 'out of memory|RESOURCE_EXHAUSTED|OOM|CUDA.?out of memory|jaxlib.*RESOURCE' "$logf" 2>/dev/null; then
        oom_hit=1
      fi
      break
    fi
    if grep -qE 'Progress on:|Created BasePyTreeCheckpointHandler|Restoring checkpoint' "$logf" 2>/dev/null; then
      saw_progress=1
    fi
    if (( saw_progress )) && (( $(date +%s) - start_ts > 1500 )); then
      if grep -qE 'Progress on: [0-9]+it' "$logf" 2>/dev/null; then
        log "Train looks healthy (progress lines present). Pipeline done."
        return 0
      fi
    fi
    if grep -qiE 'out of memory|RESOURCE_EXHAUSTED|CUDA.?out of memory' "$logf" 2>/dev/null; then
      log "OOM detected in log…"
      oom_hit=1
      pkill -f 'scripts/train.py pi05_flatten_fold_normal' 2>/dev/null || true
      sleep 5
      break
    fi
  done
  if (( oom_hit )); then
    log "OOM → set batch_size=128 and restart once"
    set_batch_size 128
    start_train
    sleep 120
    if train_healthy; then
      log "Restarted with batch_size=128; leaving train running."
      return 0
    fi
    fail "Train failed again after OOM restart"
  fi
  log "Non-OOM train death — restarting once"
  start_train
  sleep 180
  train_healthy || fail "Train failed after restart"
  log "Train restarted; leaving it running."
}

########################################
# Main
########################################
log "===== pipeline mix1027 START ====="
wait_download
retry 3 30 do_extract || fail "extract failed"
retry 2 60 do_convert || fail "convert failed"
do_patch_config
retry 2 60 do_norm || fail "norm failed"
start_train
watch_train_and_fix_oom
log "===== pipeline mix1027 END (train should be in tmux kai0_fold) ====="
log "attach: tmux attach -t kai0_fold"
log "pipeline log: $PIPE_LOG"
log "train log: $LOG_DIR/${EXP_NAME}.log"
