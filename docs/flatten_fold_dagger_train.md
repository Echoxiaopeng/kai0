# FlattenFold：收到数据集压缩包后如何训练

本文档描述本机（`/home/kian/kai0`）上，**从上传的 dagger / collect 压缩包 → LeRobot → mixture 配置 → norm stats → 开训** 的完整步骤。以 Agilex 双臂、π₀.₅ full fine-tune 为例。

---

## 0. 前置约定

| 项 | 路径 / 说明 |
|----|-------------|
| 仓库根目录 | `/home/kian/kai0` |
| 原始压缩包放置 | `data/*.tar.gz`（如 `data/kai0_dagger_10_29_novideo.tar.gz`） |
| LeRobot 数据根 | `data/FlattenFold/` |
| 已有子集示例 | `base`、`collect_1026`、`collect_1027`、`collect_1029` |
| 训练配置 | `src/openpi/training/config.py` |
| 转换脚本 | `scripts/convert_zarr_bimanual_to_lerobot.py` |
| Norm 脚本 | `scripts/compute_norm_states_fast.py` |
| 训练入口 | `scripts/train.py` |
| Python 环境 | 仓库 `.venv`（`PATH` 里优先用它） |
| Prompt | `Flatten and fold the cloth.` |
| FPS | `20` |

**压缩包内部期望结构（UMI / bimanual zarr）：**

```text
<某个目录名>/
  replay_buffer.zarr/
    data/          # 含 camera{0,1,2}_rgb、robot{0,1}_joint_pos、gripper 等
    meta/episode_ends/
  intervention/    # 可选；当前转换脚本不会写入 LeRobot
```

文件名里的 `novideo` 通常表示**没有单独 mp4**；RGB 仍在 zarr 的 `camera*_rgb` 里即可转换。

下文用占位符：

- `<DATE>`：日期标签，如 `1029`
- `<TAR>`：压缩包绝对路径
- `<COLLECT>`：新数据集名，如 `collect_1029`
- `<CONFIG>`：新 TrainConfig 名，如 `pi05_flatten_fold_mix1029`
- `<EXP>`：实验名，如 `my_fold_mix1029`
- `<INIT_CKPT>`：续训权重，如 `.../my_fold_mix1027/30000/params`

---

## 1. 检查压缩包与磁盘

```bash
cd /home/kian/kai0
ls -lah data/*.tar.gz
df -h /home/kian /dev/shm
```

建议：

- 系统盘剩余 **≥ 压缩包体积 × 2～3**（解压 + LeRobot 视频）
- `/dev/shm` 很大时（本机约 378G），**优先解压到 shm**，转完再删，磁盘更省、更快

可选：快速看包内是否有 zarr（大包可能较慢）：

```bash
tar -tzf <TAR> | head -40
# 应能看到 .../replay_buffer.zarr/...
```

---

## 2. 解压

### 推荐：解压到 `/dev/shm`

```bash
EXTRACT=/dev/shm/kai0_dagger_<DATE>
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
tar -xzf <TAR> -C "$EXTRACT"
find "$EXTRACT" -type d -name 'replay_buffer.zarr' | head
```

记下找到的 zarr 路径（或其父目录，转换脚本两种都能认）。

### 备选：解压到磁盘

```bash
EXTRACT=/home/kian/kai0/data/_extract_<DATE>
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
tar -xzf <TAR> -C "$EXTRACT"
```

### 快速核对 zarr

```bash
export PATH="/home/kian/kai0/.venv/bin:$PATH"
unset PYTHONPATH

python - <<'PY'
import zarr, numpy as np
from pathlib import Path
# 改成实际路径
z = Path("/dev/shm/kai0_dagger_<DATE>")  # 或含 replay_buffer.zarr 的目录
cands = list(z.rglob("replay_buffer.zarr"))
assert cands, "no replay_buffer.zarr"
root = zarr.open(str(cands[0]), mode="r")
ends = np.asarray(root["meta"]["episode_ends"][:])
print("zarr:", cands[0])
print("episodes:", len(ends), "frames:", int(ends[-1]))
print("data keys:", sorted(root["data"].keys()))
need = ["camera0_rgb","camera1_rgb","camera2_rgb","robot0_joint_pos","robot1_joint_pos",
        "robot0_gripper_width","robot1_gripper_width"]
missing = [k for k in need if k not in root["data"]]
print("missing:", missing)
PY
```

必须有三路相机和双臂关节/夹爪字段，否则转换会失败。

---

## 3. 转成 LeRobot 训练格式

输出目录约定：`data/FlattenFold/<COLLECT>/`（与 `base`、`collect_1026` 同级）。

```bash
cd /home/kian/kai0
export PATH="/home/kian/kai0/.venv/bin:$PATH"
unset PYTHONPATH

OUT_REPO="FlattenFold/<COLLECT>"   # 例：FlattenFold/collect_1029
rm -rf "data/$OUT_REPO"            # 若重跑，先清掉旧输出

python scripts/convert_zarr_bimanual_to_lerobot.py \
  --zarr "$EXTRACT" \
  --out-root /home/kian/kai0/data \
  --repo-id "$OUT_REPO" \
  --fps 20 \
  --prompt "Flatten and fold the cloth."
```

脚本会：

- 读 `replay_buffer.zarr`
- `observation.state` / `action` 做成 **14 维绝对关节**（下一帧 proprio 作 action）
- 相机映射：`camera1→top_head`，`camera0→hand_left`，`camera2→hand_right`
- 写出 parquet + 三路 mp4 + `meta/`

### 转换成功检查

```bash
python - <<'PY'
import json
from pathlib import Path
out = Path("/home/kian/kai0/data/FlattenFold/<COLLECT>")
info = json.loads((out / "meta/info.json").read_text())
print(info["total_episodes"], info["total_frames"], info["fps"], info["robot_type"])
assert (out / "data/chunk-000").exists()
assert (out / "videos/chunk-000").exists()
assert info["total_episodes"] > 0
PY
du -sh data/FlattenFold/<COLLECT>
```

### 清理临时解压（重要）

```bash
rm -rf "$EXTRACT"   # 尤其是 /dev/shm 上的，转完立刻删
```

> **说明：** 源包里的 `intervention/*.npy`（人/策略帧标签）**不会**进 LeRobot；当前训练是普通 BC，用人示教/纠错轨迹整包加权即可，不做帧级 intervention loss。

---

## 4. 新建 / 修改训练 Config（mixture）

编辑 `src/openpi/training/config.py`，**建议新建一个 TrainConfig**，不要直接改正在跑的旧实验配置。

### 4.1 命名

- Config：`pi05_flatten_fold_mix<DATE>`（例：`pi05_flatten_fold_mix1029`）
- Exp：`my_fold_mix<DATE>`（例：`my_fold_mix1029`）

### 4.2 `repo_id` 与 assets

`LerobotAgilexDataConfig.repo_id` 仍指向 **`data/FlattenFold/base`**（用于 norm_stats 落盘/加载路径）。  
真正采样列表在 `mixtures=` 里。

### 4.3 Mixture 权重含义

```text
P(抽到数据集 i) = weight_i / sum(weights)
```

与数据集帧数无关。例如：

| 数据 | weight | 占比 |
|------|--------|------|
| base | 1.0 | 4% |
| collect_1026 | 2.0 | 8% |
| collect_1027 | 2.0 | 8% |
| collect_1029 | 20.0 | **80%** |

合计旧数据 20%、新 dagger 80%，适合 **从已有 ckpt 短续训**。

经验建议（dagger 续训）：

- 新数据 **70%～80%**，旧数据合计 **20%～30%** 作防遗忘锚点
- `num_train_steps` 建议 **10k～30k**，不要无开又拉满 80k（新数据量通常只有几十～一百来个 episode）
- 新数据过小却权重大、步数又长 → 易过拟合

### 4.4 `weight_loader`（从哪续）

DAgger 迭代：**优先用采这批数据时线上部署的那个 ckpt** 继续训。

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/home/kian/kai0/checkpoints/.../<exp>/<step>/params"
)
```

注意路径要指到 **`.../params` 目录**（里面有 orbax 的 `ocdbt` 等），不是 step 上一级。

这是 **加载初始权重**（`--overwrite` 从 step 0 开训），不是 orbax `--resume` 恢复优化器状态。

### 4.5 配置模板

```python
TrainConfig(
    name="pi05_flatten_fold_mix<DATE>",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="/home/kian/kai0/data/FlattenFold/base",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
        mixtures=(
            LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/base", weight=1.0),
            LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/collect_1026", weight=2.0),
            LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/collect_1027", weight=2.0),
            LeRobotMixture(repo_id="/home/kian/kai0/data/FlattenFold/<COLLECT>", weight=20.0),
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "<INIT_CKPT>/params"   # 或 .../30000/params
    ),
    num_train_steps=30_000,
    save_interval=2500,
    keep_period=5_000,
    num_workers=8,
    batch_size=128,   # 4×A100；8 卡可试 256
    wandb_enabled=False,
),
```

### 4.6 校验 config 能加载

```bash
cd /home/kian/kai0
export PATH="/home/kian/kai0/.venv/bin:$PATH"
unset PYTHONPATH

python - <<'PY'
from openpi.training.config import get_config
c = get_config("pi05_flatten_fold_mix<DATE>")
w = sum(m.weight for m in c.data.mixtures)
print("steps", c.num_train_steps, "bs", c.batch_size)
print("init", c.weight_loader.params_path)
for m in c.data.mixtures:
    print(f"  {m.weight/w:5.1%}  w={m.weight:5.1f}  {m.repo_id}")
PY
```

---

## 5. 计算 Norm Stats（每次改 mixture 都要做）

Norm 会按 **mixture 的 weight** 加权统计，并写到：

```text
/home/kian/kai0/data/FlattenFold/base/norm_stats.json
```

（因为 `repo_id` 是绝对路径，`assets_dirs / repo_id` 会落到该目录。）

**先备份，再算：**

```bash
cd /home/kian/kai0
export PATH="/home/kian/kai0/.venv/bin:$PATH"
unset PYTHONPATH

NORM=data/FlattenFold/base/norm_stats.json
cp -a "$NORM" "${NORM}.bak_before_mix<DATE>"

python scripts/compute_norm_states_fast.py \
  --config-name pi05_flatten_fold_mix<DATE>
```

成功标志：日志出现 `Normalization stats saved to .../FlattenFold/base`。

> 训练启动时会把该 `norm_stats.json` 拷进 checkpoint 目录。

---

## 6. 开训

### 6.1 选 GPU

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

4 卡 + `batch_size=128` 时，选一组空闲卡，例如 `0,1,2,3`。

### 6.2 用 tmux 启动（推荐）

```bash
cd /home/kian/kai0
LOG_DIR=checkpoints/pi05_flatten_fold_mix<DATE>/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/<EXP>.log"

# 若旧会话占用同名，先处理
# tmux kill-session -t kai0_fold

tmux new-session -d -s kai0_fold -c /home/kian/kai0 bash -lc '
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export PATH=/home/kian/kai0/.venv/bin:/home/kian/.local/bin:$PATH
unset PYTHONPATH
LOG=checkpoints/pi05_flatten_fold_mix<DATE>/logs/<EXP>.log
echo "==== $(date -Is) start train ====" | tee -a "$LOG"
python scripts/train.py pi05_flatten_fold_mix<DATE> \
  --exp-name=<EXP> \
  --overwrite \
  2>&1 | tee -a "$LOG"
echo "==== $(date -Is) train exited ====" | tee -a "$LOG"
exec bash
'
```

附着查看：

```bash
tmux attach -t kai0_fold
# 或
tail -f checkpoints/pi05_flatten_fold_mix<DATE>/logs/<EXP>.log
```

### 6.3 健康检查（启动后几分钟）

日志中应依次出现：

1. `Loaded norm stats from .../FlattenFold/base`
2. `MixtureDataset: ... weight=...`（权重与 config 一致）
3. `Restoring checkpoint from <INIT_CKPT>`
4. `Progress on: x.it/30.0kit`（**总数应等于 `num_train_steps`**）

GPU 显存应明显占用（本机约 70GB+/卡）。

若 Progress 总数和 config 不一致：多半是改完 config **没重启训练**；杀掉后重新 `--overwrite` 开。

### 6.4 产出位置

```text
checkpoints/<CONFIG>/<EXP>/
  <step>/params/
  <step>/train_state/
  <step>/assets/
  norm_stats.json
```

`save_interval` / `keep_period` 控制临时保存与永久保留频率。中途可拿中间 step 做实机评测，够用就停。

---

## 7. 一页清单（复制改占位符）

```bash
cd /home/kian/kai0
export PATH="/home/kian/kai0/.venv/bin:$PATH"
unset PYTHONPATH

# --- 按次填写 ---
TAR=data/kai0_dagger_XX_XX_novideo.tar.gz
DATE=XXXX
COLLECT=collect_$DATE
CONFIG=pi05_flatten_fold_mix$DATE
EXP=my_fold_mix$DATE
INIT_CKPT=/home/kian/kai0/checkpoints/.../.../params   # 含 params 这一层
EXTRACT=/dev/shm/kai0_dagger_$DATE
GPUS=0,1,2,3

# 1) 解压
rm -rf "$EXTRACT" && mkdir -p "$EXTRACT"
tar -xzf "$TAR" -C "$EXTRACT"

# 2) 转换
rm -rf "data/FlattenFold/$COLLECT"
python scripts/convert_zarr_bimanual_to_lerobot.py \
  --zarr "$EXTRACT" \
  --out-root /home/kian/kai0/data \
  --repo-id "FlattenFold/$COLLECT" \
  --fps 20 \
  --prompt "Flatten and fold the cloth."
rm -rf "$EXTRACT"

# 3) 编辑 config.py：新建 $CONFIG，写好 mixtures / weight_loader / steps

# 4) Norm
cp -a data/FlattenFold/base/norm_stats.json \
      data/FlattenFold/base/norm_stats.json.bak_before_mix$DATE
python scripts/compute_norm_states_fast.py --config-name "$CONFIG"

# 5) 训练
mkdir -p "checkpoints/$CONFIG/logs"
tmux new-session -d -s kai0_fold -c /home/kian/kai0 \
  "export CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline; \
   export PATH=/home/kian/kai0/.venv/bin:/home/kian/.local/bin:\$PATH; unset PYTHONPATH; \
   python scripts/train.py $CONFIG --exp-name=$EXP --overwrite \
     2>&1 | tee -a checkpoints/$CONFIG/logs/$EXP.log; exec bash"
```

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| 转换缺 `camera*_rgb` | 包不是完整 zarr，或解压路径错 |
| OOM | 降低 `batch_size`（如 128→64），或减少 GPU 数并保持 batch 能被卡数整除 |
| Norm / 训练读不到 stats | 确认 `data/FlattenFold/base/norm_stats.json` 存在且刚按新 config 算过 |
| Progress 总步数不对 | 改 config 后必须重启进程；正在跑的不会热更新 |
| 想「接着 step 训」而不是重新从 0 | 用同 exp 的 `--resume`；当前 dagger 流程一般是 **weight_loader 换权重 + 新 exp + overwrite** |
| 旧实验被新 norm 影响 | 每次算 norm 前备份 `norm_stats.json`；旧 ckpt 目录里通常已有当时拷贝的一份 |

---

## 9. 和「采集」的关系（可选阅读）

- **采集**：`train_deploy_alignment/dagger/`（策略推理 → 按 `d` 介入 → 存 HDF5/zarr）
- **本机训练**：本文档（zarr → LeRobot → BC fine-tune）
- 训练侧**不区分** intervention 帧；加强新数据靠 **mixture weight** 与 **较短续训步数**
- 更完整的采集说明见 `train_deploy_alignment/dagger/README.md`

---

## 10. 本轮实例（1029）备忘

| 项 | 值 |
|----|-----|
| 压缩包 | `data/kai0_dagger_10_29_novideo.tar.gz` |
| LeRobot | `data/FlattenFold/collect_1029`（102 ep / 116792 frames） |
| Config | `pi05_flatten_fold_mix1029` |
| Mixture | base:1026:1027:1029 = 1:2:2:20 → 约 4%/8%/8%/80% |
| Init | `checkpoints/pi05_flatten_fold_normal/my_fold_mix1027/30000/params` |
| Steps | 30_000，`save_interval=2500`，`batch_size=128`，4 GPU |
| Exp / log | `my_fold_mix1029` → `checkpoints/pi05_flatten_fold_mix1029/logs/my_fold_mix1029.log` |
| tmux | `kai0_fold` |
