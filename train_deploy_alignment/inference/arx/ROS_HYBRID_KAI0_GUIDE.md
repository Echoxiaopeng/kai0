# ARX-X5 ROS 双臂推理使用教程（Kai0 / GoPro+D435）

本文档对应本机当前可用的 **无主臂 remote_slave + hybrid 相机（左右 GoPro + 中间 D435）+ Kai0 websocket 策略** 流程。  
旧版「主从臂 + 三路 RealSense」见同目录 [README.md](README.md)。

---

## 1. 架构一览

| 角色 | 做什么 |
|------|--------|
| `X5Controller` ×2 | CAN 上两只从臂，订阅“假主臂”话题做位置跟随 |
| `serve_kai0_policy.py` | GPU 上加载 checkpoint，websocket 出动作 |
| `arx_openpi_inference_rtc.py` | 采相机/关节 → 组 Kai0 obs → 收 action → 发 ROS 指令 |

话题约定（与 `dual_arm_no_master.yaml` 一致）：

| 臂 | CAN | 状态发布 | 指令订阅（由推理脚本发布） |
|----|-----|----------|---------------------------|
| 左 | `can0` | `/arm_slave_l_status` | `/arm_master_l_status` |
| 右 | `can1` | `/arm_slave_r_status` | `/arm_master_r_status` |

控制器为 `remote_slave` + `arm_end_type=0`：夹爪指令会走 `setCatch(cmd * 5)`。

---

## 2. 启动前检查

### 2.1 CAN / `slcand`（重启电脑后必做）

USB2CAN 走 **slcan**：`/dev/arxcan*` 由 udev 按序列号绑定（规则一般已写在 `/etc/udev/rules.d/arx_can.rules`，**重启后不用重写**）。  
但 **`slcand` 与 `can0`/`can1` 不会开机自启**（除非你单独配了 `arxcan-setup.service`），重启后必须重新拉起。

```bash
# 1) 插好两路 USB2CAN，确认设备节点
ls -l /dev/arxcan0 /dev/arxcan1

# 2) 创建并拉起 SocketCAN（左 can0 / 右 can1）
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ip link set can0 up
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ip link set can1 up

# 3) 确认 UP
ip link show type can
```

若没有 `/dev/arxcan*`：重插 USB；仍没有可再跑  
`/home/lenovo/kian/arx5-sdk/setup_arxcan1.sh`（会刷新 udev；脚本本身主要保证 can1）。  
`ip link` 里看不到 `can0`/`can1` 时不要 launch，否则臂控起不来。

### 2.2 其它

1. **ROS2 Humble** + 本仓库 `X5_ws` 已编译并 source：
   ```bash
   source /opt/ros/humble/setup.bash
   source /media/lenovo/data/kian/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash
   ```
2. **Python**：`conda activate kai0_inference`，已安装 `openpi-client`、相机依赖（含 umi-arx 相关库）。
3. **umi-arx 路径**（hybrid 相机）：默认  
   `--umi_root /home/lenovo/kian/umi-arx-kian`
4. **清场**：启动后会 auto-homing / 去 init，周围不要站人。

---

## 3. 推荐启动顺序（三个终端）

### 终端 A — 双臂控制器（无主臂）

```bash
source /opt/ros/humble/setup.bash
source /media/lenovo/data/kian/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash

ros2 launch arx_x5_controller open_dual_arm_no_master.launch.py
```

正常时屏幕会刷 `ARX方舟无限`（SDK 心跳，可忽略）。  
确认节点：

```bash
ros2 node list   # 应有 arm_slave_l / arm_slave_r
ros2 topic list  # 应有 arm_slave_*_status / arm_master_*_status
```

### 终端 B — 策略服务（GPU 本机或远端）

在 **kai0 仓库根目录**：

```bash
# 普通推理（无 RTC）
uv run scripts/serve_kai0_policy.py \
  --checkpoint /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_27/30000 \
  --port 8000

# 若客户端要开 --rtc_mode，服务端必须加 --rtc
uv run scripts/serve_kai0_policy.py \
  --checkpoint /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_27/30000 \
  --rtc --port 8000
```

日志里应看到 `Policy server listening ... supports_rtc=True/False`。

### 终端 C — 推理客户端

```bash
source /opt/ros/humble/setup.bash
source /media/lenovo/data/kian/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash
conda activate kai0_inference

cd /media/lenovo/data/kian/kai0/train_deploy_alignment/inference/arx/inference

python arx_openpi_inference_rtc.py \
  --host 127.0.0.1 \
  --port 8000 \
  --camera_backend hybrid \
  --camera_reorder 0,1 \
  --gripper_mode snap_kai0 \
  --gripper_feedback_open 4.61
```

常用可选参数：

```bash
# 连续夹爪（线性映射）；夹布不紧时可改回 snap_kai0
  --gripper_mode scale_kai0

# RTC（服务端必须 --rtc）
  --rtc_mode

# 控制/推理频率
  --control_frequency 20 --inference_rate 4 --chunk_size 50

# 左右腕相机对调（预览发现反了时）
  --camera_reorder 1,0
```

---

## 4. 运行时键盘（推理脚本终端内）

启动后流程：

1. Auto-homing → **zero**（关节约 0，夹爪打开）  
2. 按 **Enter** 进入 warmup（策略空跑）  
3. 平滑到 **init pose**（采集 ep0 中位关节，脚本内 `left0`/`right0`）  
4. 等待键盘：

| 键 | 作用 |
|----|------|
| **C** | 开始一局 episode（策略开始出动作） |
| **S** | 停止当前 episode，保持当前姿态 |
| **Enter** | （仅在已 S 停下后）平滑回到 **init pose** |
| **Q** | 停局 → 回到启动时的 **zero** → 再退出脚本 |
| **Ctrl+C** | 同 Q（先 zero 再退） |

建议操作：

```
C 开局 → … → S 停 → Enter 回 init → C 再开
结束：Q（回 zero 后退出）→ 再 Ctrl+C 停 ros2 launch
```

**不要**在手臂伸展、还在扛负载时直接杀 `ros2 launch`，否则电机会松、手臂下掉。

---

## 5. 相机（hybrid）

默认 `camera_backend=hybrid`：

- 左腕 / 右腕：GoPro（经 Elgato 等 V4L）  
- 头：Intel D435  

启动时会写预览图（默认目录 `/tmp/arx_ros_camera_preview/`）：

- 马赛克从左到右：**左手腕 | 头 | 右手腕**  
- 若左右反了：加 `--camera_reorder 1,0`

推理过程中默认还会起一个 **网页实时预览 + 控制台**（MJPEG）：

```text
http://127.0.0.1:8080/
```

浏览器可看三路画面、关节/夹爪状态，并用按钮或键盘控制（与终端相同）：**C** 开始、**S** 停止、**Enter/H** 回 init、**Q** 回零退出。  
换端口：`--web_camera_port 8090`；关掉：`--no_web_camera_preview`。

策略观测键（与 Kai0 训练一致）：

- `observation.images.top_head` / `hand_left` / `hand_right`  
- `observation.state`：`[left_q6, left_grip_m, right_q6, right_grip_m]`（夹爪为米）

---

## 6. 夹爪映射（重要）

| 模式 | 行为 | 建议 |
|------|------|------|
| `snap_kai0`（默认） | Kai0 开度 `<0.042m` → 全关 `-0.8`；否则全开 `5.0` | **夹布推荐** |
| `scale_kai0` | 米制线性映到 `[-0.8, 5]` | 连续开合；中段可能跟不紧 |
| `binary_ros` / `raw` | 旧逻辑 / 不转换 | 一般不用 |

说明：

- 策略输出约 `[0, 0.085]` 米（越大越开）。  
- ROS 发布约 `[-0.8, 5]`；`arm_end_type=0` 下控制器 `setCatch(cmd*5)`。  
- 全关用 `-0.8`（过闭、有力）；`0` 往往只到零位、夹不紧。  
- 本机反馈全开约 `raw≈4.61`，应用 `--gripper_feedback_open 4.61`。

调试日志（默认开）写在：

```text
train_deploy_alignment/inference/arx/inference/logs/gripper_<时间戳>/
  gripper_steps.csv   # 逐步 model / scale / snap / cmd / state / raw
  summary.txt
```

---

## 7. RTC 说明

RTC **两边都要开**才生效：

| 侧 | 正确 | 错误 |
|----|------|------|
| Server | `--rtc` | 只开普通 serve |
| Client | `--rtc_mode` | 不加该参数 |

客户端启动时会打印：

```text
[rtc] client --rtc_mode=... | server supports_rtc=...
```

真在跑 RTC 时应看到 `[rtc] chunk ok ... delay_steps=N`。  
若只有一边开，观感会几乎和非 RTC 一样。  
改过 `pi0_rtc.py` 后需 **重启** `serve_kai0_policy.py`。

---

## 8. Init / Zero 区别

| 名字 | 何时 | 内容 |
|------|------|------|
| **Zero** | 启动 auto-homing；**Q** 退出前 | 关节 ~0，夹爪打开（`smooth_return_to_zero`） |
| **Init pose** | Warmup 后；**Enter**（停局后） | 脚本里 `left0`/`right0`（与 umi collect ep0 中位一致，夹爪用 ROS `5.0`） |

改 init：编辑 `arx_openpi_inference_rtc.py` 中 `left0` / `right0`。

---

## 9. 安全退出顺序

1. 推理终端按 **Q**（回 zero 并退出推理）  
2. 再在 launch 终端 **Ctrl+C** 停控制器  
3. 最后停策略 server  

直接杀 launch → 控制器 `cleanup()` 只 `interfaces_ptr_.reset()`，无伺服，手臂会掉下来。

---

## 10. 常见问题

| 现象 | 处理 |
|------|------|
| 重启后无 `can0`/`can1` 或 launch 起不来 | 见 §2.1：先 `ls /dev/arxcan*`，再 `slcand` + `ip link set … up` |
| 夹爪一直关 / 一直开 | 检查 `--gripper_mode`；Kai0 米制不要用 `binary_ros` 的 2.5 阈值 |
| 夹不紧 | 用 `snap_kai0` + `gripper_close=-0.8`；勿用中间开度硬夹 |
| C 前手臂下坠 | 应有 idle hold；确认没在 hold 与轨迹抢指令；更新到最新脚本 |
| Enter 回 init 后按 C 又弹回旧位 | 旧 bug：会重放 RTC chunk；更新脚本后已清 chunk |
| `ConnectionClosedError 1011` + RTC | 多为 `prev_action_chunk` 长度与 `action_horizon` 不一致；重启带修复的 server，client 用 `--rtc_mode` 且 `chunk_size` 与模型一致（多为 50） |
| 相机左右反 | `--camera_reorder 1,0`，看 `/tmp/arx_ros_camera_preview/` |
| 策略 obs KeyError | 确认走 Kai0 flat keys（当前脚本已是） |

---

## 11. 最小抄写命令（本机）

```bash
# 0 CAN（重启后先做；见 §2.1）
ls -l /dev/arxcan0 /dev/arxcan1
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ip link set can0 up
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ip link set can1 up

# A 臂
source /opt/ros/humble/setup.bash
source /media/lenovo/data/kian/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash
ros2 launch arx_x5_controller open_dual_arm_no_master.launch.py

# B 策略（kai0 根目录）
uv run scripts/serve_kai0_policy.py \
  --checkpoint /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_27/30000 \
  --rtc --port 8000

# C 推理
source /opt/ros/humble/setup.bash
source /media/lenovo/data/kian/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash
conda activate kai0_inference
cd /media/lenovo/data/kian/kai0/train_deploy_alignment/inference/arx/inference
python arx_openpi_inference_rtc.py \
  --host 127.0.0.1 --port 8000 \
  --camera_backend hybrid --camera_reorder 0,1 \
  --rtc_mode \
  --gripper_mode scale_kai0 \
  --gripper_feedback_open 4.61
```

操作：Enter（warmup）→ C → S → Enter（回 init）→ C … → **Q**（回 zero 退出）→ 停 launch。
