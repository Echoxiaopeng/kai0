#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 dual-arm inference with OpenPi RTC (real-time chunking) and temporal smoothing.

- RTC: inference thread sends prev_action_chunk and inference_delay to the policy server; the server
  uses Pi0RTCConfig (pi0_rtc / pi05_rtc) for chunk alignment under latency.
- Temporal smoothing: StreamActionBuffer blends overlapping chunks at boundaries.

Uses ROS2 and ARX5 messages (arx5_arm_msg). Start the policy server with an RTC config, then run
this script with --rtc_mode. See train_deploy_alignment/inference/arx/README.md for setup.

Cameras default to the umi-arx hybrid stack (left/right GoPro + middle D435). Pass
--camera_backend realsense3 for the old three-RealSense path.
"""

import argparse
import time
import threading
import json
import numpy as np
import cv2
import os
import select
import signal
import sys
import termios
import tty
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import parse_qs, urlparse
from sensor_msgs.msg import JointState

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("Warning: pyrealsense2 not installed; realsense3 backend unavailable.")

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

# ARX5 custom messages
try:
    from arx5_arm_msg.msg import RobotStatus, RobotCmd
    print("Loaded arx5_arm_msg (RobotStatus, RobotCmd)")
except ImportError:
    print("arx5_arm_msg not found; ensure ARX5 message package is installed.")
    from sensor_msgs.msg import JointState
    RobotStatus = JointState
    RobotCmd = JointState
    print("Using JointState as fallback for RobotCmd.")


from openpi_client import image_tools, websocket_client_policy

# Globals
CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
stream_buffer = None
observation_window = deque(maxlen=2)
# Must match training / serve_kai0_policy default for flatten-fold ckpts.
lang_embeddings = "Flatten and fold the cloth."
policy_infer_lock = threading.Lock()
camera_frame_lock = threading.Lock()


def build_kai0_observation(latest_obs, config, prompt: str, **extra) -> dict:
    """Build the flat observation contract expected by serve_kai0_policy.

    Same keys as umi-arx eval_arx5_kai0_bimanual.policy_observation:
      observation.images.{top_head,hand_left,hand_right}, observation.state, prompt
    Images are RGB uint8 HWC.
    """
    imgs = [
        latest_obs["images"][config["camera_names"][0]],  # cam_high -> top_head
        latest_obs["images"][config["camera_names"][1]],  # cam_right_wrist
        latest_obs["images"][config["camera_names"][2]],  # cam_left_wrist
    ]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
    payload = {
        "observation.images.top_head": np.asarray(imgs[0], dtype=np.uint8),
        "observation.images.hand_right": np.asarray(imgs[1], dtype=np.uint8),
        "observation.images.hand_left": np.asarray(imgs[2], dtype=np.uint8),
        "observation.state": np.asarray(latest_obs["qpos"], dtype=np.float32),
        "prompt": prompt,
    }
    payload.update(extra)
    return payload


def save_camera_preview(ros_operator, out_dir: str, *, show: bool = False) -> Optional[str]:
    """Dump current camera frames for visual verification of left/right/center mapping.

    Writes BGR originals plus a labeled mosaic under ``out_dir``.
    """
    frame = ros_operator.get_frame()
    if frame is None:
        print("[camera_preview] get_frame() returned None; cannot dump.")
        return None
    imgs, _, _ = frame
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Logical mapping used by build_kai0_observation / Kai0 server.
    mapping = [
        ("01_hand_left_cam_left_wrist", "cam_left_wrist", "hand_left"),
        ("02_top_head_cam_high", "cam_high", "top_head"),
        ("03_hand_right_cam_right_wrist", "cam_right_wrist", "hand_right"),
    ]
    tiles = []
    for filename, key, policy_key in mapping:
        bgr = imgs.get(key)
        if bgr is None:
            print(f"[camera_preview] missing {key}")
            continue
        bgr = np.asarray(bgr)
        path = out / f"{filename}.jpg"
        cv2.imwrite(str(path), bgr)
        print(f"[camera_preview] wrote {path}  shape={bgr.shape} -> policy '{policy_key}'")
        labeled = bgr.copy()
        cv2.putText(
            labeled,
            f"{policy_key} ({key})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(labeled, (640, 360)))

    if tiles:
        mosaic = np.concatenate(tiles, axis=1)
        mosaic_path = out / "mosaic_left_center_right.jpg"
        cv2.imwrite(str(mosaic_path), mosaic)
        print(f"[camera_preview] mosaic: {mosaic_path}")
        print(
            "[camera_preview] Check: LEFT tile=left wrist GoPro, "
            "CENTER=D435, RIGHT=right wrist GoPro. "
            "If left/right swapped, use --camera_reorder 1,0"
        )
        if show and os.environ.get("DISPLAY"):
            cv2.imshow("camera_preview L | C | R", mosaic)
            # Non-blocking: focus the image window and press a key, or wait ~3s.
            # (Keypresses in the terminal do NOT count for OpenCV waitKey.)
            print(
                "[camera_preview] click the image window and press any key "
                "(or wait 3s) to continue..."
            )
            t_end = time.time() + 3.0
            while time.time() < t_end:
                if cv2.waitKey(50) >= 0:
                    break
            cv2.destroyAllWindows()
        return str(out)
    return None


def build_camera_preview_mosaic(
    ros_operator,
    *,
    tile_size: Tuple[int, int] = (640, 360),
) -> Optional[np.ndarray]:
    """Build labeled L|C|R mosaic from latest camera frames (BGR).

    Does not require joint readiness — cameras only — so the web page works
    even while waiting for arms.
    """
    with camera_frame_lock:
        imgs = ros_operator.get_camera_images()
    if not imgs:
        return None

    mapping = [
        ("cam_left_wrist", "hand_left"),
        ("cam_high", "top_head"),
        ("cam_right_wrist", "hand_right"),
    ]
    tiles = []
    for key, policy_key in mapping:
        bgr = imgs.get(key)
        if bgr is None:
            tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"MISSING {key}",
                (20, tile_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            labeled = np.asarray(bgr).copy()
            cv2.putText(
                labeled,
                f"{policy_key} ({key})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            tile = cv2.resize(labeled, tile_size)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


_camera_web_preview = None  # CameraWebPreview | None
_web_status_lock = threading.Lock()
_web_status: Dict[str, Any] = {
    "phase": "booting",
    "message": "Waiting for sensors / control loop...",
}
_ctrl_cmd_lock = threading.Lock()
_ctrl_cmd_queue: deque = deque()
_VALID_CTRL_CMDS = frozenset({"c", "s", "home", "q"})


def push_ctrl_cmd(cmd: str, source: str = "web") -> bool:
    """Queue a control command for the main loop (c/s/home/q)."""
    cmd = str(cmd).strip().lower()
    if cmd in ("enter", "h", "i", "init"):
        cmd = "home"
    if cmd not in _VALID_CTRL_CMDS:
        return False
    with _ctrl_cmd_lock:
        _ctrl_cmd_queue.append({"cmd": cmd, "source": source, "t": time.time()})
    return True


def drain_ctrl_cmds() -> List[Dict[str, Any]]:
    with _ctrl_cmd_lock:
        out = list(_ctrl_cmd_queue)
        _ctrl_cmd_queue.clear()
    return out


def update_web_status(**kwargs) -> None:
    """Merge fields into the JSON served at /api/status."""
    with _web_status_lock:
        _web_status.update(kwargs)
        _web_status["updated_at"] = time.time()


def get_web_status() -> Dict[str, Any]:
    with _web_status_lock:
        return dict(_web_status)


def _camera_web_index_html() -> bytes:
    """Single-page control + live camera UI."""
    return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ARX Live Control</title>
<style>
:root{
  --bg:#0e1418; --panel:#172027; --line:#2a3a44; --text:#e7eef2; --muted:#8aa0ab;
  --ok:#3dba7a; --warn:#e0a43a; --bad:#e05a4a; --accent:#3aa7c9; --btn:#24333c;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#1b2a33 0%,var(--bg) 55%);
  color:var(--text);font:14px/1.45 "IBM Plex Sans","Segoe UI",sans-serif;min-height:100vh}
.wrap{max-width:1280px;margin:0 auto;padding:16px 18px 28px}
.top{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:flex-end;justify-content:space-between;margin-bottom:14px}
.brand{font-size:22px;font-weight:700;letter-spacing:.02em}
.brand span{color:var(--accent);font-weight:600}
.sub{color:var(--muted);font-size:12px;margin-top:2px}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;
  background:var(--panel);border:1px solid var(--line);font-weight:600;font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);box-shadow:0 0 0 3px rgba(138,160,171,.15)}
.badge.run .dot{background:var(--ok);box-shadow:0 0 0 3px rgba(61,186,122,.2)}
.badge.idle .dot{background:var(--warn);box-shadow:0 0 0 3px rgba(224,164,58,.2)}
.badge.dead .dot{background:var(--bad);box-shadow:0 0 0 3px rgba(224,90,74,.2)}
.grid{display:grid;grid-template-columns:1.7fr 1fr;gap:14px}
@media (max-width:980px){.grid{grid-template-columns:1fr}}
.panel{background:linear-gradient(180deg,#1a262e 0%,var(--panel) 100%);border:1px solid var(--line);
  border-radius:12px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.panel h2{margin:0;padding:10px 14px;font-size:13px;font-weight:600;color:var(--muted);
  border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.06em}
.video-wrap{background:#000;position:relative;min-height:220px}
.video-wrap img{display:block;width:100%;height:auto;background:#000}
.hint{padding:8px 14px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 14px}
button.ctrl{appearance:none;border:1px solid var(--line);background:var(--btn);color:var(--text);
  border-radius:10px;padding:12px 10px;font:inherit;font-weight:650;cursor:pointer;
  transition:transform .08s ease, background .15s ease, border-color .15s}
button.ctrl:hover{border-color:var(--accent);background:#2b3e49}
button.ctrl:active{transform:scale(.98)}
button.ctrl kbd{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:5px;
  border:1px solid #3b4e59;background:#152028;color:var(--muted);font-size:11px;font-weight:600}
button.ctrl.go{border-color:#2f6b4d;background:#1d3a2d}
button.ctrl.stop{border-color:#7a3a34;background:#3a221f}
button.ctrl.quit{grid-column:1/-1;border-color:#5a3a34;opacity:.95}
.stats{padding:4px 14px 14px;display:grid;gap:8px}
.row{display:grid;grid-template-columns:110px 1fr;gap:8px;align-items:start;padding:7px 0;
  border-bottom:1px dashed rgba(42,58,68,.9)}
.row:last-child{border-bottom:0}
.k{color:var(--muted);font-size:12px}
.v{font-variant-numeric:tabular-nums;word-break:break-word}
.v.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:12.5px}
.cams{display:flex;flex-wrap:wrap;gap:6px}
.pill{padding:3px 8px;border-radius:999px;border:1px solid var(--line);font-size:11px;color:var(--muted)}
.pill.ok{color:#b7f0d0;border-color:#2f6b4d;background:#163126}
.pill.bad{color:#ffc9c2;border-color:#7a3a34;background:#3a221f}
.log{max-height:160px;overflow:auto;padding:0 14px 12px;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-size:11.5px;color:#b7c6ce}
.log div{padding:3px 0;border-bottom:1px solid rgba(42,58,68,.5)}
.toast{position:fixed;right:16px;bottom:16px;background:#203039;border:1px solid var(--line);
  padding:10px 14px;border-radius:10px;opacity:0;transform:translateY(8px);transition:.2s;pointer-events:none}
.toast.show{opacity:1;transform:none}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="brand">ARX <span>Live</span> · cameras + control</div>
      <div class="sub">网页按键与终端相同：C 开始 · S 停止 · Enter/H 回 init · Q 回零退出</div>
    </div>
    <div id="badge" class="badge idle"><span class="dot"></span><span id="badgeText">IDLE</span></div>
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Cameras · left wrist | D435 top | right wrist</h2>
      <div class="video-wrap"><img id="stream" src="/stream.mjpg" alt="camera mosaic"/></div>
      <div class="hint" id="camHint">MJPEG 实时流。画面黑/卡顿请查 GoPro / Elgato / D435。</div>
    </section>
    <section class="panel">
      <h2>Controls</h2>
      <div class="controls">
        <button class="ctrl go" data-cmd="c">Start episode <kbd>C</kbd></button>
        <button class="ctrl stop" data-cmd="s">Stop / hold <kbd>S</kbd></button>
        <button class="ctrl" data-cmd="home">Go init pose <kbd>Enter</kbd></button>
        <button class="ctrl quit" data-cmd="q">Zero then quit <kbd>Q</kbd></button>
      </div>
      <h2>Status</h2>
      <div class="stats" id="stats"></div>
      <h2>Event log</h2>
      <div class="log" id="log"></div>
    </section>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const statsEl = document.getElementById('stats');
const logEl = document.getElementById('log');
const badge = document.getElementById('badge');
const badgeText = document.getElementById('badgeText');
const toast = document.getElementById('toast');
const camHint = document.getElementById('camHint');
let toastTimer = null;

function row(k,v,mono=false){
  return `<div class="row"><div class="k">${k}</div><div class="v${mono?' mono':''}">${v}</div></div>`;
}
function fmt(x, d=3){
  if (x===null||x===undefined||Number.isNaN(x)) return '—';
  if (typeof x==='number') return x.toFixed(d);
  return String(x);
}
function fmtArr(a, d=3){
  if (!Array.isArray(a) || !a.length) return '—';
  return a.map(v => fmt(Number(v), d)).join(', ');
}
function addLog(msg){
  const t = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.textContent = `[${t}] ${msg}`;
  logEl.prepend(div);
  while (logEl.childElementCount > 40) logEl.removeChild(logEl.lastChild);
}
function showToast(msg){
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>toast.classList.remove('show'), 1800);
}
async function sendCmd(cmd){
  try{
    const r = await fetch('/api/cmd', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cmd})
    });
    const j = await r.json();
    const msg = j.ok ? `sent ${cmd} (${j.source||'web'})` : (`failed: ${j.error||r.status}`);
    addLog(msg); showToast(msg);
  }catch(e){ addLog('cmd error: '+e); }
}
document.querySelectorAll('button.ctrl').forEach(btn=>{
  btn.addEventListener('click', ()=>sendCmd(btn.dataset.cmd));
});
window.addEventListener('keydown', (ev)=>{
  if (ev.target && /input|textarea|select/i.test(ev.target.tagName)) return;
  const k = ev.key;
  if (k==='c' || k==='C'){ ev.preventDefault(); sendCmd('c'); }
  else if (k==='s' || k==='S'){ ev.preventDefault(); sendCmd('s'); }
  else if (k==='Enter' || k==='h' || k==='H'){ ev.preventDefault(); sendCmd('home'); }
  else if (k==='q' || k==='Q'){ ev.preventDefault(); sendCmd('q'); }
});

function render(s){
  const running = !!s.episode_running;
  const phase = s.phase || (running ? 'running' : 'idle');
  badge.className = 'badge ' + (phase==='dead'||phase==='quit' ? 'dead' : (running ? 'run' : 'idle'));
  badgeText.textContent = (running ? 'RUNNING' : String(phase).toUpperCase())
    + (s.episode_id ? ` · ep#${s.episode_id}` : '');

  const cams = s.cameras || {};
  const pills = ['cam_left_wrist','cam_high','cam_right_wrist'].map(name=>{
    const c = cams[name] || {};
    const ok = !!c.ok;
    const shape = c.shape ? c.shape.join('×') : 'missing';
    return `<span class="pill ${ok?'ok':'bad'}">${name}: ${shape}</span>`;
  }).join('');

  const age = s.frame_age_s;
  camHint.textContent = (age==null)
    ? '等待首帧…'
    : `帧龄 ${fmt(age,2)}s · encode ~${fmt(s.encode_fps,1)} FPS · JPEG q=${s.jpeg_quality||'—'}`;

  statsEl.innerHTML = [
    row('Message', s.message || '—'),
    row('Policy', `${s.policy_host||'—'}:${s.policy_port||'—'}  rtc=${s.rtc_mode?'on':'off'}`, true),
    row('Prompt', s.prompt || '—'),
    row('Step / buf', `step=${s.step??'—'}  buf=${s.buffer_len??'—'}  empty=${s.empty_actions??'—'}` , true),
    row('Gripper mode', s.gripper_mode || '—'),
    row('Grip state m', `L ${fmt(s.grip_state_L,4)} · R ${fmt(s.grip_state_R,4)}`, true),
    row('Grip raw', `L ${fmt(s.grip_raw_L,2)} · R ${fmt(s.grip_raw_R,2)}`, true),
    row('Last cmd', `L ${fmt(s.grip_cmd_L,2)} · R ${fmt(s.grip_cmd_R,2)}`, true),
    row('qpos L', fmtArr(s.qpos_left, 3), true),
    row('qpos R', fmtArr(s.qpos_right, 3), true),
    row('Cameras', `<div class="cams">${pills}</div>`),
    row('Backend', s.camera_backend || '—', true),
    row('Hold paused', String(!!s.hold_paused)),
  ].join('');
}

async function poll(){
  try{
    const r = await fetch('/api/status', {cache:'no-store'});
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  }catch(e){
    badge.className = 'badge dead';
    badgeText.textContent = 'OFFLINE';
  }
}
poll(); setInterval(poll, 400);
addLog('UI ready — click buttons or use keyboard');
</script>
</body>
</html>""".encode("utf-8")


class CameraWebPreview:
    """Background MJPEG + status/control HTTP UI for live camera verification."""

    def __init__(
        self,
        ros_operator,
        host: str = "0.0.0.0",
        port: int = 8080,
        fps: float = 10.0,
        jpeg_quality: int = 70,
        tile_width: int = 480,
        args=None,
    ):
        self.ros_operator = ros_operator
        self.args = args
        self.host = host
        self.port = int(port)
        self.fps = max(float(fps), 1.0)
        self.jpeg_quality = int(np.clip(jpeg_quality, 30, 95))
        self.tile_width = max(int(tile_width), 160)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._encoder_thread: Optional[threading.Thread] = None
        self._cam_meta_lock = threading.Lock()
        self._cam_meta: Dict[str, Any] = {}
        self._last_frame_t: Optional[float] = None
        self._encode_fps = 0.0
        update_web_status(
            phase="cameras",
            message="Camera web preview started",
            policy_host=getattr(args, "host", None) if args is not None else None,
            policy_port=getattr(args, "port", None) if args is not None else None,
            rtc_mode=bool(getattr(args, "rtc_mode", False)) if args is not None else False,
            gripper_mode=getattr(args, "gripper_mode", None) if args is not None else None,
            prompt=lang_embeddings,
            camera_backend=getattr(args, "camera_backend", None) if args is not None else None,
            jpeg_quality=self.jpeg_quality,
            web_fps_target=self.fps,
        )

    def start(self) -> "CameraWebPreview":
        if self._thread is not None:
            return self

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def _send_json(self, code: int, payload: dict):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle_cmd(self, cmd: str, source: str = "web"):
                ok = push_ctrl_cmd(cmd, source=source)
                if ok:
                    print(f"[camera_web] cmd '{cmd}' from {source}")
                    self._send_json(200, {"ok": True, "cmd": cmd, "source": source})
                else:
                    self._send_json(
                        400,
                        {"ok": False, "error": f"invalid cmd '{cmd}'", "allowed": sorted(_VALID_CTRL_CMDS)},
                    )

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self):
                path = urlparse(self.path).path
                if path != "/api/cmd":
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                cmd = ""
                try:
                    data = json.loads(raw.decode("utf-8") or "{}")
                    cmd = str(data.get("cmd", ""))
                except Exception:
                    self._send_json(400, {"ok": False, "error": "invalid JSON"})
                    return
                self._handle_cmd(cmd, source="web")

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path in ("/", "/index.html"):
                    html = _camera_web_index_html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                if path == "/api/status":
                    st = get_web_status()
                    with owner._cam_meta_lock:
                        st["cameras"] = dict(owner._cam_meta)
                        st["frame_age_s"] = (
                            None
                            if owner._last_frame_t is None
                            else max(0.0, time.time() - owner._last_frame_t)
                        )
                        st["encode_fps"] = owner._encode_fps
                    st["episode_running"] = episode_running.is_set()
                    st["hold_paused"] = hold_paused.is_set()
                    st["shutdown"] = shutdown_event.is_set()
                    self._send_json(200, st)
                    return

                if path == "/api/cmd":
                    qs = parse_qs(parsed.query)
                    cmd = (qs.get("cmd") or qs.get("c") or [""])[0]
                    self._handle_cmd(cmd, source="web-get")
                    return

                if path in ("/stream.mjpg", "/stream", "/video"):
                    boundary = b"frame"
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        f"multipart/x-mixed-replace; boundary={boundary.decode()}",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    interval = 1.0 / owner.fps
                    try:
                        while not owner._stop.is_set() and not shutdown_event.is_set():
                            jpeg = owner._get_latest_jpeg()
                            if jpeg is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(b"--" + boundary + b"\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            time.sleep(interval)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    return

                if path == "/snapshot.jpg":
                    jpeg = owner._get_latest_jpeg()
                    if jpeg is None:
                        self.send_error(503, "No camera frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return

                self.send_error(404, "Not found")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera_web_http",
            daemon=True,
        )
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="camera_web_encode",
            daemon=True,
        )
        self._thread.start()
        self._encoder_thread.start()
        print(
            f"[camera_web] UI: http://127.0.0.1:{self.port}/  "
            f"(C/S/Enter/Q from browser)  fps~{self.fps:.0f}"
        )
        return self

    def _encode_loop(self):
        interval = 1.0 / self.fps
        tile_h = int(round(self.tile_width * 360 / 640))
        while not self._stop.is_set():
            t0 = time.time()
            try:
                with camera_frame_lock:
                    imgs = self.ros_operator.get_camera_images()
                meta = {}
                for key in ("cam_left_wrist", "cam_high", "cam_right_wrist"):
                    bgr = None if not imgs else imgs.get(key)
                    if bgr is None:
                        meta[key] = {"ok": False, "shape": None}
                    else:
                        arr = np.asarray(bgr)
                        meta[key] = {"ok": True, "shape": list(arr.shape)}
                with self._cam_meta_lock:
                    self._cam_meta = meta

                mosaic = None
                if imgs:
                    mapping = [
                        ("cam_left_wrist", "hand_left"),
                        ("cam_high", "top_head"),
                        ("cam_right_wrist", "hand_right"),
                    ]
                    tiles = []
                    for key, policy_key in mapping:
                        bgr = imgs.get(key)
                        if bgr is None:
                            tile = np.zeros((tile_h, self.tile_width, 3), dtype=np.uint8)
                            cv2.putText(
                                tile, f"MISSING {key}", (16, tile_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                            )
                        else:
                            labeled = np.asarray(bgr).copy()
                            cv2.putText(
                                labeled, f"{policy_key} ({key})", (16, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
                            )
                            tile = cv2.resize(labeled, (self.tile_width, tile_h))
                        tiles.append(tile)
                    mosaic = np.concatenate(tiles, axis=1)

                if mosaic is not None:
                    ok, buf = cv2.imencode(
                        ".jpg",
                        mosaic,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if ok:
                        with self._jpeg_lock:
                            self._latest_jpeg = buf.tobytes()
                        self._last_frame_t = time.time()
            except Exception as e:
                if int(time.time() * 2) % 20 == 0:
                    print(f"[camera_web] encode error: {e}")
            dt = time.time() - t0
            if dt > 1e-6:
                inst = 1.0 / dt
                self._encode_fps = 0.8 * self._encode_fps + 0.2 * inst if self._encode_fps else inst
            time.sleep(max(0.0, interval - dt))

    def _get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def stop(self):
        self._stop.set()
        update_web_status(phase="quit", message="Preview stopped")
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        self._server = None
        self._thread = None
        self._encoder_thread = None
        print("[camera_web] preview stopped.")


def start_camera_web_preview(ros_operator, args) -> Optional[CameraWebPreview]:
    """Start browser MJPEG preview unless disabled via CLI."""
    global _camera_web_preview
    if getattr(args, "no_web_camera_preview", False):
        return None
    try:
        preview = CameraWebPreview(
            ros_operator,
            host=str(getattr(args, "web_camera_host", "0.0.0.0")),
            port=int(getattr(args, "web_camera_port", 8080)),
            fps=float(getattr(args, "web_camera_fps", 10.0)),
            jpeg_quality=int(getattr(args, "web_camera_jpeg_quality", 70)),
            tile_width=int(getattr(args, "web_camera_tile_width", 480)),
            args=args,
        )
        preview.start()
        _camera_web_preview = preview
        return preview
    except OSError as e:
        print(f"[camera_web] failed to bind preview port: {e}")
        return None


rtc_prev_chunk_lock = threading.Lock()
rtc_prev_chunk = None
delay_buffer = deque(maxlen=20)
pred_delay_steps = 0

published_actions_history = []
observed_qpos_history = []
inferred_chunks = []
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()
episode_running = threading.Event()  # set by 'c', cleared by 's'
hold_paused = threading.Event()  # set while smooth moves so hold thread doesn't fight
_idle_hold_thread = None
gripper_debug_logger = None  # set in model_inference when --gripper_debug


class TerminalKeys:
    """Non-blocking single-key reader (cbreak). Same idea as umi-arx eval."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("TTY required for C/S keyboard controls")
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self) -> List[str]:
        keys: List[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == "\x03":
                raise KeyboardInterrupt
            if char:
                keys.append(char.lower())
        return keys


class GripperDebugLogger:
    """Persist model gripper vs mapped ROS cmd vs robot feedback for offline check."""

    STEP_HEADER = (
        "step,t_s,mode,"
        "model_L_m,model_R_m,"
        "scale_cmd_L,scale_cmd_R,"
        "snap_cmd_L,snap_cmd_R,"
        "actual_cmd_L,actual_cmd_R,"
        "setCatch_L,setCatch_R,"
        "state_m_L,state_m_R,"
        "raw_L,raw_R,"
        "model_pct_L,model_pct_R,"
        "cmd_pct_L,cmd_pct_R,"
        "snap_decision_L,snap_decision_R,"
        "err_state_minus_model_L,err_state_minus_model_R,"
        "err_raw_minus_cmd_L,err_raw_minus_cmd_R"
    )
    CHUNK_HEADER = (
        "t_s,rtt_s,delay_steps,buf,"
        "chunk_T,"
        "model_L_min,model_L_mean,model_L_max,"
        "model_R_min,model_R_mean,model_R_max"
    )

    def __init__(self, log_dir: Path, args):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.t0 = time.time()
        self._lock = threading.Lock()
        self.n_steps = 0
        self.n_chunks = 0
        self.step_path = self.log_dir / "gripper_steps.csv"
        self.chunk_path = self.log_dir / "gripper_chunks.csv"
        self.console_path = self.log_dir / "console_grip.log"
        self.meta_path = self.log_dir / "meta.json"
        self.summary_path = self.log_dir / "summary.txt"

        kai0_min = float(getattr(args, "kai0_gripper_min", 0.0))
        kai0_max = float(getattr(args, "kai0_gripper_max", 0.085))
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gripper_mode": getattr(args, "gripper_mode", None),
            "gripper_open": float(getattr(args, "gripper_open", 5.0)),
            "gripper_close": float(getattr(args, "gripper_close", -0.8)),
            "gripper_snap_thresh": float(getattr(args, "gripper_snap_thresh", 0.042)),
            "kai0_gripper_min": kai0_min,
            "kai0_gripper_max": kai0_max,
            "gripper_feedback_close": float(getattr(args, "gripper_feedback_close", 0.0)),
            "gripper_feedback_open": float(getattr(args, "gripper_feedback_open", 5.0)),
            "note": (
                "model_*_m = policy action gripper (Kai0 meters). "
                "scale_cmd = linear map to ROS publish units. "
                "snap_cmd = binary open/close. "
                "actual_cmd = what we publish (joint_pos[6]). "
                "setCatch = actual_cmd * 5 (ARX remote_slave arm_end_type=0). "
                "raw = RobotStatus joint_pos[6] feedback. "
                "state_m = raw mapped back to Kai0 meters for policy state."
            ),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with open(self.step_path, "w", encoding="utf-8") as f:
            f.write(self.STEP_HEADER + "\n")
        with open(self.chunk_path, "w", encoding="utf-8") as f:
            f.write(self.CHUNK_HEADER + "\n")
        with open(self.console_path, "w", encoding="utf-8") as f:
            f.write(f"# gripper debug started {meta['created']}\n")
            f.write(json.dumps(meta) + "\n")
        print(f"[gripper_debug] logging to {self.log_dir}")

    def _map_scale(self, g: float) -> float:
        curve = getattr(self.args, "gripper_scale_curve", "linear")
        if curve == "hw":
            return scale_gripper_kai0_to_ros_hw(
                g,
                kai0_min=float(self.args.kai0_gripper_min),
                kai0_max=float(self.args.kai0_gripper_max),
                fb_close=float(getattr(self.args, "gripper_feedback_close", 0.0)),
                fb_open=float(getattr(self.args, "gripper_feedback_open", 4.61)),
                ros_close=float(self.args.gripper_close),
                ros_open=float(self.args.gripper_open),
            )
        return scale_gripper_kai0_to_ros(
            g,
            kai0_min=float(self.args.kai0_gripper_min),
            kai0_max=float(self.args.kai0_gripper_max),
            ros_close=float(self.args.gripper_close),
            ros_open=float(self.args.gripper_open),
            gamma=float(getattr(self.args, "gripper_gamma", 1.0)),
        )

    def _map_snap(self, g: float) -> float:
        thr = float(self.args.gripper_snap_thresh)
        return float(self.args.gripper_open) if g >= thr else float(self.args.gripper_close)

    def _pct_model(self, g: float) -> float:
        span = max(float(self.args.kai0_gripper_max) - float(self.args.kai0_gripper_min), 1e-6)
        return (float(g) - float(self.args.kai0_gripper_min)) / span

    def _pct_cmd(self, cmd: float) -> float:
        lo = float(self.args.gripper_close)
        hi = float(self.args.gripper_open)
        span = max(hi - lo, 1e-6)
        return (float(cmd) - lo) / span

    def log_console(self, line: str):
        with self._lock:
            with open(self.console_path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")

    def log_step(
        self,
        step: int,
        model_L: float,
        model_R: float,
        actual_L: float,
        actual_R: float,
        state_L: float,
        state_R: float,
        raw_L: Optional[float],
        raw_R: Optional[float],
    ):
        t_s = time.time() - self.t0
        scale_L, scale_R = self._map_scale(model_L), self._map_scale(model_R)
        snap_L, snap_R = self._map_snap(model_L), self._map_snap(model_R)
        thr = float(self.args.gripper_snap_thresh)
        row = (
            f"{step},{t_s:.4f},{self.args.gripper_mode},"
            f"{model_L:.6f},{model_R:.6f},"
            f"{scale_L:.6f},{scale_R:.6f},"
            f"{snap_L:.6f},{snap_R:.6f},"
            f"{actual_L:.6f},{actual_R:.6f},"
            f"{actual_L * 5.0:.6f},{actual_R * 5.0:.6f},"
            f"{state_L:.6f},{state_R:.6f},"
            f"{'' if raw_L is None else f'{raw_L:.6f}'},"
            f"{'' if raw_R is None else f'{raw_R:.6f}'},"
            f"{self._pct_model(model_L):.4f},{self._pct_model(model_R):.4f},"
            f"{self._pct_cmd(actual_L):.4f},{self._pct_cmd(actual_R):.4f},"
            f"{'open' if model_L >= thr else 'close'},"
            f"{'open' if model_R >= thr else 'close'},"
            f"{state_L - model_L:.6f},{state_R - model_R:.6f},"
            f"{(0.0 if raw_L is None else raw_L) - actual_L:.6f},"
            f"{(0.0 if raw_R is None else raw_R) - actual_R:.6f}"
        )
        with self._lock:
            with open(self.step_path, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            self.n_steps += 1

    def log_chunk(self, actions_np: np.ndarray, rtt: float, delay_steps: int, buf: int):
        a = np.asarray(actions_np, dtype=float)
        if a.ndim != 2 or a.shape[1] < 14:
            return
        t_s = time.time() - self.t0
        L, R = a[:, 6], a[:, 13]
        row = (
            f"{t_s:.4f},{rtt:.4f},{delay_steps},{buf},{a.shape[0]},"
            f"{L.min():.6f},{L.mean():.6f},{L.max():.6f},"
            f"{R.min():.6f},{R.mean():.6f},{R.max():.6f}"
        )
        line = (
            f"[rtc_grip] L[m]=[{L.min():.4f},{L.mean():.4f},{L.max():.4f}] "
            f"R[m]=[{R.min():.4f},{R.mean():.4f},{R.max():.4f}] "
            f"→scale L[{self._map_scale(L.min()):.2f},{self._map_scale(L.max()):.2f}] "
            f"R[{self._map_scale(R.min()):.2f},{self._map_scale(R.max()):.2f}]"
        )
        with self._lock:
            with open(self.chunk_path, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            with open(self.console_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.n_chunks += 1
        # Keep terminal quieter: print every 5th chunk.
        if self.n_chunks % 5 == 1:
            print(line)

    def finalize(self):
        """Write summary stats from the step CSV."""
        try:
            import csv

            if not self.step_path.exists() or self.n_steps == 0:
                self.summary_path.write_text("no steps logged\n", encoding="utf-8")
                return
            with open(self.step_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return

            def col(name):
                return np.array([float(r[name]) for r in rows], dtype=float)

            model_L, model_R = col("model_L_m"), col("model_R_m")
            cmd_L, cmd_R = col("actual_cmd_L"), col("actual_cmd_R")
            state_L, state_R = col("state_m_L"), col("state_m_R")
            scale_L, snap_L = col("scale_cmd_L"), col("snap_cmd_L")
            lines = [
                f"steps={len(rows)} chunks={self.n_chunks}",
                f"mode={self.args.gripper_mode}",
                f"model_L_m  min/mean/max = {model_L.min():.4f}/{model_L.mean():.4f}/{model_L.max():.4f}",
                f"model_R_m  min/mean/max = {model_R.min():.4f}/{model_R.mean():.4f}/{model_R.max():.4f}",
                f"actual_cmd_L min/mean/max = {cmd_L.min():.3f}/{cmd_L.mean():.3f}/{cmd_L.max():.3f}",
                f"actual_cmd_R min/mean/max = {cmd_R.min():.3f}/{cmd_R.mean():.3f}/{cmd_R.max():.3f}",
                f"state_m_L  min/mean/max = {state_L.min():.4f}/{state_L.mean():.4f}/{state_L.max():.4f}",
                f"state_m_R  min/mean/max = {state_R.min():.4f}/{state_R.mean():.4f}/{state_R.max():.4f}",
                f"mean |state-model| L/R = {np.mean(np.abs(state_L-model_L)):.4f}/{np.mean(np.abs(state_R-model_R)):.4f}",
                f"mean |actual-scale| L (0 if scale mode) = {np.mean(np.abs(cmd_L-scale_L)):.4f}",
                f"mean |actual-snap|  L (0 if snap mode)  = {np.mean(np.abs(cmd_L-snap_L)):.4f}",
                f"frac model_L < snap_thresh (want close) = {np.mean(model_L < float(self.args.gripper_snap_thresh)):.3f}",
                f"frac model_R < snap_thresh (want close) = {np.mean(model_R < float(self.args.gripper_snap_thresh)):.3f}",
                f"log_dir={self.log_dir}",
            ]
            text = "\n".join(lines) + "\n"
            self.summary_path.write_text(text, encoding="utf-8")
            print("[gripper_debug] summary:\n" + text)
        except Exception as e:
            print(f"[gripper_debug] finalize failed: {e}")


class StreamActionBuffer:
    """Chunk queue for actions; latency trim and temporal smoothing over overlaps (same design as Agilex)."""
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if str(self.smooth_method).lower() == "raw":
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return
            
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0


def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """Non-blocking inference thread; pushes chunks to stream_buffer with temporal smoothing."""
    global stream_buffer, observation_window, lang_embeddings

    # rate = rclpy.create_rate(args.inference_rate)
    rate = ros_operator.create_rate(args.inference_rate)
    consecutive_failures = 0
    max_consecutive_failures = 5

    while rclpy.ok() and not shutdown_event.is_set():
        try:
            if not episode_running.is_set():
                rate.sleep()
                continue
            time1 = time.time()
            
            update_observation_window(args, config, ros_operator)
            print("Get Observation Time", time.time() - time1, "s")
            
            if len(observation_window) == 0:
                continue
                
            latest_obs = observation_window[-1]
            payload = build_kai0_observation(latest_obs, config, lang_embeddings)

            time1 = time.time()
            with policy_infer_lock:
                actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - time1, "s")

            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
                try:
                    step_now = max(len(published_actions_history), len(observed_qpos_history))
                    with inferred_chunks_lock:
                        inferred_chunks.append({
                            "start_step": int(step_now),
                            "chunk": np.asarray(actions, dtype=float).copy()
                        })
                except Exception:
                    pass
                
                consecutive_failures = 0
            elif actions is None:
                print("actions is None")
            elif len(actions) == 0:
                print("len(actions) == 0")

            rate.sleep()

        except Exception as e:
            print(f"[inference_fn_non_blocking_fast] {e}")
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                print(f"[inference] Consecutive failures {consecutive_failures}, clearing buffer")
                stream_buffer.clear()
                consecutive_failures = 0
            
            try:
                rate.sleep()
            except:
                time.sleep(0.001)


def inference_fn_non_blocking_rtc(args, config, policy, ros_operator):
    """Async RTC inference thread: send prev_chunk and delay; push inferred chunk to stream_buffer."""
    global stream_buffer, observation_window, rtc_prev_chunk, pred_delay_steps
    rate = ros_operator.create_rate(args.inference_rate)
    chunk_size = config["chunk_size"]
    exec_h = chunk_size if getattr(args, "rtc_execute_horizon", None) is None else args.rtc_execute_horizon
    exec_h = max(1, min(exec_h, chunk_size))

    while rclpy.ok() and not shutdown_event.is_set():
        try:
            if not episode_running.is_set():
                rate.sleep()
                continue
            update_observation_window(args, config, ros_operator)
            if len(observation_window) == 0:
                rate.sleep()
                continue

            latest_obs = observation_window[-1]
            with rtc_prev_chunk_lock:
                pc = np.array(rtc_prev_chunk) if rtc_prev_chunk is not None else None
            # Keep T aligned with model action_horizon / config chunk_size.
            ah = int(config.get("chunk_size", chunk_size))
            if pc is not None:
                pc = np.asarray(pc, dtype=np.float32)
                if pc.ndim == 1:
                    pc = np.tile(pc[None, :], (ah, 1))
                elif pc.shape[0] != ah:
                    if pc.shape[0] > ah:
                        pc = pc[:ah]
                    else:
                        pc = np.concatenate(
                            [pc, np.repeat(pc[-1:], ah - pc.shape[0], axis=0)], axis=0
                        )
            rtc_extra = {
                "execute_horizon": min(int(exec_h), ah),
                "enable_rtc": True,
                "mask_prefix_delay": getattr(args, "rtc_mask_prefix_delay", False),
                "max_guidance_weight": getattr(args, "rtc_max_guidance_weight", 0.5),
                "inference_delay": int(max(0, pred_delay_steps)),
            }
            if pc is not None:
                rtc_extra["prev_action_chunk"] = pc
            payload = build_kai0_observation(
                latest_obs, config, lang_embeddings, **rtc_extra
            )

            t0 = time.time()
            with policy_infer_lock:
                out = policy.infer(payload)
            rtt = time.time() - t0
            _update_delay_buffer(rtt, args.control_frequency)

            actions = out.get("actions", None) if isinstance(out, dict) else None
            if actions is None or len(actions) == 0:
                print("[rtc] infer returned empty actions")
                rate.sleep()
                continue

            actions_np = np.asarray(actions, dtype=float)
            with rtc_prev_chunk_lock:
                rtc_prev_chunk = actions_np

            stream_buffer.integrate_new_chunk(
                actions_np,
                max_k=int(getattr(args, "latency_k", 0)),
                min_m=int(getattr(args, "min_smooth_steps", 8)),
            )
            buf_n = len(stream_buffer.cur_chunk)
            print(
                f"[rtc] chunk ok shape={actions_np.shape} "
                f"rtt={rtt:.2f}s delay_steps={pred_delay_steps} "
                f"buf={buf_n}"
            )
            dbg = gripper_debug_logger
            if dbg is not None:
                dbg.log_chunk(actions_np, rtt=rtt, delay_steps=pred_delay_steps, buf=buf_n)
        except Exception as e:
            print(f"[inference_fn_non_blocking_rtc] {e}")
            import traceback
            traceback.print_exc()
        try:
            rate.sleep()
        except Exception:
            time.sleep(0.001)


def _update_delay_buffer(rtt_sec: float, control_rate: float):
    """Record one inference RTT and update predicted delay steps."""
    global pred_delay_steps
    if rtt_sec is None or not np.isfinite(rtt_sec):
        return
    delay_buffer.append(float(rtt_sec))
    if len(delay_buffer) == 0:
        pred_delay_steps = 0
        return
    median_rtt = float(np.median(np.asarray(delay_buffer, dtype=float)))
    pred_delay_steps = int(max(0, round(median_rtt * float(control_rate))))


def start_inference_thread(args, config, policy, ros_operator):
    inference_thread = threading.Thread(
        target=inference_fn_non_blocking_fast, 
        args=(args, config, policy, ros_operator)
    )
    inference_thread.daemon = True
    inference_thread.start()
    return inference_thread


def _on_sigint(signum, frame):
    """SIGINT handler."""
    try:
        shutdown_event.set()
    except Exception:
        pass


def update_observation_window(args, config, ros_operator):
    """Update observation window from ROS/sensors."""
    global observation_window

    if len(observation_window) == 0:
        observation_window.append({
            "qpos": None,
            "images": {
                config["camera_names"][0]: None,
                config["camera_names"][1]: None,
                config["camera_names"][2]: None,
            },
        })

    frame = ros_operator.get_frame()
    if frame is None:
        return
        
    imgs, j_left, j_right = frame
    qpos = ros_operator.get_joint_positions(j_left, j_right)
    
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    img_front = jpeg_mapping(imgs['cam_high'])
    img_left = jpeg_mapping(imgs['cam_left_wrist'])
    img_right = jpeg_mapping(imgs['cam_right_wrist'])

    observation_window.append({
        "qpos": qpos,
        "images": {
            config["camera_names"][0]: img_front,
            config["camera_names"][1]: img_right,
            config["camera_names"][2]: img_left,
        },
    })
    try:
        observed_qpos_history.append(np.asarray(qpos, dtype=float).copy())
    except Exception:
        pass


class ARX5ROSController(Node):
    try:
        from arx5_arm_msg.msg import RobotStatus, RobotCmd
        print("Loaded arx5_arm_msg (RobotStatus, RobotCmd)")
    except ImportError:
        print("arx5_arm_msg not found; install ARX5 message package.")
        from sensor_msgs.msg import JointState
        RobotStatus = JointState
        RobotCmd = JointState

    def __init__(self, args):
        super().__init__("arx5_controller")
        self.args = args
        self.bridge = CvBridge()
        self.last_qpos = None
        self.qpos_lock = threading.Lock()
        self.joint_left_deque = deque(maxlen=2000)
        self.joint_right_deque = deque(maxlen=2000)
        self.pub_left = self.create_publisher(RobotStatus, args.joint_cmd_topic_left, 10)
        self.pub_right = self.create_publisher(RobotStatus, args.joint_cmd_topic_right, 10)
        self.RobotStatus = RobotStatus
        self.create_subscription(
            RobotStatus,
            '/arm_master_l_cmd',
            self.left_arm_command_callback,
            10
        )
        self.create_subscription(
            RobotStatus,
            '/arm_master_r_cmd',
            self.right_arm_command_callback,
            10
        )
        self.get_logger().info(f"Subscribed to left arm: {args.joint_state_topic_left}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_left,
            self.joint_left_callback,
            10
        )

        self.get_logger().info(f"Subscribed to right arm: {args.joint_state_topic_right}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_right,
            self.joint_right_callback,
            10
        )

        self.data_ready = {
            'joint_left': False,
            'joint_right': False,
            'cameras': False
        }
        self.data_ready_lock = threading.Lock()

        self.hybrid_cameras = None
        self.pipelines = {}
        self.init_cameras()

    def joint_left_callback(self, msg):
        """Left arm RobotStatus callback."""
        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_left'] = True

    def joint_right_callback(self, msg):
        """Right arm RobotStatus callback."""
        self.joint_right_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_right'] = True

    def left_arm_command_callback(self, msg):
        """Left arm command: merge with current right arm position and publish."""
        self.set_joint_positions(np.array(msg.joint_pos + self.joint_right_deque[-1].joint_pos))

    def right_arm_command_callback(self, msg):
        """Right arm command: merge with current left arm position and publish."""
        self.set_joint_positions(np.array(self.joint_left_deque[-1].joint_pos + msg.joint_pos))

    def get_raw_joint_positions(self, j_left: RobotStatus, j_right: RobotStatus) -> np.ndarray:
        """14-D joints in ROS publish units (gripper = catch feedback, not Kai0 m)."""
        left = list(j_left.joint_pos)
        right = list(j_right.joint_pos)
        return np.array(left + right, dtype=float)

    def get_joint_positions(self, j_left: RobotStatus, j_right: RobotStatus) -> np.ndarray:
        """Get 14-D joint positions from RobotStatus messages.

        Gripper dims are converted from ROS catch units into Kai0/UMI meters
        so the policy state matches training.
        """
        q = self.get_raw_joint_positions(j_left, j_right)
        # Keep raw catch feedback for debugging / scaling.
        self._last_raw_grip = (float(q[6]), float(q[13])) if len(q) >= 14 else (None, None)
        # ROS publishes catch feedback in ~[0, 5]; Kai0 expects ~[0, 0.085].
        # Use feedback bounds (not command over-close like -0.8) for state mapping.
        if len(q) >= 14 and getattr(self.args, "gripper_mode", "snap_kai0") != "raw":
            fb_close = float(getattr(self.args, "gripper_feedback_close", 0.0))
            fb_open = float(getattr(self.args, "gripper_feedback_open", 5.0))
            q[6] = scale_gripper_ros_to_kai0(
                q[6],
                kai0_min=float(getattr(self.args, "kai0_gripper_min", 0.0)),
                kai0_max=float(getattr(self.args, "kai0_gripper_max", 0.085)),
                ros_close=fb_close,
                ros_open=fb_open,
            )
            q[13] = scale_gripper_ros_to_kai0(
                q[13],
                kai0_min=float(getattr(self.args, "kai0_gripper_min", 0.0)),
                kai0_max=float(getattr(self.args, "kai0_gripper_max", 0.085)),
                ros_close=fb_close,
                ros_open=fb_open,
            )
        with self.qpos_lock:
            self.last_qpos = q.copy()
        return q

    def capture_hold_pose(self, pose: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Freeze a hold target. If pose is None, snapshot current measured joints once.

        Important: do NOT re-sample every tick — that follows gravity sag.
        """
        if not hasattr(self, "_hold_lock"):
            self._hold_lock = threading.Lock()
        if pose is None:
            if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
                return None
            pose = self.get_raw_joint_positions(
                self.joint_left_deque[-1], self.joint_right_deque[-1]
            )
        with self._hold_lock:
            self._hold_pose = np.asarray(pose, dtype=float).copy()
            return self._hold_pose.copy()

    def publish_hold_pose(self) -> Optional[np.ndarray]:
        """Re-publish the frozen hold target (ROS units)."""
        if not hasattr(self, "_hold_lock"):
            self._hold_lock = threading.Lock()
        with self._hold_lock:
            if getattr(self, "_hold_pose", None) is None:
                return None
            pose = self._hold_pose.copy()
        self.set_joint_positions(pose)
        return pose

    def hold_current_pose(self) -> Optional[np.ndarray]:
        """Backward-compatible: capture once if needed, then publish frozen hold."""
        if getattr(self, "_hold_pose", None) is None:
            self.capture_hold_pose()
        return self.publish_hold_pose()

    def get_frame(self):
        """Get synchronized sensor data (joints + camera images)."""
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return None
        j_left = self.joint_left_deque[-1]
        j_right = self.joint_right_deque[-1]
        with camera_frame_lock:
            imgs = self.get_camera_images()
        needed = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        if any(k not in imgs or imgs[k] is None for k in needed):
            return None

        return imgs, j_left, j_right

    def init_cameras(self):
        """Initialize cameras (hybrid GoPro+D435 by default)."""
        backend = getattr(self.args, "camera_backend", "hybrid")
        self.hybrid_cameras = None
        self.pipelines = {}
        try:
            if backend == "hybrid":
                from hybrid_gopro_rs_cameras import HybridGoProRSCameras

                reorder = getattr(self.args, "camera_reorder", "0,1")
                if isinstance(reorder, str):
                    reorder = [int(x.strip()) for x in reorder.split(",") if x.strip()]
                print("Initializing hybrid cameras (GoPro left/right + D435 center)...")
                self.hybrid_cameras = HybridGoProRSCameras(
                    camera_reorder=reorder,
                    rs_serial=getattr(self.args, "rs_serial", None) or None,
                    camera_obs_latency=float(
                        getattr(self.args, "camera_obs_latency", 0.17)
                    ),
                    umi_root=getattr(
                        self.args, "umi_root", "/home/lenovo/kian/umi-arx-kian"
                    ),
                    reset_elgato=not getattr(self.args, "no_reset_elgato", False),
                )
                self.hybrid_cameras.start(timeout=30.0)
            elif backend == "realsense3":
                if rs is None:
                    raise ImportError("pyrealsense2 is required for realsense3 backend")
                camera_serials = {
                    "cam_high": self.args.camera_front_serial,
                    "cam_left_wrist": self.args.camera_left_serial,
                    "cam_right_wrist": self.args.camera_right_serial,
                }
                print("Initializing three RealSense cameras...")
                for cam_name, serial in camera_serials.items():
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                    pipeline.start(config)
                    self.pipelines[cam_name] = pipeline
                    print(f"  {cam_name} started")
                for _ in range(30):
                    for pipeline in self.pipelines.values():
                        pipeline.wait_for_frames(timeout_ms=5000)
                print("Cameras warmed up.")
            else:
                raise ValueError(f"Unknown camera_backend={backend}")
            with self.data_ready_lock:
                self.data_ready["cameras"] = True
        except Exception as e:
            print(f"Camera init failed: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_cameras()
            self.hybrid_cameras = None
            self.pipelines = {}

    def get_camera_images(self):
        """Get images from all cameras as BGR uint8."""
        if getattr(self, "hybrid_cameras", None) is not None:
            try:
                return self.hybrid_cameras.get_bgr_images()
            except Exception as e:
                print(f"Failed to get hybrid camera images: {e}")
                return {}

        images = {}
        if not getattr(self, "pipelines", None):
            return images

        for cam_name, pipeline in self.pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    # rs.format.rgb8 -> convert to BGR for legacy BGR2RGB path
                    image = np.asanyarray(color_frame.get_data())
                    images[cam_name] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Failed to get image from {cam_name}: {e}")
        return images

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        """Wait until all sensor data is ready."""
        print("Waiting for sensor data...")
        start_time = time.time()
        while time.time() - start_time < timeout and rclpy.ok():
            with self.data_ready_lock:
                joints_ready = self.data_ready['joint_left'] and self.data_ready['joint_right']
                cameras_ready = self.data_ready['cameras']
            if joints_ready and cameras_ready:
                print("All sensor data ready.")
                return True
            time.sleep(0.5)
        print("Timeout waiting for sensor data.")
        return False

    def set_joint_positions(self, pos: np.ndarray):
        """Publish RobotStatus control commands."""
        if not rclpy.ok():
            return
        if len(pos) != 14:
            self.get_logger().warn(f"Expected 14-D, got {len(pos)}")
            return

        msg_left = RobotStatus()
        msg_right = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos[:7]]
        msg_right.joint_pos = [float(x) for x in pos[7:]]
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)

    def smooth_return_to_zero(self, duration: float = 3.0):
        """Smooth return to zero pose."""
        print("Returning to zero...")
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current joint position")
            return False

        hold_paused.set()
        try:
            _, j_left, j_right = frame
            # Raw ROS units (not Kai0 gripper meters) so cmds stay consistent.
            current_pos = self.get_raw_joint_positions(j_left, j_right)
            target_pos = np.zeros(14)
            target_pos[6] = 3.0
            target_pos[13] = 3.0

            control_hz = 50.0
            num_steps = int(duration * control_hz)

            for step in range(num_steps + 1):
                alpha = step / num_steps
                smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2

                interpolated_pos = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
                self.set_joint_positions(interpolated_pos)

                progress = int(alpha * 100)
                print(f"\rReturn to zero: {progress}%", end='', flush=True)
                time.sleep(1.0/control_hz)
            print("\nReturn to zero done.")
            open_pos = np.zeros(14)
            open_pos[6] = 5.0
            open_pos[13] = 5.0
            self.set_joint_positions(open_pos)
            self.capture_hold_pose(open_pos)
            return True
        finally:
            hold_paused.clear()

    def exit_return_to_zero(self, duration: float = 3.0):
        """Smooth return to zero using cached qpos (no camera)."""
        with self.qpos_lock:
            if self.last_qpos is None:
                print("No cached qpos; skip return to zero.")
                return False
            current_pos = self.last_qpos.copy()

        target_pos = np.zeros(14)
        control_hz = 50.0
        num_steps = int(duration * control_hz)

        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interp = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
            self.set_joint_positions(interp)

            progress = int(alpha * 100)
            print(f"\rReturn to zero: {progress}%", end='', flush=True)
            time.sleep(1.0 / control_hz)
        print("\nReturn to zero done.")
        open_pos = np.zeros(14)
        open_pos[6] = 5.0
        open_pos[13] = 5.0
        self.set_joint_positions(open_pos)
        return True

    def smooth_goto_position(self, target_pos: np.ndarray, duration: float = 3.0, hz: float = 50.0) -> bool:
        """Smooth interpolate to 14-D target position (read current joints from sensors)."""
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current joints; skip smooth move.")
            return False
        # Pause idle hold so it does not fight this trajectory (causes shake).
        hold_paused.set()
        try:
            _, j_left, j_right = frame
            current_pos = self.get_raw_joint_positions(j_left, j_right)
            target = np.asarray(target_pos, dtype=float).copy()
            max_delta = np.max(np.abs(target - current_pos))
            print(f"Max joint delta: {max_delta:.4f} rad")

            num_steps = int(duration * hz)
            for step in range(num_steps + 1):
                alpha = step / num_steps
                smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
                interp = current_pos * (1 - smooth_alpha) + target * smooth_alpha
                self.set_joint_positions(interp)

                if step % 50 == 0 or step == num_steps:
                    progress = int(alpha * 100)
                    print(f"\rSmooth move: {progress}%", end='', flush=True)
                time.sleep(1.0 / hz)
            print("\nSmooth move done.")
            self.capture_hold_pose(target)
            return True
        finally:
            hold_paused.clear()

    def cleanup_cameras(self):
        """Release camera resources."""
        if getattr(self, "hybrid_cameras", None) is not None:
            print("Stopping hybrid GoPro+RS cameras...")
            try:
                self.hybrid_cameras.stop()
            except Exception:
                pass
            self.hybrid_cameras = None
        if getattr(self, "pipelines", None):
            print("Stopping RealSense cameras...")
            for _, pipeline in self.pipelines.items():
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.pipelines = {}



def get_config(args):
    """Build config dict from args."""
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


def start_idle_hold_thread(ros_operator, hz: float = 30.0) -> threading.Thread:
    """While episode is stopped, keep publishing a frozen hold pose (anti-sag).

    Only one thread is started for the process. Smooth moves set ``hold_paused``
    so this thread does not fight the trajectory (that was the Enter shake).
    """
    global _idle_hold_thread
    if _idle_hold_thread is not None and _idle_hold_thread.is_alive():
        return _idle_hold_thread

    def _loop():
        dt = 1.0 / max(float(hz), 1.0)
        while not shutdown_event.is_set():
            if (not episode_running.is_set()) and (not hold_paused.is_set()):
                ros_operator.publish_hold_pose()
            time.sleep(dt)

    t = threading.Thread(target=_loop, name="idle_hold", daemon=True)
    t.start()
    _idle_hold_thread = t
    print(f"[hold] idle hold thread started @ {hz:.1f}Hz (frozen target until C)")
    return t


def model_inference(args, config, ros_operator):
    """Main inference loop: connect to policy server, run RTC or non-RTC inference thread, consume stream_buffer."""
    global stream_buffer, lang_embeddings, rtc_prev_chunk, gripper_debug_logger

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta = policy.get_server_metadata() or {}
    print(f"Server metadata: {meta}")
    server_rtc = bool(meta.get("supports_rtc", False))
    client_rtc = bool(getattr(args, "rtc_mode", False))
    print(
        f"[rtc] client --rtc_mode={client_rtc} | server supports_rtc={server_rtc} "
        f"(config={meta.get('config')})"
    )
    if client_rtc and not server_rtc:
        print(
            "[rtc] WARNING: client has --rtc_mode but server was started WITHOUT --rtc. "
            "prev_action_chunk will be ignored / may error. Restart server with --rtc."
        )
    elif server_rtc and not client_rtc:
        print(
            "[rtc] NOTE: server supports RTC, but client has NO --rtc_mode. "
            "Inference uses the plain path (no prev_action_chunk) → looks the same as non-RTC. "
            "Add --rtc_mode on the client to actually use RTC guidance."
        )
    elif client_rtc and server_rtc:
        print(
            "[rtc] OK: both sides RTC. Expect logs like `[rtc] chunk ok ... delay_steps=N`. "
            "RTC mainly smooths chunk seams under latency; task quality can still look similar."
        )
    else:
        print("[rtc] both sides non-RTC (plain infer + temporal buffer smoothing).")

    if getattr(args, "gripper_debug", True):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        raw_root = (getattr(args, "gripper_log_dir", "") or "").strip()
        log_root = (
            Path(raw_root).expanduser().resolve()
            if raw_root
            else Path(__file__).resolve().parent / "logs"
        )
        gripper_debug_logger = GripperDebugLogger(log_root / f"gripper_{stamp}", args)
    else:
        gripper_debug_logger = None

    max_publish_step = config["episode_len"]


    # left0  = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    # right0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    left0  = [0.024986,0.637255,0.694095,-0.620852,0.026380,-0.052849, 5.0]
    right0 = [0.051690,0.633059,0.692569,-0.632296,0.068475,0.029945, 5.0]
     
    init_pose = np.array(left0 + right0, dtype=float)
    
    frame = ros_operator.get_frame()
    if frame is None:
        print("Cannot get current joints; skip initial move.")
        ros_operator.capture_hold_pose()
    else:
        # hold_paused inside smooth_goto; updates frozen target when done.
        ros_operator.smooth_goto_position(
            target_pos=init_pose,
            duration=3.0,
            hz=50.0
        )
        print("Initial pose move done.")

    # Keep arms servoed at frozen pose through warmup / wait-for-C (singleton thread).
    episode_running.clear()
    start_idle_hold_thread(
        ros_operator, hz=float(getattr(args, "control_frequency", 30.0))
    )

    try:
        print("Warming up inference...")
        update_observation_window(args, config, ros_operator)
        if len(observation_window) == 0:
            print("Observation window empty; skip warmup.")
        else:
            latest_obs = observation_window[-1]
            payload = build_kai0_observation(latest_obs, config, lang_embeddings)
            with policy_infer_lock:
                warm_actions = policy.infer(payload)["actions"]
            warm_np = np.asarray(warm_actions)
            print(f"Warmup done. actions shape={warm_np.shape}")
            # Keep client chunk_size aligned with model action_horizon.
            if warm_np.ndim == 2 and warm_np.shape[0] > 0 and warm_np.shape[0] != int(config["chunk_size"]):
                print(
                    f"[rtc] sync chunk_size {config['chunk_size']} -> {warm_np.shape[0]} "
                    f"(model action_horizon)"
                )
                config["chunk_size"] = int(warm_np.shape[0])
                args.chunk_size = int(warm_np.shape[0])
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()

    if args.rtc_mode and len(observation_window) > 0:
        latest_obs = observation_window[-1]
        proprio = latest_obs.get("qpos", None)
        if proprio is not None:
            with rtc_prev_chunk_lock:
                rtc_prev_chunk = np.tile(
                    np.asarray(proprio, dtype=float)[None, : config["state_dim"]],
                    (config["chunk_size"], 1),
                )
            # Compile/warm the RTC path before the control loop, otherwise the
            # first RTC infer can take tens of seconds and the main loop
            # empties the buffer and repeatedly returns to zero.
            try:
                print("Warming up RTC path (may take a while on first run)...")
                chunk_size = config["chunk_size"]
                exec_h = chunk_size if getattr(args, "rtc_execute_horizon", None) is None else args.rtc_execute_horizon
                exec_h = max(1, min(int(exec_h), chunk_size))
                with rtc_prev_chunk_lock:
                    pc = np.array(rtc_prev_chunk) if rtc_prev_chunk is not None else None
                if pc is not None:
                    pc = np.asarray(pc, dtype=np.float32)
                    # Ensure (T, A) with T == chunk_size for RTC guidance.
                    if pc.ndim == 1:
                        pc = np.tile(pc[None, :], (chunk_size, 1))
                    elif pc.shape[0] != chunk_size:
                        if pc.shape[0] > chunk_size:
                            pc = pc[:chunk_size]
                        else:
                            pad = np.repeat(pc[-1:], chunk_size - pc.shape[0], axis=0)
                            pc = np.concatenate([pc, pad], axis=0)
                rtc_extra = {
                    "execute_horizon": exec_h,
                    "enable_rtc": True,
                    "mask_prefix_delay": getattr(args, "rtc_mask_prefix_delay", False),
                    "max_guidance_weight": getattr(args, "rtc_max_guidance_weight", 0.5),
                    "inference_delay": 0,
                }
                if pc is not None:
                    rtc_extra["prev_action_chunk"] = pc
                rtc_payload = build_kai0_observation(
                    latest_obs, config, lang_embeddings, **rtc_extra
                )
                t_rtc = time.time()
                with policy_infer_lock:
                    rtc_warm = policy.infer(rtc_payload)["actions"]
                print(
                    f"RTC warmup done in {time.time() - t_rtc:.1f}s, "
                    f"shape={np.asarray(rtc_warm).shape}"
                )
                with rtc_prev_chunk_lock:
                    rtc_prev_chunk = np.asarray(rtc_warm, dtype=float)
            except Exception as e:
                print(f"RTC warmup failed: {e}")
                import traceback
                traceback.print_exc()

    stream_buffer = StreamActionBuffer(
        max_chunks=args.buffer_max_chunks,
        decay_alpha=args.exp_decay_alpha,
        state_dim=config["state_dim"],
        smooth_method="raw" if (args.rtc_mode and getattr(args, "rtc_disable_smoothing", False)) else "temporal",
    )

    # Seed buffer with RTC warmup chunk so control can start immediately.
    if args.rtc_mode:
        with rtc_prev_chunk_lock:
            seed = None if rtc_prev_chunk is None else np.asarray(rtc_prev_chunk, dtype=float).copy()
        if seed is not None and seed.size > 0:
            stream_buffer.integrate_new_chunk(
                seed,
                max_k=0,
                min_m=int(getattr(args, "min_smooth_steps", 8)),
            )
            print(f"Seeded action buffer with {len(stream_buffer.cur_chunk)} steps")

    if args.rtc_mode:
        inference_thread = threading.Thread(
            target=inference_fn_non_blocking_rtc,
            args=(args, config, policy, ros_operator),
            daemon=True,
        )
        inference_thread.start()
        print("RTC inference thread started.")
    else:
        inference_thread = start_inference_thread(args, config, policy, ros_operator)

    rate = ros_operator.create_rate(args.control_frequency)
    step = 0
    episode_id = 0
    consecutive_empty_actions = 0
    # ~30Hz * 20s: allow slow first RTC compiles without flapping home.
    max_empty_actions = int(max(600, args.control_frequency * 20))
    got_first_action = False
    # Do not execute until user presses C.
    episode_running.clear()
    if stream_buffer is not None:
        stream_buffer.clear()

    def _reset_rtc_prev_from_pose(pose_14: np.ndarray):
        """RTC guidance prior only — do NOT push into the execute buffer."""
        global rtc_prev_chunk
        if not args.rtc_mode:
            return
        ah = int(config.get("chunk_size", getattr(args, "chunk_size", 50)))
        q = np.asarray(pose_14, dtype=float).reshape(-1)[: int(config["state_dim"])]
        with rtc_prev_chunk_lock:
            rtc_prev_chunk = np.tile(q[None, :], (ah, 1))

    def _start_episode():
        nonlocal step, consecutive_empty_actions, got_first_action, episode_id
        episode_id += 1
        step = 0
        consecutive_empty_actions = 0
        got_first_action = False
        if stream_buffer is not None:
            stream_buffer.clear()
        # Use current hold/init pose as RTC prior only. Never re-play the old
        # action chunk — that yanked the arms back after Enter→home→C.
        hold = getattr(ros_operator, "_hold_pose", None)
        seed_pose = np.asarray(hold if hold is not None else init_pose, dtype=float)
        _reset_rtc_prev_from_pose(seed_pose)
        episode_running.set()
        msg = (
            f"episode #{episode_id} STARTED from current pose "
            f"(waiting for fresh policy chunk; press S to stop)"
        )
        print(f"[ctrl] {msg}")
        update_web_status(
            phase="running",
            message=msg,
            episode_id=episode_id,
            step=0,
            empty_actions=0,
        )

    def _stop_episode(reason: str = "S"):
        nonlocal consecutive_empty_actions, got_first_action
        if not episode_running.is_set() and reason == "S":
            print("[ctrl] already stopped; Enter=go home, C=start, Q=quit")
            update_web_status(message="already stopped; Enter=home, C=start, Q=quit")
            return
        # Freeze last commanded pose (or measured once) BEFORE clearing running,
        # so the idle hold thread keeps a fixed anti-sag target.
        if published_actions_history:
            ros_operator.capture_hold_pose(published_actions_history[-1])
        else:
            ros_operator.capture_hold_pose()
        episode_running.clear()
        if stream_buffer is not None:
            stream_buffer.clear()
        consecutive_empty_actions = 0
        got_first_action = False
        # Drop stale RTC chunk so a later C cannot replay pre-stop motions.
        hold = getattr(ros_operator, "_hold_pose", None)
        if hold is not None:
            _reset_rtc_prev_from_pose(hold)
        else:
            with rtc_prev_chunk_lock:
                rtc_prev_chunk = None
        ros_operator.publish_hold_pose()
        msg = (
            f"episode STOPPED ({reason}); holding pose. "
            f"Enter=return to init, C=start, Q=quit"
        )
        print(f"[ctrl] {msg}")
        update_web_status(
            phase="idle",
            message=msg,
            episode_id=episode_id,
            empty_actions=0,
        )

    def _return_to_init(*, force: bool = False) -> bool:
        """Smooth move back to init_pose; then hold there.

        If ``force`` (used by Q/quit), stop the episode first when needed.
        """
        if episode_running.is_set():
            if not force:
                print("[ctrl] episode still running; press S first, then Enter to go home")
                return False
            _stop_episode("Q")
        print("[ctrl] Returning to initial pose...")
        if stream_buffer is not None:
            stream_buffer.clear()
        ok = ros_operator.smooth_goto_position(
            target_pos=init_pose,
            duration=3.0,
            hz=50.0,
        )
        if ok:
            ros_operator.capture_hold_pose(init_pose)
            ros_operator.publish_hold_pose()
            _reset_rtc_prev_from_pose(init_pose)
            print("[ctrl] at initial pose (holding). Press C to start, Q to quit")
            update_web_status(
                phase="idle",
                message="at initial pose (holding). Press C to start, Q to quit",
            )
            return True
        print("[ctrl] return-to-init failed; still holding previous pose")
        update_web_status(message="return-to-init failed; still holding previous pose")
        return False

    def _quit_to_zero():
        """Stop episode if needed, return to startup zero pose, then exit."""
        print("[ctrl] Q: return to zero then quit...")
        update_web_status(phase="quit", message="Q: returning to zero then quit...")
        try:
            if episode_running.is_set():
                _stop_episode("Q")
            if stream_buffer is not None:
                stream_buffer.clear()
            ok = ros_operator.smooth_return_to_zero(duration=3.0)
            if ok:
                # smooth_return_to_zero already capture_hold_pose(open zeros)
                for _ in range(10):
                    ros_operator.publish_hold_pose()
                    time.sleep(0.05)
            else:
                print("[ctrl] return-to-zero failed; quitting anyway")
        except Exception as e:
            print(f"[ctrl] go-zero-before-quit failed: {e}")
        shutdown_event.set()

    def _apply_ctrl_cmd(cmd: str, source: str = "kbd"):
        if cmd == "q":
            _quit_to_zero()
            return
        if cmd == "s":
            _stop_episode("S" if source.startswith("kbd") else f"web:{source}")
            return
        if cmd == "home":
            update_web_status(message=f"home requested ({source})")
            _return_to_init()
            return
        if cmd == "c":
            if episode_running.is_set():
                print("[ctrl] already running; press S to stop first")
                update_web_status(message="already running; press S to stop first")
            else:
                _start_episode()

    keys = None
    try:
        keys = TerminalKeys()
        keys.__enter__()
        use_keys = True
    except Exception as e:
        print(f"[ctrl] keyboard unavailable ({e}); use web UI buttons or auto-wait")
        use_keys = False
        # Prefer web/terminal later; do not auto-start if web preview is up.
        if _camera_web_preview is None:
            print("[ctrl] no web UI either; auto-starting episode")
            _start_episode()

    print("Starting control loop...")
    print("[ctrl] keys: C=start, S=stop, Enter=init pose (after S), Q=zero then quit")
    print("[ctrl] same keys/buttons also work on the camera web page")
    update_web_status(
        phase="idle",
        message="waiting for C to start (terminal or web)",
        episode_id=episode_id,
        step=0,
        prompt=lang_embeddings,
        policy_host=args.host,
        policy_port=args.port,
        rtc_mode=bool(args.rtc_mode),
        gripper_mode=args.gripper_mode,
        camera_backend=getattr(args, "camera_backend", None),
    )
    if use_keys or _camera_web_preview is not None:
        print("[ctrl] waiting for C to start...")
    try:
        while rclpy.ok() and step < max_publish_step and not shutdown_event.is_set():
            if use_keys and keys is not None:
                try:
                    for key in keys.read():
                        if key == "q":
                            push_ctrl_cmd("q", source="kbd")
                        elif key == "s":
                            push_ctrl_cmd("s", source="kbd")
                        elif key in ("\n", "\r"):
                            push_ctrl_cmd("home", source="kbd")
                        elif key == "c":
                            push_ctrl_cmd("c", source="kbd")
                except KeyboardInterrupt:
                    push_ctrl_cmd("q", source="kbd")

            for item in drain_ctrl_cmds():
                _apply_ctrl_cmd(item["cmd"], source=item.get("source", "web"))
                if shutdown_event.is_set():
                    break
            if shutdown_event.is_set():
                break

            frame = ros_operator.get_frame()
            if frame is None:
                rate.sleep()
                continue

            imgs, j_left, j_right = frame
            qpos = ros_operator.get_joint_positions(j_left, j_right)
            observed_qpos_history.append(qpos.copy())
            raw = getattr(ros_operator, "_last_raw_grip", (None, None))
            last_cmd = published_actions_history[-1] if published_actions_history else None
            buf_len = 0
            if stream_buffer is not None:
                try:
                    with stream_buffer.lock:
                        buf_len = len(stream_buffer.cur_chunk)
                except Exception:
                    buf_len = 0
            update_web_status(
                episode_id=episode_id,
                step=step,
                empty_actions=consecutive_empty_actions,
                buffer_len=buf_len,
                qpos_left=[float(x) for x in qpos[:6]],
                qpos_right=[float(x) for x in qpos[7:13]],
                grip_state_L=float(qpos[6]),
                grip_state_R=float(qpos[13]),
                grip_raw_L=raw[0],
                grip_raw_R=raw[1],
                grip_cmd_L=None if last_cmd is None else float(last_cmd[6]),
                grip_cmd_R=None if last_cmd is None else float(last_cmd[13]),
            )

            if not episode_running.is_set():
                # Idle hold thread already servos frozen pose; keep a tick here too.
                ros_operator.publish_hold_pose()
                rate.sleep()
                continue

            act = stream_buffer.pop_next_action()
            if act is not None:
                consecutive_empty_actions = 0
                got_first_action = True
                if args.use_eef_correction:
                    act = apply_eef_correction(act, qpos, args)

                pre_g = (float(act[6]), float(act[13]))
                act = apply_gripper_for_ros(act, args)

                ros_operator.set_joint_positions(act)
                published_actions_history.append(act.copy())

                step += 1
                raw = getattr(ros_operator, "_last_raw_grip", (None, None))
                dbg = gripper_debug_logger
                if dbg is not None:
                    dbg.log_step(
                        step=step,
                        model_L=pre_g[0],
                        model_R=pre_g[1],
                        actual_L=float(act[6]),
                        actual_R=float(act[13]),
                        state_L=float(qpos[6]),
                        state_R=float(qpos[13]),
                        raw_L=raw[0],
                        raw_R=raw[1],
                    )
                # Print every N steps, or whenever model vs feedback diverge a lot.
                print_every = int(getattr(args, "gripper_log_every", 10))
                diverge = (
                    abs(pre_g[0] - float(qpos[6])) > 0.02
                    or abs(pre_g[1] - float(qpos[13])) > 0.02
                )
                if step % print_every == 0 or (dbg is not None and diverge and step % 5 == 0):
                    if getattr(args, "gripper_scale_curve", "linear") == "hw":
                        _sk = dict(
                            kai0_min=float(args.kai0_gripper_min),
                            kai0_max=float(args.kai0_gripper_max),
                            fb_close=float(args.gripper_feedback_close),
                            fb_open=float(args.gripper_feedback_open),
                            ros_close=float(args.gripper_close),
                            ros_open=float(args.gripper_open),
                        )
                        scale_L = scale_gripper_kai0_to_ros_hw(pre_g[0], **_sk)
                        scale_R = scale_gripper_kai0_to_ros_hw(pre_g[1], **_sk)
                    else:
                        _sk = dict(
                            kai0_min=float(args.kai0_gripper_min),
                            kai0_max=float(args.kai0_gripper_max),
                            ros_close=float(args.gripper_close),
                            ros_open=float(args.gripper_open),
                            gamma=float(getattr(args, "gripper_gamma", 1.0)),
                        )
                        scale_L = scale_gripper_kai0_to_ros(pre_g[0], **_sk)
                        scale_R = scale_gripper_kai0_to_ros(pre_g[1], **_sk)
                    thr = float(args.gripper_snap_thresh)
                    snap_L = float(args.gripper_open) if pre_g[0] >= thr else float(args.gripper_close)
                    snap_R = float(args.gripper_open) if pre_g[1] >= thr else float(args.gripper_close)
                    line = (
                        f"[grip] ep={episode_id} step={step} mode={args.gripper_mode} "
                        f"model_m={pre_g[0]:.4f}/{pre_g[1]:.4f} "
                        f"scale={scale_L:.2f}/{scale_R:.2f} "
                        f"snap={snap_L:.2f}/{snap_R:.2f} "
                        f"cmd={act[6]:.3f}/{act[13]:.3f} "
                        f"setCatch={act[6]*5:.2f}/{act[13]*5:.2f} "
                        f"state_m={qpos[6]:.4f}/{qpos[13]:.4f} "
                        f"raw={raw[0]}/{raw[1]}"
                    )
                    print(line)
                    if dbg is not None:
                        dbg.log_console(line)
            else:
                consecutive_empty_actions += 1
                # Hold init/current pose until the first fresh policy chunk arrives.
                if not got_first_action:
                    ros_operator.publish_hold_pose()
                if consecutive_empty_actions % 30 == 0:
                    print(
                        f"[main] waiting for actions "
                        f"({consecutive_empty_actions}/{max_empty_actions})"
                    )
                if got_first_action and consecutive_empty_actions >= max_empty_actions:
                    print(f"[main] No actions for {consecutive_empty_actions} steps; auto-stop episode")
                    _stop_episode("empty_buffer")

            rate.sleep()
                
    except Exception as e:
        print(f"[main] Loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        episode_running.clear()
        shutdown_event.set()
        if keys is not None:
            try:
                keys.__exit__(None, None, None)
            except Exception:
                pass
        if inference_thread.is_alive():
            inference_thread.join(timeout=2.0)
        if gripper_debug_logger is not None:
            try:
                gripper_debug_logger.finalize()
            except Exception as e:
                print(f"[gripper_debug] finalize error: {e}")
        try:
            if _camera_web_preview is not None:
                _camera_web_preview.stop()
        except Exception as e:
            print(f"[camera_web] stop error: {e}")
        ros_operator.cleanup_cameras()

        print("ARX5 inference controller shut down.")

        return inference_thread




def apply_eef_correction(act: np.ndarray, qpos: np.ndarray, args) -> np.ndarray:
    """Apply end-effector micro correction."""
    left0, right0 = qpos[:6], qpos[7:13]
    dl = np.array(args.eef_corr_left)
    dr = np.array(args.eef_corr_right)
    
    left6 = apply_micro_correction(left0, dl, "base",
                                   args.eef_lambda,
                                   args.eef_step_limit_m,
                                   args.eef_joint_step_limit)
    right6 = apply_micro_correction(right0, dr, "base",
                                    args.eef_lambda,
                                    args.eef_step_limit_m,
                                    args.eef_joint_step_limit)
    
    act2 = act.copy()
    act2[:6], act2[7:13] = left6, right6
    return act2

def apply_gripper_binary(act: np.ndarray, close_val: float = 0.0, open_val: float = 5.0, thresh: float = 2.5) -> np.ndarray:
    """Apply gripper binary threshold in ROS catch units (0/5)."""
    act2 = act.copy()
    act2[6] = open_val if act[6] >= thresh else close_val
    act2[13] = open_val if act[13] >= thresh else close_val
    return act2


def scale_gripper_kai0_to_ros(
    g: float,
    *,
    kai0_min: float = 0.0,
    kai0_max: float = 0.085,
    ros_close: float = 0.0,
    ros_open: float = 5.0,
    gamma: float = 1.0,
) -> float:
    """Linear (or power) map Kai0 meters → ROS publish units."""
    span = max(float(kai0_max) - float(kai0_min), 1e-6)
    t = (float(np.clip(g, kai0_min, kai0_max)) - float(kai0_min)) / span
    if gamma != 1.0:
        t = float(t) ** float(gamma)
    return float(ros_close + t * (float(ros_open) - float(ros_close)))


# Empirically from gripper_20261030_114806 steady cmd→raw under remote_slave
# setCatch(cmd*5): almost binary. Proportional motion only for cmd ≲ 0.4;
# cmd ≳ 0.6 already reads fully open (~4.61). Linear map to 5 wastes the mid.
# Knots: desired feedback raw → publish cmd that achieved it when steady.
_ARX_HW_RAW_KNOTS = np.array([0.00, 0.30, 1.32, 2.50, 4.61], dtype=float)
_ARX_HW_CMD_KNOTS = np.array([-0.80, 0.10, 0.30, 0.40, 0.75], dtype=float)


def scale_gripper_kai0_to_ros_hw(
    g: float,
    *,
    kai0_min: float = 0.0,
    kai0_max: float = 0.085,
    fb_close: float = 0.0,
    fb_open: float = 4.61,
    ros_close: float = -0.8,
    ros_open: float = 5.0,
) -> float:
    """Map Kai0 meters through the measured ARX catch response curve.

    Converts desired width → target feedback raw, then inverts the steady
    cmd→raw curve so mid widths land in the tiny proportional band (~[-0.8, 0.4])
    instead of the deadzone [0.6, 5] where the jaw stays fully open.
    """
    span = max(float(kai0_max) - float(kai0_min), 1e-6)
    t = (float(np.clip(g, kai0_min, kai0_max)) - float(kai0_min)) / span
    # Near-closed: over-close for grasp force (same as snap/stock scripts).
    if t <= 0.10:
        return float(ros_close)
    # Near-open: use full open publish (0.75 already opens, 5.0 matches snap).
    if t >= 0.92:
        return float(ros_open)
    target_raw = float(fb_close) + t * (float(fb_open) - float(fb_close))
    cmd = float(np.interp(target_raw, _ARX_HW_RAW_KNOTS, _ARX_HW_CMD_KNOTS))
    return float(np.clip(cmd, min(float(ros_close), float(ros_open)), max(float(ros_close), float(ros_open))))


def scale_gripper_ros_to_kai0(
    g: float,
    *,
    kai0_min: float = 0.0,
    kai0_max: float = 0.085,
    ros_close: float = 0.0,
    ros_open: float = 5.0,
) -> float:
    """Map ARX ROS catch feedback back to Kai0/UMI gripper width.

    ``remote_slave`` with ``arm_end_type=0`` does ``setCatch(msg*5)``, so
    feedback may be in that amplified SDK space (~0-25). Detect and undo.
    """
    g = float(g)
    ros_close = float(ros_close)
    ros_open = float(ros_open)
    if g > ros_open * 1.5:
        g = g / 5.0
    span = max(ros_open - ros_close, 1e-6)
    t = (float(np.clip(g, ros_close, ros_open)) - ros_close) / span
    return float(kai0_min + t * (float(kai0_max) - float(kai0_min)))


def apply_gripper_for_ros(act: np.ndarray, args) -> np.ndarray:
    """Convert policy gripper dims to ROS command units.

    Kai0 flatten/fold actions use ~[0, 0.085] meters. ARX ``remote_slave``
    with ``arm_end_type=0`` does ``setCatch(cmd * 5)``. Measured response is
    nearly binary (cmd≳0.6 → full open); ``scale_kai0`` defaults to a hardware
    curve so mid widths use the proportional band. ``snap_kai0`` forces open/close.
    """
    act2 = np.asarray(act, dtype=float).copy()
    mode = getattr(args, "gripper_mode", "snap_kai0")
    kai0_min = float(getattr(args, "kai0_gripper_min", 0.0))
    kai0_max = float(getattr(args, "kai0_gripper_max", 0.085))
    # Match temporal_smooth / dagger: over-close for force under setCatch(msg*5).
    ros_close = float(getattr(args, "gripper_close", -0.8))
    ros_open = float(getattr(args, "gripper_open", 5.0))
    if mode == "raw":
        return act2
    if mode == "binary_ros":
        return apply_gripper_binary(
            act2,
            close_val=ros_close,
            open_val=ros_open,
            thresh=float(getattr(args, "gripper_thresh", 2.5)),
        )
    snap_thresh = float(
        getattr(
            args,
            "gripper_snap_thresh",
            0.5 * (kai0_min + kai0_max),
        )
    )
    curve = getattr(args, "gripper_scale_curve", "linear")
    gamma = float(getattr(args, "gripper_gamma", 1.0))
    fb_close = float(getattr(args, "gripper_feedback_close", 0.0))
    fb_open = float(getattr(args, "gripper_feedback_open", 4.61))
    for idx in (6, 13):
        g = float(act2[idx])
        if mode == "snap_kai0":
            # Below thresh -> full close; above -> full open (firm cloth grasp).
            act2[idx] = ros_open if g >= snap_thresh else ros_close
        elif curve == "hw":
            act2[idx] = scale_gripper_kai0_to_ros_hw(
                g,
                kai0_min=kai0_min,
                kai0_max=kai0_max,
                fb_close=fb_close,
                fb_open=fb_open,
                ros_close=ros_close,
                ros_open=ros_open,
            )
        else:
            # Default: linear Kai0 meters -> ROS catch publish units.
            act2[idx] = scale_gripper_kai0_to_ros(
                g,
                kai0_min=kai0_min,
                kai0_max=kai0_max,
                ros_close=ros_close,
                ros_open=ros_open,
                gamma=gamma,
            )
    return act2


def main():

    def _on_sigint(sig, frame):
        shutdown_event.set()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ARX-X5 dual-arm OpenPi inference (RTC / temporal smoothing).")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--joint_cmd_topic_right", default="/arm_master_r_status")
    parser.add_argument("--joint_state_topic_right", default="/arm_slave_r_status")

    parser.add_argument("--camera_front_serial", type=str, default='152122073503')
    parser.add_argument("--camera_left_serial", type=str, default='213622070289')
    parser.add_argument("--camera_right_serial", type=str, default='152122073474')
    parser.add_argument(
        "--camera_backend",
        type=str,
        default="hybrid",
        choices=["hybrid", "realsense3"],
        help="hybrid=GoPro wrists+D435 center (umi-arx); realsense3=three RealSense serials.",
    )
    parser.add_argument(
        "--camera_reorder",
        type=str,
        default="0,1",
        help="Wrist GoPro V4L indices after filtering RealSense (same as eval --camera-reorder).",
    )
    parser.add_argument(
        "--rs_serial",
        type=str,
        default=None,
        help="Optional D435 serial for hybrid backend.",
    )
    parser.add_argument(
        "--camera_obs_latency",
        type=float,
        default=0.17,
        help="Camera receive latency used by hybrid backend.",
    )
    parser.add_argument(
        "--umi_root",
        type=str,
        default="/home/lenovo/kian/umi-arx-kian",
        help="Path to umi-arx-kian (hybrid camera stack).",
    )
    parser.add_argument(
        "--no_reset_elgato",
        action="store_true",
        help="Skip Elgato USB reset when starting hybrid cameras.",
    )
    parser.add_argument(
        "--save_camera_dir",
        type=str,
        default="/tmp/arx_ros_camera_preview",
        help="Directory to dump camera preview images after sensors are ready.",
    )
    parser.add_argument(
        "--no_save_camera_preview",
        action="store_true",
        help="Skip dumping camera preview images.",
    )
    parser.add_argument(
        "--show_camera_preview",
        action="store_true",
        help="Also pop up a mosaic window (needs DISPLAY).",
    )
    parser.add_argument(
        "--no_web_camera_preview",
        action="store_true",
        help="Disable live MJPEG camera page (enabled by default).",
    )
    parser.add_argument(
        "--web_camera_host",
        type=str,
        default="0.0.0.0",
        help="Bind address for live camera web preview.",
    )
    parser.add_argument(
        "--web_camera_port",
        type=int,
        default=8080,
        help="HTTP port for live camera web preview (open http://HOST:PORT/).",
    )
    parser.add_argument(
        "--web_camera_fps",
        type=float,
        default=10.0,
        help="Target FPS for camera web preview encode/stream.",
    )
    parser.add_argument(
        "--web_camera_jpeg_quality",
        type=int,
        default=70,
        help="JPEG quality 30-95 for camera web preview.",
    )
    parser.add_argument(
        "--web_camera_tile_width",
        type=int,
        default=480,
        help="Width of each camera tile in the web mosaic.",
    )
    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control_frequency", type=float, default=20.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)

    parser.add_argument("--rtc_mode", action="store_true", help="Enable real-time chunking (server must use RTC config).")
    parser.add_argument("--rtc_mask_prefix_delay", action="store_true", help="RTC: mask delay prefix.")
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=0.5, help="RTC max guidance weight.")
    parser.add_argument("--rtc_execute_horizon", type=int, default=None, help="RTC execute horizon (default chunk_size).")
    parser.add_argument("--rtc_disable_smoothing", action="store_true", help="RTC: disable chunk smoothing (latency trim only).")
    parser.add_argument("--use_temporal_smoothing", action="store_true", default=True)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=10)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)

    parser.add_argument(
        "--gripper_mode",
        type=str,
        default="snap_kai0",
        choices=["snap_kai0", "scale_kai0", "binary_ros", "raw"],
        help="snap_kai0: firm open/close from Kai0 meters (recommended for cloth); "
             "scale_kai0: continuous map (default curve=linear); "
             "binary_ros: threshold in ROS units; raw: no conversion.",
    )
    parser.add_argument(
        "--gripper_scale_curve",
        type=str,
        default="linear",
        choices=["hw", "linear"],
        help="scale_kai0 only: linear=map onto [--gripper_close,--gripper_open] (default); "
             "hw=experimental ARX catch curve (often wrong for grasp force).",
    )
    parser.add_argument(
        "--gripper_gamma",
        type=float,
        default=1.0,
        help="scale_kai0 + linear only: power on normalized width (t**gamma).",
    )
    parser.add_argument(
        "--gripper_snap_thresh",
        type=float,
        default=0.042,
        help="snap_kai0: Kai0 gripper width >= thresh -> full open, else full close.",
    )
    parser.add_argument(
        "--gripper_open",
        type=float,
        default=5.0,
        help="ROS catch command for fully open (ARX remote_slave units).",
    )
    parser.add_argument(
        "--gripper_close",
        type=float,
        default=-0.8,
        help="ROS catch close command. Default -0.8 matches ARX scripts; "
             "remote_slave does setCatch(cmd*5) so -0.8 -> -4 for firm grip. "
             "0 only reaches zero position and often feels soft.",
    )
    parser.add_argument(
        "--gripper_feedback_close",
        type=float,
        default=0.0,
        help="ROS catch feedback value treated as fully closed (state mapping).",
    )
    parser.add_argument(
        "--gripper_feedback_open",
        type=float,
        default=4.61,
        help="ROS catch feedback value treated as fully open (state mapping). "
             "Measured ~4.61 on this ARX stack.",
    )
    parser.add_argument(
        "--gripper_thresh",
        type=float,
        default=2.5,
        help="binary_ros only: infer value >= thresh -> open.",
    )
    parser.add_argument("--kai0_gripper_min", type=float, default=0.0)
    parser.add_argument(
        "--kai0_gripper_max",
        type=float,
        default=0.085,
        help="Kai0/UMI open gripper width in meters (same as eval --gripper-max).",
    )
    parser.add_argument(
        "--gripper_debug",
        action="store_true",
        default=True,
        help="Save per-step gripper CSV: model meters vs scale/snap/actual cmd vs feedback.",
    )
    parser.add_argument(
        "--no_gripper_debug",
        action="store_true",
        help="Disable gripper debug CSV logging.",
    )
    parser.add_argument(
        "--gripper_log_dir",
        type=str,
        default="",
        help="Root dir for gripper logs (default: <script>/logs/).",
    )
    parser.add_argument(
        "--gripper_log_every",
        type=int,
        default=10,
        help="Print gripper debug line every N control steps.",
    )
    parser.add_argument("--use_eef_correction", action="store_true")
    parser.add_argument("--eef_corr_left", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_corr_right", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_lambda", type=float, default=0.001)
    parser.add_argument("--eef_step_limit_m", type=float, default=0.01)
    parser.add_argument("--eef_joint_step_limit", nargs=6, type=float, default=[0.1]*6)

    parser.add_argument("--auto_homing", action="store_true", default=True, help="Return to zero on startup.")
    parser.add_argument("--exit_homing", action="store_true", help="Return to zero on exit.")

    args = parser.parse_args()
    if args.no_gripper_debug:
        args.gripper_debug = False
    print(
        f"[gripper] mode={args.gripper_mode} curve={args.gripper_scale_curve} "
        f"open={args.gripper_open} close={args.gripper_close} "
        f"fb_open={args.gripper_feedback_open} "
        f"snap_thresh={args.gripper_snap_thresh} "
        f"debug={args.gripper_debug}"
    )
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        rclpy.init()
        ros_operator = ARX5ROSController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
        spin_thread.start()
        print("ROS spin thread started.")

        if not ros_operator.wait_for_data_ready(timeout=15.0):
            print("Sensor data not ready; exiting.")
            return

        if not args.no_save_camera_preview:
            save_camera_preview(
                ros_operator,
                args.save_camera_dir,
                show=bool(args.show_camera_preview),
            )

        start_camera_web_preview(ros_operator, args)

        if args.auto_homing:
            print("Auto homing...")
            ros_operator.smooth_return_to_zero(duration=3.0)
            time.sleep(1.0)

        # Hold frozen pose while waiting for Enter (before model_inference hold thread).
        ros_operator.capture_hold_pose()
        episode_running.clear()
        pre_hold = start_idle_hold_thread(
            ros_operator, hz=float(getattr(args, "control_frequency", 30.0))
        )

        print("Arms ready. Warmup next; then C=start, S=stop, Enter=init pose, Q=zero then quit.")
        input("Press Enter to continue to warmup...")

        config = get_config(args)
        inference_thread = model_inference(args, config, ros_operator)

    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            shutdown_event.set()
            try:
                if gripper_debug_logger is not None:
                    gripper_debug_logger.finalize()
            except Exception as e:
                print(f"Gripper debug finalize error: {e}")
            try:
                if _camera_web_preview is not None:
                    _camera_web_preview.stop()
            except Exception as e:
                print(f"Camera web preview stop error: {e}")
            try:
                if "ros_operator" in locals() and ros_operator is not None:
                    ros_operator.cleanup_cameras()
            except Exception as e:
                print(f"Camera cleanup error: {e}")
            if rclpy.ok():
                rclpy.shutdown()
            print("ROS2 shutdown.")
        except Exception as e:
            print(f"ROS2 shutdown error: {e}")
        print("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
