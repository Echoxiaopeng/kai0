"""GoPro wrist (V4L/Elgato) + middle D435 camera stack for ARX ROS inference.

Reuses umi-arx-kian HybridMultiCamera without importing the ZMQ robot env:
  camera0 / cam_left_wrist  = left GoPro
  camera1 / cam_high        = D435
  camera2 / cam_right_wrist = right GoPro

Returns BGR uint8 frames so existing BGR2RGB paths in the inference scripts stay valid.
"""

from __future__ import annotations

import math
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_UMI_ROOT = Path("/home/lenovo/kian/umi-arx-kian")
_RS_V4L_MARKERS = ("realsense", "intel_r", "intel_corp", "depth_camera")


def _ensure_umi_on_path(umi_root: Path) -> Path:
    root = Path(umi_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"umi-arx root not found: {root}. Pass --umi_root or install the camera stack."
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def filter_wrist_v4l_paths(v4l_paths: List[str]) -> List[str]:
    kept = []
    for path in v4l_paths:
        name = path.lower()
        if any(m in name for m in _RS_V4L_MARKERS):
            print(f"[HybridGoProRS] skip RealSense V4L device: {path}")
            continue
        kept.append(path)
    return kept


def _geometry_for_transform(input_res, output_res):
    iw, ih = input_res
    ow, oh = output_res
    interp_method = cv2.INTER_AREA
    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    w_slice_start = (rw - ow) // 2
    h_slice_start = (rh - oh) // 2
    return rw, rh, w_slice_start, h_slice_start, ow, oh, interp_method


def make_uvc_transform(input_res, obs_image_resolution, *, obs_float32: bool):
    from utils.cv2_util import get_image_transform
    from utils.cv_util import draw_predefined_mask

    def tf(data, input_res=input_res):
        img = data["color"]
        f = get_image_transform(
            input_res=input_res,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        img = np.ascontiguousarray(f(img))
        img = draw_predefined_mask(
            img,
            color=(0, 0, 0),
            mirror=False,
            gripper=False,
            finger=False,
            use_aa=False,
        )
        if obs_float32:
            img = img.astype(np.float32) / 255
        data["color"] = img
        return data

    return tf


def make_rs_transform(input_res, obs_image_resolution, *, obs_float32: bool):
    from utils.cv2_util import get_image_transform

    color_tf = get_image_transform(
        input_res=input_res,
        output_res=obs_image_resolution,
        bgr_to_rgb=True,
    )
    rw, rh, w0, h0, ow, oh, _ = _geometry_for_transform(
        input_res, obs_image_resolution
    )
    w_slice = slice(w0, w0 + ow)
    h_slice = slice(h0, h0 + oh)

    def tf(data, input_res=input_res):
        img = data["color"]
        if img.shape[:2] == (input_res[1], input_res[0]):
            out = color_tf(img)
        else:
            out = img
        if obs_float32 and out.dtype != np.float32:
            out = out.astype(np.float32) / 255
        data["color"] = out
        if "depth" in data:
            depth = data["depth"]
            if depth.ndim == 2 and depth.shape == (input_res[1], input_res[0]):
                depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
                depth = depth[h_slice, w_slice]
            data["depth"] = depth
        return data

    return tf


class HybridGoProRSCameras:
    """Thin camera-only wrapper around umi-arx HybridMultiCamera."""

    def __init__(
        self,
        *,
        camera_reorder: Sequence[int] = (0, 1),
        rs_serial: Optional[str] = None,
        camera_obs_latency: float = 0.17,
        obs_image_resolution: Tuple[int, int] = (224, 224),
        rs_resolution: Tuple[int, int] = (640, 480),
        rs_fps: int = 30,
        max_obs_buffer_size: int = 60,
        umi_root: Path | str = DEFAULT_UMI_ROOT,
        reset_elgato: bool = True,
    ):
        self._umi_root = _ensure_umi_on_path(Path(umi_root))

        from peripherals.hybrid_multi_camera import HybridMultiCamera
        from peripherals.multi_uvc_camera import MultiUvcCamera
        from peripherals.realsense_camera import RealSenseCamera, require_realsense_device
        from peripherals.video_recorder import VideoRecorder
        from utils.usb_util import get_sorted_v4l_paths, reset_all_elgato_devices

        reorder = [int(x) for x in camera_reorder]
        if len(reorder) != 2:
            raise ValueError(f"camera_reorder needs 2 indices, got {reorder}")

        chosen_rs = require_realsense_device(rs_serial)
        print(
            f"[HybridGoProRS] RealSense OK: {chosen_rs['name']} "
            f"serial={chosen_rs['serial']}"
        )

        if reset_elgato:
            reset_all_elgato_devices()
            print("[HybridGoProRS] waiting for Elgato after USB reset...")
            time.sleep(2.5)

        wrist_v4l = filter_wrist_v4l_paths(get_sorted_v4l_paths())
        print(f"[HybridGoProRS] wrist V4L candidates ({len(wrist_v4l)}):")
        for i, p in enumerate(wrist_v4l):
            print(f"  [{i}] {p}")
        try:
            wrist_paths = [wrist_v4l[i] for i in reorder]
        except IndexError as e:
            raise RuntimeError(
                f"wrist camera index out of range: camera_reorder={reorder}, "
                f"available={len(wrist_v4l)}"
            ) from e

        self._shm = SharedMemoryManager()
        self._shm.start()

        resolution: List[Tuple[int, int]] = []
        capture_fps: List[int] = []
        cap_buffer_size: List[int] = []
        video_recorder = []
        transform = []
        for path in wrist_paths:
            print("[HybridGoProRS] wrist path:", path)
            if "Cam_Link_4K" in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(
                VideoRecorder.create_h264(fps=fps, input_pix_fmt="bgr24")
            )
            transform.append(
                make_uvc_transform(
                    res,
                    obs_image_resolution,
                    obs_float32=False,
                )
            )

        uvc_camera = MultiUvcCamera(
            dev_video_paths=wrist_paths,
            shm_manager=self._shm,
            resolution=resolution,
            capture_fps=capture_fps,
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=float(camera_obs_latency),
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        rs_res = tuple(rs_resolution)
        rs_transform = make_rs_transform(
            rs_res,
            obs_image_resolution,
            obs_float32=False,
        )
        rs_recorder = VideoRecorder.create_h264(
            fps=int(rs_fps), input_pix_fmt="bgr24"
        )
        realsense_camera = RealSenseCamera(
            shm_manager=self._shm,
            serial_number=rs_serial,
            resolution=rs_res,
            capture_fps=int(rs_fps),
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=float(camera_obs_latency),
            enable_depth=False,
            transform=rs_transform,
            video_recorder=rs_recorder,
            verbose=False,
        )

        self._camera = HybridMultiCamera(
            uvc_camera=uvc_camera,
            realsense_camera=realsense_camera,
            rs_index=1,
        )
        self._last_out = None
        self._started = False

    @property
    def is_ready(self) -> bool:
        return bool(self._started and self._camera.is_ready)

    def start(self, timeout: float = 20.0) -> None:
        self._camera.start(wait=False)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._camera.is_ready:
                self._started = True
                for _ in range(10):
                    try:
                        self.get_bgr_images()
                    except Exception:
                        time.sleep(0.05)
                        continue
                    time.sleep(0.03)
                print("[HybridGoProRS] cameras ready.")
                return
            time.sleep(0.1)
        raise TimeoutError("Hybrid GoPro+RS cameras not ready within timeout")

    def get_bgr_images(self) -> Dict[str, np.ndarray]:
        """Return {cam_high, cam_left_wrist, cam_right_wrist} as BGR uint8."""
        if not self._started:
            raise RuntimeError("cameras not started")
        self._last_out = self._camera.get(k=1, out=self._last_out)

        def _to_bgr(frame: np.ndarray) -> np.ndarray:
            img = np.asarray(frame)
            if img.ndim == 4:
                img = img[-1]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # Hybrid transforms emit RGB; convert to BGR for legacy ROS scripts.
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return {
            "cam_left_wrist": _to_bgr(self._last_out[0]["color"]),
            "cam_high": _to_bgr(self._last_out[1]["color"]),
            "cam_right_wrist": _to_bgr(self._last_out[2]["color"]),
        }

    def stop(self) -> None:
        try:
            if self._started:
                self._camera.stop(wait=True)
        except Exception:
            pass
        try:
            self._shm.shutdown()
        except Exception:
            pass
        self._started = False
        print("[HybridGoProRS] cameras stopped.")
