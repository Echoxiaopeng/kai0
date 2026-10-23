#!/usr/bin/env python3
"""Convert bimanual UMI-style replay_buffer.zarr to Kai0 Agilex LeRobot format.

Source fields (per frame):
  robot{0,1}_joint_pos [6], robot{0,1}_gripper_width, camera{0,1,2}_rgb [H,W,3]
  action in zarr is EE pose (pos+rot+gripper)x2 — NOT used for joint training.

Target (kai0 Agilex / FlattenFold-compatible):
  observation.state [14] = [j0(6), g0, j1(6), g1]
  action [14]           = next-frame absolute joint state (last frame repeats)
  images: top_head <- camera1, hand_left <- camera0, hand_right <- camera2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from tqdm import tqdm


FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {"dtype": "float32", "shape": [14]},
    "action": {"dtype": "float32", "shape": [14]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
}


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def encode_mp4(images: np.ndarray, path: Path, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = images[0].shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "28", "preset": "ultrafast"}
        for img in images:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def load_cam_uint8(data, key: str, start: int, end: int, batch: int = 256) -> np.ndarray:
    """Load RGB float/uint frames in batches to limit peak memory."""
    T = end - start
    out = np.empty((T, 224, 224, 3), dtype=np.uint8)
    for i in range(0, T, batch):
        sl = slice(start + i, min(end, start + i + batch))
        chunk = np.asarray(data[key][sl])
        out[i : i + chunk.shape[0]] = to_uint8_rgb(chunk)
    return out


def build_joint_state(data, start: int, end: int) -> np.ndarray:
    j0 = np.asarray(data["robot0_joint_pos"][start:end], dtype=np.float32)
    j1 = np.asarray(data["robot1_joint_pos"][start:end], dtype=np.float32)
    g0 = np.asarray(data["robot0_gripper_width"][start:end], dtype=np.float32).reshape(-1, 1)
    g1 = np.asarray(data["robot1_gripper_width"][start:end], dtype=np.float32).reshape(-1, 1)
    return np.concatenate([j0, g0, j1, g1], axis=-1)


def convert_one_zarr(
    zarr_path: Path,
    out_root: Path,
    repo_id: str,
    prompt: str,
    fps: int,
    episode_offset: int,
    global_index_offset: int,
) -> tuple[int, int]:
    """Convert one replay_buffer.zarr. Returns (n_episodes, n_frames)."""
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])

    out_ds = out_root / repo_id
    data_chunk = out_ds / "data" / "chunk-000"
    vid_chunk = out_ds / "videos" / "chunk-000"
    meta_dir = out_ds / "meta"
    for cam in ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"):
        (vid_chunk / cam).mkdir(parents=True, exist_ok=True)
    data_chunk.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # resume-friendly: skip episodes whose parquet already exists (by global episode id)
    total_frames_written = 0
    for p in sorted(data_chunk.glob("episode_*.parquet")):
        total_frames_written += len(pq.read_table(p, columns=["index"]))

    # When appending another zarr with episode_offset, global_index_offset should be
    # the total frames already in the dataset; fall back to counting disk if 0 and offset>0.
    if global_index_offset == 0 and episode_offset > 0:
        global_index = total_frames_written
    else:
        global_index = global_index_offset

    for local_i in tqdm(range(len(ends)), desc=f"{zarr_path.parent.name}"):
        s, e = int(starts[local_i]), int(ends[local_i])
        T = e - s
        if T <= 1:
            continue

        ep_idx = episode_offset + local_i
        ep_name = f"episode_{ep_idx:06d}"
        if (data_chunk / f"{ep_name}.parquet").exists():
            continue

        state = build_joint_state(data, s, e)
        # absolute joint action = next proprio (last frame repeats)
        action = np.concatenate([state[1:], state[-1:]], axis=0)

        for zkey, vkey in (
            ("camera1_rgb", "observation.images.top_head"),
            ("camera0_rgb", "observation.images.hand_left"),
            ("camera2_rgb", "observation.images.hand_right"),
        ):
            cams = load_cam_uint8(data, zkey, s, e)
            encode_mp4(cams, vid_chunk / vkey / f"{ep_name}.mp4", fps)
            del cams

        timestamps = np.arange(T, dtype=np.float32) / float(fps)
        frame_index = np.arange(T, dtype=np.int64)
        episode_index = np.full(T, ep_idx, dtype=np.int64)
        index = np.arange(global_index, global_index + T, dtype=np.int64)
        task_index = np.zeros(T, dtype=np.int64)
        global_index += T
        total_frames_written += T

        state_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(state.reshape(-1), type=pa.float32()), 14
        )
        action_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), 14
        )
        table = pa.table(
            {
                "observation.state": state_arr,
                "action": action_arr,
                "timestamp": timestamps,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": index,
                "task_index": task_index,
            }
        )
        pq.write_table(table, data_chunk / f"{ep_name}.parquet")
        print(f"  wrote {ep_name} T={T}", flush=True)

    # write / update meta
    n_eps = episode_offset + len(ends)
    # recount from disk for accuracy
    parquets = sorted(data_chunk.glob("episode_*.parquet"))
    lengths = []
    total = 0
    episodes_meta = []
    for p in parquets:
        n = len(pq.read_table(p, columns=["episode_index"]))
        ep_id = int(p.stem.split("_")[-1])
        episodes_meta.append(
            {
                "episode_index": ep_id,
                "tasks": [prompt],
                "length": n,
            }
        )
        lengths.append(n)
        total += n

    with open(meta_dir / "info.json", "w") as f:
        json.dump(
            {
                "codebase_version": "v2.1",
                "robot_type": "agilex",
                "total_episodes": len(parquets),
                "total_frames": total,
                "total_tasks": 1,
                "total_videos": len(parquets) * 3,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": fps,
                "splits": {"train": f"0:{len(parquets)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": FEATURES,
            },
            f,
            indent=2,
        )

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": prompt}) + "\n")

    # episodes_stats minimal
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for row, n in zip(episodes_meta, lengths):
            f.write(
                json.dumps(
                    {
                        "episode_index": row["episode_index"],
                        "stats": {
                            "observation.state": {"min": [], "max": [], "mean": [], "std": [], "count": [n]},
                            "action": {"min": [], "max": [], "mean": [], "std": [], "count": [n]},
                        },
                    }
                )
                + "\n"
            )

    return len(parquets), total


def find_zarr(extract_dir: Path) -> Path:
    cands = list(extract_dir.rglob("replay_buffer.zarr"))
    if not cands:
        raise FileNotFoundError(f"No replay_buffer.zarr under {extract_dir}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=Path, required=True, help="Path to replay_buffer.zarr or its parent extract dir")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--repo-id", type=str, default="FlattenFold/base")
    ap.add_argument("--prompt", type=str, default="Flatten and fold the cloth.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--global-index-offset", type=int, default=0)
    args = ap.parse_args()

    zpath = args.zarr
    if zpath.name != "replay_buffer.zarr":
        zpath = find_zarr(zpath)

    n_eps, n_frames = convert_one_zarr(
        zpath,
        args.out_root,
        args.repo_id,
        args.prompt,
        args.fps,
        args.episode_offset,
        args.global_index_offset,
    )
    print(f"Done: episodes={n_eps} frames={n_frames} -> {args.out_root / args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
