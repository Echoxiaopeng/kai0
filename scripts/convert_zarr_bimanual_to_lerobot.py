#!/usr/bin/env python3
"""Convert bimanual UMI-style replay_buffer.zarr to Kai0 Agilex LeRobot format.

Thin CLI wrapper around umi-arx-kian.modules.convert_zarr_bimanual_to_lerobot
(falls back to a local import path if the module is available under that repo).

Example:
  python scripts/convert_zarr_bimanual_to_lerobot.py \\
    --zarr /path/to/dataset \\
    --out-root /media/lenovo/data/kian \\
    --repo-id FlattenFold/collect_1029 \\
    --fps 20 \\
    --prompt "Flatten and fold the cloth."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer the shared implementation in umi-arx-kian when present.
_UMI = Path("/home/lenovo/kian/umi-arx-kian")
if _UMI.is_dir() and str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))

from modules.convert_zarr_bimanual_to_lerobot import (  # noqa: E402
    convert_one_zarr,
    find_zarr,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zarr",
        type=Path,
        required=True,
        help="Path to replay_buffer.zarr or its parent extract dir",
    )
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Final dataset root")
    ap.add_argument("--repo-id", type=str, default="FlattenFold/base")
    ap.add_argument("--prompt", type=str, default="Flatten and fold the cloth.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--global-index-offset", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    zpath = args.zarr
    if zpath.name != "replay_buffer.zarr":
        zpath = find_zarr(zpath)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.out_root is not None:
        out_dir = Path(args.out_root) / args.repo_id
    else:
        ap.error("需要 --out-dir 或 --out-root")
        return 2

    n_eps, n_frames = convert_one_zarr(
        zpath,
        out_dir,
        prompt=args.prompt,
        fps=args.fps,
        episode_offset=args.episode_offset,
        global_index_offset=args.global_index_offset,
        overwrite=bool(args.overwrite),
    )
    print(f"Done: episodes={n_eps} frames={n_frames} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
