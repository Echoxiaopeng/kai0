#!/usr/bin/env python3
"""Serve the local Kai0 flatten/fold checkpoint over websocket.

Run from the Kai0 repository:

    uv run scripts/serve_kai0_policy.py \
      --checkpoint checkpoints/pi05_flatten_fold_normal/my_fold_from_mixed1/5000 \
      --port 8000

RTC (real-time chunking) — same JAX checkpoint, Pi0RTC model class:

    uv run scripts/serve_kai0_policy.py \
      --checkpoint /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_26/10000 \
      --rtc --port 8000



## 测试日志
目前/media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_26/10000效果最佳

# 2026/10/30测试日志
checkpoint: /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_normal_10_27/30000
效果：目前最优，针对个别特殊情况还是无法恢复。

# 2026/11/06测试日志
checkpoint: 
/media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_cont_dagger/dagger_human/15000
效果：目前最优，也是针对个别特殊情况还是无法恢复。

# 2026/11/10测试日志
/media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_mix_human_low/5000
效果：有些情况比较犹豫
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi import transforms
from openpi.policies import platform_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config

DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi05_flatten_fold_normal/my_fold_from_mixed1/5000"
)
DEFAULT_CONFIG = "pi05_flatten_fold_normal"
RTC_CONFIG = "pi05_rtc_flatten_fold_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            f"Train config name. Default: {DEFAULT_CONFIG}; "
            f"with --rtc: {RTC_CONFIG}. Explicit --config always wins."
        ),
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help=f"Serve Pi0RTC ({RTC_CONFIG}) for prev_action_chunk / delay guidance.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Norm stats JSON. Defaults to checkpoint/assets/norm_stats.json, then checkpoint.parent/norm_stats.json.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("legacy", "platform-joint14"),
        default="legacy",
        help=(
            "Preprocessing contract. legacy preserves the existing Kai0 pipeline (default). "
            "platform-joint14 selects the OpenPI-UMI composition used by the new platform: "
            "raw state14, min/max normalization, and a 2D state tokenizer input. "
            "Select explicitly to match training; norm-stat dimensions do not choose the mode."
        ),
    )
    parser.add_argument("--default-prompt", default="Flatten and fold the cloth.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def resolve_norm_stats_path(checkpoint: Path, norm_stats: Path | None = None) -> Path:
    if norm_stats is not None:
        candidates = (norm_stats.expanduser().resolve(),)
    else:
        candidates = (
            checkpoint / "assets" / "norm_stats.json",
            checkpoint.parent / "norm_stats.json",
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Norm stats not found. Checked: " + ", ".join(str(path) for path in candidates))


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint / 'params'}")

    if args.config is not None:
        config_name = args.config
    elif args.rtc:
        config_name = RTC_CONFIG
    else:
        config_name = DEFAULT_CONFIG

    if args.rtc and "rtc" not in config_name.lower():
        logging.warning(
            "--rtc is set but config=%s does not look like an RTC config; "
            "Pi0RTC guidance requires Pi0RTCConfig (e.g. %s).",
            config_name,
            RTC_CONFIG,
        )

    norm_stats_path = resolve_norm_stats_path(checkpoint, args.norm_stats)

    config = training_config.get_config(config_name)
    # Training repack also requires ``action``; inference observations do not
    # contain it, so use the inference-only subset of the same contract.
    # RTC fields (prev_action_chunk, inference_delay, ...) stay on the raw obs
    # and are read by Policy.infer — do not put them in this repack.
    inference_repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "top_head": "observation.images.top_head",
                        "hand_left": "observation.images.hand_left",
                        "hand_right": "observation.images.hand_right",
                    },
                    "state": "observation.state",
                    "prompt": "prompt",
                }
            )
        ]
    )
    logging.info(
        "Loading config=%s checkpoint=%s rtc=%s preprocessing=%s",
        config_name,
        checkpoint,
        args.rtc,
        args.preprocessing,
    )
    logging.info("Loading norm stats=%s", norm_stats_path)
    if args.preprocessing == "platform-joint14":
        policy = platform_policy.create_trained_platform_policy(
            config,
            checkpoint,
            repack_transforms=inference_repack,
            default_prompt=args.default_prompt,
            norm_stats_path=norm_stats_path,
        )
    else:
        stats = normalize.deserialize_json(norm_stats_path.read_text())
        policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            repack_transforms=inference_repack,
            default_prompt=args.default_prompt,
            norm_stats=stats,
        )

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "config": config_name,
            "checkpoint": str(checkpoint),
            "preprocessing": args.preprocessing,
            "norm_stats_path": str(norm_stats_path),
            "supports_rtc": bool(args.rtc),
            "rtc_config": RTC_CONFIG if args.rtc else None,
            "observation_contract": (
                "top_head+hand_left+hand_right images; "
                "left_q6+left_gripper+right_q6+right_gripper state"
            ),
            "action_contract": "absolute dual-arm joints [T,14]",
            "rtc_fields": (
                "prev_action_chunk, inference_delay, execute_horizon, "
                "enable_rtc, max_guidance_weight, mask_prefix_delay"
                if args.rtc
                else None
            ),
        }
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info(
        "Policy server listening on ws://%s:%d (supports_rtc=%s)",
        args.host,
        args.port,
        args.rtc,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
