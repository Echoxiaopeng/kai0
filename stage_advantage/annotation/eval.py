"""
Evaluate a trained Advantage Estimator on a LeRobot dataset and write
predicted advantage values back into new parquet files.

Expected dataset layout:
    dataset_root/
      ├── data/
      ├── videos/
      └── meta/

Usage:
    python eval.py <model_type> <model_name> <repo_id>
        [--ckpt_dir DIR] [--ckpt_steps N] [--output-dirname NAME]
        [--only-episodes 1,2,3] [--overwrite]
"""
import os
import argparse
from evaluator import SimpleValueEvaluator
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional, Set
import pyarrow as pa
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# Model configuration registry: maps (model_type, model_name) to checkpoint info.
# Only two variants: PI06 (single-timestep) and KAI0 (two-timestep stage-level).
# Update ckpt_dir / ckpt_steps to point to your trained Advantage Estimator checkpoints.
MODELS_CONFIG_MAP = {
    'Flatten-Fold': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            # 补训 base_cont_from30k @ 10000（自标 base 数据续训）
            'ckpt_dir': '/data3/kian_workspace/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/base_cont_from30k',
            'ckpt_steps': 10000
        },
    },
    'demo_A': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
    },
    'demo_B': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Advantage Estimator on a LeRobot dataset'
    )
    parser.add_argument('model_type', type=str, choices=['Flatten-Fold', 'demo_A', 'demo_B'],
                        help='Model type: Flatten-Fold / demo_A / demo_B')
    parser.add_argument('model_name', type=str, choices=['PI06', 'KAI0'],
                        help='Model variant: PI06 (single-timestep) / KAI0 (two-timestep stage-level)')
    parser.add_argument('repo_id', type=str,
                        help='Path to the LeRobot dataset')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Override checkpoint parent dir (contains <steps>/model.safetensors)')
    parser.add_argument('--ckpt_steps', type=int, default=None,
                        help='Override checkpoint step subdirectory')
    parser.add_argument('--output-dirname', type=str, default=None,
                        help='Override output label dirname under repo (default data_{name}_{steps})')
    parser.add_argument('--only-episodes', type=str, default=None,
                        help='Comma-separated episode indices to label (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete existing output parquet before writing')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Inference batch size (default: 400 for PI06, 96 for KAI0)')
    return parser.parse_args()


def _parse_only_episodes(raw: Optional[str]) -> Optional[Set[int]]:
    if not raw or not str(raw).strip():
        return None
    out: Set[int] = set()
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out or None


def edit_parquet_file(src_parquet: Path, output_path: Path, advantages_dict: Dict[str, list]):
    """Read source parquet, append predicted advantage columns, and write to output_path."""
    table = pq.read_table(src_parquet)
    advantages_table = pa.Table.from_pylist(advantages_dict)

    cols_to_add = ["relative_advantage", "absolute_value", "absolute_advantage"]
    new_columns = {}
    for col in cols_to_add:
        if col not in table.column_names and col in advantages_table.column_names:
            new_columns[col] = advantages_table[col]
    if new_columns:
        for name, column in new_columns.items():
            table = table.append_column(name, column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def main():
    args = parse_args()

    model_type = args.model_type
    model_name = args.model_name
    repo_id = Path(args.repo_id)

    # Look up model configuration
    if model_type not in MODELS_CONFIG_MAP:
        raise ValueError(f"Unknown model_type: {model_type}, available: {list(MODELS_CONFIG_MAP.keys())}")
    if model_name not in MODELS_CONFIG_MAP[model_type]:
        raise ValueError(f"Unknown model_name: {model_name}, available: {list(MODELS_CONFIG_MAP[model_type].keys())}")

    model_cfg = dict(MODELS_CONFIG_MAP[model_type][model_name])
    if args.ckpt_dir:
        model_cfg['ckpt_dir'] = args.ckpt_dir
    if args.ckpt_steps is not None:
        model_cfg['ckpt_steps'] = int(args.ckpt_steps)

    only_eps = _parse_only_episodes(args.only_episodes)
    out_dirname = args.output_dirname or f"data_{model_cfg['name']}_{model_cfg['ckpt_steps']}"

    print(f"Model type: {model_type}")
    print(f"Model name: {model_name}")
    print(f"ckpt_dir/steps: {model_cfg['ckpt_dir']}/{model_cfg['ckpt_steps']}")
    print(f"output_dirname: {out_dirname}")
    print(f"only_episodes: {sorted(only_eps) if only_eps else 'ALL'}")
    print(f"overwrite: {bool(args.overwrite)}")
    print(f"Using dataset at {repo_id}")

    # Interval for relative advantage computation (frames to look ahead)
    relative_interval = 50

    config_name = model_cfg['config_name']
    ckpt_dir = f"{model_cfg['ckpt_dir']}/{model_cfg['ckpt_steps']}"
    is_1timestep = (model_cfg['name'] == 'PI06')
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 400 if is_1timestep else 96
    print(f"batch_size: {batch_size}")

    # Initialize the evaluator
    evaluator = SimpleValueEvaluator(
        config_name=config_name,
        ckpt_dir=ckpt_dir,
        num_workers=64,  # Parallel threads for video loading; adjust based on CPU cores
    )

    dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=repo_id)
    if only_eps is not None:
        ep_indices = sorted(i for i in only_eps if 0 <= i < dataset_metadata.total_episodes)
    else:
        ep_indices = list(range(dataset_metadata.total_episodes))

    for i in tqdm(ep_indices, desc="Evaluating episodes"):
        parquet_file = repo_id / dataset_metadata.data_path.format(
            episode_chunk=i // dataset_metadata.chunks_size, episode_index=i
        )
        if not parquet_file.exists():
            print(f"Parquet file {parquet_file} not found, skipping")
            continue

        # Resolve video paths for all three camera views
        top_video = repo_id / dataset_metadata.video_path.format(
            episode_chunk=i // dataset_metadata.chunks_size, episode_index=i,
            video_key='observation.images.top_head'
        )
        left_video = repo_id / dataset_metadata.video_path.format(
            episode_chunk=i // dataset_metadata.chunks_size, episode_index=i,
            video_key='observation.images.hand_left'
        )
        right_video = repo_id / dataset_metadata.video_path.format(
            episode_chunk=i // dataset_metadata.chunks_size, episode_index=i,
            video_key='observation.images.hand_right'
        )
        if not top_video.exists() or not left_video.exists() or not right_video.exists():
            print(f"Missing video file(s) for episode {i}, skipping")
            continue

        video_paths = (top_video, left_video, right_video)

        # Read frame index range from parquet
        frame_indices = pq.read_table(parquet_file)['frame_index'].to_pylist()
        min_frame_index = frame_indices[0]
        max_frame_index = frame_indices[-1]

        # Output path: data_<model_name>_<ckpt_steps>/chunk-*/episode_*.parquet
        output_path = repo_id / out_dirname / parquet_file.relative_to(repo_id / "data")
        if output_path.exists():
            if args.overwrite:
                output_path.unlink()
                print(f"Removed existing {output_path}")
            else:
                print(f"Output {output_path} already exists, skipping")
                continue

        # Run inference
        if is_1timestep:
            results = evaluator.evaluate_video_1timestep_advantage(
                video_paths=video_paths,
                prompt="Flatten and fold the cloth.",
                batch_size=batch_size,
                frame_interval=1,       # 1 = evaluate every frame
                min_frame_index=min_frame_index,
                max_frame_index=max_frame_index,
                prefetch=True,
            )
        else:
            results = evaluator.evaluate_video_2timesteps_advantages(
                video_paths=video_paths,
                prompt="Flatten and fold the cloth.",
                batch_size=batch_size,
                frame_interval=1,       # 1 = evaluate every frame
                relative_interval=relative_interval,
                min_frame_index=min_frame_index,
                max_frame_index=max_frame_index,
                prefetch=True,
            )

        # Write results back as new parquet with advantage columns
        edit_parquet_file(
            src_parquet=parquet_file,
            output_path=output_path,
            advantages_dict=results,
        )


if __name__ == "__main__":
    main()
