"""Compute normalization statistics for a config - Fast Version.

This script uses a simplified approach to compute normalization statistics,
directly reading parquet files and accumulating data in memory, then computing 
statistics using the same RunningStats class as the org version for exact matching.

"""
import os
import numpy as np
from tqdm import tqdm
import tyro
import pandas as pd
from pathlib import Path

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms


def pad_to_dim(data, target_dim):
    """Pad data to target dimension."""
    if isinstance(data, list):
        data = np.array(data)
    data = np.asarray(data)
    
    if data.shape[-1] >= target_dim:
        return data[..., :target_dim]
    
    padding = np.zeros((*data.shape[:-1], target_dim - data.shape[-1]))
    return np.concatenate([data, padding], axis=-1)


def process_state(state, action_dim):
    """Process state following the FakeInputs logic."""
    # Pad to action dimension
    state = pad_to_dim(state, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    state = np.where(state > np.pi, 0, state)
    state = np.where(state < -np.pi, 0, state)
    
    return state


def process_actions(actions, action_dim):
    """Process actions following the FakeInputs logic."""
    # Pad to action dimension
    actions = pad_to_dim(actions, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    actions = np.where(actions > np.pi, 0, actions)
    actions = np.where(actions < -np.pi, 0, actions)
    
    # Return full action_dim dimensions (not trimmed to 14)
    return actions


def _mixture_dirs(data_config: _config.DataConfig, base_dir: str | None) -> list[tuple[str, float]]:
    """Return (data_dir, sample_weight) pairs for norm-stat computation.

    When config has `mixtures`, use those dirs and weights so stats match training mix.
    Otherwise fall back to a single `base_dir` / `repo_id`.
    """
    if base_dir is not None:
        return [(base_dir, 1.0)]
    if data_config.mixtures:
        return [(m.repo_id, float(m.weight)) for m in data_config.mixtures]
    if data_config.repo_id is None:
        raise ValueError("Either base_dir must be provided or config must have repo_id/mixtures")
    return [(data_config.repo_id, 1.0)]


def _weight_repeat(weight: float, weights: list[float]) -> int:
    """Integer repeat count so relative frequencies ≈ mixture weights."""
    scale = 1.0 / min(weights)
    return max(1, int(round(weight * scale)))


def main(config_name: str, base_dir: str | None = None, max_frames: int | None = None):
    """
    Compute normalization statistics for a config.
    
    Args:
        config_name: Name of the config to use
        base_dir: Base directory containing the data. If None, will be read from config
            (all `mixtures` repos when set, else `repo_id`).
        max_frames: Maximum number of frames to process. If None, processes all data.
    """
    # Load config
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    action_dim = config.model.action_dim

    dirs = _mixture_dirs(data_config, base_dir)
    for d, _ in dirs:
        if not Path(d).exists():
            raise ValueError(f"Base directory does not exist: {d}")

    weights = [w for _, w in dirs]
    print(f"Action dimension: {action_dim}")
    print("Reading data from:")
    for d, w in dirs:
        print(f"  {d}  (weight={w}, repeat={_weight_repeat(w, weights)})")

    # Keys to collect
    keys = ["state", "actions"]
    collected_data = {key: [] for key in keys}

    # Column names in the parquet files
    state_col = "observation.state"
    action_col = "action"

    total_frames = 0
    files_processed = 0
    hit_max = False

    for data_dir, weight in dirs:
        parquet_files = []
        for root, _dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        parquet_files.sort()
        repeat = _weight_repeat(weight, weights)
        print(f"\n[{data_dir}] {len(parquet_files)} parquet files, repeat x{repeat}")

        for file_path in tqdm(parquet_files, desc=Path(data_dir).name):
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

            if state_col not in df.columns or action_col not in df.columns:
                continue

            try:
                states_list = []
                actions_list = []

                for i in range(len(df)):
                    try:
                        state = process_state(np.array(df[state_col].iloc[i]), action_dim)
                        action = process_actions(np.array(df[action_col].iloc[i]), action_dim)
                        states_list.append(state)
                        actions_list.append(action)
                        total_frames += 1
                        if max_frames is not None and total_frames >= max_frames:
                            break
                    except Exception:
                        continue

                if states_list:
                    stacked_s = np.stack(states_list)
                    stacked_a = np.stack(actions_list)
                    for _ in range(repeat):
                        collected_data["state"].append(stacked_s)
                        collected_data["actions"].append(stacked_a)

                files_processed += 1
                if max_frames is not None and total_frames >= max_frames:
                    print(f"\nReached max_frames limit: {max_frames}")
                    hit_max = True
                    break
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue

        if hit_max:
            break

    print(f"\nProcessed {files_processed} files with {total_frames} unique frames "
          f"(mixture-weighted repeats applied before stats)")
    
    # Compute statistics using RunningStats (same as org version)
    print("\nComputing statistics...")
    print("Initializing RunningStats objects...")
    stats = {key: normalize.RunningStats() for key in keys}
    
    # Concatenate all data first
    print("Concatenating collected data...")
    all_data = {}
    for key in keys:
        if collected_data[key]:
            print(f"Concatenating {len(collected_data[key])} batches for '{key}'...")
            all_data[key] = np.concatenate(collected_data[key], axis=0)
            print(f"  Shape after concatenation: {all_data[key].shape}")
            
            # Ensure data is padded to action_dim
            if all_data[key].shape[-1] < action_dim:
                print(f"  Padding from {all_data[key].shape[-1]} to {action_dim} dimensions...")
                padding = np.zeros((*all_data[key].shape[:-1], action_dim - all_data[key].shape[-1]))
                all_data[key] = np.concatenate([all_data[key], padding], axis=-1)
                print(f"  Shape after padding: {all_data[key].shape}")
    
    # Feed data to RunningStats in fixed batches of 32 (same as org version)
    # This ensures identical floating-point accumulation
    batch_size = 32
    print("Feeding data to RunningStats...")
    for key in keys:
        if key in all_data:
            data = all_data[key]
            num_samples = len(data)
            num_batches = (num_samples + batch_size - 1) // batch_size
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing {key}", total=num_batches):
                batch = data[i:i+batch_size]
                stats[key].update(batch)
    
    # Get statistics from RunningStats
    norm_stats = {}
    for key in keys:
        if collected_data[key]:
            stat_result = stats[key].get_statistics()
            norm_stats[key] = stat_result
            
            print(f"\n{key} statistics:")
            print(f"  Mean: {stat_result.mean}")
            print(f"  Std: {stat_result.std}")
            print(f"  Q01: {stat_result.q01}")
            print(f"  Q99: {stat_result.q99}")
        else:
            print(f"Warning: No data collected for key '{key}'")
    
    # Save statistics
    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"✅ Normalization stats saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

