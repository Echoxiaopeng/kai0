"""Export a clean LeRobot dataset: merge advantage value columns, hard-drop fail episodes."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VALUE_COLS = ("absolute_value", "absolute_advantage", "relative_advantage")


def default_export_folder_name(dataset: Path, label_dirname: str) -> str:
    tag = label_dirname.replace("data_", "") if str(label_dirname).startswith("data_") else str(label_dirname or "label")
    return f"{dataset.name}__adv_{tag}_clean"


def default_export_out_dir(dataset: Path, label_dirname: str) -> Path:
    """Sibling of the source dataset: <parent>/<dataset>__adv_<tag>_clean."""
    return (dataset.resolve().parent / default_export_folder_name(dataset.resolve(), label_dirname)).resolve()


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _merge_value_parquet(src_parquet: Path, label_parquet: Path | None, out_parquet: Path, new_ep: int, global_start: int) -> int:
    table = pq.read_table(src_parquet)
    n = table.num_rows
    names = list(table.column_names)
    arrays = []
    for name in names:
        if name == "episode_index":
            arrays.append(pa.array([new_ep] * n, type=table.column(name).type))
        elif name == "index":
            arrays.append(pa.array(list(range(global_start, global_start + n)), type=table.column(name).type))
        elif name in VALUE_COLS:
            continue  # will re-append from label if present
        else:
            arrays.append(table.column(name))
    keep = [n for n in names if n not in VALUE_COLS]
    # rewrite episode_index / index already handled in arrays; keep names match
    keep_final = []
    arrays_final = []
    for name, arr in zip(keep, arrays):
        keep_final.append(name)
        arrays_final.append(arr)

    if label_parquet is not None and label_parquet.exists():
        lab = pq.read_table(label_parquet)
        if lab.num_rows != n:
            raise ValueError(f"row mismatch src={n} label={lab.num_rows} for {src_parquet.name}")
        for col in VALUE_COLS:
            if col in lab.column_names:
                keep_final.append(col)
                arrays_final.append(lab.column(col))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays_final, names=keep_final), out_parquet)
    return n


def export_clean_dataset(
    dataset: Path,
    label_dirname: str,
    reviews: dict,
    out_dir: Path | None = None,
    camera_keys: list[str] | None = None,
) -> dict:
    """Hard-exclude mark==fail episodes; write values into new dataset data/.

    reviews: advantage_review JSON episodes map {ep: {mark, ...}}
    """
    dataset = dataset.resolve()
    label_root = dataset / label_dirname
    if not (dataset / "data").is_dir():
        raise ValueError(f"missing data/: {dataset}")
    if not label_root.is_dir():
        raise ValueError(f"missing label dir: {label_root}")

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    chunks_size = int(info.get("chunks_size") or 1000)
    video_keys = camera_keys or [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]

    if out_dir is None:
        out_dir = default_export_out_dir(dataset, label_dirname)
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"out_dir already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted((dataset / "data").rglob("episode_*.parquet"))
    kept_map = []  # (old_ep, src_parquet, label_parquet)
    excluded = []
    for parquet in src_files:
        m = parquet.stem  # episode_000031
        try:
            ep_idx = int(m.split("_")[-1])
        except ValueError:
            continue
        mark = None
        rev = reviews.get(str(ep_idx)) if isinstance(reviews, dict) else None
        if isinstance(rev, dict):
            mark = str(rev.get("mark") or "").lower()
        if mark == "fail":
            excluded.append(ep_idx)
            continue
        # Require a label parquet for kept episodes
        rel = parquet.relative_to(dataset / "data")
        label_parquet = label_root / rel
        if not label_parquet.exists():
            # also try same episode name under any chunk
            cands = list(label_root.rglob(parquet.name))
            label_parquet = cands[0] if cands else None
        if label_parquet is None or not Path(label_parquet).exists():
            # no label → skip (cannot write values)
            excluded.append(ep_idx)
            continue
        kept_map.append((ep_idx, parquet, Path(label_parquet)))

    episodes_jsonl = []
    global_index = 0
    total_frames = 0
    video_mode_counts = {"hardlink": 0, "copy": 0, "missing": 0}

    for new_ep, (old_ep, src_pq, lab_pq) in enumerate(kept_map):
        chunk = new_ep // chunks_size
        chunk_name = f"chunk-{chunk:03d}"
        out_pq = out_dir / "data" / chunk_name / f"episode_{new_ep:06d}.parquet"
        n = _merge_value_parquet(src_pq, lab_pq, out_pq, new_ep, global_index)
        # videos
        src_chunk = src_pq.relative_to(dataset / "data").parts[0]
        for vk in video_keys:
            src_v = dataset / "videos" / src_chunk / vk / f"episode_{old_ep:06d}.mp4"
            dst_v = out_dir / "videos" / chunk_name / vk / f"episode_{new_ep:06d}.mp4"
            if src_v.exists():
                mode = _link_or_copy(src_v, dst_v)
                video_mode_counts[mode] = video_mode_counts.get(mode, 0) + 1
            else:
                video_mode_counts["missing"] += 1
        episodes_jsonl.append(
            {
                "episode_index": new_ep,
                "tasks": [],
                "length": n,
            }
        )
        # try preserve tasks from source episodes.jsonl if present
        global_index += n
        total_frames += n

    # meta
    meta_out = out_dir / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    # tasks.jsonl
    tasks_src = dataset / "meta" / "tasks.jsonl"
    if tasks_src.exists():
        shutil.copy2(tasks_src, meta_out / "tasks.jsonl")
    else:
        (meta_out / "tasks.jsonl").write_text('{"task_index": 0, "task": "Flatten and fold the cloth."}\n', encoding="utf-8")

    # enrich episodes.jsonl with task_index 0
    for row in episodes_jsonl:
        row.setdefault("tasks", [0])
        if not row["tasks"]:
            row["tasks"] = [0]

    _write_jsonl(meta_out / "episodes.jsonl", episodes_jsonl)

    new_info = dict(info) if info else {}
    new_info["total_episodes"] = len(kept_map)
    new_info["total_frames"] = total_frames
    new_info["total_chunks"] = max(1, (len(kept_map) + chunks_size - 1) // chunks_size) if kept_map else 0
    new_info["total_videos"] = len(kept_map) * len(video_keys)
    new_info["splits"] = {"train": f"0:{len(kept_map)}"}
    features = dict(new_info.get("features") or {})
    for col in VALUE_COLS:
        features[col] = {"dtype": "float32", "shape": [1], "names": None}
    new_info["features"] = features
    (meta_out / "info.json").write_text(json.dumps(new_info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    mapping = [{"old_episode_index": old, "new_episode_index": new} for new, (old, _, _) in enumerate(kept_map)]
    manifest = {
        "source_dataset": str(dataset),
        "label_dirname": label_dirname,
        "out_dir": str(out_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_kept": len(kept_map),
        "n_excluded": len(excluded),
        "excluded_episode_indices": excluded,
        "mapping": mapping,
        "video_mode_counts": video_mode_counts,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
