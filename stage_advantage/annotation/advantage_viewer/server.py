#!/usr/bin/env python3
"""Advantage review + stage_progress_gt boundary annotation viewer.

Usage:
  .venv/bin/python stage_advantage/annotation/advantage_viewer/server.py \
    --dataset /path/to/lerobot_dataset \
    --label-dirname data_KAI0_10000 \
    --port 18765

cd /data3/kian_workspace/kai0
.venv/bin/python -u stage_advantage/annotation/advantage_viewer/server.py \
  --dataset /data3/kian_workspace/RLinf/FlattenFold/flatten_fold_clothes_base_lerobot \
  --label-dirname data_KAI0_10000 \
  --host 0.0.0.0 --port 18765

"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from export_clean import default_export_out_dir, export_clean_dataset

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
ANN_NAME = "stage_boundary_annotations.json"
KAI0_ROOT = ROOT.parents[2]  # .../kai0
ANNOTATION_DIR = ROOT.parent  # .../stage_advantage/annotation
RELABEL_LOG_DIR = KAI0_ROOT / "experiment" / "ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD" / "log"

# Extra ckpt parents to scan for relabel model picker (in addition to MODELS_CONFIG_MAP).
EXTRA_CKPT_PARENTS = [
    Path("/data3/kian_workspace/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/base_cont_from30k"),
    Path("/data3/kian_workspace/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/task_a_adv_2gpu"),
]

# In-memory relabel job status (single job at a time is enough for the viewer).
_RELABEL_JOB: dict = {}


DEFAULT_ROOTS = [
    Path("/data3/kian_workspace/RLinf/FlattenFold"),
    Path("/data3/kian_workspace/kai0/data/Task_A"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Advantage / stage GT annotation viewer")
    p.add_argument("--dataset", type=Path, default=None, help="initial dataset")
    p.add_argument("--roots", type=Path, nargs="*", default=None, help="parent dirs to scan")
    p.add_argument("--label-dirname", type=str, default="data_KAI0_10000")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--camera", type=str, default="observation.images.top_head")
    return p.parse_args()


def is_lerobot_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "data").is_dir() and (path / "meta").is_dir()


def read_dataset_fps(dataset: Path, default: float = 30.0) -> float:
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        return default
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info.get("fps", default))
        return fps if fps > 0 else default
    except Exception:  # noqa: BLE001
        return default


def _dir_has_advantage_parquet(root: Path) -> bool:
    """True if this tree looks like advantage-labeled parquets."""
    sample = next(root.rglob("episode_*.parquet"), None)
    if sample is None:
        return False
    try:
        names = pq.read_schema(sample).names
    except Exception:  # noqa: BLE001
        return False
    return "absolute_value" in names or "absolute_advantage" in names


def discover_label_dirs(dataset: Path) -> list[str]:
    """Label sources for review mode: data_* dirs, and main data/ if it already has advantage cols.

    Official Task_A/advantage stores absolute_* inside data/ itself (no data_KAI0_* folder).
    """
    names: list[str] = []
    # Official / already-merged advantage datasets
    data_root = dataset / "data"
    if data_root.is_dir() and _dir_has_advantage_parquet(data_root):
        names.append("data")
    for p in sorted(dataset.iterdir()):
        if not (p.is_dir() and p.name.startswith("data_")):
            continue
        if next(p.rglob("episode_*.parquet"), None) is not None:
            names.append(p.name)
    return names


def count_episode_parquets(data_root: Path) -> int:
    if not data_root.is_dir():
        return 0
    return sum(1 for _ in data_root.rglob("episode_*.parquet"))


def discover_datasets(roots: list[Path], extra: Path | None = None) -> list[dict]:
    found: dict[str, Path] = {}
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        if is_lerobot_dataset(root):
            found[str(root)] = root
        for child in sorted(root.iterdir()):
            if is_lerobot_dataset(child):
                found[str(child.resolve())] = child.resolve()
    if extra is not None:
        extra = extra.resolve()
        if is_lerobot_dataset(extra):
            found[str(extra)] = extra
    out = []
    for path in sorted(found.values(), key=lambda p: (p.parent.name, p.name)):
        out.append(
            {
                "id": path.name,
                "path": str(path),
                "parent": str(path.parent),
                "n_source": count_episode_parquets(path / "data"),
                "label_dirs": discover_label_dirs(path),
                "has_videos": (path / "videos").is_dir(),
            }
        )
    return out


class AppState:
    def __init__(self, dataset: Path, label_dirname: str, camera: str, roots: list[Path]):
        self.dataset = dataset.resolve()
        self.label_dirname = label_dirname
        self.camera = camera
        self.roots = [r.resolve() for r in roots]

    def select(self, dataset_path: str | Path, label_dirname: str | None = None) -> None:
        path = Path(dataset_path).resolve()
        if not is_lerobot_dataset(path):
            raise ValueError(f"not a lerobot dataset: {path}")
        under_root = any(path == r or str(path).startswith(str(r) + "/") for r in self.roots)
        if not under_root and path != self.dataset:
            raise ValueError(f"dataset outside configured roots: {path}")
        self.dataset = path
        labels = discover_label_dirs(path)
        if label_dirname and label_dirname in labels:
            self.label_dirname = label_dirname
        elif self.label_dirname not in labels:
            # Prefer side-car data_KAI0_* if present, else main data/ with advantage cols.
            preferred = next((x for x in labels if x.startswith("data_")), None)
            self.label_dirname = preferred or (labels[0] if labels else self.label_dirname)


def episode_index_from_name(name: str) -> int | None:
    m = re.search(r"episode_(\d+)", name)
    return int(m.group(1)) if m else None


def ann_path(dataset: Path) -> Path:
    return dataset / ANN_NAME


def review_path(dataset: Path, label_dirname: str) -> Path:
    """Human pass/fail marks for a specific estimator label tree."""
    safe = re.sub(r"[^\w.\-]+", "_", label_dirname or "data")
    return dataset / f"advantage_review_{safe}.json"


def load_annotations(dataset: Path) -> dict:
    path = ann_path(dataset)
    if not path.exists():
        return {"episodes": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_annotations(dataset: Path, data: dict) -> None:
    path = ann_path(dataset)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_reviews(dataset: Path, label_dirname: str) -> dict:
    path = review_path(dataset, label_dirname)
    if not path.exists():
        return {"label_dirname": label_dirname, "episodes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"label_dirname": label_dirname, "episodes": {}}
    if not isinstance(data, dict):
        return {"label_dirname": label_dirname, "episodes": {}}
    data.setdefault("label_dirname", label_dirname)
    data.setdefault("episodes", {})
    if not isinstance(data["episodes"], dict):
        data["episodes"] = {}
    return data


def save_reviews(dataset: Path, label_dirname: str, data: dict) -> Path:
    path = review_path(dataset, label_dirname)
    payload = {
        "label_dirname": label_dirname,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "episodes": data.get("episodes", {}),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def upsert_review_mark(
    dataset: Path,
    label_dirname: str,
    episode_index: int,
    mark: str | None,
    note: str = "",
) -> dict:
    """Set or clear a human review mark. mark in {pass, fail, None/'' to clear}."""
    data = load_reviews(dataset, label_dirname)
    data.setdefault("episodes", {})
    key = str(int(episode_index))
    mark_norm = (mark or "").strip().lower()
    note = (note or "").strip()
    if mark_norm in ("", "clear", "none", "null"):
        data["episodes"].pop(key, None)
    else:
        if mark_norm not in ("pass", "fail"):
            raise ValueError(f"mark must be pass/fail/clear, got {mark!r}")
        data["episodes"][key] = {
            "mark": mark_norm,
            "note": note,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    path = save_reviews(dataset, label_dirname, data)
    return {"ok": True, "path": str(path), "episode": data["episodes"].get(key), "cleared": key not in data["episodes"]}


def _human_review_counts(dataset: Path, label_dirname: str) -> dict:
    eps = load_reviews(dataset, label_dirname).get("episodes", {})
    counts = {"pass": 0, "fail": 0, "total": 0}
    for v in eps.values():
        if not isinstance(v, dict):
            continue
        m = v.get("mark")
        if m in ("pass", "fail"):
            counts[m] += 1
            counts["total"] += 1
    return counts


def compute_stage_progress_gt(n_frames: int, splits: list[int]) -> list[float]:
    """Interpolate stage_progress_gt from boundary split frames.

    splits: frame indices where a NEW stage starts (exclusive of 0).
    Example 2-stage with split=400 on n=800 → stage0 [0,400), stage1 [400,800).
    """
    if n_frames <= 0:
        return []
    splits = sorted({int(s) for s in splits if 0 < int(s) < n_frames})
    edges = [0] + splits + [n_frames]
    k_stages = len(edges) - 1
    out = [0.0] * n_frames
    for k in range(k_stages):
        start, end = edges[k], edges[k + 1]
        length = end - start
        if length <= 0:
            continue
        for i, f in enumerate(range(start, end)):
            if length == 1:
                frac = 1.0
            else:
                frac = i / (length - 1)
            out[f] = k / k_stages + (1.0 / k_stages) * frac
    out[-1] = 1.0
    return out


def compute_progress_gt(n_frames: int) -> list[float]:
    if n_frames <= 1:
        return [0.0] * n_frames
    return [i / (n_frames - 1) for i in range(n_frames)]


def list_source_episodes(dataset: Path) -> list[dict]:
    data_root = dataset / "data"
    episodes = []
    if not data_root.exists():
        return episodes
    anns = load_annotations(dataset).get("episodes", {})
    for parquet in sorted(data_root.rglob("episode_*.parquet")):
        ep_idx = episode_index_from_name(parquet.name)
        if ep_idx is None:
            continue
        rel = parquet.relative_to(data_root)
        chunk = rel.parts[0] if len(rel.parts) > 1 else "chunk-000"
        try:
            pf = pq.ParquetFile(parquet)
            n = int(pf.metadata.num_rows)
            has_gt = "stage_progress_gt" in pf.schema_arrow.names
        except Exception:  # noqa: BLE001
            n = 0
            has_gt = False
        ann = anns.get(str(ep_idx), {})
        episodes.append(
            {
                "episode_index": ep_idx,
                "chunk": chunk,
                "n_frames": int(n),
                "parquet": str(parquet),
                "splits": ann.get("splits", []),
                "written": bool(ann.get("written", False)),
                "has_stage_progress_gt": has_gt,
                "status": (
                    "written"
                    if ann.get("written")
                    else ("draft" if ann.get("splits") else ("has_gt" if has_gt else "todo"))
                ),
            }
        )
    episodes.sort(key=lambda x: x["episode_index"])
    return episodes


def _verdict_from_curves(abs_v: np.ndarray, abs_a: np.ndarray) -> tuple[str, dict]:
    """Auto QC calibrated on human fail marks (base advantage_review_data_KAI0_10000).

    Fail episodes vs unmarked: lower/negative early, lower late, higher neg_frac,
    more late_neg, smaller delta/end_drop. Old thirds-only rule missed most fails.
    """
    n = len(abs_v)
    thirds = np.array_split(abs_v, 3)
    early, mid, late = [float(x.mean()) if len(x) else 0.0 for x in thirds]
    progress_delta = late - early
    mono = early < mid < late
    neg_frac = float(np.mean(abs_a < 0)) if len(abs_a) else 0.0
    end_n = max(1, n // 10)
    end_drop = float(abs_v[-end_n:].mean() - abs_v[:end_n].mean())
    feats = {
        "early_mean": round(early, 4),
        "mid_mean": round(mid, 4),
        "late_mean": round(late, 4),
        "adv_mean": round(float(abs_a.mean()) if len(abs_a) else 0.0, 4),
        "progress_delta": round(float(progress_delta), 4),
        "neg_frac": round(neg_frac, 4),
        "end_drop": round(end_drop, 4),
        "mono": bool(mono),
    }
    if (
        late < 0
        or early < -0.05
        or neg_frac > 0.32
        or (late < 0.35 and progress_delta < 0.40)
        or end_drop < 0.40
    ):
        verdict = "suspicious"
    elif mono and progress_delta > 0.40 and early > 0.10 and late > 0.50 and neg_frac < 0.25:
        verdict = "likely_ok"
    else:
        verdict = "weak_ok"
    return verdict, feats


def _summarize_labeled_parquet(parquet: Path, label_root: Path) -> dict | None:
    ep_idx = episode_index_from_name(parquet.name)
    if ep_idx is None:
        return None
    rel = parquet.relative_to(label_root)
    chunk = rel.parts[0] if len(rel.parts) > 1 else "chunk-000"
    try:
        try:
            table = pq.read_table(
                parquet, columns=["absolute_value", "absolute_advantage", "task_index"]
            )
        except Exception:
            table = pq.read_table(parquet, columns=["absolute_value", "absolute_advantage"])
        abs_v = table["absolute_value"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        abs_a = table["absolute_advantage"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        n = len(abs_v)
        if n == 0:
            return None
        verdict, feats = _verdict_from_curves(abs_v, abs_a)
        out = {
            "episode_index": ep_idx,
            "chunk": chunk,
            "n_frames": int(n),
            **feats,
            "verdict": verdict,
            "parquet": str(parquet),
            "has_task_index": False,
            "pos_frames": 0,
            "neg_frames": 0,
            "pos_pct": None,
            "pos_mean_adv": None,
            "neg_mean_adv": None,
        }
        if "task_index" in table.column_names:
            ti = table["task_index"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            pos = ti == 1
            neg = ti == 0
            n_pos = int(pos.sum())
            n_neg = int(neg.sum())
            out["has_task_index"] = True
            out["pos_frames"] = n_pos
            out["neg_frames"] = n_neg
            out["pos_pct"] = round(100.0 * n_pos / n, 1) if n else None
            out["pos_mean_adv"] = round(float(abs_a[pos].mean()), 4) if n_pos else None
            out["neg_mean_adv"] = round(float(abs_a[neg].mean()), 4) if n_neg else None
        return out
    except Exception as e:  # noqa: BLE001
        return {
            "episode_index": ep_idx,
            "chunk": chunk,
            "error": str(e),
            "verdict": "error",
            "parquet": str(parquet),
        }


def _qc_vs_human(labeled: list[dict]) -> dict:
    """Recall / false-positive of suspicious vs human marks (when present)."""
    fails = [e for e in labeled if e.get("human_mark") == "fail"]
    passes = [e for e in labeled if e.get("human_mark") == "pass"]
    non_fail_marked = passes  # only count explicit pass for FPR among marked
    unmarked = [e for e in labeled if not e.get("human_mark")]
    fail_sus = sum(1 for e in fails if e.get("verdict") == "suspicious")
    pass_sus = sum(1 for e in passes if e.get("verdict") == "suspicious")
    unmarked_sus = sum(1 for e in unmarked if e.get("verdict") == "suspicious")
    return {
        "n_fail": len(fails),
        "n_pass": len(passes),
        "n_unmarked": len(unmarked),
        "fail_suspicious": fail_sus,
        "fail_recall": round(fail_sus / len(fails), 4) if fails else None,
        "pass_suspicious": pass_sus,
        "unmarked_suspicious": unmarked_sus,
        "unmarked_suspicious_rate": round(unmarked_sus / len(unmarked), 4) if unmarked else None,
        "marked_pass_false_positive": pass_sus,
        "n_marked_pass": len(non_fail_marked),
    }


def list_labeled_episodes(dataset: Path, label_dirname: str) -> list[dict]:
    label_root = dataset / label_dirname
    if not label_root.exists():
        return []
    files = sorted(label_root.rglob("episode_*.parquet"))
    if not files:
        return []
    # Official Task_A has thousands of eps; parallelize IO-bound parquet reads.
    workers = min(16, max(4, (len(files) // 64) or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        episodes = [r for r in pool.map(lambda p: _summarize_labeled_parquet(p, label_root), files) if r]
    reviews = load_reviews(dataset, label_dirname).get("episodes", {})
    for e in episodes:
        rev = reviews.get(str(e["episode_index"]))
        mark = rev.get("mark") if isinstance(rev, dict) else None
        mark = mark if mark in ("pass", "fail") else None
        e["human_mark"] = mark
        e["excluded"] = mark == "fail"
    episodes.sort(key=lambda x: x["episode_index"])
    return episodes


def load_series_from_parquet(parquet_path: Path, columns: list[str]) -> dict:
    table = pq.read_table(parquet_path)
    cols = {}
    for name in columns:
        if name in table.column_names:
            cols[name] = [float(x) if x is not None else 0.0 for x in table[name].to_pylist()]
    n = table.num_rows
    if "frame_index" not in cols:
        cols["frame_index"] = list(range(n))
    cols["n_frames"] = n
    return cols


def _is_allowed_media_path(dataset: Path, file_path: Path) -> bool:
    """Allow files under dataset, or under a videos/ symlink target (e.g. *_awbc → clean)."""
    try:
        ds = dataset.resolve()
        fp = file_path.resolve()
    except OSError:
        return False
    ds_s = str(ds)
    fp_s = str(fp)
    if fp_s == ds_s or fp_s.startswith(ds_s + os.sep):
        return True
    videos = dataset / "videos"
    if videos.exists() or videos.is_symlink():
        try:
            vroot = videos.resolve()
            v_s = str(vroot)
            if fp_s == v_s or fp_s.startswith(v_s + os.sep):
                return True
        except OSError:
            pass
    return False


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _data_search_roots(dataset: Path, label_dirname: str | None = None) -> list[Path]:
    roots = [dataset / "data"]
    if label_dirname:
        roots.append(dataset / label_dirname)
    for p in dataset.iterdir():
        if p.is_dir() and p.name.startswith("data_") and p not in roots:
            roots.append(p)
    return [r for r in roots if r.is_dir()]


def _rewrite_parquet_episode_and_index(parquet_path: Path, new_ep: int, global_start: int) -> int:
    """Set episode_index / index columns; return number of rows."""
    table = pq.read_table(parquet_path)
    n = table.num_rows
    names = table.column_names
    arrays = []
    for name in names:
        if name == "episode_index":
            arrays.append(pa.array([new_ep] * n, type=table.column(name).type))
        elif name == "index":
            arrays.append(pa.array(list(range(global_start, global_start + n)), type=table.column(name).type))
        else:
            arrays.append(table.column(name))
    pq.write_table(pa.Table.from_arrays(arrays, names=names), parquet_path)
    return n


def _rename_episode_files(dataset: Path, old_ep: int, new_ep: int, search_roots: list[Path]) -> list[str]:
    """Rename parquet(+bak) and videos from old_ep -> new_ep. Caller must avoid collisions."""
    changed: list[str] = []
    old_stem = f"episode_{old_ep:06d}"
    new_stem = f"episode_{new_ep:06d}"

    for root in search_roots:
        for src in list(root.rglob(f"{old_stem}.parquet")):
            dst = src.with_name(f"{new_stem}.parquet")
            if dst.exists():
                raise FileExistsError(f"renumber collision: {dst}")
            bak = Path(str(src) + ".pre_stage_gt.bak")
            src.rename(dst)
            changed.append(f"{src.relative_to(dataset)} -> {dst.relative_to(dataset)}")
            if bak.exists():
                dst_bak = Path(str(dst) + ".pre_stage_gt.bak")
                bak.rename(dst_bak)
                changed.append(f"{bak.relative_to(dataset)} -> {dst_bak.relative_to(dataset)}")

    videos_root = dataset / "videos"
    if videos_root.is_dir():
        for src in list(videos_root.rglob(f"{old_stem}.mp4")):
            dst = src.with_name(f"{new_stem}.mp4")
            if dst.exists():
                raise FileExistsError(f"renumber collision: {dst}")
            src.rename(dst)
            changed.append(f"{src.relative_to(dataset)} -> {dst.relative_to(dataset)}")
    return changed


def _reindex_all_parquets(dataset: Path, search_roots: list[Path]) -> int:
    """Rewrite episode_index + contiguous global index for every episode parquet. Return total frames."""
    primary = dataset / "data"
    if not primary.is_dir():
        return 0
    ordered = sorted(
        primary.rglob("episode_*.parquet"),
        key=lambda p: episode_index_from_name(p.name) or -1,
    )
    total = 0
    cursor = 0
    for parquet in ordered:
        ep = episode_index_from_name(parquet.name)
        if ep is None:
            continue
        n = _rewrite_parquet_episode_and_index(parquet, ep, cursor)
        for root in search_roots:
            for path in root.rglob(f"episode_{ep:06d}.parquet"):
                if path.resolve() == parquet.resolve():
                    continue
                _rewrite_parquet_episode_and_index(path, ep, cursor)
            for bak in root.rglob(f"episode_{ep:06d}.parquet.pre_stage_gt.bak"):
                _rewrite_parquet_episode_and_index(bak, ep, cursor)
        cursor += n
        total += n
    return total


def delete_episode_from_dataset(dataset: Path, episode_index: int, label_dirname: str | None = None) -> dict:
    """Delete one episode, then shift later episodes down so indices stay contiguous 0..N-1."""
    ep = int(episode_index)
    stem = f"episode_{ep:06d}"
    deleted: list[str] = []
    renumbered: list[str] = []
    missing: list[str] = []
    search_roots = _data_search_roots(dataset, label_dirname)

    n_frames = 0
    for root in search_roots:
        for parquet in list(root.rglob(f"{stem}.parquet")):
            try:
                n_frames = max(n_frames, int(pq.ParquetFile(parquet).metadata.num_rows))
            except Exception:  # noqa: BLE001
                pass
            parquet.unlink(missing_ok=True)
            deleted.append(str(parquet.relative_to(dataset)))
            bak = Path(str(parquet) + ".pre_stage_gt.bak")
            if bak.exists():
                bak.unlink()
                deleted.append(str(bak.relative_to(dataset)))

    n_videos = 0
    videos_root = dataset / "videos"
    if videos_root.is_dir():
        for mp4 in list(videos_root.rglob(f"{stem}.mp4")):
            mp4.unlink(missing_ok=True)
            deleted.append(str(mp4.relative_to(dataset)))
            n_videos += 1

    if not deleted:
        missing.append(f"{stem}.*")

    # Discover remaining episode ids from primary data/
    primary = dataset / "data"
    remaining = sorted(
        {
            i
            for p in (primary.rglob("episode_*.parquet") if primary.is_dir() else [])
            for i in [episode_index_from_name(p.name)]
            if i is not None
        }
    )
    # Shift higher ids down via temp ids to avoid filename collisions
    to_shift = [i for i in remaining if i > ep]
    temp_base = 1_000_000
    for old_i in sorted(to_shift, reverse=True):
        _rename_episode_files(dataset, old_i, temp_base + old_i, search_roots)
    for old_i in sorted(to_shift):
        renumbered.extend(_rename_episode_files(dataset, temp_base + old_i, old_i - 1, search_roots))

    # Rewrite parquet internal episode_index + global index
    total_frames = _reindex_all_parquets(dataset, search_roots)

    # Annotations: drop deleted, shift keys > ep
    ann = load_annotations(dataset)
    old_eps = ann.get("episodes", {})
    new_eps: dict = {}
    for k, v in old_eps.items():
        try:
            ki = int(k)
        except ValueError:
            continue
        if ki == ep:
            continue
        new_k = ki - 1 if ki > ep else ki
        vv = dict(v)
        if isinstance(vv.get("parquet"), str):
            vv["parquet"] = re.sub(r"episode_\d{6}", f"episode_{new_k:06d}", vv["parquet"])
        new_eps[str(new_k)] = vv
    ann["episodes"] = new_eps
    save_annotations(dataset, ann)
    deleted.append(ANN_NAME)

    # Meta jsonl
    ep_meta = dataset / "meta" / "episodes.jsonl"
    ep_stats = dataset / "meta" / "episodes_stats.jsonl"
    info_path = dataset / "meta" / "info.json"
    removed_length = 0
    if ep_meta.exists():
        rows = _read_jsonl(ep_meta)
        keep = []
        for r in rows:
            ri = int(r.get("episode_index", -1))
            if ri == ep:
                removed_length = int(r.get("length") or n_frames or 0)
                continue
            if ri > ep:
                r = {**r, "episode_index": ri - 1}
            keep.append(r)
        keep.sort(key=lambda x: int(x.get("episode_index", 0)))
        _write_jsonl(ep_meta, keep)
        deleted.append("meta/episodes.jsonl")
    if ep_stats.exists():
        rows = []
        for r in _read_jsonl(ep_stats):
            ri = int(r.get("episode_index", -1))
            if ri == ep:
                continue
            if ri > ep:
                r = {**r, "episode_index": ri - 1}
            rows.append(r)
        rows.sort(key=lambda x: int(x.get("episode_index", 0)))
        _write_jsonl(ep_stats, rows)
        deleted.append("meta/episodes_stats.jsonl")

    total_eps = len(remaining)  # remaining was before rename but after delete; count is correct
    # remaining listed files after delete, before rename — same count
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_episodes"] = total_eps
        if total_frames > 0:
            info["total_frames"] = total_frames
        elif removed_length or n_frames:
            info["total_frames"] = max(0, int(info.get("total_frames", 0)) - (removed_length or n_frames))
        if n_videos:
            info["total_videos"] = max(0, int(info.get("total_videos", 0)) - n_videos)
        splits = info.get("splits")
        if isinstance(splits, dict) and "train" in splits:
            info["splits"]["train"] = f"0:{total_eps}"
        info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        deleted.append("meta/info.json")

    return {
        "episode_index": ep,
        "deleted": deleted,
        "renumbered": renumbered,
        "renumbered_count": len(to_shift),
        "missing": missing,
        "n_frames_removed": removed_length or n_frames,
        "n_videos_removed": n_videos,
        "total_episodes": total_eps,
        "total_frames": total_frames,
    }


def write_stage_gt_to_parquet(parquet_path: Path, stage_gt: list[float], progress_gt: list[float]) -> None:
    table = pq.read_table(parquet_path)
    n = table.num_rows
    if len(stage_gt) != n or len(progress_gt) != n:
        raise ValueError(f"length mismatch parquet={n} stage_gt={len(stage_gt)}")

    # backup once
    bak = parquet_path.with_suffix(parquet_path.suffix + ".pre_stage_gt.bak")
    if not bak.exists():
        shutil.copy2(parquet_path, bak)

    names = table.column_names
    arrays = []
    for name in names:
        if name in ("stage_progress_gt", "progress_gt"):
            continue
        arrays.append(table.column(name))
    keep_names = [n for n in names if n not in ("stage_progress_gt", "progress_gt")]
    keep_names.extend(["progress_gt", "stage_progress_gt"])
    arrays.append(pa.array(progress_gt, type=pa.float32()))
    arrays.append(pa.array(stage_gt, type=pa.float32()))
    new_table = pa.Table.from_arrays(arrays, names=keep_names)
    pq.write_table(new_table, parquet_path)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def _load_models_config_map() -> dict:
    """Lightweight mirror of eval.MODELS_CONFIG_MAP (avoid importing heavy eval deps)."""
    return {
        "Flatten-Fold": {
            "KAI0": {
                "name": "KAI0",
                "config_name": "ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",
                "ckpt_dir": str(EXTRA_CKPT_PARENTS[0]),
                "ckpt_steps": 10000,
            },
            "PI06": {
                "name": "PI06",
                "config_name": "ADVANTAGE_TORCH_PI06_FLATTEN_FOLD",
                "ckpt_dir": str(KAI0_ROOT / "experiment" / "ADVANTAGE_TORCH_PI06_FLATTEN_FOLD" / "run1"),
                "ckpt_steps": 100000,
            },
        },
    }


def _scan_ckpt_steps(ckpt_dir: Path) -> list[int]:
    if not ckpt_dir.is_dir():
        return []
    steps = []
    for child in ckpt_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "model.safetensors").exists() or (child / "model.pt").exists():
            steps.append(int(child.name))
    return sorted(steps)


def list_relabel_models() -> list[dict]:
    """Build picker list: registry entries × scanned steps + EXTRA_CKPT_PARENTS."""
    cfg_map = _load_models_config_map()
    out: list[dict] = []
    seen: set[tuple] = set()

    def add(model_type: str, model_name: str, ckpt_dir: str, ckpt_steps: int, config_name: str):
        key = (model_type, model_name, str(Path(ckpt_dir).resolve()) if Path(ckpt_dir).exists() else ckpt_dir, int(ckpt_steps))
        if key in seen:
            return
        seen.add(key)
        weights = Path(ckpt_dir) / str(ckpt_steps)
        has = (weights / "model.safetensors").exists() or (weights / "model.pt").exists()
        out.append(
            {
                "model_type": model_type,
                "model_name": model_name,
                "ckpt_dir": str(ckpt_dir),
                "ckpt_steps": int(ckpt_steps),
                "config_name": config_name,
                "label": f"{model_type}/{model_name}@{ckpt_steps}",
                "has_weights": has,
            }
        )

    for model_type, variants in cfg_map.items():
        for model_name, cfg in variants.items():
            ckpt_dir = cfg.get("ckpt_dir")
            config_name = cfg.get("config_name", "")
            name = cfg.get("name", model_name)
            steps = _scan_ckpt_steps(Path(ckpt_dir)) if ckpt_dir else []
            if not steps and cfg.get("ckpt_steps") is not None:
                steps = [int(cfg["ckpt_steps"])]
            for st in steps:
                add(model_type, name, ckpt_dir, st, config_name)

    for parent in EXTRA_CKPT_PARENTS:
        for st in _scan_ckpt_steps(parent):
            add("Flatten-Fold", "KAI0", str(parent), st, "ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD")

    out.sort(key=lambda x: (x["model_type"], x["model_name"], x["ckpt_dir"], x["ckpt_steps"]))
    return out


def _episodes_for_relabel_scope(dataset: Path, label_dirname: str, scope: str) -> list[int]:
    labeled = list_labeled_episodes(dataset, label_dirname)
    scope = (scope or "fail").strip().lower()
    if scope == "all":
        return [int(e["episode_index"]) for e in labeled]
    if scope == "suspicious":
        return [int(e["episode_index"]) for e in labeled if e.get("verdict") == "suspicious" or e.get("human_mark") == "fail"]
    # default fail
    return [int(e["episode_index"]) for e in labeled if e.get("human_mark") == "fail"]


def list_gpus(min_free_mib: int = 35000) -> dict:
    """Query nvidia-smi; mark GPUs with enough free memory as idle/recommended."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        return {"gpus": [], "error": str(e), "min_free_mib": min_free_mib}

    gpus = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            used = int(float(parts[2]))
            total = int(float(parts[3]))
            free = int(float(parts[4]))
            util = int(float(parts[5]))
        except ValueError:
            continue
        idle = free >= min_free_mib
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": free,
                "utilization": util,
                "idle": idle,
                "label": f"GPU {idx} · free {free}MiB / {total}MiB · util {util}%{' · 空闲' if idle else ''}",
            }
        )
    idle_idxs = [g["index"] for g in gpus if g["idle"]]
    return {
        "gpus": gpus,
        "idle": idle_idxs,
        "recommended": idle_idxs[0] if idle_idxs else (gpus[0]["index"] if gpus else None),
        "min_free_mib": min_free_mib,
    }


def _parse_tqdm_progress(log_text: str) -> dict | None:
    """Parse last tqdm-like '12/34' or '45%' from eval log."""
    # Prefer explicit fraction near the end.
    frac = None
    pct = None
    for line in reversed(log_text.splitlines()[-80:]):
        m = re.search(r"(\d+)\s*/\s*(\d+)", line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total > 0 and done <= total * 2:  # sanity
                frac = (done, total)
                break
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        if m2 and pct is None:
            pct = float(m2.group(1))
    if frac:
        done, total = frac
        return {"done": done, "total": total, "percent": round(100.0 * done / total, 1) if total else 0.0}
    if pct is not None:
        return {"done": None, "total": None, "percent": pct}
    return None


def _count_relabel_done(dataset: Path, output_dirname: str, episode_indices: list[int] | None) -> int:
    out_root = dataset / output_dirname
    if not out_root.is_dir():
        return 0
    if episode_indices:
        want = {f"episode_{int(i):06d}.parquet" for i in episode_indices}
        return sum(1 for p in out_root.rglob("episode_*.parquet") if p.name in want)
    return sum(1 for _ in out_root.rglob("episode_*.parquet"))


def start_relabel_job(
    dataset: Path,
    *,
    model_type: str,
    model_name: str,
    ckpt_dir: str | None,
    ckpt_steps: int | None,
    scope: str,
    output_dirname: str | None,
    gpu: str,
    overwrite: bool,
    label_dirname: str,
) -> dict:
    global _RELABEL_JOB
    if _RELABEL_JOB.get("pid"):
        old_pid = int(_RELABEL_JOB["pid"])
        if _pid_is_running(old_pid):
            raise RuntimeError(f"relabel already running pid={old_pid}")
        try:
            os.waitpid(old_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        _RELABEL_JOB = {}

    eps = _episodes_for_relabel_scope(dataset, label_dirname, scope)
    if not eps and scope != "all":
        raise ValueError(f"no episodes for scope={scope}")

    steps = int(ckpt_steps) if ckpt_steps is not None else None
    out_dirname = output_dirname or (
        f"data_{model_name}_{steps}_relabel" if steps is not None else f"data_{model_name}_relabel"
    )
    RELABEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = RELABEL_LOG_DIR / f"relabel_{dataset.name}_{job_id}.log"
    pid_path = RELABEL_LOG_DIR / f"relabel_{dataset.name}_{job_id}.pid"

    py = KAI0_ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    cmd = [
        str(py),
        "-u",
        str(ANNOTATION_DIR / "eval.py"),
        model_type,
        model_name,
        str(dataset),
    ]
    if ckpt_dir:
        cmd += ["--ckpt_dir", str(ckpt_dir)]
    if steps is not None:
        cmd += ["--ckpt_steps", str(steps)]
    cmd += ["--output-dirname", out_dirname]
    if eps:
        cmd += ["--only-episodes", ",".join(str(i) for i in eps)]
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    # Reduce fragmentation OOMs when co-tenant jobs leave ~50–60GB free.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("HF_HOME", str(KAI0_ROOT / ".cache" / "hf"))
    env.setdefault("HF_DATASETS_CACHE", str(KAI0_ROOT / ".cache" / "hf" / "datasets"))
    env.setdefault("XDG_CACHE_HOME", str(KAI0_ROOT / ".cache" / "xdg"))

    # Slightly smaller default batch than standalone labeling to fit shared GPUs.
    if "--batch-size" not in cmd:
        cmd += ["--batch-size", "80"]

    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("CMD: " + " ".join(cmd) + "\n")
    log_f.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
    log_f.write(f"PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ANNOTATION_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    total = len(eps) if eps else None
    _RELABEL_JOB = {
        "job_id": job_id,
        "pid": proc.pid,
        "log": str(log_path),
        "pid_file": str(pid_path),
        "dataset": str(dataset),
        "output_dirname": out_dirname,
        "n_episodes": total if total is not None else "all",
        "total": total,
        "episode_indices": eps,
        "scope": scope,
        "model": {
            "model_type": model_type,
            "model_name": model_name,
            "ckpt_dir": ckpt_dir,
            "ckpt_steps": steps,
        },
        "gpu": str(gpu),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_log_handle": log_f,  # keep open
    }
    return {k: v for k, v in _RELABEL_JOB.items() if not k.startswith("_")}


def _pid_is_running(pid: int) -> bool:
    """True if pid exists and is not a zombie."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # field 3 is state: R/S/D/... ; Z = zombie
        state = stat_path.read_text(encoding="utf-8").split()[2]
        if state == "Z":
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def stop_relabel_job() -> dict:
    """Stop the current relabel process group (SIGTERM then SIGKILL)."""
    global _RELABEL_JOB
    job = _RELABEL_JOB or {}
    pid = int(job.get("pid") or 0)
    if not pid:
        return {"ok": True, "stopped": False, "reason": "no job"}
    was_running = _pid_is_running(pid)
    log_path = job.get("log")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[viewer] stop requested at {datetime.now(timezone.utc).isoformat()}\n")
        except Exception:  # noqa: BLE001
            pass

    def _kill(sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        except PermissionError:
            os.kill(pid, sig)

    if was_running:
        _kill(signal.SIGTERM)
        deadline = time.time() + 8
        while time.time() < deadline and _pid_is_running(pid):
            time.sleep(0.25)
        if _pid_is_running(pid):
            _kill(signal.SIGKILL)
            time.sleep(0.3)
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    handle = job.get("_log_handle")
    if handle:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    _RELABEL_JOB = {
        **{k: v for k, v in job.items() if not k.startswith("_")},
        "stopped": True,
        "stopped_at": datetime.now(timezone.utc).isoformat(),
        "_log_handle": None,
    }
    still = _pid_is_running(pid)
    return {
        "ok": True,
        "stopped": True,
        "was_running": was_running,
        "still_running": still,
        "pid": pid,
        "output_dirname": job.get("output_dirname"),
        "log": job.get("log"),
    }


def relabel_job_status() -> dict:
    job = {k: v for k, v in _RELABEL_JOB.items() if not k.startswith("_")} if _RELABEL_JOB else {}
    if not job:
        return {"running": False, "done": 0, "total": 0, "percent": 0.0}
    pid = int(job.get("pid") or 0)
    alive = _pid_is_running(pid)
    # Reap zombie children if any (viewer is parent of Popen).
    if not alive and pid:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    tail = ""
    log_text = ""
    log_path = job.get("log")
    if log_path and Path(log_path).exists():
        try:
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            lines = log_text.splitlines()
            tail = "\n".join(lines[-40:])
        except Exception:  # noqa: BLE001
            tail = ""

    eps = job.get("episode_indices") or []
    total = job.get("total")
    if total is None and isinstance(job.get("n_episodes"), int):
        total = job["n_episodes"]
    done = 0
    ds = job.get("dataset")
    out_dirname = job.get("output_dirname")
    if ds and out_dirname:
        try:
            done = _count_relabel_done(Path(ds), out_dirname, list(eps) if eps else None)
        except Exception:  # noqa: BLE001
            done = 0

    tqdm_prog = _parse_tqdm_progress(log_text) if log_text else None
    # Prefer on-disk parquet count; only use tqdm when it matches job total.
    if done == 0 and tqdm_prog and tqdm_prog.get("done") is not None:
        t_done = int(tqdm_prog["done"])
        t_total = tqdm_prog.get("total")
        if total and t_total and int(t_total) == int(total):
            done = t_done
        elif not total and t_total:
            done = t_done
            total = int(t_total)
    percent = 0.0
    if total:
        percent = round(100.0 * min(done, total) / total, 1)
    elif tqdm_prog and tqdm_prog.get("percent") is not None:
        percent = float(tqdm_prog["percent"])

    last_line = ""
    if tail:
        for line in reversed(tail.splitlines()):
            if line.strip():
                last_line = line.strip()
                break

    failed = (not alive) and ("OutOfMemoryError" in log_text or "Traceback" in log_text)
    stopped = bool(job.get("stopped")) and not alive

    return {
        **job,
        "running": alive,
        "failed": failed,
        "stopped": stopped,
        "done": done,
        "total": total,
        "percent": percent,
        "last_line": last_line,
        "log_tail": tail,
    }


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.path.startswith("/api/") or self.path in ("/", "/index.html"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _meta_payload(self):
            dataset = state.dataset
            labeled = list_labeled_episodes(dataset, state.label_dirname)
            source = list_source_episodes(dataset)
            n_excluded = sum(1 for e in labeled if e.get("excluded"))
            return {
                "dataset": str(dataset),
                "dataset_id": dataset.name,
                "label_dirname": state.label_dirname,
                "label_dirs": discover_label_dirs(dataset),
                "camera": state.camera,
                "fps": read_dataset_fps(dataset),
                "n_labeled": len(labeled),
                "n_source": len(source),
                "n_excluded": n_excluded,
                "n_kept": len(labeled) - n_excluded,
                "n_annotated_draft": sum(1 for e in source if e["status"] == "draft"),
                "n_annotated_written": sum(1 for e in source if e["status"] == "written"),
                "verdict_counts": {
                    k: sum(1 for e in labeled if e.get("verdict") == k)
                    for k in ("likely_ok", "weak_ok", "suspicious", "error")
                },
                "qc_vs_human": _qc_vs_human(labeled),
                "annotations_file": str(ann_path(dataset)),
                "review_file": str(review_path(dataset, state.label_dirname)),
                "human_review_counts": _human_review_counts(dataset, state.label_dirname),
                "roots": [str(r) for r in state.roots],
                "export_default_out_dir": str(default_export_out_dir(dataset, state.label_dirname)),
                "export_default_folder_name": default_export_out_dir(dataset, state.label_dirname).name,
            }

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                body = _read_json_body(self)
            except Exception as e:  # noqa: BLE001
                self._json(400, {"error": f"invalid json: {e}"})
                return

            if path == "/api/select":
                try:
                    ds = body.get("dataset") or body.get("path")
                    if not ds:
                        raise ValueError("dataset required")
                    state.select(ds, body.get("label_dirname"))
                    if body.get("camera"):
                        state.camera = str(body["camera"])
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, {"ok": True, **self._meta_payload()})
                return

            dataset = state.dataset

            if path == "/api/annotate/save":
                ep_idx = int(body["episode_index"])
                splits = sorted({int(x) for x in body.get("splits", [])})
                n_frames = int(body.get("n_frames") or 0)
                data = load_annotations(dataset)
                data.setdefault("episodes", {})
                prev = data["episodes"].get(str(ep_idx), {})
                data["episodes"][str(ep_idx)] = {
                    "splits": splits,
                    "n_frames": n_frames or prev.get("n_frames"),
                    "written": False,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                save_annotations(dataset, data)
                preview = compute_stage_progress_gt(n_frames, splits) if n_frames else []
                self._json(200, {"ok": True, "episode_index": ep_idx, "splits": splits, "preview": preview})
                return

            if path == "/api/annotate/write":
                ep_idx = int(body["episode_index"])
                splits = sorted({int(x) for x in body.get("splits", [])})
                src_eps = {e["episode_index"]: e for e in list_source_episodes(dataset)}
                if ep_idx not in src_eps:
                    self._json(404, {"error": f"episode {ep_idx} not found"})
                    return
                meta = src_eps[ep_idx]
                n = meta["n_frames"]
                if not splits:
                    self._json(400, {"error": "需要至少一个分界帧（2 段任务点一次折叠开始帧）"})
                    return
                stage_gt = compute_stage_progress_gt(n, splits)
                prog_gt = compute_progress_gt(n)
                write_stage_gt_to_parquet(Path(meta["parquet"]), stage_gt, prog_gt)
                data = load_annotations(dataset)
                data.setdefault("episodes", {})
                data["episodes"][str(ep_idx)] = {
                    "splits": splits,
                    "n_frames": n,
                    "written": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "parquet": meta["parquet"],
                }
                save_annotations(dataset, data)
                self._json(
                    200,
                    {
                        "ok": True,
                        "episode_index": ep_idx,
                        "splits": splits,
                        "parquet": meta["parquet"],
                        "backup": meta["parquet"] + ".pre_stage_gt.bak",
                        "stage_progress_gt_preview": stage_gt[:: max(1, n // 20)],
                    },
                )
                return

            if path == "/api/annotate/clear":
                ep_idx = int(body["episode_index"])
                data = load_annotations(dataset)
                data.get("episodes", {}).pop(str(ep_idx), None)
                save_annotations(dataset, data)
                self._json(200, {"ok": True})
                return

            if path == "/api/review/mark":
                try:
                    ep_idx = int(body["episode_index"])
                    mark = body.get("mark")
                    note = body.get("note", "")
                    result = upsert_review_mark(dataset, state.label_dirname, ep_idx, mark, note)
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"error": str(e)})
                    return
                self._json(
                    200,
                    {
                        **result,
                        "label_dirname": state.label_dirname,
                        "human_review_counts": _human_review_counts(dataset, state.label_dirname),
                    },
                )
                return

            if path == "/api/review/import_local":
                # One-shot migrate: merge browser localStorage dumps into the json file.
                # body: {episodes: {ep: {mark, note, ts?}}}
                incoming = body.get("episodes") or {}
                if not isinstance(incoming, dict):
                    self._json(400, {"error": "episodes must be an object"})
                    return
                data = load_reviews(dataset, state.label_dirname)
                data.setdefault("episodes", {})
                imported = 0
                for k, v in incoming.items():
                    if not isinstance(v, dict):
                        continue
                    mark = str(v.get("mark", "")).strip().lower()
                    if mark not in ("pass", "fail"):
                        continue
                    key = str(int(k))
                    if key in data["episodes"]:
                        continue  # keep existing server marks
                    data["episodes"][key] = {
                        "mark": mark,
                        "note": str(v.get("note") or "").strip(),
                        "ts": v.get("ts") or datetime.now(timezone.utc).isoformat(),
                        "imported_from": "localStorage",
                    }
                    imported += 1
                path_out = save_reviews(dataset, state.label_dirname, data)
                self._json(
                    200,
                    {
                        "ok": True,
                        "imported": imported,
                        "path": str(path_out),
                        "human_review_counts": _human_review_counts(dataset, state.label_dirname),
                    },
                )
                return

            if path == "/api/episode/delete":
                try:
                    ep_idx = int(body["episode_index"])
                except Exception:  # noqa: BLE001
                    self._json(400, {"error": "episode_index required"})
                    return
                try:
                    result = delete_episode_from_dataset(dataset, ep_idx, state.label_dirname)
                except Exception as e:  # noqa: BLE001
                    self._json(500, {"error": str(e)})
                    return
                self._json(200, {"ok": True, **result, **self._meta_payload()})
                return

            if path == "/api/export_clean_dataset":
                try:
                    out_dir = body.get("out_dir") or None
                    out_name = (body.get("out_name") or body.get("folder_name") or "").strip()
                    label = body.get("label_dirname") or state.label_dirname
                    if out_dir:
                        resolved_out = Path(out_dir).expanduser()
                    elif out_name:
                        # Folder name only → place as sibling of current dataset.
                        name = Path(out_name).name  # prevent path traversal via parent dirs
                        if not name or name in (".", ".."):
                            raise ValueError("invalid out_name")
                        resolved_out = dataset.parent / name
                    else:
                        resolved_out = None
                    reviews = load_reviews(dataset, label).get("episodes", {})
                    manifest = export_clean_dataset(
                        dataset,
                        label,
                        reviews,
                        out_dir=resolved_out,
                    )
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, {"ok": True, **manifest})
                return

            if path == "/api/relabel":
                try:
                    model_type = str(body.get("model_type") or "Flatten-Fold")
                    model_name = str(body.get("model_name") or "KAI0")
                    ckpt_dir = body.get("ckpt_dir")
                    ckpt_steps = body.get("ckpt_steps")
                    if ckpt_steps is not None:
                        ckpt_steps = int(ckpt_steps)
                    scope = str(body.get("scope") or "fail")
                    output_dirname = body.get("output_dirname")
                    gpu = str(body.get("gpu") if body.get("gpu") is not None else "0")
                    overwrite = bool(body.get("overwrite") or False)
                    result = start_relabel_job(
                        dataset,
                        model_type=model_type,
                        model_name=model_name,
                        ckpt_dir=ckpt_dir,
                        ckpt_steps=ckpt_steps,
                        scope=scope,
                        output_dirname=output_dirname,
                        gpu=gpu,
                        overwrite=overwrite,
                        label_dirname=state.label_dirname,
                    )
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, {"ok": True, **result})
                return

            if path == "/api/relabel/stop":
                try:
                    result = stop_relabel_job()
                except Exception as e:  # noqa: BLE001
                    self._json(500, {"error": str(e)})
                    return
                self._json(200, {**result, **relabel_job_status()})
                return

            self._json(404, {"error": "unknown post path", "path": path})

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query)
            dataset = state.dataset
            label_dirname = state.label_dirname
            camera = state.camera

            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                return

            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                file_path = (STATIC / rel).resolve()
                if not str(file_path).startswith(str(STATIC.resolve())) or not file_path.exists():
                    self._json(404, {"error": "not found"})
                    return
                mime, _ = mimetypes.guess_type(str(file_path))
                self._send(200, file_path.read_bytes(), mime or "application/octet-stream")
                return

            if path == "/api/datasets":
                self._json(
                    200,
                    {
                        "datasets": discover_datasets(state.roots, extra=state.dataset),
                        "current": str(state.dataset),
                        "label_dirname": state.label_dirname,
                    },
                )
                return

            if path == "/api/meta":
                self._json(200, self._meta_payload())
                return

            if path == "/api/review":
                data = load_reviews(dataset, label_dirname)
                self._json(
                    200,
                    {
                        "label_dirname": label_dirname,
                        "path": str(review_path(dataset, label_dirname)),
                        "episodes": data.get("episodes", {}),
                        "human_review_counts": _human_review_counts(dataset, label_dirname),
                    },
                )
                return

            if path == "/api/episodes":
                self._json(200, {"episodes": list_labeled_episodes(dataset, label_dirname)})
                return

            if path == "/api/relabel/models":
                models = list_relabel_models()
                default = next(
                    (m for m in models if m["model_name"] == "KAI0" and m["ckpt_steps"] == 10000 and m.get("has_weights")),
                    models[0] if models else None,
                )
                self._json(200, {"models": models, "default": default})
                return

            if path == "/api/relabel/status":
                self._json(200, relabel_job_status())
                return

            if path == "/api/gpus":
                try:
                    min_free = int(qs.get("min_free_mib", ["35000"])[0])
                except ValueError:
                    min_free = 35000
                self._json(200, list_gpus(min_free_mib=min_free))
                return

            if path == "/api/annotate/episodes":
                self._json(200, {"episodes": list_source_episodes(dataset)})
                return

            if path == "/api/annotate/preview":
                try:
                    ep_idx = int(qs.get("episode_index", [""])[0])
                    n_frames = int(qs.get("n_frames", ["0"])[0])
                except ValueError:
                    self._json(400, {"error": "bad params"})
                    return
                splits_raw = qs.get("splits", [""])[0]
                splits = [int(x) for x in splits_raw.split(",") if x.strip().isdigit()]
                if n_frames <= 0:
                    src = {e["episode_index"]: e for e in list_source_episodes(dataset)}
                    n_frames = src.get(ep_idx, {}).get("n_frames", 0)
                preview = compute_stage_progress_gt(n_frames, splits)
                self._json(200, {"episode_index": ep_idx, "splits": splits, "n_frames": n_frames, "preview": preview})
                return

            if path == "/api/episode":
                try:
                    ep_idx = int(qs.get("episode_index", [""])[0])
                except ValueError:
                    self._json(400, {"error": "episode_index required"})
                    return
                mode = qs.get("mode", ["review"])[0]
                if mode == "annotate":
                    src = {e["episode_index"]: e for e in list_source_episodes(dataset)}
                    if ep_idx not in src:
                        self._json(404, {"error": f"episode {ep_idx} not found"})
                        return
                    meta = src[ep_idx]
                    series = load_series_from_parquet(
                        Path(meta["parquet"]),
                        ["frame_index", "stage_progress_gt", "progress_gt"],
                    )
                    splits = meta.get("splits") or []
                    preview = compute_stage_progress_gt(meta["n_frames"], splits)
                    video_rel = f"videos/{meta['chunk']}/{camera}/episode_{ep_idx:06d}.mp4"
                    video_path = dataset / video_rel
                    self._json(
                        200,
                        {
                            "mode": "annotate",
                            "meta": meta,
                            "series": series,
                            "preview_stage_progress_gt": preview,
                            "video_url": f"/video/{video_rel}" if video_path.exists() else None,
                            "video_exists": video_path.exists(),
                        },
                    )
                    return

                eps = {e["episode_index"]: e for e in list_labeled_episodes(dataset, label_dirname)}
                if ep_idx not in eps:
                    src = {e["episode_index"]: e for e in list_source_episodes(dataset)}
                    if ep_idx not in src:
                        self._json(404, {"error": f"episode {ep_idx} not found"})
                        return
                    meta = src[ep_idx]
                    series = {"n_frames": meta["n_frames"], "frame_index": list(range(meta["n_frames"]))}
                    video_rel = f"videos/{meta['chunk']}/{camera}/episode_{ep_idx:06d}.mp4"
                    video_path = dataset / video_rel
                    self._json(
                        200,
                        {
                            "mode": "review",
                            "meta": {**meta, "verdict": "unlabeled"},
                            "series": series,
                            "video_url": f"/video/{video_rel}" if video_path.exists() else None,
                            "video_exists": video_path.exists(),
                        },
                    )
                    return
                meta = eps[ep_idx]
                series = load_series_from_parquet(
                    Path(meta["parquet"]),
                    [
                        "frame_index",
                        "absolute_value",
                        "absolute_advantage",
                        "relative_advantage",
                        "task_index",
                    ],
                )
                video_rel = f"videos/{meta['chunk']}/{camera}/episode_{ep_idx:06d}.mp4"
                video_path = dataset / video_rel
                self._json(
                    200,
                    {
                        "mode": "review",
                        "meta": meta,
                        "series": series,
                        "video_url": f"/video/{video_rel}" if video_path.exists() else None,
                        "video_exists": video_path.exists(),
                    },
                )
                return

            if path.startswith("/video/"):
                rel = path[len("/video/") :]
                # Resolve after join so symlink videos/ (awbc → clean) still works.
                candidate = dataset / rel
                try:
                    file_path = candidate.resolve()
                except OSError:
                    self._json(404, {"error": "video not found"})
                    return
                if not file_path.exists() or not _is_allowed_media_path(dataset, file_path):
                    self._json(404, {"error": "video not found", "rel": rel})
                    return
                mime, _ = mimetypes.guess_type(str(file_path))
                mime = mime or "video/mp4"
                file_size = file_path.stat().st_size
                range_header = self.headers.get("Range")
                if range_header:
                    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
                    if not m:
                        self.send_error(400, "invalid range")
                        return
                    start_s, end_s = m.group(1), m.group(2)
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end:
                        self.send_error(416, "invalid range")
                        return
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", mime)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with file_path.open("rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with file_path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return

            self._json(404, {"error": "unknown path", "path": path})

    return Handler


def main():
    args = parse_args()
    roots = list(args.roots) if args.roots else list(DEFAULT_ROOTS)
    if args.dataset is not None:
        dataset = args.dataset.resolve()
        if not is_lerobot_dataset(dataset):
            raise SystemExit(f"dataset not found / invalid: {dataset}")
        # ensure parent root is searchable
        if dataset.parent not in roots and dataset not in roots:
            roots.append(dataset.parent)
    else:
        discovered = discover_datasets(roots)
        if not discovered:
            raise SystemExit(f"no datasets found under roots={roots}")
        # prefer merged base with labels if present
        preferred = next(
            (d for d in discovered if d["id"] == "flatten_fold_clothes_base_lerobot"),
            next((d for d in discovered if d.get("label_dirs")), discovered[0]),
        )
        dataset = Path(preferred["path"])

    state = AppState(dataset, args.label_dirname, args.camera, roots)
    # Resolve label dir against what actually exists on disk (e.g. 30000 → 10000 after merge).
    state.select(state.dataset, args.label_dirname)
    catalogs = discover_datasets(state.roots, extra=state.dataset)
    print(f"roots        : {[str(r) for r in state.roots]}")
    print(f"datasets     : {len(catalogs)} → {[d['id'] for d in catalogs]}")
    print(f"dataset      : {state.dataset}")
    print(f"labels       : {state.dataset / state.label_dirname}")
    print(f"annotations  : {ann_path(state.dataset)}")
    print(f"source eps   : {len(list_source_episodes(state.dataset))}")
    print(f"labeled eps  : {len(list_labeled_episodes(state.dataset, state.label_dirname))}")
    print(f"open         : http://127.0.0.1:{args.port}/")
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
