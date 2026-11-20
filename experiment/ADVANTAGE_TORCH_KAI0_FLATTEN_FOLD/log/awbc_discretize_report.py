#!/usr/bin/env python3
"""Generate awbc_discretize_check.md after *_awbc discretize."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/data3/kian_workspace/RLinf/FlattenFold")
OUT = Path(
    "/data3/kian_workspace/kai0/experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/log/awbc_discretize_check.md"
)
AWBC_DIRS = [
    "flatten_fold_clothes_base_lerobot_adv_KAI0_10000_clean_awbc",
    "flatten_fold_clothes_bw_1116_lerobot__adv_KAI0_10000_clean_awbc",
    "kai0_dagger_human_clothes_1106_lerobot_adv_KAI0_10000_clean_awbc",
    "kai0_dagger_human_clothes_b_1117_lerobot__adv_KAI0_10000_clean_awbc",
    "kai0_dagger_human_clothes_mix1110_lerobot_adv_KAI0_10000_clean_awbc",
]


def load_tasks(meta_path: Path) -> str:
    p = meta_path / "tasks.jsonl"
    return p.read_text(encoding="utf-8")


def frame_stats(ds: Path):
    files = sorted(ds.glob("data/chunk-*/episode_*.parquet"))
    counts: Counter = Counter()
    total = 0
    ep_stats = []
    for f in files:
        table = pq.read_table(f, columns=["task_index", "absolute_advantage"])
        ti = table.column("task_index").to_numpy()
        adv = table.column("absolute_advantage").to_numpy()
        c = Counter(int(x) for x in ti)
        counts.update(c)
        n = len(ti)
        total += n
        pos = int(c.get(1, 0))
        neg = int(c.get(0, 0))
        pos_mean = float(adv[ti == 1].mean()) if pos else float("nan")
        neg_mean = float(adv[ti == 0].mean()) if neg else float("nan")
        ep_stats.append(
            {
                "file": f.name,
                "n": n,
                "pos": pos,
                "neg": neg,
                "pos_pct": 100.0 * pos / n if n else 0.0,
                "pos_mean_adv": pos_mean,
                "neg_mean_adv": neg_mean,
            }
        )
    return counts, total, ep_stats, len(files)


def main() -> None:
    lines: list[str] = []
    lines.append("# AWBC discretize check")
    lines.append("")
    lines.append("Settings: `--threshold 30 --chunk-size 50 --discretion-type binary --advantage-source absolute_advantage`")
    lines.append("Base only: `--stage-nums 2`.")
    lines.append("")
    lines.append("Binary expectation: roughly ~70% negative (task_index=0) / ~30% positive (task_index=1).")
    lines.append("")

    for name in AWBC_DIRS:
        ds = ROOT / name
        lines.append(f"## `{name}`")
        lines.append("")
        if not ds.is_dir():
            lines.append("**MISSING**")
            lines.append("")
            continue

        lines.append("### meta/tasks.jsonl")
        lines.append("```")
        lines.append(load_tasks(ds / "meta").rstrip())
        lines.append("```")
        lines.append("")

        counts, total, ep_stats, n_ep = frame_stats(ds)
        lines.append(f"Episodes: **{n_ep}**, frames: **{total}**")
        lines.append("")
        lines.append("| task_index | count | pct |")
        lines.append("|---|---:|---:|")
        for k in sorted(counts):
            pct = 100.0 * counts[k] / total if total else 0.0
            lines.append(f"| {k} | {counts[k]} | {pct:.2f}% |")
        lines.append("")

        # sample 5 episodes (prefer mixed pos/neg)
        mixed = [e for e in ep_stats if e["pos"] > 0 and e["neg"] > 0]
        pool = mixed if len(mixed) >= 5 else ep_stats
        rng = random.Random(0)
        sample = pool if len(pool) <= 5 else rng.sample(pool, 5)
        sample = sorted(sample, key=lambda e: e["file"])

        lines.append("### Sample episodes (pos/neg vs abs_adv mean)")
        lines.append("")
        lines.append("| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |")
        lines.append("|---|---:|---:|---:|---:|:---:|")
        for e in sample:
            ok = (
                e["pos"] > 0
                and e["neg"] > 0
                and e["pos_mean_adv"] > e["neg_mean_adv"]
            )
            lines.append(
                f"| `{e['file']}` | {e['n']} | {e['pos_pct']:.1f}% | "
                f"{e['pos_mean_adv']:.4f} | {e['neg_mean_adv']:.4f} | {'yes' if ok else 'NO'} |"
            )
        lines.append("")

    lines.append("## Viewer commands")
    lines.append("")
    lines.append("Values live under `data/` (copied from clean). Example:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /data3/kian_workspace/kai0")
    lines.append(
        ".venv/bin/python -u stage_advantage/annotation/advantage_viewer/server.py \\"
    )
    lines.append(
        "  --dataset /data3/kian_workspace/RLinf/FlattenFold/flatten_fold_clothes_base_lerobot_adv_KAI0_10000_clean_awbc \\"
    )
    lines.append("  --label-dirname data")
    lines.append("```")
    lines.append("")
    lines.append("Other `*_awbc` paths under `/data3/kian_workspace/RLinf/FlattenFold/`.")
    lines.append("")
    lines.append("---")
    lines.append("Reply **OK** to proceed with config + 2-GPU train, or ask to change threshold / re-run.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
