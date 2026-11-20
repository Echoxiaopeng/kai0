# AWBC discretize check

Settings: `--threshold 30 --chunk-size 50 --discretion-type binary --advantage-source absolute_advantage`

**Base note:** plan suggested `--stage-nums 2`, but only 303/571 base eps have `stage_progress_gt`. First pass left 268 eps unlabeled under stage mode, so base was **re-discretized with `stage_nums=1`** for full coverage. Other four datasets: no stage-nums (as planned).

Binary expectation: roughly ~70% negative (task_index=0) / ~30% positive (task_index=1).

## `flatten_fold_clothes_base_lerobot_adv_KAI0_10000_clean_awbc`

### meta/tasks.jsonl
```
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

Episodes: **571**, frames: **518950**

| task_index | count | pct |
|---|---:|---:|
| 0 | 363265 | 70.00% |
| 1 | 155685 | 30.00% |

### Sample episodes (pos/neg vs abs_adv mean)

| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |
|---|---:|---:|---:|---:|:---:|
| `episode_000041.parquet` | 992 | 37.7% | 0.1388 | 0.0040 | yes |
| `episode_000265.parquet` | 627 | 47.8% | 0.1062 | 0.0406 | yes |
| `episode_000394.parquet` | 1630 | 26.6% | 0.1204 | -0.0055 | yes |
| `episode_000430.parquet` | 804 | 30.3% | 0.1177 | 0.0237 | yes |
| `episode_000523.parquet` | 639 | 33.5% | 0.1273 | 0.0129 | yes |

## `flatten_fold_clothes_bw_1116_lerobot__adv_KAI0_10000_clean_awbc`

### meta/tasks.jsonl
```
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

Episodes: **68**, frames: **54570**

| task_index | count | pct |
|---|---:|---:|
| 0 | 38199 | 70.00% |
| 1 | 16371 | 30.00% |

### Sample episodes (pos/neg vs abs_adv mean)

| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |
|---|---:|---:|---:|---:|:---:|
| `episode_000005.parquet` | 844 | 21.8% | 0.1788 | 0.0211 | yes |
| `episode_000033.parquet` | 1003 | 23.9% | 0.2528 | -0.0682 | yes |
| `episode_000049.parquet` | 907 | 28.0% | 0.2041 | -0.0123 | yes |
| `episode_000053.parquet` | 790 | 36.5% | 0.4131 | -0.2527 | yes |
| `episode_000065.parquet` | 725 | 21.2% | 0.2039 | -0.0618 | yes |

## `kai0_dagger_human_clothes_1106_lerobot_adv_KAI0_10000_clean_awbc`

### meta/tasks.jsonl
```
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

Episodes: **55**, frames: **31778**

| task_index | count | pct |
|---|---:|---:|
| 0 | 22244 | 70.00% |
| 1 | 9534 | 30.00% |

### Sample episodes (pos/neg vs abs_adv mean)

| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |
|---|---:|---:|---:|---:|:---:|
| `episode_000002.parquet` | 547 | 28.5% | 0.1315 | -0.0878 | yes |
| `episode_000024.parquet` | 862 | 20.8% | 0.1300 | -0.0069 | yes |
| `episode_000026.parquet` | 696 | 23.9% | 0.1294 | 0.0068 | yes |
| `episode_000048.parquet` | 473 | 37.2% | 0.1298 | 0.0161 | yes |
| `episode_000054.parquet` | 572 | 18.7% | 0.1223 | -0.0034 | yes |

## `kai0_dagger_human_clothes_b_1117_lerobot__adv_KAI0_10000_clean_awbc`

### meta/tasks.jsonl
```
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

Episodes: **21**, frames: **10706**

| task_index | count | pct |
|---|---:|---:|
| 0 | 7494 | 70.00% |
| 1 | 3212 | 30.00% |

### Sample episodes (pos/neg vs abs_adv mean)

| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |
|---|---:|---:|---:|---:|:---:|
| `episode_000001.parquet` | 652 | 32.1% | 0.1751 | -0.0067 | yes |
| `episode_000008.parquet` | 476 | 26.5% | 0.1742 | -0.1204 | yes |
| `episode_000012.parquet` | 488 | 22.1% | 0.1792 | -0.0826 | yes |
| `episode_000013.parquet` | 364 | 42.0% | 0.2116 | -0.2478 | yes |
| `episode_000016.parquet` | 525 | 29.1% | 0.1775 | 0.0107 | yes |

## `kai0_dagger_human_clothes_mix1110_lerobot_adv_KAI0_10000_clean_awbc`

### meta/tasks.jsonl
```
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

Episodes: **38**, frames: **19994**

| task_index | count | pct |
|---|---:|---:|
| 0 | 13996 | 70.00% |
| 1 | 5998 | 30.00% |

### Sample episodes (pos/neg vs abs_adv mean)

| episode | frames | pos% | pos_mean_adv | neg_mean_adv | pos>neg? |
|---|---:|---:|---:|---:|:---:|
| `episode_000002.parquet` | 691 | 34.3% | 0.1495 | 0.0099 | yes |
| `episode_000016.parquet` | 588 | 24.0% | 0.1309 | 0.0352 | yes |
| `episode_000024.parquet` | 685 | 20.1% | 0.1434 | 0.0235 | yes |
| `episode_000026.parquet` | 279 | 61.6% | 0.1726 | 0.0246 | yes |
| `episode_000032.parquet` | 468 | 30.6% | 0.1269 | 0.0139 | yes |

## Viewer commands（可检查离散 task_index）

打开 `*_awbc` 后图表会显示：
- **绿色底带** = `task_index=1`（Advantage positive）
- **浅红底带** = `task_index=0`（Advantage negative）
- 右侧阶梯线 `task_index (1=pos)`；读数区当前帧 AWBC positive/negative
- 侧栏每集 `pos %` 与 μ⁺/μ⁻

```bash
cd /data3/kian_workspace/kai0
.venv/bin/python -u stage_advantage/annotation/advantage_viewer/server.py \
  --dataset /data3/kian_workspace/RLinf/FlattenFold/flatten_fold_clothes_base_lerobot_adv_KAI0_10000_clean_awbc \
  --label-dirname data \
  --host 0.0.0.0 --port 18765
```

Other `*_awbc` paths under `/data3/kian_workspace/RLinf/FlattenFold/`.

---
Reply **OK** to proceed with 2-GPU train (`pi05_flatten_fold_awbc_clean_mix`), or ask to change threshold / re-run.
