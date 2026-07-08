#!/usr/bin/env python3
"""把 84 个窗口级 CSV 拼成建模用长表，并保存成 .npz。

按用户已确认的特征方案：
- EEG 只用 28 个 z-score 版（原始 28 个 EEG 排除）
- 心率 5 列（hr_mean/std/min/max/slope，hr_n 是元信息排除）
- 眼动瞳孔/注视 6 列 + AOI 9 列
- 眨眼 6 列（过滤后的，_raw 审计列排除）
- 日志只用 _win 版（_cum、_recent_60s、_time_since_last 全部排除）

输出：工作目录/03_建模/dataset/
  - X.npy         (12624 × F)  特征矩阵
  - y.npy         (12624,)     窗口级 NASA 标签（同 sample_id 内共享）
  - groups.npy    (12624,)     sample_id 整数编码，GroupKFold 用
  - sample_ids.npy (12624,)    原始字符串 sample_id
  - feature_names.json         列名清单
  - dataset_audit.md           数据审计报告
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "01_预处理" / "output_30s_step5s_final"
OUT_DIR = HERE / "dataset"


# ---- 特征白名单（按前缀 / 精确列名混合过滤）----
HR_KEEP = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_slope_bpm_per_min"]

EYE_KEEP = [
    # 瞳孔/注视
    "eye_pupil_filtered_mean", "eye_pupil_filtered_std",
    "eye_valid_ratio", "eye_fixation_ratio", "eye_saccade_ratio",
    "eye_eyes_not_found_ratio",
    # AOI（9 个，含情境意识代理）
    "eye_aoi_interval_n", "eye_aoi_unique_hit_n", "eye_aoi_total_fix_ms",
    "eye_aoi_fixation_n", "eye_aoi_fixation_density_per_sec",
    "eye_aoi_coverage_ratio", "eye_aoi_max_share", "eye_aoi_entropy",
    "eye_aoi_pupil_weighted_mean",
]

BLINK_KEEP = [
    "blink_count", "blink_rate_per_min",
    "blink_duration_mean_ms", "blink_duration_std_ms",
    "blink_duration_median_ms", "blink_total_duration_ratio",
]

# EEG：所有 _z_within_subject 列（28 个）
# 日志：所有 _win 结尾的列（12 个）
# 排除：_cum / _recent_60s / _time_since_last / _raw / eye_n / hr_n / anchor_source


def build_feature_columns(sample_df: pd.DataFrame) -> list[str]:
    cols = list(sample_df.columns)
    features: list[str] = []

    # EEG z-score only
    features += [c for c in cols if c.startswith("eeg_") and c.endswith("_z_within_subject")]
    # 心率
    features += [c for c in HR_KEEP if c in cols]
    # 眼动
    features += [c for c in EYE_KEEP if c in cols]
    # 眨眼
    features += [c for c in BLINK_KEEP if c in cols]
    # 日志 win only（严格过滤所有 _cum / _recent_60s / _time_since_last）
    for c in cols:
        if not c.startswith("log_"):
            continue
        if "_cum" in c or "_recent_60s" in c or "_time_since_last" in c:
            continue
        if c.endswith("_win"):
            features.append(c)
    return features


def load_all_samples() -> tuple[pd.DataFrame, list[str]]:
    csv_files = sorted(DATA_DIR.glob("subject_*_task_*.csv"))
    if not csv_files:
        raise SystemExit(f"未找到样本文件：{DATA_DIR}")

    dfs = []
    feature_cols: list[str] | None = None
    for f in csv_files:
        df = pd.read_csv(f)
        if feature_cols is None:
            feature_cols = build_feature_columns(df)
        # 保证所有文件列一致
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise RuntimeError(f"{f.name} 缺失特征列：{missing}")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df, feature_cols


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[make_dataset] loading from {DATA_DIR}")
    all_df, feature_cols = load_all_samples()
    print(f"[make_dataset] {len(all_df)} windows across {all_df['sample_id'].nunique()} samples")
    print(f"[make_dataset] {len(feature_cols)} features")

    X = all_df[feature_cols].to_numpy(dtype=np.float64)
    y = all_df["nasa_tlx_weighted_task_label"].to_numpy(dtype=np.float64)
    sample_ids = all_df["sample_id"].to_numpy()

    # groups：把 sample_id 编成 0..83
    unique_ids = pd.unique(all_df["sample_id"])
    id_to_int = {s: i for i, s in enumerate(unique_ids)}
    groups = np.array([id_to_int[s] for s in sample_ids], dtype=np.int64)

    # 保存
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "groups.npy", groups)
    np.save(OUT_DIR / "sample_ids.npy", sample_ids)
    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # 审计报告
    audit_lines = []
    audit_lines.append("# 建模数据集审计报告\n")
    audit_lines.append(f"- 数据源：`{DATA_DIR.relative_to(HERE.parent.parent)}`\n")
    audit_lines.append(f"- 总窗口数：{len(all_df)}\n")
    audit_lines.append(f"- 样本数（sample_id）：{all_df['sample_id'].nunique()}\n")
    audit_lines.append(f"- 特征列数：{len(feature_cols)}\n")
    audit_lines.append(f"- 标签列：`nasa_tlx_weighted_task_label`\n")
    audit_lines.append(f"- 标签范围：[{y.min():.3f}, {y.max():.3f}]，均值 {y.mean():.3f}\n\n")

    # 缺失率统计
    audit_lines.append("## 特征缺失率（前 20 高）\n\n")
    miss_pct = all_df[feature_cols].isna().mean().sort_values(ascending=False) * 100
    audit_lines.append("| 特征 | 缺失率 |\n|---|---:|\n")
    for name, pct in miss_pct.head(20).items():
        audit_lines.append(f"| `{name}` | {pct:.2f}% |\n")
    audit_lines.append(f"\n总体缺失率：{all_df[feature_cols].isna().to_numpy().mean() * 100:.3f}%\n\n")

    # 分类型统计
    eeg_n = sum(1 for c in feature_cols if c.startswith("eeg_"))
    hr_n = sum(1 for c in feature_cols if c.startswith("hr_"))
    eye_n = sum(1 for c in feature_cols if c.startswith("eye_"))
    blink_n = sum(1 for c in feature_cols if c.startswith("blink_"))
    log_n = sum(1 for c in feature_cols if c.startswith("log_"))
    audit_lines.append("## 特征分类\n\n")
    audit_lines.append("| 类别 | 列数 |\n|---|---:|\n")
    audit_lines.append(f"| EEG（z-score） | {eeg_n} |\n")
    audit_lines.append(f"| 心率 | {hr_n} |\n")
    audit_lines.append(f"| 眼动（瞳孔+注视+AOI） | {eye_n} |\n")
    audit_lines.append(f"| 眨眼（过滤后） | {blink_n} |\n")
    audit_lines.append(f"| 日志（仅 win） | {log_n} |\n")
    audit_lines.append(f"| **合计** | **{len(feature_cols)}** |\n")

    (OUT_DIR / "dataset_audit.md").write_text("".join(audit_lines), encoding="utf-8")

    print(f"[make_dataset] saved to {OUT_DIR}")
    print(f"  X.shape = {X.shape}")
    print(f"  y range = [{y.min():.3f}, {y.max():.3f}]")
    print(f"  groups: {len(np.unique(groups))} unique")
    print(f"  overall missing: {all_df[feature_cols].isna().to_numpy().mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
