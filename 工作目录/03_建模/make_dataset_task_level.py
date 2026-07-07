#!/usr/bin/env python3
"""P0 实验：把 84 个窗口级 CSV 聚合成 84 行任务级建模表。

聚合方式（每个 sample_id、每个原始特征各产 4 个统计量）：
    - _mean   窗口内均值（反映任务整体水平）
    - _std    窗口内标准差（反映波动性）
    - _median 窗口内中位数（对离群窗口鲁棒的中心估计）
    - _slope  简单线性回归斜率（反映任务过程中的漂移趋势，单位：per_window_index）

因此最终特征维度 = 原始 66 列 × 4 统计量 = 264 列。

设计要点：
1. 聚合统计量在**任务内**做，不涉及跨任务信息，因此不引入泄漏
2. slope 用简单最小二乘 (x = 窗口序号 0..n-1)，n<2 时置 NaN
3. 保留 sample_id / subject / task_difficulty 等元信息，便于消融按被试分组

输出到 工作目录/03_建模/dataset_task/：
    - X_task.npy         (84 × 264)
    - y_task.npy         (84,)   NASA 加权标签
    - groups_task.npy    (84,)   subject 编号（用于按被试 GroupKFold）
    - sample_task.npy    (84,)   sample_id 字符串
    - feature_names_task.json    列名清单
    - dataset_audit_task.md      审计报告
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "01_预处理" / "output_30s_step5s_final"
OUT_DIR = HERE / "dataset_task"

# 复用窗口级白名单（保持与 make_dataset.py 一致）
HR_KEEP = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_slope_bpm_per_min"]
EYE_KEEP = [
    "eye_pupil_filtered_mean", "eye_pupil_filtered_std",
    "eye_valid_ratio", "eye_fixation_ratio", "eye_saccade_ratio",
    "eye_eyes_not_found_ratio",
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

STATS = ("mean", "std", "median", "slope")


def build_raw_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    features: list[str] = []
    features += [c for c in cols if c.startswith("eeg_") and c.endswith("_z_within_subject")]
    features += [c for c in HR_KEEP if c in cols]
    features += [c for c in EYE_KEEP if c in cols]
    features += [c for c in BLINK_KEEP if c in cols]
    for c in cols:
        if not c.startswith("log_"):
            continue
        if "_cum" in c or "_recent_60s" in c or "_time_since_last" in c:
            continue
        if c.endswith("_win"):
            features.append(c)
    return features


def _slope(values: np.ndarray) -> float:
    """对窗口序列做简单最小二乘斜率。NaN 位置剔除后拟合。"""
    v = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(v)
    if mask.sum() < 2:
        return np.nan
    y = v[mask]
    x = np.arange(len(v), dtype=np.float64)[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def aggregate_one_sample(df: pd.DataFrame, raw_features: list[str]) -> dict:
    """把一个 sample_id 的窗口 DataFrame 聚合成 1 行字典。"""
    row: dict = {}
    for feat in raw_features:
        vals = df[feat].to_numpy(dtype=np.float64)
        # mean / std / median 用 nan-safe
        with np.errstate(all="ignore"):
            row[f"{feat}__mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{feat}__std"] = float(np.nanstd(vals, ddof=0)) if np.isfinite(vals).any() else np.nan
            row[f"{feat}__median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        row[f"{feat}__slope"] = _slope(vals)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[make_dataset_task] loading from {DATA_DIR}")
    csv_files = sorted(DATA_DIR.glob("subject_*_task_*.csv"))
    if not csv_files:
        raise SystemExit(f"未找到样本文件：{DATA_DIR}")

    # 读第一个文件确定原始特征列
    first_df = pd.read_csv(csv_files[0])
    raw_features = build_raw_feature_columns(first_df)
    print(f"[make_dataset_task] 原始特征列：{len(raw_features)}")
    print(f"[make_dataset_task] 聚合统计量：{STATS} (共 {len(raw_features) * len(STATS)} 列)")

    rows: list[dict] = []
    meta_rows: list[dict] = []

    for f in csv_files:
        df = pd.read_csv(f)

        # 校验特征列一致
        missing = set(raw_features) - set(df.columns)
        if missing:
            raise RuntimeError(f"{f.name} 缺失特征列：{missing}")

        sample_id = df["sample_id"].iloc[0]
        subject = int(df["subject"].iloc[0])
        task = str(df["task"].iloc[0])
        y_label = float(df["nasa_tlx_weighted_task_label"].iloc[0])
        difficulty = str(df["task_difficulty"].iloc[0])
        n_windows = len(df)

        agg = aggregate_one_sample(df, raw_features)
        agg_row = {
            "sample_id": sample_id,
            "subject": subject,
            "task": task,
            "task_difficulty": difficulty,
            "n_windows": n_windows,
            "y_nasa": y_label,
            **agg,
        }
        rows.append(agg_row)
        meta_rows.append({
            "sample_id": sample_id,
            "subject": subject,
            "task": task,
            "task_difficulty": difficulty,
            "n_windows": n_windows,
            "y_nasa": y_label,
        })

    task_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta_rows)

    # 特征列顺序：外层按 raw_features 顺序，内层按 STATS 顺序
    feature_cols = []
    for feat in raw_features:
        for stat in STATS:
            feature_cols.append(f"{feat}__{stat}")

    # 校验：所有列存在
    missing_feat = set(feature_cols) - set(task_df.columns)
    if missing_feat:
        raise RuntimeError(f"任务级表缺失聚合列：{missing_feat}")

    X = task_df[feature_cols].to_numpy(dtype=np.float64)
    y = task_df["y_nasa"].to_numpy(dtype=np.float64)
    groups_subject = task_df["subject"].to_numpy(dtype=np.int64)  # 用于按被试的 GroupKFold
    sample_ids = task_df["sample_id"].to_numpy()

    # 每行对应一个 sample_id，把 sample_id 也编成整数便于评估函数使用
    unique_sids = pd.unique(task_df["sample_id"])
    sid_to_int = {s: i for i, s in enumerate(unique_sids)}
    sample_int = np.array([sid_to_int[s] for s in sample_ids], dtype=np.int64)

    # 保存
    np.save(OUT_DIR / "X_task.npy", X)
    np.save(OUT_DIR / "y_task.npy", y)
    np.save(OUT_DIR / "groups_task.npy", groups_subject)
    np.save(OUT_DIR / "sample_task.npy", sample_ids)
    np.save(OUT_DIR / "sample_int_task.npy", sample_int)
    task_df.to_csv(OUT_DIR / "task_level_table.csv", index=False)
    with open(OUT_DIR / "feature_names_task.json", "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp, ensure_ascii=False, indent=2)

    # 审计
    lines = []
    lines.append("# 任务级建模数据集审计报告\n\n")
    lines.append(f"- 数据源：`{DATA_DIR.relative_to(HERE.parent.parent)}`\n")
    lines.append(f"- 样本数（sample_id）：{len(task_df)}\n")
    lines.append(f"- 独立被试数：{task_df['subject'].nunique()}\n")
    lines.append(f"- 原始特征列数：{len(raw_features)}\n")
    lines.append(f"- 聚合统计量：{', '.join(STATS)}\n")
    lines.append(f"- 任务级特征列数：{len(feature_cols)}（= {len(raw_features)} × {len(STATS)}）\n")
    lines.append(f"- 标签范围：[{y.min():.3f}, {y.max():.3f}]，均值 {y.mean():.3f}，std {y.std():.3f}\n\n")

    lines.append("## 样本分布\n\n")
    lines.append("### 按被试\n\n")
    per_subj = task_df.groupby("subject").size().reset_index(name="n_samples")
    lines.append(f"- 被试数：{len(per_subj)}\n")
    lines.append(f"- 每被试样本数：min={per_subj['n_samples'].min()}, max={per_subj['n_samples'].max()}, mean={per_subj['n_samples'].mean():.2f}\n\n")

    lines.append("### 按任务难度\n\n")
    diff_stat = task_df.groupby("task_difficulty").agg(
        n=("sample_id", "count"),
        y_mean=("y_nasa", "mean"),
        y_std=("y_nasa", "std"),
    ).reset_index()
    lines.append("| 难度 | 样本数 | NASA 均值 | NASA std |\n|---|---:|---:|---:|\n")
    for _, r in diff_stat.iterrows():
        lines.append(f"| {r['task_difficulty']} | {r['n']} | {r['y_mean']:.3f} | {r['y_std']:.3f} |\n")
    lines.append("\n")

    # 每个样本的窗口数分布
    lines.append("### 每个样本的窗口数\n\n")
    lines.append(f"- min={task_df['n_windows'].min()}, max={task_df['n_windows'].max()}, "
                 f"mean={task_df['n_windows'].mean():.1f}, median={task_df['n_windows'].median():.0f}\n\n")

    # 缺失率（聚合后应该显著减少）
    miss_pct = task_df[feature_cols].isna().mean().sort_values(ascending=False) * 100
    lines.append("## 特征缺失率（聚合后，前 20 高）\n\n")
    lines.append("| 特征 | 缺失率 |\n|---|---:|\n")
    for name, pct in miss_pct.head(20).items():
        lines.append(f"| `{name}` | {pct:.2f}% |\n")
    lines.append(f"\n整体缺失率：{task_df[feature_cols].isna().to_numpy().mean() * 100:.3f}%\n\n")

    # 说明 slope 高缺失的可能原因
    slope_cols = [c for c in feature_cols if c.endswith("__slope")]
    slope_miss = task_df[slope_cols].isna().mean().mean() * 100
    lines.append(f"备注：`__slope` 列整体缺失率 {slope_miss:.2f}%（当某特征在整任务内全 NaN 或仅 1 个窗口有效时会置 NaN）\n")

    (OUT_DIR / "dataset_audit_task.md").write_text("".join(lines), encoding="utf-8")

    print(f"[make_dataset_task] saved to {OUT_DIR}")
    print(f"  X.shape = {X.shape}")
    print(f"  y range = [{y.min():.3f}, {y.max():.3f}]")
    print(f"  unique subjects = {len(np.unique(groups_subject))}")
    print(f"  overall missing = {task_df[feature_cols].isna().to_numpy().mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
