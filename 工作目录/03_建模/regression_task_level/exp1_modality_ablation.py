#!/usr/bin/env python3
"""P1 模态消融实验。

在 84 × 264 任务级表上，做 3 类实验：
1. Leave-One-Modality-Out：从 Full 中去掉某一模态的所有统计量
2. Only-One-Modality：只保留某一模态
3. Only-One-Statistic：只保留某一统计量（mean/std/median/slope）

每类都用 3 个代表模型跑：RandomForest_shallow / XGBoost_shallow / Ridge_alpha100
（Ridge 加进来是为了看"高维小样本下强正则线性模型对特征子集的敏感度"）

主指标：pooled MAE 与 pooled R²（84 样本 5×GroupKFold by subject）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import (  # noqa: E402
    RANDOM_STATE, median_impute_fold, median_impute_and_scale, pooled_cv,
)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp1"
N_SPLITS = 5


# 模态定义（按前缀识别原始列，然后所有统计量都属于同一模态）
def build_modality_masks(feature_names: list[str]) -> dict[str, np.ndarray]:
    """返回每个模态的布尔掩码（对 264 列）。"""
    def match(prefix_check):
        return np.array([prefix_check(c) for c in feature_names])

    def is_eye_pupil(c: str) -> bool:
        # eye_* 但不是 aoi
        return c.startswith("eye_") and "aoi" not in c

    masks = {
        "EEG": match(lambda c: c.startswith("eeg_")),
        "HR": match(lambda c: c.startswith("hr_")),
        "EyePupil": match(is_eye_pupil),      # 瞳孔/注视/valid/saccade 等（6 列 × 4 stats = 24）
        "AOI": match(lambda c: c.startswith("eye_aoi_")),
        "Blink": match(lambda c: c.startswith("blink_")),
        "Log": match(lambda c: c.startswith("log_")),
    }
    return masks


def build_stat_masks(feature_names: list[str]) -> dict[str, np.ndarray]:
    """返回每个统计量的布尔掩码。"""
    stats = ["mean", "std", "median", "slope"]
    return {s: np.array([c.endswith(f"__{s}") for c in feature_names]) for s in stats}


def run_all_models(X, y, groups, subset_name):
    """在给定特征子集上跑 3 个模型，返回结果列表。"""
    models = [
        ("RF_shallow", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold),
        ("Ridge_alpha100", lambda: Ridge(alpha=100.0, random_state=RANDOM_STATE),
         median_impute_and_scale),
    ]
    # XGBoost 单独处理（延迟导入）
    from xgboost import XGBRegressor
    models.append((
        "XGB_shallow",
        lambda: XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                             reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
        None,
    ))

    out = []
    for m_name, factory, prep in models:
        res = pooled_cv(factory, X, y, groups, N_SPLITS, prep,
                        name=f"{subset_name}__{m_name}")
        out.append({
            "subset": subset_name,
            "model": m_name,
            "n_features": res.n_features,
            "pooled_mae": res.pooled_mae,
            "pooled_r2": res.pooled_r2,
            "fold_mae_mean": res.fold_mae_mean,
            "fold_mae_std": res.fold_mae_std,
            "fold_r2_mean": res.fold_r2_mean,
            "fold_r2_std": res.fold_r2_std,
        })
    return out


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_task.npy")
    y = np.load(DATA_DIR / "y_task.npy")
    groups = np.load(DATA_DIR / "groups_task.npy")
    with open(DATA_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp1] X={X.shape}, y=[{y.min():.2f},{y.max():.2f}], subjects={len(np.unique(groups))}")

    modality_masks = build_modality_masks(feature_names)
    stat_masks = build_stat_masks(feature_names)

    print("[exp1] 模态特征数：")
    for mod, mask in modality_masks.items():
        print(f"  {mod}: {mask.sum()} 列")
    total_marked = sum(m.sum() for m in modality_masks.values())
    print(f"  合计标记 = {total_marked} / 264（未标记 = {264 - total_marked}）")

    all_rows = []

    # ---- Full baseline ----
    print("\n=== Full (264d) ===")
    all_rows += run_all_models(X, y, groups, subset_name="Full_264")

    # ---- Leave-One-Modality-Out ----
    print("\n=== Leave-One-Modality-Out ===")
    for mod, mask in modality_masks.items():
        idx = np.where(~mask)[0]
        X_sub = X[:, idx]
        subset_name = f"minus_{mod}"
        print(f"  {subset_name}: {X_sub.shape[1]} features")
        all_rows += run_all_models(X_sub, y, groups, subset_name=subset_name)

    # ---- Only-One-Modality ----
    print("\n=== Only-One-Modality ===")
    for mod, mask in modality_masks.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        X_sub = X[:, idx]
        subset_name = f"only_{mod}"
        print(f"  {subset_name}: {X_sub.shape[1]} features")
        all_rows += run_all_models(X_sub, y, groups, subset_name=subset_name)

    # ---- Only-One-Statistic ----
    print("\n=== Only-One-Statistic ===")
    for stat, mask in stat_masks.items():
        idx = np.where(mask)[0]
        X_sub = X[:, idx]
        subset_name = f"only_{stat}"
        print(f"  {subset_name}: {X_sub.shape[1]} features")
        all_rows += run_all_models(X_sub, y, groups, subset_name=subset_name)

    # ---- 保存与报告 ----
    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    write_markdown(all_rows, modality_masks, stat_masks)
    print(f"\n[exp1] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(rows, modality_masks, stat_masks):
    lines = []
    lines.append("# P1 模态消融实验报告\n\n")
    lines.append("**设置**：84 样本 × 264 特征，5×GroupKFold by subject，主指标 pooled MAE / R²\n\n")

    # 模态定义表
    lines.append("## 模态定义（264 列覆盖情况）\n\n")
    lines.append("| 模态 | 特征列数 | 覆盖范围 |\n|---|---:|---|\n")
    for mod, mask in modality_masks.items():
        lines.append(f"| {mod} | {int(mask.sum())} | 4 统计量 × N 原始列 |\n")
    lines.append("\n")

    # 分子表：Full
    lines.append("## Full baseline\n\n")
    _write_subset_table(lines, rows, ["Full_264"])

    # 分子表：Leave-One-Modality-Out
    lines.append("## Leave-One-Modality-Out（去除某一模态后的表现）\n\n")
    _write_subset_table(lines, rows,
                        [f"minus_{m}" for m in modality_masks.keys()])
    lines.append("解读：与 Full 相比 pooled R² 下降越多 → 该模态贡献越大\n\n")

    # 分子表：Only-One-Modality
    lines.append("## Only-One-Modality（仅使用某一模态）\n\n")
    _write_subset_table(lines, rows,
                        [f"only_{m}" for m in modality_masks.keys()])
    lines.append("解读：单模态 pooled R² 越高 → 该模态独立预测能力越强\n\n")

    # 分子表：Only-One-Statistic
    lines.append("## Only-One-Statistic（仅使用某一统计量的所有 66 列）\n\n")
    _write_subset_table(lines, rows, [f"only_{s}" for s in stat_masks.keys()])
    lines.append("解读：判断 mean/std/median/slope 四种聚合方式的相对价值\n\n")

    # 汇总排序（按 XGB pooled R² 全部实验）
    lines.append("## 全部实验按 XGBoost pooled R² 排序\n\n")
    xgb_rows = [r for r in rows if r["model"] == "XGB_shallow"]
    xgb_rows_sorted = sorted(xgb_rows, key=lambda x: -x["pooled_r2"])
    lines.append("| 排名 | subset | n_features | pooled MAE | pooled R² |\n|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(xgb_rows_sorted, 1):
        lines.append(f"| {i} | {r['subset']} | {r['n_features']} | {r['pooled_mae']:.3f} | {r['pooled_r2']:+.3f} |\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


def _write_subset_table(lines, all_rows, subset_names):
    lines.append("| subset | n_feat | RF pooled R² | Ridge pooled R² | XGB pooled R² | XGB MAE |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name in subset_names:
        row_map = {r["model"]: r for r in all_rows if r["subset"] == name}
        if not row_map:
            continue
        xgb = row_map.get("XGB_shallow")
        rf = row_map.get("RF_shallow")
        rg = row_map.get("Ridge_alpha100")
        n_feat = xgb["n_features"] if xgb else "-"
        lines.append(
            f"| {name} | {n_feat} | "
            f"{rf['pooled_r2']:+.3f} | "
            f"{rg['pooled_r2']:+.3f} | "
            f"{xgb['pooled_r2']:+.3f} | "
            f"{xgb['pooled_mae']:.3f} |\n"
        )
    lines.append("\n")


if __name__ == "__main__":
    main()
