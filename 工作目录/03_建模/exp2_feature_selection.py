#!/usr/bin/env python3
"""P2 特征筛选实验。

关键约束：**筛选必须在训练折内做**，绝不用全数据先筛后跑 CV。

实验矩阵：
    - 排序方法：MI / RF_importance / Permutation（3 种）
    - Top-K：{5, 10, 15, 20, 30, 50, 80, 130}（8 个值）
    - 模型：Ridge_alpha10 / RF_shallow / XGB_shallow（3 种）
    - 共 3 × 8 × 3 = 72 次跑，加上 Full baseline 共 75 次

产出：
    - K vs pooled R² 曲线数据
    - 稳定性分析：Top-K 里在 5 折都被选中的特征清单
    - 各模型最佳 K 对应的选中特征名 Top-20
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from exp_utils import (
    RANDOM_STATE, RANKERS, median_impute_fold, median_impute_and_scale,
    pooled_cv, pooled_cv_with_selection,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset_task"
REPORT_DIR = HERE / "exp2_feature_selection"
N_SPLITS = 5

TOP_KS = [5, 10, 15, 20, 30, 50, 80, 130]


def make_models():
    from xgboost import XGBRegressor
    return [
        ("Ridge_alpha10", lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("RF_shallow", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1), median_impute_fold),
        ("XGB_shallow", lambda: XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1), None),
    ]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_task.npy")
    y = np.load(DATA_DIR / "y_task.npy")
    groups = np.load(DATA_DIR / "groups_task.npy")
    with open(DATA_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp2] X={X.shape}, 264-d full")

    models = make_models()

    # Full baseline
    print("\n=== Full 264d baseline ===")
    baseline_rows = []
    for m_name, factory, prep in models:
        res = pooled_cv(factory, X, y, groups, N_SPLITS, prep, name=f"Full_{m_name}")
        print(f"  {m_name}: MAE={res.pooled_mae:.3f}  R²={res.pooled_r2:+.3f}")
        baseline_rows.append({
            "ranker": "None", "k": 264, "model": m_name,
            "pooled_mae": res.pooled_mae, "pooled_r2": res.pooled_r2,
            "fold_r2_mean": res.fold_r2_mean, "fold_r2_std": res.fold_r2_std,
        })

    # 大网格
    all_rows = list(baseline_rows)
    selected_stats = {}  # {(ranker, k, model): list of 5 selected idx arrays}

    for ranker_name, ranker_fn in RANKERS.items():
        print(f"\n=== Ranker: {ranker_name} ===")
        for k in TOP_KS:
            for m_name, factory, prep in models:
                res, selected = pooled_cv_with_selection(
                    factory, X, y, groups, N_SPLITS,
                    top_k=k, ranker=ranker_fn, preprocessor=prep,
                    name=f"{ranker_name}_top{k}_{m_name}",
                )
                selected_stats[(ranker_name, k, m_name)] = selected
                all_rows.append({
                    "ranker": ranker_name, "k": k, "model": m_name,
                    "pooled_mae": res.pooled_mae, "pooled_r2": res.pooled_r2,
                    "fold_r2_mean": res.fold_r2_mean, "fold_r2_std": res.fold_r2_std,
                })
                print(f"  K={k:3d}  {m_name:16s}  MAE={res.pooled_mae:.3f}  R²={res.pooled_r2:+.3f}")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # 稳定性 & 特征命中榜（针对每 (ranker, model) 找到最佳 K）
    stability = {}
    for (ranker, k, m_name), selected in selected_stats.items():
        # 计数：在 5 折中被选中的次数
        counter = Counter()
        for arr in selected:
            for idx in arr:
                counter[idx] += 1
        stable_5 = [i for i, c in counter.items() if c == 5]
        stability[(ranker, k, m_name)] = {
            "counter": counter,
            "stable_5_count": len(stable_5),
            "stable_5_indices": stable_5,
        }

    write_markdown(all_rows, stability, feature_names, models)
    print(f"\n[exp2] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(rows, stability, feature_names, models):
    lines = []
    lines.append("# P2 特征筛选实验报告\n\n")
    lines.append("**设置**：84×264 任务级表，折内筛选（fit only on train fold），5×GroupKFold by subject\n\n")

    # Best-K per (ranker, model)
    lines.append("## 各 (排序方法, 模型) 组合的最佳 K\n\n")
    lines.append("| 排序 | 模型 | best K | pooled R² | pooled MAE | (Full 264 R²) |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")

    # 组织成表：model × ranker
    for m_name, _, _ in models:
        full_r2 = None
        for r in rows:
            if r["ranker"] == "None" and r["model"] == m_name:
                full_r2 = r["pooled_r2"]
                break
        for ranker in ["MI", "RF_importance", "Permutation"]:
            sub = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name]
            best = max(sub, key=lambda x: x["pooled_r2"])
            lines.append(
                f"| {ranker} | {m_name} | {best['k']} | "
                f"{best['pooled_r2']:+.3f} | {best['pooled_mae']:.3f} | "
                f"{full_r2:+.3f} |\n"
            )
    lines.append("\n")

    # K vs R² 曲线（每模型分别汇总 3 个 ranker）
    lines.append("## K vs pooled R² 曲线\n\n")
    for m_name, _, _ in models:
        lines.append(f"### {m_name}\n\n")
        lines.append("| K | MI R² | RF_importance R² | Permutation R² |\n|---:|---:|---:|---:|\n")
        for k in TOP_KS:
            row_dict = {r["ranker"]: r for r in rows
                        if r["k"] == k and r["model"] == m_name}
            mi = row_dict.get("MI", {}).get("pooled_r2", np.nan)
            rf = row_dict.get("RF_importance", {}).get("pooled_r2", np.nan)
            pm = row_dict.get("Permutation", {}).get("pooled_r2", np.nan)
            lines.append(f"| {k} | {mi:+.3f} | {rf:+.3f} | {pm:+.3f} |\n")
        # Full
        full = [r for r in rows if r["ranker"] == "None" and r["model"] == m_name][0]
        lines.append(f"| 264 | {full['pooled_r2']:+.3f} | {full['pooled_r2']:+.3f} | {full['pooled_r2']:+.3f} |\n\n")

    # 稳定性 Top-20 for 每 ranker 在最优 K 下
    lines.append("## 稳定选中的特征（每 (ranker, model) 组合，取最佳 K）\n\n")
    lines.append("*stable_5 = 在 5 折训练中都被选中；对应\"极稳健\"信号*\n\n")

    for m_name, _, _ in models:
        for ranker in ["MI", "RF_importance", "Permutation"]:
            sub = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name]
            best = max(sub, key=lambda x: x["pooled_r2"])
            key = (ranker, best["k"], m_name)
            stab = stability[key]
            lines.append(f"### {ranker} + {m_name} @ K={best['k']} (R²={best['pooled_r2']:+.3f})\n\n")
            lines.append(f"- stable_5 count: **{stab['stable_5_count']}** / {best['k']}\n\n")
            # 打印命中次数 Top-20
            top20 = stab["counter"].most_common(20)
            lines.append("| 特征 | 命中折数 |\n|---|---:|\n")
            for idx, cnt in top20:
                lines.append(f"| `{feature_names[idx]}` | {cnt} |\n")
            lines.append("\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
