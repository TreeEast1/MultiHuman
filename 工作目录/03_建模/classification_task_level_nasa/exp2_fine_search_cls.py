#!/usr/bin/env python3
"""P2 扩展：NASA 三分类 精细化 K 搜索。

背景：
  现有 P2 (`exp2_feature_selection_cls.py`) 搜索 K 集合为
  {5, 10, 15, 20, 30, 50, 80, 130}，粒度较粗，最佳点附近可能
  错过更优 K。本脚本在 K∈[15, 60] 区间加密，并在两端各取一点
  做边界确认，最终锁定 (ranker, K, model) 的最佳组合。

矩阵：
  ranker ∈ {MI, RF_importance, Permutation}
  K      ∈ {15, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55, 60, 70, 80}
  model  ∈ {LR_L2_strong, RF_shallow, XGB_shallow}
共 3 × 16 × 3 = 144 组。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE, RANKERS_CLS,
    median_impute_fold, median_impute_and_scale,
    pooled_cv_cls_with_selection,
)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp2_fine"
N_SPLITS = 5

# 加密 K 集合：在原 {5,10,15,20,30,50,80,130} 基础上，K∈[15,80] 区间加密
TOP_KS = [15, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55, 60, 70, 80]


def make_models(y_str, y_int):
    from xgboost import XGBClassifier
    return [
        ("LR_L2_strong",
         lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
         median_impute_and_scale, y_str),
        ("RF_shallow",
         lambda: RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                        random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold, y_str),
        ("XGB_shallow",
         lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=3,
                               reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
         None, y_int),
    ]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y_str = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp2_fine_cls] X={X.shape}  TOP_KS={TOP_KS}")
    models = make_models(y_str, y_int)

    all_rows = []
    selected_stats = {}

    for ranker_name, ranker_fn in RANKERS_CLS.items():
        print(f"\n=== Ranker: {ranker_name} ===")
        for k in TOP_KS:
            for m_name, factory, prep, y_use in models:
                res, selected = pooled_cv_cls_with_selection(
                    factory, X, y_use, groups, N_SPLITS,
                    top_k=k, ranker=ranker_fn, preprocessor=prep,
                    name=f"{ranker_name}_top{k}_{m_name}",
                )
                selected_stats[(ranker_name, k, m_name)] = selected
                all_rows.append({
                    "ranker": ranker_name, "k": k, "model": m_name,
                    "pooled_acc": res.pooled_acc,
                    "pooled_macro_f1": res.pooled_macro_f1,
                    "pooled_weighted_f1": res.pooled_weighted_f1,
                    "fold_macro_f1_mean": res.fold_macro_f1_mean,
                    "fold_macro_f1_std": res.fold_macro_f1_std,
                })
                print(f"  K={k:3d}  {m_name:16s}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # 稳定性统计
    stability = {}
    for (ranker, k, m_name), selected in selected_stats.items():
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
    print(f"\n[exp2_fine_cls] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(rows, stability, feature_names, models):
    lines = []
    lines.append("# P2 扩展：NASA 三分类 精细化 K 搜索\n\n")
    lines.append(f"**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject\n\n")
    lines.append(f"**K 集合**（共 {len(TOP_KS)} 点）：{TOP_KS}\n\n")
    lines.append("**ranker × K × model** 全组合扫描；下表汇总每个 (ranker, model) 在所有 K 上的最佳 K 与对应精度。\n\n")

    # 1) 每个 (ranker, model) 的最佳 K
    lines.append("## 1. 各 (排序, 模型) 组合的最佳 K\n\n")
    lines.append("| 排序 | 模型 | best K | pooled Acc | pooled Macro-F1 |\n|---|---|---:|---:|---:|\n")
    best_by_rm = {}
    for m_name, _, _, _ in models:
        for ranker in ["MI", "RF_importance", "Permutation"]:
            sub = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name]
            best = max(sub, key=lambda x: x["pooled_macro_f1"])
            best_by_rm[(ranker, m_name)] = best
            lines.append(
                f"| {ranker} | {m_name} | {best['k']} | "
                f"{best['pooled_acc']:.3f} | {best['pooled_macro_f1']:.3f} |\n"
            )
    lines.append("\n")

    # 2) 全局 Top-10 组合
    lines.append("## 2. 全局 Top-10 (ranker, K, model) 组合\n\n")
    lines.append("| rank | ranker | K | model | pooled Acc | pooled Macro-F1 |\n|---:|---|---:|---|---:|---:|\n")
    rows_sorted = sorted(rows, key=lambda x: -x["pooled_macro_f1"])
    for i, r in enumerate(rows_sorted[:10], 1):
        lines.append(
            f"| {i} | {r['ranker']} | {r['k']} | {r['model']} | "
            f"{r['pooled_acc']:.3f} | {r['pooled_macro_f1']:.3f} |\n"
        )
    lines.append("\n")

    # 3) K vs pooled Macro-F1 曲线（每个 model 一张表，列=ranker）
    lines.append("## 3. K vs pooled Macro-F1 曲线\n\n")
    for m_name, _, _, _ in models:
        lines.append(f"### {m_name}\n\n")
        lines.append("| K | MI | RF_importance | Permutation |\n|---:|---:|---:|---:|\n")
        for k in TOP_KS:
            rm = {r["ranker"]: r for r in rows if r["k"] == k and r["model"] == m_name}
            mi = rm.get("MI", {}).get("pooled_macro_f1", np.nan)
            rf = rm.get("RF_importance", {}).get("pooled_macro_f1", np.nan)
            pm = rm.get("Permutation", {}).get("pooled_macro_f1", np.nan)
            lines.append(f"| {k} | {mi:.3f} | {rf:.3f} | {pm:.3f} |\n")
        lines.append("\n")

    # 4) 全局最佳组合的稳定特征
    global_best = rows_sorted[0]
    key = (global_best["ranker"], global_best["k"], global_best["model"])
    stab = stability[key]
    lines.append(f"## 4. 全局最佳组合：{global_best['ranker']} + {global_best['model']} @ K={global_best['k']} (Macro-F1={global_best['pooled_macro_f1']:.3f})\n\n")
    lines.append(f"- stable_5 count: **{stab['stable_5_count']}** / {global_best['k']}\n\n")
    top20 = stab["counter"].most_common(20)
    lines.append("| 特征 | 命中折数 |\n|---|---:|\n")
    for idx, cnt in top20:
        lines.append(f"| `{feature_names[idx]}` | {cnt} |\n")
    lines.append("\n")

    # 5) 每个 (ranker, model) 最佳 K 的稳定特征
    lines.append("## 5. 各 (排序, 模型) 最佳 K 的稳定特征（命中 5/5 折）\n\n")
    for m_name, _, _, _ in models:
        for ranker in ["MI", "RF_importance", "Permutation"]:
            best = best_by_rm[(ranker, m_name)]
            key = (ranker, best["k"], m_name)
            stab = stability[key]
            lines.append(f"### {ranker} + {m_name} @ K={best['k']} (Macro-F1={best['pooled_macro_f1']:.3f})\n\n")
            lines.append(f"- stable_5 count: **{stab['stable_5_count']}** / {best['k']}\n\n")
            top15 = stab["counter"].most_common(15)
            lines.append("| 特征 | 命中折数 |\n|---|---:|\n")
            for idx, cnt in top15:
                lines.append(f"| `{feature_names[idx]}` | {cnt} |\n")
            lines.append("\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
