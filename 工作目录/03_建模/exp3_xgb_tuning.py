#!/usr/bin/env python3
"""P3 XGBoost 调参实验。

基于 P2 得到的最佳筛选方案：MI + Top-K，扫小网格。
（K 取 P2 中 XGB 表现最好的那个）

网格：
    - max_depth ∈ {2, 3, 4}
    - learning_rate ∈ {0.02, 0.05, 0.1}
    - reg_lambda ∈ {1.0, 2.0, 5.0}
    - n_estimators ∈ {300, 500, 800}
共 81 次配置，全部 pooled CV（每次 5 折）。

同时对 RandomForest 做一个小规模调参对照：
    - max_depth ∈ {3, 4, 6}
    - min_samples_leaf ∈ {2, 3, 5}
    - n_estimators ∈ {300, 500}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from exp_utils import (
    RANDOM_STATE, RANKERS, median_impute_fold, pooled_cv_with_selection,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset_task"
REPORT_DIR = HERE / "exp3_xgb_tuning"
EXP2_JSON = HERE / "exp2_feature_selection" / "results.json"
N_SPLITS = 5


def pick_best_k_for_xgb(ranker: str = "MI") -> int:
    """从 exp2 结果读出 XGB_shallow 在指定 ranker 下的最佳 K。"""
    with open(EXP2_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    sub = [r for r in rows if r["ranker"] == ranker and r["model"] == "XGB_shallow"]
    if not sub:
        return 20
    best = max(sub, key=lambda x: x["pooled_r2"])
    return best["k"]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_task.npy")
    y = np.load(DATA_DIR / "y_task.npy")
    groups = np.load(DATA_DIR / "groups_task.npy")
    with open(DATA_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    from xgboost import XGBRegressor

    # 选定筛选方案
    ranker_name = "MI"
    best_k = pick_best_k_for_xgb(ranker_name)
    print(f"[exp3] 采用 {ranker_name} 折内筛选，K={best_k}")

    # XGBoost 网格
    xgb_grid = []
    for depth in [2, 3, 4]:
        for lr in [0.02, 0.05, 0.1]:
            for reg in [1.0, 2.0, 5.0]:
                for n in [300, 500, 800]:
                    xgb_grid.append(dict(max_depth=depth, learning_rate=lr,
                                         reg_lambda=reg, n_estimators=n))

    print(f"[exp3] XGB 网格共 {len(xgb_grid)} 组")

    xgb_results = []
    for i, cfg in enumerate(xgb_grid):
        def factory(cfg=cfg):
            return XGBRegressor(
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", n_jobs=-1,
                random_state=RANDOM_STATE, **cfg,
            )
        res, _ = pooled_cv_with_selection(
            factory, X, y, groups, N_SPLITS,
            top_k=best_k, ranker=RANKERS[ranker_name],
            preprocessor=None,
        )
        xgb_results.append({
            "cfg": cfg, "pooled_mae": res.pooled_mae, "pooled_r2": res.pooled_r2,
            "fold_r2_mean": res.fold_r2_mean, "fold_r2_std": res.fold_r2_std,
        })
        if (i + 1) % 9 == 0:
            print(f"  progress {i+1}/{len(xgb_grid)}  best so far R²={max(r['pooled_r2'] for r in xgb_results):+.3f}")

    xgb_sorted = sorted(xgb_results, key=lambda x: -x["pooled_r2"])
    best_xgb = xgb_sorted[0]
    print(f"\n[exp3] XGB 最佳：{best_xgb['cfg']}  R²={best_xgb['pooled_r2']:+.3f}  MAE={best_xgb['pooled_mae']:.3f}")

    # RF 网格
    rf_grid = []
    for depth in [3, 4, 6, None]:
        for msl in [2, 3, 5]:
            for n in [300, 500]:
                rf_grid.append(dict(max_depth=depth, min_samples_leaf=msl, n_estimators=n))

    print(f"\n[exp3] RF 网格共 {len(rf_grid)} 组")
    rf_results = []
    for i, cfg in enumerate(rf_grid):
        def factory(cfg=cfg):
            return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        res, _ = pooled_cv_with_selection(
            factory, X, y, groups, N_SPLITS,
            top_k=best_k, ranker=RANKERS[ranker_name],
            preprocessor=median_impute_fold,
        )
        rf_results.append({
            "cfg": {k: (str(v) if v is None else v) for k, v in cfg.items()},
            "pooled_mae": res.pooled_mae, "pooled_r2": res.pooled_r2,
            "fold_r2_mean": res.fold_r2_mean, "fold_r2_std": res.fold_r2_std,
        })
    rf_sorted = sorted(rf_results, key=lambda x: -x["pooled_r2"])
    best_rf = rf_sorted[0]
    print(f"[exp3] RF 最佳：{best_rf['cfg']}  R²={best_rf['pooled_r2']:+.3f}  MAE={best_rf['pooled_mae']:.3f}")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {"ranker": ranker_name, "top_k": best_k},
            "xgb": xgb_results,
            "rf": rf_results,
        }, f, ensure_ascii=False, indent=2)

    write_markdown(ranker_name, best_k, xgb_sorted, rf_sorted)
    print(f"\n[exp3] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(ranker_name, best_k, xgb_sorted, rf_sorted):
    lines = []
    lines.append(f"# P3 调参实验报告（{ranker_name} + Top-{best_k}）\n\n")
    lines.append(f"**设置**：折内 {ranker_name} 筛选 top-{best_k}；84 样本 5×GroupKFold by subject；pooled 指标\n\n")

    lines.append("## XGBoost Top-10 配置\n\n")
    lines.append("| rank | cfg | pooled MAE | pooled R² | fold R² (mean±std) |\n|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(xgb_sorted[:10], 1):
        cfg_str = ", ".join(f"{k}={v}" for k, v in r["cfg"].items())
        lines.append(
            f"| {i} | {cfg_str} | {r['pooled_mae']:.3f} | "
            f"{r['pooled_r2']:+.3f} | {r['fold_r2_mean']:+.3f}±{r['fold_r2_std']:.3f} |\n"
        )
    lines.append("\n")

    lines.append("## XGBoost Bottom-3 配置（做对照）\n\n")
    lines.append("| cfg | pooled MAE | pooled R² |\n|---|---:|---:|\n")
    for r in xgb_sorted[-3:]:
        cfg_str = ", ".join(f"{k}={v}" for k, v in r["cfg"].items())
        lines.append(f"| {cfg_str} | {r['pooled_mae']:.3f} | {r['pooled_r2']:+.3f} |\n")
    lines.append("\n")

    lines.append("## RandomForest Top-5\n\n")
    lines.append("| rank | cfg | pooled MAE | pooled R² |\n|---:|---|---:|---:|\n")
    for i, r in enumerate(rf_sorted[:5], 1):
        cfg_str = ", ".join(f"{k}={v}" for k, v in r["cfg"].items())
        lines.append(f"| {i} | {cfg_str} | {r['pooled_mae']:.3f} | {r['pooled_r2']:+.3f} |\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
