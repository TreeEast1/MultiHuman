#!/usr/bin/env python3
"""P3 分类调参实验。

基于 P2 得到的最佳筛选方案（默认 MI）扫超参：
    XGBoost 网格（约 81 组）：
      max_depth ∈ {2, 3, 4}
      learning_rate ∈ {0.02, 0.05, 0.1}
      reg_lambda ∈ {1.0, 2.0, 5.0}
      n_estimators ∈ {300, 500, 800}
    RandomForest 网格（约 24 组）：
      max_depth ∈ {3, 4, 6, None}
      min_samples_leaf ∈ {2, 3, 5}
      n_estimators ∈ {300, 500}
    LogisticRegression 网格（21 组）：
      C ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0}
      penalty ∈ {'l2'} × solver ∈ {'lbfgs','liblinear','saga(l1)'}
    SVC 网格（16 组）：
      C ∈ {0.5, 1, 3, 10}
      gamma ∈ {'scale', 0.01, 0.05, 0.1}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE, RANKERS_CLS,
    median_impute_fold, median_impute_and_scale,
    pooled_cv_cls_with_selection,
)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp3"
EXP2_JSON = HERE / "reports_exp2" / "results.json"
N_SPLITS = 5


def pick_best_k(model_name="XGB_shallow", ranker="MI"):
    with open(EXP2_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    sub = [r for r in rows if r["ranker"] == ranker and r["model"] == model_name]
    if not sub:
        return 20
    return max(sub, key=lambda x: x["pooled_macro_f1"])["k"]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y_str = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    from xgboost import XGBClassifier

    ranker_name = "MI"
    ranker_fn = RANKERS_CLS[ranker_name]
    best_k_xgb = pick_best_k("XGB_shallow", ranker_name)
    best_k_rf = pick_best_k("RF_shallow", ranker_name)
    best_k_lr = pick_best_k("LR_L2_strong", ranker_name)
    print(f"[exp3_cls] 用 {ranker_name}；XGB K={best_k_xgb}, RF K={best_k_rf}, LR K={best_k_lr}")

    # ---- XGB 网格 ----
    xgb_grid = []
    for depth in [2, 3, 4]:
        for lr in [0.02, 0.05, 0.1]:
            for reg in [1.0, 2.0, 5.0]:
                for n in [300, 500, 800]:
                    xgb_grid.append(dict(max_depth=depth, learning_rate=lr, reg_lambda=reg, n_estimators=n))
    print(f"[exp3_cls] XGB 网格 {len(xgb_grid)} 组，K={best_k_xgb}")

    xgb_results = []
    for i, cfg in enumerate(xgb_grid):
        def factory(cfg=cfg):
            return XGBClassifier(subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", n_jobs=-1,
                                 random_state=RANDOM_STATE, **cfg)
        res, _ = pooled_cv_cls_with_selection(
            factory, X, y_int, groups, N_SPLITS,
            top_k=best_k_xgb, ranker=ranker_fn, preprocessor=None,
        )
        xgb_results.append({
            "cfg": cfg, "pooled_acc": res.pooled_acc,
            "pooled_macro_f1": res.pooled_macro_f1,
            "fold_macro_f1_mean": res.fold_macro_f1_mean,
            "fold_macro_f1_std": res.fold_macro_f1_std,
        })
        if (i + 1) % 9 == 0:
            best_so_far = max(r["pooled_macro_f1"] for r in xgb_results)
            print(f"  XGB progress {i+1}/{len(xgb_grid)}  best F1 so far = {best_so_far:.3f}")
    xgb_sorted = sorted(xgb_results, key=lambda x: -x["pooled_macro_f1"])
    print(f"[exp3_cls] XGB 最佳：{xgb_sorted[0]['cfg']}  F1={xgb_sorted[0]['pooled_macro_f1']:.3f}")

    # ---- RF 网格 ----
    rf_grid = []
    for depth in [3, 4, 6, None]:
        for msl in [2, 3, 5]:
            for n in [300, 500]:
                rf_grid.append(dict(max_depth=depth, min_samples_leaf=msl, n_estimators=n))
    print(f"\n[exp3_cls] RF 网格 {len(rf_grid)} 组，K={best_k_rf}")
    rf_results = []
    for cfg in rf_grid:
        def factory(cfg=cfg):
            return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        res, _ = pooled_cv_cls_with_selection(
            factory, X, y_str, groups, N_SPLITS,
            top_k=best_k_rf, ranker=ranker_fn, preprocessor=median_impute_fold,
        )
        rf_results.append({
            "cfg": {k: (str(v) if v is None else v) for k, v in cfg.items()},
            "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
            "fold_macro_f1_mean": res.fold_macro_f1_mean,
            "fold_macro_f1_std": res.fold_macro_f1_std,
        })
    rf_sorted = sorted(rf_results, key=lambda x: -x["pooled_macro_f1"])
    print(f"[exp3_cls] RF 最佳：{rf_sorted[0]['cfg']}  F1={rf_sorted[0]['pooled_macro_f1']:.3f}")

    # ---- LogisticRegression 网格 ----
    lr_grid = []
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for penalty, solver in [("l2", "lbfgs"), ("l2", "liblinear"), ("l1", "liblinear")]:
            lr_grid.append(dict(C=C, penalty=penalty, solver=solver))
    print(f"\n[exp3_cls] LR 网格 {len(lr_grid)} 组，K={best_k_lr}")
    lr_results = []
    for cfg in lr_grid:
        def factory(cfg=cfg):
            return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **cfg)
        try:
            res, _ = pooled_cv_cls_with_selection(
                factory, X, y_str, groups, N_SPLITS,
                top_k=best_k_lr, ranker=ranker_fn, preprocessor=median_impute_and_scale,
            )
            lr_results.append({
                "cfg": cfg, "pooled_acc": res.pooled_acc,
                "pooled_macro_f1": res.pooled_macro_f1,
                "fold_macro_f1_mean": res.fold_macro_f1_mean,
                "fold_macro_f1_std": res.fold_macro_f1_std,
            })
        except Exception as e:
            lr_results.append({"cfg": cfg, "error": str(e)})
    lr_valid = [r for r in lr_results if "pooled_macro_f1" in r]
    lr_sorted = sorted(lr_valid, key=lambda x: -x["pooled_macro_f1"])
    print(f"[exp3_cls] LR 最佳：{lr_sorted[0]['cfg']}  F1={lr_sorted[0]['pooled_macro_f1']:.3f}")

    # ---- SVC 网格 ----
    svc_grid = []
    for C in [0.5, 1.0, 3.0, 10.0]:
        for gamma in ["scale", 0.01, 0.05, 0.1]:
            svc_grid.append(dict(C=C, gamma=gamma))
    print(f"\n[exp3_cls] SVC 网格 {len(svc_grid)} 组，K={best_k_lr}")
    svc_results = []
    for cfg in svc_grid:
        def factory(cfg=cfg):
            return SVC(kernel="rbf", random_state=RANDOM_STATE, **cfg)
        res, _ = pooled_cv_cls_with_selection(
            factory, X, y_str, groups, N_SPLITS,
            top_k=best_k_lr, ranker=ranker_fn, preprocessor=median_impute_and_scale,
        )
        svc_results.append({
            "cfg": {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()},
            "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
            "fold_macro_f1_mean": res.fold_macro_f1_mean,
            "fold_macro_f1_std": res.fold_macro_f1_std,
        })
    svc_sorted = sorted(svc_results, key=lambda x: -x["pooled_macro_f1"])
    print(f"[exp3_cls] SVC 最佳：{svc_sorted[0]['cfg']}  F1={svc_sorted[0]['pooled_macro_f1']:.3f}")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {"ranker": ranker_name,
                       "top_k_xgb": best_k_xgb, "top_k_rf": best_k_rf, "top_k_lr": best_k_lr},
            "xgb": xgb_results, "rf": rf_results,
            "lr": lr_results, "svc": svc_results,
        }, f, ensure_ascii=False, indent=2)

    write_markdown(ranker_name, best_k_xgb, best_k_rf, best_k_lr,
                   xgb_sorted, rf_sorted, lr_sorted, svc_sorted)
    print(f"\n[exp3_cls] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(ranker_name, k_xgb, k_rf, k_lr,
                   xgb_sorted, rf_sorted, lr_sorted, svc_sorted):
    lines = []
    lines.append(f"# P3 分类调参报告（{ranker_name} 折内筛选）\n\n")
    lines.append(f"**筛选**：XGB K={k_xgb}, RF K={k_rf}, LR/SVC K={k_lr}\n")
    lines.append("**评估**：StratifiedGroupKFold(5) by subject，pooled Macro-F1\n\n")

    def _top_table(sorted_list, top=10):
        out = "| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |\n|---:|---|---:|---:|---:|\n"
        for i, r in enumerate(sorted_list[:top], 1):
            cfg_str = ", ".join(f"{k}={v}" for k, v in r["cfg"].items())
            out += (f"| {i} | {cfg_str} | {r['pooled_acc']:.3f} | "
                    f"{r['pooled_macro_f1']:.3f} | "
                    f"{r['fold_macro_f1_mean']:.3f}±{r['fold_macro_f1_std']:.3f} |\n")
        return out

    lines.append("## XGBoost Top-10\n\n")
    lines.append(_top_table(xgb_sorted))

    lines.append("\n## RandomForest Top-10\n\n")
    lines.append(_top_table(rf_sorted))

    lines.append("\n## LogisticRegression Top-10\n\n")
    lines.append(_top_table(lr_sorted))

    lines.append("\n## SVC-RBF Top-10\n\n")
    lines.append(_top_table(svc_sorted))

    # 综合排名
    lines.append("\n## 四类模型 Top-1 对比\n\n")
    lines.append("| 模型 | 最佳 cfg | pooled Acc | pooled Macro-F1 |\n|---|---|---:|---:|\n")
    for name, arr in [("XGBoost", xgb_sorted), ("RandomForest", rf_sorted),
                       ("LogisticRegression", lr_sorted), ("SVC-RBF", svc_sorted)]:
        b = arr[0]
        cfg_str = ", ".join(f"{k}={v}" for k, v in b["cfg"].items())
        lines.append(f"| {name} | {cfg_str} | {b['pooled_acc']:.3f} | {b['pooled_macro_f1']:.3f} |\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
