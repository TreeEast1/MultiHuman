#!/usr/bin/env python3
"""P1 分类模态消融实验。

在 84 × 264 任务级表上，做 3 类实验：
1. Full baseline
2. Leave-One-Modality-Out：EEG / HR / EyePupil / AOI / Blink / Log
3. Only-One-Modality
4. Only-One-Statistic：mean / std / median / slope

每类都用 4 个代表模型跑：LogisticRegression_L2_strong / RF_shallow / XGB_shallow / SVC_RBF

主指标：pooled Accuracy / Macro-F1
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
    RANDOM_STATE, median_impute_fold, median_impute_and_scale, pooled_cv_cls,
)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp1"
N_SPLITS = 5


def build_modality_masks(feature_names):
    def match(prefix_check):
        return np.array([prefix_check(c) for c in feature_names])

    def is_eye_pupil(c):
        return c.startswith("eye_") and "aoi" not in c

    return {
        "EEG": match(lambda c: c.startswith("eeg_")),
        "HR": match(lambda c: c.startswith("hr_")),
        "EyePupil": match(is_eye_pupil),
        "AOI": match(lambda c: c.startswith("eye_aoi_")),
        "Blink": match(lambda c: c.startswith("blink_")),
        "Log": match(lambda c: c.startswith("log_")),
    }


def build_stat_masks(feature_names):
    stats = ["mean", "std", "median", "slope"]
    return {s: np.array([c.endswith(f"__{s}") for c in feature_names]) for s in stats}


def run_all_models(X, y, y_int, groups, subset_name):
    from xgboost import XGBClassifier
    experiments = [
        ("LR_L2_strong",
         lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
         median_impute_and_scale, y),
        ("SVC_RBF",
         lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
         median_impute_and_scale, y),
        ("RF_shallow",
         lambda: RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                        random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold, y),
        ("XGB_shallow",
         lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=3,
                               reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
         None, y_int),
    ]
    out = []
    for m_name, factory, prep, y_this in experiments:
        res = pooled_cv_cls(factory, X, y_this, groups, N_SPLITS, prep,
                            name=f"{subset_name}__{m_name}")
        out.append({
            "subset": subset_name, "model": m_name,
            "n_features": res.n_features,
            "pooled_acc": res.pooled_acc,
            "pooled_macro_f1": res.pooled_macro_f1,
            "pooled_weighted_f1": res.pooled_weighted_f1,
            "fold_acc_mean": res.fold_acc_mean, "fold_acc_std": res.fold_acc_std,
            "fold_macro_f1_mean": res.fold_macro_f1_mean, "fold_macro_f1_std": res.fold_macro_f1_std,
        })
    return out


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp1_cls] X={X.shape}, subjects={len(np.unique(groups))}")
    modality_masks = build_modality_masks(feature_names)
    stat_masks = build_stat_masks(feature_names)

    for mod, mask in modality_masks.items():
        print(f"  {mod}: {mask.sum()} 列")

    all_rows = []

    print("\n=== Full ===")
    all_rows += run_all_models(X, y, y_int, groups, "Full_264")

    print("\n=== Leave-One-Modality-Out ===")
    for mod, mask in modality_masks.items():
        idx = np.where(~mask)[0]
        print(f"  minus_{mod}: {len(idx)}")
        all_rows += run_all_models(X[:, idx], y, y_int, groups, f"minus_{mod}")

    print("\n=== Only-One-Modality ===")
    for mod, mask in modality_masks.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        print(f"  only_{mod}: {len(idx)}")
        all_rows += run_all_models(X[:, idx], y, y_int, groups, f"only_{mod}")

    print("\n=== Only-One-Statistic ===")
    for stat, mask in stat_masks.items():
        idx = np.where(mask)[0]
        print(f"  only_{stat}: {len(idx)}")
        all_rows += run_all_models(X[:, idx], y, y_int, groups, f"only_{stat}")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    write_markdown(all_rows, modality_masks, stat_masks)
    print(f"\n[exp1_cls] 报告写入 {REPORT_DIR}/report.md")


def write_markdown(rows, modality_masks, stat_masks):
    lines = []
    lines.append("# P1 分类模态消融实验报告\n\n")
    lines.append("**设置**：84 样本 × 264 特征，StratifiedGroupKFold(5) by subject，主指标 pooled Macro-F1\n\n")

    lines.append("## Full baseline\n\n")
    _write_subset_table(lines, rows, ["Full_264"])

    lines.append("## Leave-One-Modality-Out（去除某一模态后表现）\n\n")
    _write_subset_table(lines, rows, [f"minus_{m}" for m in modality_masks.keys()])
    lines.append("解读：与 Full 相比 Macro-F1 下降越多 → 该模态贡献越大\n\n")

    lines.append("## Only-One-Modality（仅使用某一模态）\n\n")
    _write_subset_table(lines, rows, [f"only_{m}" for m in modality_masks.keys()])
    lines.append("解读：单模态 Macro-F1 越高 → 该模态独立预测能力越强\n\n")

    lines.append("## Only-One-Statistic（仅使用某一统计量的所有 66 列）\n\n")
    _write_subset_table(lines, rows, [f"only_{s}" for s in stat_masks.keys()])
    lines.append("解读：判断 mean/std/median/slope 四种聚合方式的相对价值\n\n")

    lines.append("## 全部实验按 XGB Macro-F1 排序\n\n")
    xgb_rows = [r for r in rows if r["model"] == "XGB_shallow"]
    xgb_sorted = sorted(xgb_rows, key=lambda x: -x["pooled_macro_f1"])
    lines.append("| 排名 | subset | n_feat | pooled Acc | pooled Macro-F1 |\n|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(xgb_sorted, 1):
        lines.append(f"| {i} | {r['subset']} | {r['n_features']} | {r['pooled_acc']:.3f} | {r['pooled_macro_f1']:.3f} |\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


def _write_subset_table(lines, all_rows, subset_names):
    lines.append("| subset | n_feat | LR F1 | SVC F1 | RF F1 | XGB F1 | XGB Acc |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name in subset_names:
        row_map = {r["model"]: r for r in all_rows if r["subset"] == name}
        if not row_map:
            continue
        lr = row_map.get("LR_L2_strong")
        sv = row_map.get("SVC_RBF")
        rf = row_map.get("RF_shallow")
        xgb = row_map.get("XGB_shallow")
        n_feat = xgb["n_features"] if xgb else "-"
        lines.append(
            f"| {name} | {n_feat} | "
            f"{lr['pooled_macro_f1']:.3f} | "
            f"{sv['pooled_macro_f1']:.3f} | "
            f"{rf['pooled_macro_f1']:.3f} | "
            f"{xgb['pooled_macro_f1']:.3f} | "
            f"{xgb['pooled_acc']:.3f} |\n"
        )
    lines.append("\n")


if __name__ == "__main__":
    main()
