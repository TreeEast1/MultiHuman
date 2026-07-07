#!/usr/bin/env python3
"""跑 4 个 baseline，输出对比报告。

模型清单：
    A. 均值预测（零信息下限）——所有测试样本预测训练集 NASA 均值
    B. 单变量线性回归——只用 eeg_frontal_theta_alpha_z_within_subject 一个特征
    C. RandomForestRegressor（默认参数）
    D. XGBoostRegressor（默认参数）+ 原生支持 NaN

评估协议：见 evaluate.py（GroupKFold n_splits=5，任务级 MAE/R² 为主指标）

缺失值处理策略：
    - 线性回归、RandomForest：训练折内中位数填充（fit 时算，transform 到测试折，避免泄漏）
    - XGBoost：原生吃 NaN，不填充
    - 均值预测：不需要看 X
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from evaluate import group_kfold_evaluate, result_to_dict


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
REPORT_DIR = HERE / "baseline_reports"
RANDOM_STATE = 0
N_SPLITS = 5


def median_impute_fold(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """训练折算中位数，填到训练+测试折。"""
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)
    return X_tr, X_te


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    print(f"[baseline] loading dataset from {DATASET_DIR}")
    X = np.load(DATASET_DIR / "X.npy")
    y = np.load(DATASET_DIR / "y.npy")
    groups = np.load(DATASET_DIR / "groups.npy")
    with open(DATASET_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"  X.shape = {X.shape}")
    print(f"  y range = [{y.min():.3f}, {y.max():.3f}]  mean = {y.mean():.3f}")
    print(f"  groups  = {len(np.unique(groups))} 个 sample_id")
    print()

    # 找单变量 baseline 用的特征列索引
    target_col = "eeg_frontal_theta_alpha_z_within_subject"
    single_idx = feature_names.index(target_col)
    print(f"[baseline] 单变量 baseline 使用特征：`{target_col}` (idx={single_idx})")
    print()

    all_results = []

    # ---- A. 均值预测 ----
    print("=" * 100)
    print("[A] MeanPredictor（均值预测，零信息下限）")
    print("=" * 100)
    r = group_kfold_evaluate(
        model_factory=lambda: DummyRegressor(strategy="mean"),
        X=X, y=y, groups=groups,
        model_name="MeanPredictor",
        n_splits=N_SPLITS,
    )
    print(r.summary_line())
    all_results.append(r)
    print()

    # ---- B. 单变量线性回归（EEG frontal theta/alpha z-score）----
    print("=" * 100)
    print(f"[B] Linear (single feature: {target_col})")
    print("=" * 100)
    X_single = X[:, [single_idx]]
    r = group_kfold_evaluate(
        model_factory=lambda: LinearRegression(),
        X=X_single, y=y, groups=groups,
        model_name="Linear_Single_FrontalThetaAlpha_z",
        n_splits=N_SPLITS,
        preprocessor=median_impute_fold,
    )
    print(r.summary_line())
    all_results.append(r)
    print()

    # ---- C. 线性回归（全特征）----
    print("=" * 100)
    print("[C] Linear (all features, median-imputed)")
    print("=" * 100)
    r = group_kfold_evaluate(
        model_factory=lambda: LinearRegression(),
        X=X, y=y, groups=groups,
        model_name="Linear_AllFeatures",
        n_splits=N_SPLITS,
        preprocessor=median_impute_fold,
    )
    print(r.summary_line())
    all_results.append(r)
    print()

    # ---- D. RandomForest ----
    print("=" * 100)
    print("[D] RandomForest (default, median-imputed)")
    print("=" * 100)
    r = group_kfold_evaluate(
        model_factory=lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        X=X, y=y, groups=groups,
        model_name="RandomForest_default",
        n_splits=N_SPLITS,
        preprocessor=median_impute_fold,
    )
    print(r.summary_line())
    all_results.append(r)
    print()

    # ---- E. XGBoost（原生支持 NaN） ----
    print("=" * 100)
    print("[E] XGBoost (default, native NaN handling)")
    print("=" * 100)
    from xgboost import XGBRegressor  # 延迟导入，避免上面 baseline 无 xgb 也能跑
    r = group_kfold_evaluate(
        model_factory=lambda: XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        ),
        X=X, y=y, groups=groups,
        model_name="XGBoost_default",
        n_splits=N_SPLITS,
        preprocessor=None,  # 不填充，XGBoost 原生吃 NaN
    )
    print(r.summary_line())
    all_results.append(r)
    print()

    # ---- 写报告 ----
    write_report(all_results, X.shape, y, groups)


def write_report(all_results, X_shape, y, groups):
    lines = []
    lines.append("# Baseline 对比报告\n\n")
    lines.append(f"**评估协议**：GroupKFold n_splits={N_SPLITS}，按 sample_id 分组\n")
    lines.append(f"**主指标**：任务级 MAE / R²（sample_id 内窗口预测取中位数聚合）\n")
    lines.append(f"**对照指标**：窗口级 MAE / R²（不作主判断依据）\n\n")

    lines.append("## 数据集\n\n")
    lines.append(f"- 窗口数：{X_shape[0]}\n")
    lines.append(f"- 特征数：{X_shape[1]}\n")
    lines.append(f"- 任务数：{len(np.unique(groups))}\n")
    lines.append(f"- NASA 范围：[{y.min():.3f}, {y.max():.3f}]  均值 {y.mean():.3f}  std {y.std():.3f}\n\n")

    lines.append("## 汇总（{}折均值±std）\n\n".format(N_SPLITS))
    lines.append("| 模型 | 任务级 MAE ↓ | 任务级 R² ↑ | 窗口级 MAE ↓ | 窗口级 R² ↑ |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in all_results:
        lines.append(
            f"| {r.model_name} | "
            f"{r.mae_task_mean:.3f} ± {r.mae_task_std:.3f} | "
            f"{r.r2_task_mean:+.3f} ± {r.r2_task_std:.3f} | "
            f"{r.mae_window_mean:.3f} ± {r.mae_window_std:.3f} | "
            f"{r.r2_window_mean:+.3f} ± {r.r2_window_std:.3f} |\n"
        )

    # 逐折详情
    lines.append("\n## 逐折详情\n\n")
    for r in all_results:
        lines.append(f"### {r.model_name}\n\n")
        lines.append("| fold | train samples | test samples | task MAE | task R² | win MAE | win R² |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for f in r.folds:
            lines.append(
                f"| {f.fold} | {f.n_train_samples} | {f.n_test_samples} | "
                f"{f.mae_task:.3f} | {f.r2_task:+.3f} | "
                f"{f.mae_window:.3f} | {f.r2_window:+.3f} |\n"
            )
        lines.append("\n")

    # 解读指引
    lines.append("## 解读指引\n\n")
    lines.append("- **零信息下限（MeanPredictor）**：任何模型必须**任务级 MAE 低于**它，才谈得上有效学习。\n")
    lines.append("- **单变量 vs 全特征**：如果全特征 R² 比单变量高很多 → 多模态融合有效；差不多 → 大部分特征贡献冗余。\n")
    lines.append("- **线性 vs 树模型**：树模型显著更好 → 存在非线性关系；差不多 → 线性可解释模型可能已够用。\n")
    lines.append("- **XGBoost vs RandomForest**：XGB 通常表现最好，且原生吃 NaN 无需插值——如果两者差距很小说明数据集不够复杂或还没到调参环节。\n")
    lines.append("- **任务级 vs 窗口级 R²**：窗口级 R² 系统性高于任务级 → 模型在利用同任务窗口相似性；本项目应以任务级为准。\n\n")

    # 保存 markdown 与 json
    report_md = REPORT_DIR / "baseline_report.md"
    report_md.write_text("".join(lines), encoding="utf-8")

    json_out = [result_to_dict(r) for r in all_results]
    with open(REPORT_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"[baseline] 报告已写入 {report_md}")


if __name__ == "__main__":
    main()
