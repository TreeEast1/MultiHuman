#!/usr/bin/env python3
"""P0 任务级建模 baseline。

评估协议（**已修复**）：
    - 划分：GroupKFold(n_splits=5, groups=subject)——26 名被试分到 5 折，同被试的所有任务只在同折
    - 任务级聚合：按 sample_id（每行=1任务），与划分 group 解耦
    - 每行本身就是任务级样本，任务级指标 = 窗口级指标

补充口径：
    - "各折 R² 均值"：传统汇报方式，但小样本（~17 任务/折）方差大
    - "pooled R²/MAE"：把所有折的测试预测拼一起，在 84 个 (真值, 预测) 上算总指标，更稳健
      这是回归论文小样本 CV 的推荐主指标（Kohavi 1995; Bengio & Grandvalet 2004）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from evaluate import group_kfold_evaluate, result_to_dict  # noqa: E402

DATASET_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_baseline"
RANDOM_STATE = 0
N_SPLITS = 5


def median_impute_fold(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_test)


def median_impute_and_scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    return Xtr, Xte


def pooled_cv(model_factory, X, y, groups_subject, n_splits, preprocessor=None):
    """返回 pooled 预测（长度=n_samples），以及各折的详细。
    与 group_kfold_evaluate 类似，但收集所有折预测。"""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred_all = np.full(len(y), np.nan)
    fold_details = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups_subject)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if preprocessor is not None:
            X_tr, X_te = preprocessor(X_tr, X_te)
        m = model_factory()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_pred_all[test_idx] = y_pred
        fold_details.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_subjects": int(len(np.unique(groups_subject[train_idx]))),
            "n_test_subjects": int(len(np.unique(groups_subject[test_idx]))),
            "fold_mae": float(mean_absolute_error(y_te, y_pred)),
            "fold_r2": float(r2_score(y_te, y_pred)) if len(y_te) > 1 else float("nan"),
        })
    # 所有 84 个样本都应被预测过
    assert not np.isnan(y_pred_all).any(), "有样本未落入任何测试折"
    pooled_mae = float(mean_absolute_error(y, y_pred_all))
    pooled_r2 = float(r2_score(y, y_pred_all))
    fold_mae_arr = np.array([f["fold_mae"] for f in fold_details])
    fold_r2_arr = np.array([f["fold_r2"] for f in fold_details])
    return {
        "pooled_mae": pooled_mae,
        "pooled_r2": pooled_r2,
        "fold_mae_mean": float(fold_mae_arr.mean()),
        "fold_mae_std": float(fold_mae_arr.std()),
        "fold_r2_mean": float(np.nanmean(fold_r2_arr)),
        "fold_r2_std": float(np.nanstd(fold_r2_arr)),
        "fold_details": fold_details,
        "y_pred_all": y_pred_all,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[baseline_task] loading from {DATASET_DIR}")
    X = np.load(DATASET_DIR / "X_task.npy")
    y = np.load(DATASET_DIR / "y_task.npy")
    groups_subj = np.load(DATASET_DIR / "groups_task.npy")
    sample_ids = np.load(DATASET_DIR / "sample_task.npy", allow_pickle=True)
    with open(DATASET_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"  X.shape = {X.shape}")
    print(f"  y range = [{y.min():.3f}, {y.max():.3f}]  mean = {y.mean():.3f}  std = {y.std():.3f}")
    print(f"  subjects = {len(np.unique(groups_subj))}, samples = {len(y)}")
    print()

    # 单变量列
    target_col = "eeg_frontal_theta_alpha_z_within_subject__mean"
    if target_col not in feature_names:
        target_col = [c for c in feature_names if "frontal_theta_alpha" in c and c.endswith("__mean")][0]
    single_idx = feature_names.index(target_col)
    print(f"[baseline_task] 单变量 baseline 特征：`{target_col}` (idx={single_idx})")
    print()

    experiments = [
        # (name, X_input, model_factory, preprocessor)
        ("MeanPredictor", X, lambda: DummyRegressor(strategy="mean"), None),
        (f"Linear_Single_FrontalThetaAlpha", X[:, [single_idx]],
         lambda: LinearRegression(), median_impute_fold),
        ("Linear_AllFeatures", X, lambda: LinearRegression(), median_impute_and_scale),
        ("Ridge_alpha1", X, lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE), median_impute_and_scale),
        ("Ridge_alpha10", X, lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE), median_impute_and_scale),
        ("Ridge_alpha100", X, lambda: Ridge(alpha=100.0, random_state=RANDOM_STATE), median_impute_and_scale),
        ("RandomForest_default", X,
         lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold),
        ("RandomForest_shallow", X,
         lambda: RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                       random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold),
    ]

    from xgboost import XGBRegressor
    experiments.append(
        ("XGBoost_default", X,
         lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
         None)
    )
    experiments.append(
        ("XGBoost_shallow", X,
         lambda: XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                              reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
         None)
    )

    all_results = []
    for name, X_input, factory, prep in experiments:
        print("=" * 100)
        print(f"[{name}]  X.shape = {X_input.shape}")
        print("=" * 100)
        res = pooled_cv(factory, X_input, y, groups_subj, N_SPLITS, prep)
        print(f"  pooled MAE = {res['pooled_mae']:.3f}   pooled R² = {res['pooled_r2']:+.3f}")
        print(f"  fold  MAE = {res['fold_mae_mean']:.3f} ± {res['fold_mae_std']:.3f}")
        print(f"  fold  R²  = {res['fold_r2_mean']:+.3f} ± {res['fold_r2_std']:.3f}")
        all_results.append({"name": name, **res})
        print()

    # 参考：Mean 预测的 baseline_mae 是 y 与 y_mean 的 MAE
    mean_absolute_deviation = float(np.mean(np.abs(y - y.mean())))
    print(f"参考：|y - mean(y)| 的均值 = {mean_absolute_deviation:.3f}（若 pooled MAE 低于此则模型有实质贡献）")

    write_report(all_results, X.shape, y, groups_subj, mean_absolute_deviation)


def write_report(all_results, X_shape, y, groups, mad_baseline):
    lines = []
    lines.append("# P0 任务级建模 Baseline 报告（84 行 × 264 特征，5×GroupKFold by subject）\n\n")
    lines.append("**评估协议**\n")
    lines.append(f"- 划分：GroupKFold(n_splits={N_SPLITS}, groups=subject)，26 名被试分 5 折\n")
    lines.append("- 主指标 **pooled MAE / R²**：所有折的测试预测拼成 84 个 (真值,预测) 后统一计算\n")
    lines.append("- 参考指标 fold-mean：各折分别算再取均值，波动大仅供对照\n\n")

    lines.append("## 数据集\n\n")
    lines.append(f"- 样本数：{X_shape[0]}\n")
    lines.append(f"- 特征维度：{X_shape[1]}\n")
    lines.append(f"- 独立被试数：{len(np.unique(groups))}\n")
    lines.append(f"- NASA 范围：[{y.min():.3f}, {y.max():.3f}]  均值 {y.mean():.3f}  std {y.std():.3f}\n")
    lines.append(f"- 参考基线 |y - mean(y)| 均值：**{mad_baseline:.3f}**（任何模型 pooled MAE 应低于此才有实质贡献）\n\n")

    lines.append("## 汇总（按 pooled R² 排序）\n\n")
    lines.append("| 模型 | pooled MAE ↓ | pooled R² ↑ | fold MAE (mean±std) | fold R² (mean±std) |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in sorted(all_results, key=lambda x: -x["pooled_r2"]):
        lines.append(
            f"| {r['name']} | "
            f"{r['pooled_mae']:.3f} | "
            f"{r['pooled_r2']:+.3f} | "
            f"{r['fold_mae_mean']:.3f} ± {r['fold_mae_std']:.3f} | "
            f"{r['fold_r2_mean']:+.3f} ± {r['fold_r2_std']:.3f} |\n"
        )

    lines.append("\n## 逐折详情\n\n")
    for r in all_results:
        lines.append(f"### {r['name']}\n\n")
        lines.append("| fold | train subs | test subs | n_test | fold MAE | fold R² |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for f in r["fold_details"]:
            lines.append(
                f"| {f['fold']} | {f['n_train_subjects']} | {f['n_test_subjects']} | "
                f"{f['n_test']} | {f['fold_mae']:.3f} | {f['fold_r2']:+.3f} |\n"
            )
        lines.append("\n")

    lines.append("## 与窗口级 baseline 的对照\n\n")
    lines.append("| 建模粒度 | 最佳模型 | 任务级 MAE | 任务级 R² |\n|---|---|---:|---:|\n")
    lines.append("| 窗口级（12624 行 × 66 特征） | RandomForest | 1.170 | +0.126 |\n")
    best = max(all_results, key=lambda x: x["pooled_r2"])
    lines.append(f"| **任务级（84 行 × 264 特征）** | **{best['name']}** | **{best['pooled_mae']:.3f}** | **{best['pooled_r2']:+.3f}** |\n\n")

    lines.append("## 解读\n\n")
    lines.append(f"- 零信息基线 |y-mean(y)| = {mad_baseline:.3f}\n")
    lines.append("- pooled R² > 0 才算模型真正学到了预测跨被试 NASA 的能力\n")
    lines.append("- 若 pooled R² < 0：说明当前特征方案在 84 样本量下无法泛化到未见被试\n")
    lines.append("- 若 pooled R² > 0 但 fold R² 波动大：说明少数被试的可预测性差异是主导误差源\n")

    (REPORT_DIR / "baseline_report.md").write_text("".join(lines), encoding="utf-8")

    # JSON（去掉 numpy array）
    json_ready = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k != "y_pred_all"}
        json_ready.append(d)
    with open(REPORT_DIR / "baseline_results.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready, fp, ensure_ascii=False, indent=2)
    print(f"[baseline_task] 报告已写入 {REPORT_DIR}/baseline_report.md")


if __name__ == "__main__":
    main()
