#!/usr/bin/env python3
"""统一的评估协议——GroupKFold + 任务级聚合。

用法：
    from evaluate import group_kfold_evaluate
    result = group_kfold_evaluate(model_factory, X, y, groups, sample_ids, n_splits=5)

model_factory 是一个 callable，每次调用返回一个新鲜的、未训练的模型实例
（例如 lambda: RandomForestRegressor(n_estimators=100, random_state=0)）。

评估协议约定（本项目建模一律遵守）：
1. 划分单位：GroupKFold(groups=sample_id)，同一被试-任务的所有窗口只出现在同一折
2. 主指标：任务级 MAE + R²（把测试折内每个 sample_id 的窗口预测聚合到 1 个数，与真值比对）
3. 窗口级 MAE + R² 作为对照报告（不作为主判断依据）
4. 聚合方式：窗口预测在 sample_id 内取中位数（对离群窗口鲁棒，比均值稳）
5. 随机种子固定
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class FoldResult:
    fold: int
    n_train_samples: int
    n_test_samples: int
    n_train_windows: int
    n_test_windows: int
    # 窗口级
    mae_window: float
    r2_window: float
    # 任务级（sample_id 内中位数聚合）
    mae_task: float
    r2_task: float


@dataclass
class EvalResult:
    model_name: str
    n_splits: int
    folds: list[FoldResult]
    # 汇总（任务级为主）
    mae_task_mean: float
    mae_task_std: float
    r2_task_mean: float
    r2_task_std: float
    mae_window_mean: float
    mae_window_std: float
    r2_window_mean: float
    r2_window_std: float

    def summary_line(self) -> str:
        return (
            f"{self.model_name:<24s} | "
            f"任务级 MAE {self.mae_task_mean:.3f}±{self.mae_task_std:.3f} | "
            f"任务级 R² {self.r2_task_mean:+.3f}±{self.r2_task_std:.3f} | "
            f"窗口级 MAE {self.mae_window_mean:.3f}±{self.mae_window_std:.3f} | "
            f"窗口级 R² {self.r2_window_mean:+.3f}±{self.r2_window_std:.3f}"
        )


def _aggregate_task_level(y_true_win: np.ndarray, y_pred_win: np.ndarray,
                           groups_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """窗口级 → 任务级：每个 group 取中位数。真值取任一（同 group 内相同）。"""
    unique_g = np.unique(groups_win)
    y_true_task = np.empty(len(unique_g))
    y_pred_task = np.empty(len(unique_g))
    for i, g in enumerate(unique_g):
        mask = groups_win == g
        y_true_task[i] = y_true_win[mask][0]  # 同 group 内标签一致
        y_pred_task[i] = np.median(y_pred_win[mask])
    return y_true_task, y_pred_task


def group_kfold_evaluate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    fit_kwargs: dict | None = None,
    preprocessor: Callable | None = None,
) -> EvalResult:
    """执行 GroupKFold 评估。

    preprocessor: 可选的预处理函数，签名为
        (X_train, X_test) -> (X_train_prep, X_test_prep)
        用于折内做缺失值填充等（fit 只用训练折统计量）。
    """
    fit_kwargs = fit_kwargs or {}
    gkf = GroupKFold(n_splits=n_splits)

    fold_results: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr, g_te = groups[train_idx], groups[test_idx]

        # 折内预处理（避免测试集信息泄漏到训练）
        if preprocessor is not None:
            X_tr, X_te = preprocessor(X_tr, X_te)

        model = model_factory()
        model.fit(X_tr, y_tr, **fit_kwargs)
        y_pred_win = model.predict(X_te)

        # 窗口级指标
        mae_win = mean_absolute_error(y_te, y_pred_win)
        r2_win = r2_score(y_te, y_pred_win)

        # 任务级聚合
        y_true_task, y_pred_task = _aggregate_task_level(y_te, y_pred_win, g_te)
        mae_task = mean_absolute_error(y_true_task, y_pred_task)
        r2_task = r2_score(y_true_task, y_pred_task) if len(y_true_task) > 1 else float("nan")

        fold_results.append(FoldResult(
            fold=fold_idx,
            n_train_samples=int(len(np.unique(g_tr))),
            n_test_samples=int(len(np.unique(g_te))),
            n_train_windows=int(len(train_idx)),
            n_test_windows=int(len(test_idx)),
            mae_window=float(mae_win),
            r2_window=float(r2_win),
            mae_task=float(mae_task),
            r2_task=float(r2_task),
        ))

    mae_task_vals = np.array([f.mae_task for f in fold_results])
    r2_task_vals = np.array([f.r2_task for f in fold_results])
    mae_win_vals = np.array([f.mae_window for f in fold_results])
    r2_win_vals = np.array([f.r2_window for f in fold_results])

    return EvalResult(
        model_name=model_name,
        n_splits=n_splits,
        folds=fold_results,
        mae_task_mean=float(mae_task_vals.mean()),
        mae_task_std=float(mae_task_vals.std()),
        r2_task_mean=float(np.nanmean(r2_task_vals)),
        r2_task_std=float(np.nanstd(r2_task_vals)),
        mae_window_mean=float(mae_win_vals.mean()),
        mae_window_std=float(mae_win_vals.std()),
        r2_window_mean=float(r2_win_vals.mean()),
        r2_window_std=float(r2_win_vals.std()),
    )


def result_to_dict(r: EvalResult) -> dict:
    """便于 json 序列化"""
    d = asdict(r)
    return d
