#!/usr/bin/env python3
"""消融/筛选/调参实验共用的工具。

关键点：
1. pooled_cv：把 5 折测试预测拼成 84 个 (真值, 预测)，一次算 MAE / R²（小样本主指标）
2. 折内预处理：中位数填充、标准化都必须 fit 只用训练折，避免泄漏
3. select_topk_inside_fold：折内特征筛选（MI / RF importance / Permutation），
   完全在训练集里选，防止测试信号泄漏到特征选择
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 0


# ---------------- 预处理 ---------------- #

def median_impute_fold(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_test)


def median_impute_and_scale(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    return Xtr, Xte


# ---------------- CV 主流程 ---------------- #

@dataclass
class CVResult:
    name: str
    n_features: int
    pooled_mae: float
    pooled_r2: float
    fold_mae_mean: float
    fold_mae_std: float
    fold_r2_mean: float
    fold_r2_std: float
    fold_details: list
    y_pred_pooled: np.ndarray


def pooled_cv(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    preprocessor: Callable | None = None,
    name: str = "",
) -> CVResult:
    """标准 pooled group k-fold 评估。"""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred_all = np.full(len(y), np.nan)
    fold_details = []
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if preprocessor is not None:
            X_tr, X_te = preprocessor(X_tr, X_te)
        m = model_factory()
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)
        y_pred_all[te] = y_hat
        fold_details.append({
            "fold": fold_idx,
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_test_subjects": int(len(np.unique(groups[te]))),
            "n_test": int(len(te)),
            "fold_mae": float(mean_absolute_error(y_te, y_hat)),
            "fold_r2": float(r2_score(y_te, y_hat)) if len(y_te) > 1 else float("nan"),
        })
    assert not np.isnan(y_pred_all).any()
    fmae = np.array([f["fold_mae"] for f in fold_details])
    fr2 = np.array([f["fold_r2"] for f in fold_details])
    return CVResult(
        name=name,
        n_features=int(X.shape[1]),
        pooled_mae=float(mean_absolute_error(y, y_pred_all)),
        pooled_r2=float(r2_score(y, y_pred_all)),
        fold_mae_mean=float(fmae.mean()),
        fold_mae_std=float(fmae.std()),
        fold_r2_mean=float(np.nanmean(fr2)),
        fold_r2_std=float(np.nanstd(fr2)),
        fold_details=fold_details,
        y_pred_pooled=y_pred_all,
    )


# ---------------- 折内特征筛选 ---------------- #

def _rank_by_mi(X_tr_prep: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """返回按 MI 从高到低的特征索引数组。"""
    mi = mutual_info_regression(X_tr_prep, y_tr, random_state=RANDOM_STATE)
    return np.argsort(-mi)


def _rank_by_rf_importance(X_tr_prep: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_prep, y_tr)
    return np.argsort(-rf.feature_importances_)


def _rank_by_permutation(X_tr_prep: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_prep, y_tr)
    perm = permutation_importance(rf, X_tr_prep, y_tr, n_repeats=5,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    return np.argsort(-perm.importances_mean)


RANKERS = {
    "MI": _rank_by_mi,
    "RF_importance": _rank_by_rf_importance,
    "Permutation": _rank_by_permutation,
}


def pooled_cv_with_selection(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    top_k: int,
    ranker: Callable,
    preprocessor: Callable | None,
    name: str = "",
) -> tuple[CVResult, list[np.ndarray]]:
    """折内做特征筛选后跑 pooled CV。返回结果 + 每折选中的索引。"""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred_all = np.full(len(y), np.nan)
    fold_details = []
    selected_per_fold = []
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr_full, X_te_full = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 特征筛选之前要先做基本的中位数填充（否则 MI/RF importance 遇 NaN 出错）
        # 注意：这个填充只在训练折上 fit，测试折 transform
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_full)
        X_te_imp = imputer.transform(X_te_full)

        # 折内 rank
        rank_idx = ranker(X_tr_imp, y_tr)
        top_idx = rank_idx[:top_k]
        selected_per_fold.append(top_idx.copy())

        X_tr_sel = X_tr_imp[:, top_idx]
        X_te_sel = X_te_imp[:, top_idx]

        # 后续预处理（如标准化）在选中特征上做
        if preprocessor is not None:
            X_tr_sel, X_te_sel = preprocessor(X_tr_sel, X_te_sel)

        m = model_factory()
        m.fit(X_tr_sel, y_tr)
        y_hat = m.predict(X_te_sel)
        y_pred_all[te] = y_hat
        fold_details.append({
            "fold": fold_idx,
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_test_subjects": int(len(np.unique(groups[te]))),
            "n_test": int(len(te)),
            "fold_mae": float(mean_absolute_error(y_te, y_hat)),
            "fold_r2": float(r2_score(y_te, y_hat)) if len(y_te) > 1 else float("nan"),
            "selected_idx": top_idx.tolist(),
        })

    assert not np.isnan(y_pred_all).any()
    fmae = np.array([f["fold_mae"] for f in fold_details])
    fr2 = np.array([f["fold_r2"] for f in fold_details])
    res = CVResult(
        name=name,
        n_features=int(top_k),
        pooled_mae=float(mean_absolute_error(y, y_pred_all)),
        pooled_r2=float(r2_score(y, y_pred_all)),
        fold_mae_mean=float(fmae.mean()),
        fold_mae_std=float(fmae.std()),
        fold_r2_mean=float(np.nanmean(fr2)),
        fold_r2_std=float(np.nanstd(fr2)),
        fold_details=fold_details,
        y_pred_pooled=y_pred_all,
    )
    return res, selected_per_fold
