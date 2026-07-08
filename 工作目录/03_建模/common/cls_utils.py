#!/usr/bin/env python3
"""分类实验共用工具。

关键点：
1. 使用 StratifiedGroupKFold 保证每折训练/测试集都覆盖 3 类难度，
   同时保证同一 subject 的所有任务只出现在同一折（防被试个体信息泄漏）。
2. 主指标：pooled Accuracy / Macro-F1（合并 5 折预测计算）
3. 折内特征筛选：MI / RF importance / Permutation，训练折内单独筛，防泄漏
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold
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
class ClsCVResult:
    name: str
    n_features: int
    n_classes: int
    pooled_acc: float
    pooled_macro_f1: float
    pooled_weighted_f1: float
    pooled_per_class_f1: dict            # {class: f1}
    fold_acc_mean: float
    fold_acc_std: float
    fold_macro_f1_mean: float
    fold_macro_f1_std: float
    confusion: list                       # confusion matrix as nested list
    class_labels: list
    fold_details: list
    y_pred_pooled: np.ndarray


def pooled_cv_cls(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    preprocessor: Callable | None = None,
    name: str = "",
) -> ClsCVResult:
    """StratifiedGroupKFold 分类 CV。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(list(np.unique(y)))
    # 用与 y 相同 dtype 初始化，避免 str/int 混合触发 sklearn 报错
    y_pred_all = np.empty(len(y), dtype=y.dtype)
    filled = np.zeros(len(y), dtype=bool)

    fold_details = []
    for fold_idx, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if preprocessor is not None:
            X_tr, X_te = preprocessor(X_tr, X_te)
        m = model_factory()
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)
        y_pred_all[te] = y_hat
        filled[te] = True

        acc = accuracy_score(y_te, y_hat)
        mac_f1 = f1_score(y_te, y_hat, average="macro", labels=class_labels, zero_division=0)
        train_class_counts = {str(c): int((y_tr == c).sum()) for c in class_labels}
        test_class_counts = {str(c): int((y_te == c).sum()) for c in class_labels}
        fold_details.append({
            "fold": fold_idx,
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_test_subjects": int(len(np.unique(groups[te]))),
            "n_test": int(len(te)),
            "train_class_counts": train_class_counts,
            "test_class_counts": test_class_counts,
            "fold_acc": float(acc),
            "fold_macro_f1": float(mac_f1),
        })

    assert filled.all(), "有样本未被预测"

    # pooled metrics
    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_mac_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_w_f1 = float(f1_score(y, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    per_class = f1_score(y, y_pred_all, average=None, labels=class_labels, zero_division=0)
    per_class_dict = {c: float(v) for c, v in zip(class_labels, per_class)}
    cm = confusion_matrix(y, y_pred_all, labels=class_labels)

    facc = np.array([f["fold_acc"] for f in fold_details])
    ff1 = np.array([f["fold_macro_f1"] for f in fold_details])

    return ClsCVResult(
        name=name,
        n_features=int(X.shape[1]),
        n_classes=len(class_labels),
        pooled_acc=pooled_acc,
        pooled_macro_f1=pooled_mac_f1,
        pooled_weighted_f1=pooled_w_f1,
        pooled_per_class_f1=per_class_dict,
        fold_acc_mean=float(facc.mean()),
        fold_acc_std=float(facc.std()),
        fold_macro_f1_mean=float(ff1.mean()),
        fold_macro_f1_std=float(ff1.std()),
        confusion=cm.tolist(),
        class_labels=list(class_labels),
        fold_details=fold_details,
        y_pred_pooled=y_pred_all,
    )


# ---------------- 折内特征筛选 ---------------- #

def _rank_by_mi_cls(X_tr_prep, y_tr):
    mi = mutual_info_classif(X_tr_prep, y_tr, random_state=RANDOM_STATE)
    return np.argsort(-mi)


def _rank_by_rf_importance_cls(X_tr_prep, y_tr):
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_prep, y_tr)
    return np.argsort(-rf.feature_importances_)


def _rank_by_permutation_cls(X_tr_prep, y_tr):
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_prep, y_tr)
    perm = permutation_importance(rf, X_tr_prep, y_tr, n_repeats=5,
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring="f1_macro")
    return np.argsort(-perm.importances_mean)


RANKERS_CLS = {
    "MI": _rank_by_mi_cls,
    "RF_importance": _rank_by_rf_importance_cls,
    "Permutation": _rank_by_permutation_cls,
}


def pooled_cv_cls_with_selection(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    top_k: int,
    ranker: Callable,
    preprocessor: Callable | None,
    name: str = "",
) -> tuple[ClsCVResult, list[np.ndarray]]:
    """折内做特征筛选后跑 pooled 分类 CV。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(list(np.unique(y)))
    y_pred_all = np.empty(len(y), dtype=y.dtype)
    filled = np.zeros(len(y), dtype=bool)
    fold_details = []
    selected_per_fold = []
    for fold_idx, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        X_tr_full, X_te_full = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_full)
        X_te_imp = imputer.transform(X_te_full)

        rank_idx = ranker(X_tr_imp, y_tr)
        top_idx = rank_idx[:top_k]
        selected_per_fold.append(top_idx.copy())

        X_tr_sel = X_tr_imp[:, top_idx]
        X_te_sel = X_te_imp[:, top_idx]

        if preprocessor is not None:
            X_tr_sel, X_te_sel = preprocessor(X_tr_sel, X_te_sel)

        m = model_factory()
        m.fit(X_tr_sel, y_tr)
        y_hat = m.predict(X_te_sel)
        y_pred_all[te] = y_hat
        filled[te] = True

        acc = accuracy_score(y_te, y_hat)
        mac_f1 = f1_score(y_te, y_hat, average="macro", labels=class_labels, zero_division=0)
        fold_details.append({
            "fold": fold_idx,
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_test_subjects": int(len(np.unique(groups[te]))),
            "n_test": int(len(te)),
            "fold_acc": float(acc),
            "fold_macro_f1": float(mac_f1),
            "selected_idx": top_idx.tolist(),
        })

    assert filled.all()

    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_mac_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_w_f1 = float(f1_score(y, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    per_class = f1_score(y, y_pred_all, average=None, labels=class_labels, zero_division=0)
    per_class_dict = {c: float(v) for c, v in zip(class_labels, per_class)}
    cm = confusion_matrix(y, y_pred_all, labels=class_labels)

    facc = np.array([f["fold_acc"] for f in fold_details])
    ff1 = np.array([f["fold_macro_f1"] for f in fold_details])

    res = ClsCVResult(
        name=name, n_features=int(top_k), n_classes=len(class_labels),
        pooled_acc=pooled_acc, pooled_macro_f1=pooled_mac_f1,
        pooled_weighted_f1=pooled_w_f1, pooled_per_class_f1=per_class_dict,
        fold_acc_mean=float(facc.mean()), fold_acc_std=float(facc.std()),
        fold_macro_f1_mean=float(ff1.mean()), fold_macro_f1_std=float(ff1.std()),
        confusion=cm.tolist(), class_labels=list(class_labels),
        fold_details=fold_details, y_pred_pooled=y_pred_all,
    )
    return res, selected_per_fold
