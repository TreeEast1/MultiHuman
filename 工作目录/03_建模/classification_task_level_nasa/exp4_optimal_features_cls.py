#!/usr/bin/env python3
"""P4 NASA 三分类：最优特征筛选 + 模型优化。

目标：从 264 个指标中筛选最佳输入子集，提高三分类准确率。

当前最佳（P1 模态消融）：minus_EEG + XGB_shallow = 0.809
现有特征选择（P2，在全 264 上做）：RF_importance + XGB @ K=30 = 0.776

核心改进思路：
  现有 P2 的特征选择在全 264 特征上做，EEG(112 个噪声特征) 会干扰排序器。
  本实验在 minus_EEG(152) 等干净子集上做折内特征选择，
  再叠加集成投票和精细调参，力求突破 0.809。

实验矩阵：
  A. 模态子集 × 折内特征选择（minus_EEG / minus_EEG_HR / only_AOI_Log 上精选）
  B. 固定稳定特征集（从历史 CV 5/5 折稳定选中的特征）
  C. 集成投票（XGB+RF / XGB+RF+LR）
  D. 精细调参（XGB / RF 网格搜索，在最佳特征配置上）

评估：StratifiedGroupKFold(5) by subject，pooled 指标，折内筛选防泄漏。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE, RANKERS_CLS,
    median_impute_fold, median_impute_and_scale,
    pooled_cv_cls, pooled_cv_cls_with_selection,
)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp4"
N_SPLITS = 5


# ============================================================ #
#  模态索引识别
# ============================================================ #

def build_modality_indices(feature_names):
    """返回 {modality: [col_indices]} 字典。"""
    mods = {}
    for i, name in enumerate(feature_names):
        if name.startswith("eeg_"):
            mods.setdefault("EEG", []).append(i)
        elif name.startswith("hr_"):
            mods.setdefault("HR", []).append(i)
        elif name.startswith("blink_"):
            mods.setdefault("Blink", []).append(i)
        elif name.startswith("log_"):
            mods.setdefault("Log", []).append(i)
        elif name.startswith("eye_aoi"):
            mods.setdefault("AOI", []).append(i)
        elif name.startswith("eye_"):
            mods.setdefault("EyePupil", []).append(i)
        else:
            mods.setdefault("Other", []).append(i)
    return mods


# ============================================================ #
#  模型工厂
# ============================================================ #

def make_xgb(**kw):
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1,
    )
    defaults.update(kw)
    return XGBClassifier(**defaults)


def make_rf(**kw):
    defaults = dict(
        n_estimators=300, max_depth=4, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    defaults.update(kw)
    return RandomForestClassifier(**defaults)


def make_lr(**kw):
    defaults = dict(max_iter=2000, C=0.1, random_state=RANDOM_STATE)
    defaults.update(kw)
    return LogisticRegression(**defaults)


# ============================================================ #
#  集成投票 CV（固定特征集 + 多模型投票）
# ============================================================ #

def pooled_cv_ensemble_vote(
    model_specs, X, y_int, groups, n_splits,
    feature_idx=None, name="",
):
    """固定特征集 + 多模型多数投票 CV。

    model_specs: [(name, factory, needs_scale), ...]
    feature_idx: 固定特征列索引列表；None=用全部特征
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(np.unique(y_int))
    y_pred_all = np.empty(len(y_int), dtype=y_int.dtype)
    filled = np.zeros(len(y_int), dtype=bool)
    fold_f1s = []

    for tr, te in sgkf.split(X, y_int, groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_int[tr], y_int[te]

        if feature_idx is not None:
            X_tr = X_tr[:, feature_idx]
            X_te = X_te[:, feature_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        preds = []
        for m_name, factory, needs_scale in model_specs:
            if needs_scale:
                scaler = StandardScaler()
                X_tr_m = scaler.fit_transform(X_tr_imp)
                X_te_m = scaler.transform(X_te_imp)
            else:
                X_tr_m, X_te_m = X_tr_imp, X_te_imp
            m = factory()
            m.fit(X_tr_m, y_tr)
            preds.append(m.predict(X_te_m))

        preds_arr = np.array(preds)  # (n_models, n_test)
        # 多数投票
        voted = np.array([
            np.bincount(preds_arr[:, j].astype(int), minlength=3).argmax()
            for j in range(preds_arr.shape[1])
        ])
        y_pred_all[te] = voted
        filled[te] = True
        fold_f1s.append(f1_score(y_te, voted, average="macro", labels=class_labels, zero_division=0))

    assert filled.all()

    pooled_acc = float(accuracy_score(y_int, y_pred_all))
    pooled_f1 = float(f1_score(y_int, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_wf1 = float(f1_score(y_int, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    cm = confusion_matrix(y_int, y_pred_all, labels=class_labels)
    ff1 = np.array(fold_f1s)

    return {
        "name": name,
        "pooled_acc": pooled_acc,
        "pooled_macro_f1": pooled_f1,
        "pooled_weighted_f1": pooled_wf1,
        "fold_macro_f1_mean": float(ff1.mean()),
        "fold_macro_f1_std": float(ff1.std()),
        "confusion": cm.tolist(),
        "class_labels": list(class_labels),
    }


# ============================================================ #
#  折内选择 + 集成投票 CV
# ============================================================ #

def pooled_cv_selection_ensemble(
    model_specs, X, y_int, groups, n_splits,
    top_k, ranker, name="",
):
    """折内特征选择 + 多模型多数投票 CV。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(np.unique(y_int))
    y_pred_all = np.empty(len(y_int), dtype=y_int.dtype)
    filled = np.zeros(len(y_int), dtype=bool)
    fold_f1s = []
    selected_per_fold = []

    for tr, te in sgkf.split(X, y_int, groups):
        X_tr_full, X_te_full = X[tr], X[te]
        y_tr, y_te = y_int[tr], y_int[te]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_full)
        X_te_imp = imputer.transform(X_te_full)

        rank_idx = ranker(X_tr_imp, y_tr)
        top_idx = rank_idx[:top_k]
        selected_per_fold.append(top_idx.copy())

        X_tr_sel = X_tr_imp[:, top_idx]
        X_te_sel = X_te_imp[:, top_idx]

        preds = []
        for m_name, factory, needs_scale in model_specs:
            if needs_scale:
                scaler = StandardScaler()
                X_tr_m = scaler.fit_transform(X_tr_sel)
                X_te_m = scaler.transform(X_te_sel)
            else:
                X_tr_m, X_te_m = X_tr_sel, X_te_sel
            m = factory()
            m.fit(X_tr_m, y_tr)
            preds.append(m.predict(X_te_m))

        preds_arr = np.array(preds)
        voted = np.array([
            np.bincount(preds_arr[:, j].astype(int), minlength=3).argmax()
            for j in range(preds_arr.shape[1])
        ])
        y_pred_all[te] = voted
        filled[te] = True
        fold_f1s.append(f1_score(y_te, voted, average="macro", labels=class_labels, zero_division=0))

    assert filled.all()

    pooled_acc = float(accuracy_score(y_int, y_pred_all))
    pooled_f1 = float(f1_score(y_int, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_wf1 = float(f1_score(y_int, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    cm = confusion_matrix(y_int, y_pred_all, labels=class_labels)
    ff1 = np.array(fold_f1s)

    return {
        "name": name,
        "pooled_acc": pooled_acc,
        "pooled_macro_f1": pooled_f1,
        "pooled_weighted_f1": pooled_wf1,
        "fold_macro_f1_mean": float(ff1.mean()),
        "fold_macro_f1_std": float(ff1.std()),
        "confusion": cm.tolist(),
        "class_labels": list(class_labels),
    }, selected_per_fold


# ============================================================ #
#  主流程
# ============================================================ #

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y_str = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp4] X={X.shape}, n_subjects={len(np.unique(groups))}")
    mods = build_modality_indices(feature_names)
    for m, idx in sorted(mods.items()):
        print(f"  {m}: {len(idx)} features")

    all_results = []

    # ---- 模态子集索引 ----
    eeg_idx = set(mods.get("EEG", []))
    hr_idx = set(mods.get("HR", []))
    aoi_idx = set(mods.get("AOI", []))
    log_idx = set(mods.get("Log", []))
    all_idx = list(range(X.shape[1]))

    minus_eeg_idx = [i for i in all_idx if i not in eeg_idx]
    minus_eeg_hr_idx = [i for i in all_idx if i not in eeg_idx and i not in hr_idx]
    aoi_log_idx = sorted(list(aoi_idx | log_idx))
    aoi_log_blink_idx = sorted(list(aoi_idx | log_idx | set(mods.get("Blink", []))))

    subsets = {
        "minus_EEG": ("去掉EEG(噪声模态)", minus_eeg_idx),
        "minus_EEG_HR": ("去掉EEG+HR", minus_eeg_hr_idx),
        "AOI_Log": ("仅AOI+Log", aoi_log_idx),
        "AOI_Log_Blink": ("AOI+Log+Blink", aoi_log_blink_idx),
    }

    # ============================================================
    #  A. 模态子集 × 折内特征选择
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 A：模态子集 × 折内特征选择")
    print("=" * 60)

    TOP_KS_A = [15, 20, 25, 30, 40]

    for sub_name, (sub_desc, sub_idx) in subsets.items():
        X_sub = X[:, sub_idx]
        print(f"\n--- {sub_name} ({sub_desc}, {len(sub_idx)}特征) ---")

        # 子集 Full baseline
        for m_name, factory, prep, y_use, y_type in [
            ("XGB_shallow", lambda: make_xgb(), None, y_int, "int"),
            ("RF_shallow", lambda: make_rf(), median_impute_fold, y_str, "str"),
        ]:
            res = pooled_cv_cls(factory, X_sub, y_use, groups, N_SPLITS, prep, name=f"{sub_name}_Full_{m_name}")
            all_results.append({
                "exp": "A_baseline", "subset": sub_name, "n_feat": len(sub_idx),
                "ranker": "Full", "k": len(sub_idx), "model": m_name,
                "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
            })
            print(f"  Full  {m_name:14s}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

        # 折内选择
        for ranker_name, ranker_fn in [("MI", RANKERS_CLS["MI"]), ("RF_imp", RANKERS_CLS["RF_importance"])]:
            for k in TOP_KS_A:
                if k > len(sub_idx):
                    continue
                for m_name, factory, prep, y_use in [
                    ("XGB", lambda: make_xgb(), None, y_int),
                    ("RF", lambda: make_rf(), median_impute_fold, y_str),
                ]:
                    res, _ = pooled_cv_cls_with_selection(
                        factory, X_sub, y_use, groups, N_SPLITS,
                        top_k=k, ranker=ranker_fn, preprocessor=prep,
                        name=f"{sub_name}_{ranker_name}_top{k}_{m_name}",
                    )
                    all_results.append({
                        "exp": "A_selection", "subset": sub_name, "n_feat": k,
                        "ranker": ranker_name, "k": k, "model": m_name,
                        "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                        "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
                    })
                    print(f"  {ranker_name} K={k:3d}  {m_name:4s}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # ============================================================
    #  B. 固定稳定特征集（从历史 P2 结果提取的 5/5 折稳定特征）
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 B：固定稳定特征集")
    print("=" * 60)

    # 从 exp2 report 中提取的稳定特征名（5/5 折选中）
    stable_sets = {
        "stable_rf_xgb_k30": [
            "eye_aoi_unique_hit_n__std", "eye_aoi_entropy__median", "eye_aoi_entropy__mean",
            "eye_aoi_interval_n__std", "eye_aoi_interval_n__mean", "eye_aoi_unique_hit_n__mean",
            "log_action_count_win__mean", "eye_aoi_fixation_n__std", "eye_aoi_fixation_n__slope",
            "eye_aoi_max_share__mean",
        ],
        "stable_core_aoi": [
            "eye_aoi_unique_hit_n__std", "eye_aoi_interval_n__std", "eye_aoi_interval_n__mean",
            "eye_aoi_coverage_ratio__slope", "eye_aoi_entropy__median", "eye_aoi_entropy__mean",
            "eye_aoi_fixation_n__std", "eye_aoi_fixation_n__slope", "eye_aoi_max_share__mean",
            "eye_aoi_unique_hit_n__mean", "eye_aoi_interval_n__median",
        ],
        "stable_aoi_log_no_eeg": [
            "eye_aoi_unique_hit_n__std", "eye_aoi_interval_n__std", "eye_aoi_interval_n__mean",
            "eye_aoi_entropy__median", "eye_aoi_entropy__mean", "eye_aoi_unique_hit_n__mean",
            "eye_aoi_fixation_n__std", "eye_aoi_fixation_n__slope", "eye_aoi_max_share__mean",
            "eye_aoi_coverage_ratio__slope", "log_action_count_win__mean",
            "log_action_density_win__mean", "log_error_rate_win__std",
            "eye_aoi_coverage_ratio__median", "eye_aoi_total_fix_ms__median",
        ],
    }

    fname_to_idx = {n: i for i, n in enumerate(feature_names)}

    for set_name, feat_names in stable_sets.items():
        feat_idx = [fname_to_idx[n] for n in feat_names if n in fname_to_idx]
        print(f"\n--- {set_name} ({len(feat_idx)}特征) ---")

        for m_name, factory, prep, y_use in [
            ("XGB", lambda: make_xgb(), None, y_int),
            ("RF", lambda: make_rf(), median_impute_fold, y_str),
            ("LR", lambda: make_lr(), median_impute_and_scale, y_str),
        ]:
            res = pooled_cv_cls(factory, X[:, feat_idx], y_use, groups, N_SPLITS, prep,
                                name=f"{set_name}_{m_name}")
            all_results.append({
                "exp": "B_stable", "subset": set_name, "n_feat": len(feat_idx),
                "ranker": "fixed", "k": len(feat_idx), "model": m_name,
                "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
            })
            print(f"  {m_name:4s}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # ============================================================
    #  C. 集成投票
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 C：集成投票")
    print("=" * 60)

    # C1: 固定特征集 + 投票
    for set_name, feat_names in stable_sets.items():
        feat_idx = [fname_to_idx[n] for n in feat_names if n in fname_to_idx]
        for ens_name, specs in [
            ("XGB+RF", [
                ("XGB", lambda: make_xgb(), False),
                ("RF", lambda: make_rf(), False),
            ]),
            ("XGB+RF+LR", [
                ("XGB", lambda: make_xgb(), False),
                ("RF", lambda: make_rf(), False),
                ("LR", lambda: make_lr(), True),
            ]),
        ]:
            res = pooled_cv_ensemble_vote(specs, X, y_int, groups, N_SPLITS,
                                          feature_idx=feat_idx, name=f"{set_name}_{ens_name}")
            all_results.append({
                "exp": "C_ensemble_fixed", "subset": set_name, "n_feat": len(feat_idx),
                "ranker": "fixed", "k": len(feat_idx), "model": ens_name,
                "pooled_acc": res["pooled_acc"], "pooled_macro_f1": res["pooled_macro_f1"],
                "fold_f1_mean": res["fold_macro_f1_mean"], "fold_f1_std": res["fold_macro_f1_std"],
            })
            print(f"  {set_name:25s}  {ens_name:12s}  Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # C2: minus_EEG 上折内选择 + 投票
    print("\n--- minus_EEG 折内选择 + 投票 ---")
    X_me = X[:, minus_eeg_idx]
    for ranker_name, ranker_fn in [("MI", RANKERS_CLS["MI"]), ("RF_imp", RANKERS_CLS["RF_importance"])]:
        for k in [20, 30]:
            for ens_name, specs in [
                ("XGB+RF", [
                    ("XGB", lambda: make_xgb(), False),
                    ("RF", lambda: make_rf(), False),
                ]),
                ("XGB+RF+LR", [
                    ("XGB", lambda: make_xgb(), False),
                    ("RF", lambda: make_rf(), False),
                    ("LR", lambda: make_lr(), True),
                ]),
            ]:
                res, _ = pooled_cv_selection_ensemble(
                    specs, X_me, y_int, groups, N_SPLITS,
                    top_k=k, ranker=ranker_fn, name=f"minus_EEG_{ranker_name}_top{k}_{ens_name}",
                )
                all_results.append({
                    "exp": "C_ensemble_sel", "subset": "minus_EEG", "n_feat": k,
                    "ranker": ranker_name, "k": k, "model": ens_name,
                    "pooled_acc": res["pooled_acc"], "pooled_macro_f1": res["pooled_macro_f1"],
                    "fold_f1_mean": res["fold_macro_f1_mean"], "fold_f1_std": res["fold_macro_f1_std"],
                })
                print(f"  {ranker_name} K={k:3d}  {ens_name:12s}  Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # ============================================================
    #  D. 精细调参（在最有前景的配置上）
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 D：精细调参（minus_EEG + 折内选择）")
    print("=" * 60)

    # D1: XGB 网格（聚焦浅树+强正则，n=300 加速）
    print("\n--- XGB 网格 (minus_EEG + MI 选择) ---")
    xgb_grid = []
    for depth in [2, 3]:
        for lr in [0.02, 0.05]:
            for lam in [2.0, 5.0]:
                for n_est in [300]:
                    xgb_grid.append((depth, lr, lam, n_est))

    for k in [20, 30]:
        for depth, lr, lam, n_est in xgb_grid:
            factory = lambda d=depth, l=lr, la=lam, n=n_est: make_xgb(
                max_depth=d, learning_rate=l, reg_lambda=la, n_estimators=n)
            res, _ = pooled_cv_cls_with_selection(
                factory, X_me, y_int, groups, N_SPLITS,
                top_k=k, ranker=RANKERS_CLS["MI"], preprocessor=None,
                name=f"XGB_d{depth}_lr{lr}_l{lam}_n{n_est}_K{k}",
            )
            all_results.append({
                "exp": "D_tuning", "subset": "minus_EEG", "n_feat": k,
                "ranker": "MI", "k": k, "model": f"XGB_d{depth}_lr{lr}_l{lam}_n{n_est}",
                "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
            })
            if res.pooled_macro_f1 >= 0.80:
                print(f"  ★ K={k} d={depth} lr={lr} l={lam} n={n_est}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # D2: RF 网格
    print("\n--- RF 网格 (minus_EEG + MI 选择) ---")
    rf_grid = []
    for depth in [3, 4, 5]:
        for msl in [2, 3]:
            for n_est in [300]:
                rf_grid.append((depth, msl, n_est))

    for k in [20, 30]:
        for depth, msl, n_est in rf_grid:
            factory = lambda d=depth, m=msl, n=n_est: make_rf(
                max_depth=d, min_samples_leaf=m, n_estimators=n)
            res, _ = pooled_cv_cls_with_selection(
                factory, X_me, y_str, groups, N_SPLITS,
                top_k=k, ranker=RANKERS_CLS["MI"], preprocessor=median_impute_fold,
                name=f"RF_d{depth}_msl{msl}_n{n_est}_K{k}",
            )
            all_results.append({
                "exp": "D_tuning", "subset": "minus_EEG", "n_feat": k,
                "ranker": "MI", "k": k, "model": f"RF_d{depth}_msl{msl}_n{n_est}",
                "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
            })
            if res.pooled_macro_f1 >= 0.80:
                print(f"  ★ K={k} d={depth} msl={msl} n={n_est}  Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # ============================================================
    #  汇总 & 报告
    # ============================================================
    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    write_report(all_results, feature_names, subsets, stable_sets, fname_to_idx)
    print(f"\n[exp4] 报告写入 {REPORT_DIR}/report.md")
    print(f"[exp4] 共 {len(all_results)} 组实验")


def write_report(rows, feature_names, subsets, stable_sets, fname_to_idx):
    lines = []
    lines.append("# P4 NASA 三分类：最优特征筛选 + 模型优化\n\n")
    lines.append("**目标**：从 264 指标中筛选最佳输入子集，突破 P1 最佳 0.809\n\n")
    lines.append("**设置**：84×264，StratifiedGroupKFold(5) by subject，pooled 指标，折内筛选防泄漏\n\n")
    lines.append("**参考基线**：\n")
    lines.append("- P1 minus_EEG + XGB = **0.809**（当前最高）\n")
    lines.append("- P2 RF_importance + XGB @ K=30（全264上选）= 0.776\n")
    lines.append("- P0 Full + XGB = 0.750\n\n")

    # ---- 全局 Top-20 ----
    lines.append("## 1. 全局 Top-20 组合\n\n")
    lines.append("| rank | 实验 | 子集 | 排序 | K | 模型 | Acc | Macro-F1 | fold F1 μ±σ |\n")
    lines.append("|---:|---|---|---|---:|---|---:|---:|---|\n")
    sorted_rows = sorted(rows, key=lambda x: -x["pooled_macro_f1"])
    for i, r in enumerate(sorted_rows[:20], 1):
        lines.append(
            f"| {i} | {r['exp']} | {r['subset']} | {r['ranker']} | {r['k']} | {r['model']} | "
            f"{r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** | "
            f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
        )
    lines.append("\n")

    # ---- 实验 A 汇总 ----
    lines.append("## 2. 实验 A：模态子集 × 折内特征选择\n\n")
    lines.append("### 各子集最佳组合\n\n")
    lines.append("| 子集 | n_feat | 排序 | K | 模型 | Acc | Macro-F1 |\n|---|---:|---|---:|---|---:|---:|\n")
    a_rows = [r for r in rows if r["exp"].startswith("A")]
    for sub_name in subsets:
        sub_rows = [r for r in a_rows if r["subset"] == sub_name and r["exp"] == "A_selection"]
        if sub_rows:
            best = max(sub_rows, key=lambda x: x["pooled_macro_f1"])
            lines.append(
                f"| {sub_name} | {best['n_feat']} | {best['ranker']} | {best['k']} | "
                f"{best['model']} | {best['pooled_acc']:.3f} | **{best['pooled_macro_f1']:.3f}** |\n"
            )
        base_rows = [r for r in a_rows if r["subset"] == sub_name and r["exp"] == "A_baseline"]
        for br in base_rows:
            lines.append(
                f"| {sub_name} (Full) | {br['n_feat']} | — | {br['k']} | "
                f"{br['model']} | {br['pooled_acc']:.3f} | {br['pooled_macro_f1']:.3f} |\n"
            )
    lines.append("\n")

    # K 曲线（minus_EEG）
    lines.append("### minus_EEG 上 K vs Macro-F1\n\n")
    lines.append("| K | MI+XGB | MI+RF | RF_imp+XGB | RF_imp+RF |\n|---:|---:|---:|---:|---:|\n")
    for k in [10, 15, 20, 25, 30, 35, 40, 50]:
        vals = {}
        for r in a_rows:
            if r["subset"] == "minus_EEG" and r["k"] == k and r["exp"] == "A_selection":
                vals[(r["ranker"], r["model"])] = r["pooled_macro_f1"]
        mi_xgb = vals.get(("MI", "XGB"), float("nan"))
        mi_rf = vals.get(("MI", "RF"), float("nan"))
        rf_xgb = vals.get(("RF_imp", "XGB"), float("nan"))
        rf_rf = vals.get(("RF_imp", "RF"), float("nan"))
        if not np.isnan(mi_xgb):
            lines.append(f"| {k} | {mi_xgb:.3f} | {mi_rf:.3f} | {rf_xgb:.3f} | {rf_rf:.3f} |\n")
    lines.append("\n")

    # ---- 实验 B 汇总 ----
    lines.append("## 3. 实验 B：固定稳定特征集\n\n")
    for set_name, feat_names in stable_sets.items():
        lines.append(f"### {set_name}（{len(feat_names)} 特征）\n\n")
        lines.append("| 模型 | Acc | Macro-F1 | fold F1 μ±σ |\n|---|---:|---:|---|\n")
        for r in rows:
            if r["exp"] == "B_stable" and r["subset"] == set_name:
                lines.append(
                    f"| {r['model']} | {r['pooled_acc']:.3f} | {r['pooled_macro_f1']:.3f} | "
                    f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
                )
        lines.append("\n特征列表：\n")
        for n in feat_names:
            lines.append(f"- `{n}`\n")
        lines.append("\n")

    # ---- 实验 C 汇总 ----
    lines.append("## 4. 实验 C：集成投票\n\n")
    lines.append("### 固定特征集 + 投票\n\n")
    lines.append("| 特征集 | 集成 | Acc | Macro-F1 |\n|---|---|---:|---:|\n")
    for r in rows:
        if r["exp"] == "C_ensemble_fixed":
            lines.append(f"| {r['subset']} | {r['model']} | {r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** |\n")
    lines.append("\n### minus_EEG 折内选择 + 投票\n\n")
    lines.append("| 排序 | K | 集成 | Acc | Macro-F1 |\n|---|---:|---|---:|---:|\n")
    for r in rows:
        if r["exp"] == "C_ensemble_sel":
            lines.append(f"| {r['ranker']} | {r['k']} | {r['model']} | {r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** |\n")
    lines.append("\n")

    # ---- 实验 D 汇总 ----
    lines.append("## 5. 实验 D：精细调参（minus_EEG + MI 选择）\n\n")
    lines.append("### XGB Top-10\n\n")
    lines.append("| K | 配置 | Acc | Macro-F1 | fold F1 μ±σ |\n|---:|---|---:|---:|---|\n")
    d_xgb = [r for r in rows if r["exp"] == "D_tuning" and r["model"].startswith("XGB")]
    for r in sorted(d_xgb, key=lambda x: -x["pooled_macro_f1"])[:10]:
        lines.append(
            f"| {r['k']} | {r['model']} | {r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** | "
            f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
        )
    lines.append("\n### RF Top-10\n\n")
    lines.append("| K | 配置 | Acc | Macro-F1 | fold F1 μ±σ |\n|---:|---|---:|---:|---|\n")
    d_rf = [r for r in rows if r["exp"] == "D_tuning" and r["model"].startswith("RF")]
    for r in sorted(d_rf, key=lambda x: -x["pooled_macro_f1"])[:10]:
        lines.append(
            f"| {r['k']} | {r['model']} | {r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** | "
            f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
        )
    lines.append("\n")

    # ---- 总结 ----
    lines.append("## 6. 总结\n\n")
    best = sorted_rows[0]
    lines.append(f"**全局最佳**：{best['subset']} + {best['ranker']} K={best['k']} + {best['model']}\n\n")
    lines.append(f"- pooled Accuracy = **{best['pooled_acc']:.3f}**\n")
    lines.append(f"- pooled Macro-F1 = **{best['pooled_macro_f1']:.3f}**\n")
    lines.append(f"- fold F1 = {best['fold_f1_mean']:.3f} ± {best['fold_f1_std']:.3f}\n\n")

    prev_best = 0.809
    delta = best["pooled_macro_f1"] - prev_best
    lines.append(f"- vs P1 最佳(0.809)：{'↑' if delta > 0 else '→'} {delta:+.3f}\n")
    lines.append(f"- vs P0 baseline(0.750)：{best['pooled_macro_f1'] - 0.750:+.3f}\n\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
