#!/usr/bin/env python3
"""P4b NASA 三分类：稳定特征基底 + 自适应补充 + Stacking。

P4 发现：
  - minus_EEG + MI K=30 + XGB = 0.809（30特征，持平P1最佳）
  - 固定15特征(stable_aoi_log_no_eeg) + XGB = 0.798（15特征，差0.011）

本实验尝试突破 0.809：
  A. 稳定基底(15) + 折内MI自适应补充(从剩余minus_EEG中选K个)
  B. Stacking 集成（XGB+RF base, LR meta）
  C. 固定15特征 + 更多调参
  D. minus_EEG + RF_importance 折内选择 + 精细调参（P4只调了MI）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
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


def make_xgb(**kw):
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=300, learning_rate=0.02, max_depth=3,
        reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1,
    )
    defaults.update(kw)
    return XGBClassifier(**defaults)


def make_rf(**kw):
    defaults = dict(
        n_estimators=300, max_depth=5, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    defaults.update(kw)
    return RandomForestClassifier(**defaults)


# 固定15特征（P4最佳稳定集）
STABLE_15 = [
    "eye_aoi_unique_hit_n__std", "eye_aoi_interval_n__std",
    "eye_aoi_interval_n__mean", "eye_aoi_entropy__median",
    "eye_aoi_entropy__mean", "eye_aoi_unique_hit_n__mean",
    "eye_aoi_fixation_n__std", "eye_aoi_fixation_n__slope",
    "eye_aoi_max_share__mean", "eye_aoi_coverage_ratio__slope",
    "log_action_count_win__mean", "log_action_density_win__mean",
    "log_error_rate_win__std", "eye_aoi_coverage_ratio__median",
    "eye_aoi_total_fix_ms__median",
]


def build_modality_indices(feature_names):
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
    return mods


# ============================================================ #
#  稳定基底 + 自适应补充 CV
# ============================================================ #

def pooled_cv_stable_plus_adaptive(
    model_factory, X, y, groups, n_splits,
    stable_idx, candidate_idx, top_k_adaptive, ranker,
    preprocessor=None, needs_scale=False, name="",
):
    """固定稳定特征 + 折内从候选集中MI选择补充特征。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(np.unique(y))
    y_pred_all = np.empty(len(y), dtype=y.dtype)
    filled = np.zeros(len(y), dtype=bool)
    fold_f1s = []

    for tr, te in sgkf.split(X, y, groups):
        X_tr_full, X_te_full = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_full)
        X_te_imp = imputer.transform(X_te_full)

        # 从候选集中折内排序
        X_tr_cand = X_tr_imp[:, candidate_idx]
        rank_idx = ranker(X_tr_cand, y_tr)
        candidate_arr = np.array(candidate_idx)
        adaptive_top = candidate_arr[rank_idx[:top_k_adaptive]]

        # 合并稳定特征 + 自适应特征
        combined = sorted(set(stable_idx) | set(adaptive_top))
        X_tr_sel = X_tr_imp[:, combined]
        X_te_sel = X_te_imp[:, combined]

        if preprocessor is not None:
            X_tr_sel, X_te_sel = preprocessor(X_tr_sel, X_te_sel)
        elif needs_scale:
            scaler = StandardScaler()
            X_tr_sel = scaler.fit_transform(X_tr_sel)
            X_te_sel = scaler.transform(X_te_sel)

        m = model_factory()
        m.fit(X_tr_sel, y_tr)
        y_hat = m.predict(X_te_sel)
        y_pred_all[te] = y_hat
        filled[te] = True
        fold_f1s.append(f1_score(y_te, y_hat, average="macro", labels=class_labels, zero_division=0))

    assert filled.all()
    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    ff1 = np.array(fold_f1s)
    return {
        "name": name, "pooled_acc": pooled_acc, "pooled_macro_f1": pooled_f1,
        "fold_f1_mean": float(ff1.mean()), "fold_f1_std": float(ff1.std()),
    }


# ============================================================ #
#  Stacking CV
# ============================================================ #

def pooled_cv_stacking(
    X, y, groups, n_splits, feature_idx=None, name="",
):
    """Stacking: XGB + RF base → LR meta，固定特征集。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(np.unique(y))
    y_pred_all = np.empty(len(y), dtype=y.dtype)
    filled = np.zeros(len(y), dtype=bool)
    fold_f1s = []

    for tr, te in sgkf.split(X, y, groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        if feature_idx is not None:
            X_tr = X_tr[:, feature_idx]
            X_te = X_te[:, feature_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        from xgboost import XGBClassifier
        estimators = [
            ("xgb", XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=3,
                                   reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
                                   random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]
        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
            cv=3, n_jobs=-1,
        )
        stacker.fit(X_tr_imp, y_tr)
        y_hat = stacker.predict(X_te_imp)
        y_pred_all[te] = y_hat
        filled[te] = True
        fold_f1s.append(f1_score(y_te, y_hat, average="macro", labels=class_labels, zero_division=0))

    assert filled.all()
    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    ff1 = np.array(fold_f1s)
    return {
        "name": name, "pooled_acc": pooled_acc, "pooled_macro_f1": pooled_f1,
        "fold_f1_mean": float(ff1.mean()), "fold_f1_std": float(ff1.std()),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y_str = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[exp4b] X={X.shape}")
    mods = build_modality_indices(feature_names)
    eeg_idx = set(mods.get("EEG", []))
    all_idx = list(range(X.shape[1]))
    minus_eeg_idx = [i for i in all_idx if i not in eeg_idx]

    fname_to_idx = {n: i for i, n in enumerate(feature_names)}
    stable_15_idx = [fname_to_idx[n] for n in STABLE_15 if n in fname_to_idx]
    # 候选特征 = minus_EEG 中不在 stable_15 的特征
    candidate_idx = [i for i in minus_eeg_idx if i not in set(stable_15_idx)]

    all_results = []

    # ============================================================
    #  A. 稳定基底(15) + 自适应补充
    # ============================================================
    print("=" * 60)
    print("实验 A：稳定基底(15) + 折内MI自适应补充")
    print("=" * 60)

    for k_add in [0, 5, 10, 15, 20, 25, 30]:
        res = pooled_cv_stable_plus_adaptive(
            lambda: make_xgb(), X, y_int, groups, N_SPLITS,
            stable_15_idx, candidate_idx, k_add, RANKERS_CLS["MI"],
            needs_scale=False, name=f"stable15+MI{k_add}_XGB",
        )
        all_results.append({
            "exp": "A_stable_adaptive", "model": "XGB",
            "stable_k": 15, "adaptive_k": k_add, "total_k": 15 + k_add,
            **res,
        })
        print(f"  stable15 + MI top{k_add:2d}  XGB  total={15+k_add:3d}  "
              f"Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # 也试 RF_importance 排序的补充
    for k_add in [5, 10, 15, 20]:
        res = pooled_cv_stable_plus_adaptive(
            lambda: make_xgb(), X, y_int, groups, N_SPLITS,
            stable_15_idx, candidate_idx, k_add, RANKERS_CLS["RF_importance"],
            needs_scale=False, name=f"stable15+RFimp{k_add}_XGB",
        )
        all_results.append({
            "exp": "A_stable_adaptive", "model": "XGB",
            "stable_k": 15, "adaptive_k": k_add, "total_k": 15 + k_add,
            **res,
        })
        print(f"  stable15 + RFimp top{k_add:2d}  XGB  total={15+k_add:3d}  "
              f"Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # ============================================================
    #  B. Stacking 集成
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 B：Stacking 集成")
    print("=" * 60)

    # B1: 固定15特征
    res = pooled_cv_stacking(X, y_str, groups, N_SPLITS, feature_idx=stable_15_idx,
                             name="stack_stable15")
    all_results.append({"exp": "B_stacking", "subset": "stable15", "n_feat": 15, **res})
    print(f"  Stacking stable15(15)  Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # B2: minus_EEG Full
    res = pooled_cv_stacking(X, y_str, groups, N_SPLITS, feature_idx=minus_eeg_idx,
                             name="stack_minus_EEG")
    all_results.append({"exp": "B_stacking", "subset": "minus_EEG", "n_feat": 152, **res})
    print(f"  Stacking minus_EEG(152)  Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # B3: Full 264
    res = pooled_cv_stacking(X, y_str, groups, N_SPLITS, feature_idx=None,
                             name="stack_full264")
    all_results.append({"exp": "B_stacking", "subset": "full264", "n_feat": 264, **res})
    print(f"  Stacking full264(264)  Acc={res['pooled_acc']:.3f}  F1={res['pooled_macro_f1']:.3f}")

    # ============================================================
    #  C. minus_EEG + RF_importance 折内选择 + XGB调参
    #     （P4只调了MI排序，这里补RF_importance）
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 C：minus_EEG + RF_importance 选择 + XGB调参")
    print("=" * 60)

    X_me = X[:, minus_eeg_idx]
    for k in [15, 20, 25, 30]:
        for depth in [2, 3]:
            for lr in [0.02, 0.05]:
                for lam in [2.0, 5.0]:
                    factory = lambda d=depth, l=lr, la=lam: make_xgb(
                        max_depth=d, learning_rate=l, reg_lambda=la)
                    res, _ = pooled_cv_cls_with_selection(
                        factory, X_me, y_int, groups, N_SPLITS,
                        top_k=k, ranker=RANKERS_CLS["RF_importance"], preprocessor=None,
                        name=f"RFimp_K{k}_d{depth}_lr{lr}_l{lam}",
                    )
                    all_results.append({
                        "exp": "C_rfimp_tuning", "subset": "minus_EEG",
                        "ranker": "RF_imp", "k": k,
                        "model": f"XGB_d{depth}_lr{lr}_l{lam}",
                        "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                        "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
                    })
                    if res.pooled_macro_f1 >= 0.80:
                        print(f"  ★ RFimp K={k} d={depth} lr={lr} l={lam}  "
                              f"Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # ============================================================
    #  D. 固定15特征 + XGB调参
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 D：固定15特征 + XGB调参")
    print("=" * 60)

    for depth in [2, 3, 4]:
        for lr in [0.01, 0.02, 0.05]:
            for lam in [2.0, 5.0, 10.0]:
                for n_est in [300, 500]:
                    factory = lambda d=depth, l=lr, la=lam, n=n_est: make_xgb(
                        max_depth=d, learning_rate=l, reg_lambda=la, n_estimators=n)
                    res = pooled_cv_cls(factory, X[:, stable_15_idx], y_int, groups,
                                        N_SPLITS, None, name=f"stable15_d{depth}_lr{lr}_l{lam}_n{n_est}")
                    all_results.append({
                        "exp": "D_stable15_tuning", "subset": "stable15", "n_feat": 15,
                        "model": f"XGB_d{depth}_lr{lr}_l{lam}_n{n_est}",
                        "pooled_acc": res.pooled_acc, "pooled_macro_f1": res.pooled_macro_f1,
                        "fold_f1_mean": res.fold_macro_f1_mean, "fold_f1_std": res.fold_macro_f1_std,
                    })
                    if res.pooled_macro_f1 >= 0.80:
                        print(f"  ★ d={depth} lr={lr} l={lam} n={n_est}  "
                              f"Acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")

    # ============================================================
    #  汇总
    # ============================================================
    with open(REPORT_DIR / "results_b.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    write_report_b(all_results, feature_names)
    print(f"\n[exp4b] 共 {len(all_results)} 组实验，报告写入 {REPORT_DIR}/report_b.md")


def write_report_b(rows, feature_names):
    lines = []
    lines.append("# P4b NASA 三分类：稳定基底+自适应补充+Stacking+调参\n\n")
    lines.append("**目标**：突破 P4 最佳 0.809\n\n")
    lines.append("**参考基线**：P4 minus_EEG + MI K=30 + XGB(d3,lr0.02,λ5) = **0.809**\n\n")

    # Top-20
    lines.append("## 1. 全局 Top-20\n\n")
    lines.append("| rank | 实验 | 描述 | 模型 | Acc | Macro-F1 | fold F1 μ±σ |\n|---:|---|---|---|---:|---:|---|\n")
    sorted_rows = sorted(rows, key=lambda x: -x.get("pooled_macro_f1", 0))
    for i, r in enumerate(sorted_rows[:20], 1):
        desc = r.get("name", f"{r.get('subset','')} K={r.get('k','?')}")
        if r["exp"] == "A_stable_adaptive":
            desc = f"stable15+{r.get('adaptive_k',0)} adaptive"
        elif r["exp"] == "B_stacking":
            desc = f"stacking {r.get('subset','')}"
        elif r["exp"] == "C_rfimp_tuning":
            desc = f"minus_EEG RFimp K={r.get('k','?')}"
        elif r["exp"] == "D_stable15_tuning":
            desc = f"stable15"
        lines.append(
            f"| {i} | {r['exp']} | {desc} | {r.get('model','stack')} | "
            f"{r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** | "
            f"{r.get('fold_f1_mean',0):.3f}±{r.get('fold_f1_std',0):.3f} |\n"
        )
    lines.append("\n")

    # 实验 A
    lines.append("## 2. 稳定基底(15) + 自适应补充\n\n")
    lines.append("| 基底 | 自适应K | 排序 | 总特征数 | Acc | Macro-F1 |\n|---|---:|---|---:|---:|---:|\n")
    for r in rows:
        if r["exp"] == "A_stable_adaptive":
            ranker = "MI" if "MI" in r.get("name", "") else "RF_imp"
            lines.append(
                f"| stable15 | {r['adaptive_k']} | {ranker} | {r['total_k']} | "
                f"{r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** |\n"
            )
    lines.append("\n")

    # 实验 B
    lines.append("## 3. Stacking 集成\n\n")
    lines.append("| 特征集 | n_feat | Acc | Macro-F1 | fold F1 μ±σ |\n|---|---:|---:|---:|---|\n")
    for r in rows:
        if r["exp"] == "B_stacking":
            lines.append(
                f"| {r['subset']} | {r['n_feat']} | {r['pooled_acc']:.3f} | "
                f"**{r['pooled_macro_f1']:.3f}** | {r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
            )
    lines.append("\n")

    # 实验 D Top-10
    lines.append("## 4. 固定15特征 + XGB调参 Top-10\n\n")
    lines.append("| 配置 | Acc | Macro-F1 | fold F1 μ±σ |\n|---|---:|---:|---|\n")
    d_rows = [r for r in rows if r["exp"] == "D_stable15_tuning"]
    for r in sorted(d_rows, key=lambda x: -x["pooled_macro_f1"])[:10]:
        lines.append(
            f"| {r['model']} | {r['pooled_acc']:.3f} | **{r['pooled_macro_f1']:.3f}** | "
            f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
        )
    lines.append("\n")

    # 总结
    best = sorted_rows[0]
    lines.append("## 5. 总结\n\n")
    lines.append(f"**全局最佳**：{best.get('name', best.get('model', ''))}\n")
    lines.append(f"- pooled Acc = **{best['pooled_acc']:.3f}**\n")
    lines.append(f"- pooled Macro-F1 = **{best['pooled_macro_f1']:.3f}**\n")
    lines.append(f"- fold F1 = {best.get('fold_f1_mean',0):.3f} ± {best.get('fold_f1_std',0):.3f}\n\n")
    lines.append(f"- vs P4 最佳(0.809)：{best['pooled_macro_f1'] - 0.809:+.3f}\n")
    lines.append(f"- vs P0 baseline(0.750)：{best['pooled_macro_f1'] - 0.750:+.3f}\n\n")

    (REPORT_DIR / "report_b.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
