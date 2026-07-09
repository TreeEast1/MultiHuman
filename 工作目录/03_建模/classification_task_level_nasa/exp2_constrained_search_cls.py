#!/usr/bin/env python3
"""P2 扩展 v2：NASA 三分类 带模态约束的精细化 K 搜索。

背景：
  翁老师要求：训练模型时 眼动/脑电/心率/行为 4 个维度
  每个都必须至少入选 1 个特征。原 P2 在 K=30（全局最佳
  组合 RF_importance + XGB_shallow）的 Top-30 中 0 个心率
  特征，违反约束。本脚本做：
    1) 沿用原 ranker（MI / RF_importance）做 Top-K 选，
       选完后做"模态补足"：缺失模态用该模态排名最高的特征
       替换 Top-K 中最弱者；
    2) 输出"无约束" vs "带约束" 两套 K vs Macro-F1 曲线；
    3) 锁定"带约束全局最佳 (ranker, K, model) 组合"，
       配合 P3 调参最佳模型做最终验证。

矩阵：
  ranker ∈ {MI, RF_importance}        (Permutation 太慢，省略)
  K      ∈ {15,20,25,30,35,40,45,50,55,60,70,80}
  model  ∈ {LR_L2_strong, RF_shallow, XGB_shallow}
  共 2 × 12 × 3 × 2(约束开关) = 144 组实验

模态规则（与翁老师原话一致：眼动/脑电/心率/行为）：
  eye  =  eye_*  ∪  blink_*        (84)
  eeg  =  eeg_*                    (112)
  hr   =  hr_*                     (20)
  behav = log_*                    (48)
  合计 264。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE, RANKERS_CLS,
    median_impute_fold, median_impute_and_scale,
    pooled_cv_cls_with_selection,
)

# 强制无缓冲，立刻看到进度
import functools
print = functools.partial(print, flush=True)

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp2_constrained"
N_SPLITS = 5

TOP_KS = [15, 20, 25, 30, 35, 40, 50, 60, 80]
RANKERS = ["MI", "RF_importance"]  # 跳过 Permutation（太慢且前面 P2 里通常不是最佳）
MODALITY_RANKS = ["eye", "eeg", "hr", "behav"]


def modality_of(name: str) -> str:
    if name.startswith("eye_") or name.startswith("blink_"):
        return "eye"
    if name.startswith("eeg_"):
        return "eeg"
    if name.startswith("hr_"):
        return "hr"
    if name.startswith("log_"):
        return "behav"
    return "other"


def make_models(y_str, y_int):
    from xgboost import XGBClassifier
    return [
        ("LR_L2_strong",
         lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
         median_impute_and_scale, y_str),
        ("RF_shallow",
         lambda: RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                        random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold, y_str),
        ("XGB_shallow",
         lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=3,
                               reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
         None, y_int),
    ]


def select_top_k_with_constraint(rank_idx: np.ndarray,
                                 mod_of_idx: np.ndarray,
                                 K: int,
                                 min_per_mod: int = 1) -> np.ndarray:
    """按 rank_idx 顺序选 K 个特征；若最终 4 模态未全覆盖，
    则用"该模态在 rank_idx 中排名最高、且未入选"的特征，
    替换 Top-K 中"在 rank_idx 中排名最靠后（即最弱）"的那一个。
    保持总特征数严格 = K。
    """
    # 1) 选前 K 个
    top = list(rank_idx[:K])

    # 2) 检查缺失模态
    present = {mod_of_idx[i] for i in top}
    missing = [m for m in MODALITY_RANKS if m not in present]
    if not missing:
        return np.array(top)

    # 3) 找出 Top-K 中"最弱"的索引（按 rank_idx 的位置最靠后）
    pos_in_rank = {idx: pos for pos, idx in enumerate(rank_idx)}
    top_sorted_by_weakness = sorted(top, key=lambda i: -pos_in_rank[i])

    # 4) 补齐
    top_set = set(top)
    for m in missing:
        for idx in rank_idx:
            if mod_of_idx[idx] != m:
                continue
            if idx in top_set:
                continue
            # 替换最弱
            victim = top_sorted_by_weakness.pop(0)
            top_set.discard(victim)
            top.remove(victim)
            top.append(int(idx))
            top_set.add(int(idx))
            break

    return np.array(top)


def _run_with_per_fold_selection(X, y, groups, top_k_list_per_fold, factory, prep, name):
    """复刻 cls_utils.pooled_cv_cls_with_selection 的核心流程，
    但允许每折用不同的 selected_idx 列表。"""
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    class_labels = sorted(list(np.unique(y)))
    y_pred_all = np.empty(len(y), dtype=y.dtype)
    filled = np.zeros(len(y), dtype=bool)
    fold_details = []

    for fold_idx, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        X_tr_full, X_te_full = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_full)
        X_te_imp = imputer.transform(X_te_full)

        top_idx = top_k_list_per_fold[fold_idx]
        X_tr_sel = X_tr_imp[:, top_idx]
        X_te_sel = X_te_imp[:, top_idx]

        if prep is not None:
            X_tr_sel, X_te_sel = prep(X_tr_sel, X_te_sel)

        m = factory()
        m.fit(X_tr_sel, y_tr)
        y_hat = m.predict(X_te_sel)
        y_pred_all[te] = y_hat
        filled[te] = True

        acc = accuracy_score(y_te, y_hat)
        mac_f1 = f1_score(y_te, y_hat, average="macro", labels=class_labels, zero_division=0)
        fold_details.append({
            "fold": fold_idx,
            "fold_acc": float(acc),
            "fold_macro_f1": float(mac_f1),
            "selected_idx": top_idx.tolist(),
        })

    assert filled.all()
    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_mac_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_w_f1 = float(f1_score(y, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    per_class = f1_score(y, y_pred_all, average=None, labels=class_labels, zero_division=0)
    cm = confusion_matrix(y, y_pred_all, labels=class_labels)
    facc = np.array([f["fold_acc"] for f in fold_details])
    ff1 = np.array([f["fold_macro_f1"] for f in fold_details])

    return {
        "name": name, "n_features": int(len(top_k_list_per_fold[0])),
        "pooled_acc": pooled_acc, "pooled_macro_f1": pooled_mac_f1,
        "pooled_weighted_f1": pooled_w_f1,
        "pooled_per_class_f1": {c: float(v) for c, v in zip(class_labels, per_class)},
        "fold_acc_mean": float(facc.mean()), "fold_acc_std": float(facc.std()),
        "fold_macro_f1_mean": float(ff1.mean()), "fold_macro_f1_std": float(ff1.std()),
        "confusion": cm.tolist(), "class_labels": list(class_labels),
        "fold_details": fold_details, "y_pred_pooled": y_pred_all,
    }


def pooled_cv_cls_with_constraint(
    model_factory, X, y, groups, n_splits, top_k, ranker, preprocessor,
    mod_of_idx, name="", constraint=True,
):
    """与 cls_utils.pooled_cv_cls_with_selection 一致，但可选约束。
    constraint=True 时用 select_top_k_with_constraint 选每折特征。
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
        if constraint:
            top_idx = select_top_k_with_constraint(rank_idx, mod_of_idx, top_k, min_per_mod=1)
        else:
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
            "fold_acc": float(acc),
            "fold_macro_f1": float(mac_f1),
            "selected_idx": top_idx.tolist(),
        })

    assert filled.all()
    pooled_acc = float(accuracy_score(y, y_pred_all))
    pooled_mac_f1 = float(f1_score(y, y_pred_all, average="macro", labels=class_labels, zero_division=0))
    pooled_w_f1 = float(f1_score(y, y_pred_all, average="weighted", labels=class_labels, zero_division=0))
    per_class = f1_score(y, y_pred_all, average=None, labels=class_labels, zero_division=0)
    cm = confusion_matrix(y, y_pred_all, labels=class_labels)
    facc = np.array([f["fold_acc"] for f in fold_details])
    ff1 = np.array([f["fold_macro_f1"] for f in fold_details])

    res = {
        "name": name, "n_features": int(top_k),
        "pooled_acc": pooled_acc, "pooled_macro_f1": pooled_mac_f1,
        "pooled_weighted_f1": pooled_w_f1,
        "pooled_per_class_f1": {c: float(v) for c, v in zip(class_labels, per_class)},
        "fold_acc_mean": float(facc.mean()), "fold_acc_std": float(facc.std()),
        "fold_macro_f1_mean": float(ff1.mean()), "fold_macro_f1_std": float(ff1.std()),
        "confusion": cm.tolist(), "class_labels": list(class_labels),
        "fold_details": fold_details, "y_pred_pooled": y_pred_all,
    }
    return res, selected_per_fold


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "X_cls.npy")
    y_str = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    mod_of_idx = np.array([modality_of(n) for n in feature_names])
    print(f"[exp2_constrained] X={X.shape}  TOP_KS={TOP_KS}  RANKERS={RANKERS}")
    print(f"[exp2_constrained] modality counts: {Counter(mod_of_idx.tolist())}")

    models = make_models(y_str, y_int)

    all_rows = []
    selected_stats = {}
    t0 = time.time()

    for ranker_name in RANKERS:
        ranker_fn = RANKERS_CLS[ranker_name]
        print(f"\n=== Ranker: {ranker_name} ===")
        for k in TOP_KS:
            for m_name, factory, prep, y_use in models:
                # 无约束
                res_unc, sel_unc = pooled_cv_cls_with_constraint(
                    factory, X, y_use, groups, N_SPLITS,
                    top_k=k, ranker=ranker_fn, preprocessor=prep,
                    mod_of_idx=mod_of_idx, constraint=False,
                    name=f"{ranker_name}_top{k}_{m_name}_unconstrained",
                )
                all_rows.append({
                    "ranker": ranker_name, "k": k, "model": m_name,
                    "constraint": False,
                    "pooled_acc": res_unc["pooled_acc"],
                    "pooled_macro_f1": res_unc["pooled_macro_f1"],
                    "pooled_weighted_f1": res_unc["pooled_weighted_f1"],
                    "fold_macro_f1_mean": res_unc["fold_macro_f1_mean"],
                    "fold_macro_f1_std": res_unc["fold_macro_f1_std"],
                })

                # 带约束
                res_con, sel_con = pooled_cv_cls_with_constraint(
                    factory, X, y_use, groups, N_SPLITS,
                    top_k=k, ranker=ranker_fn, preprocessor=prep,
                    mod_of_idx=mod_of_idx, constraint=True,
                    name=f"{ranker_name}_top{k}_{m_name}_constrained",
                )
                all_rows.append({
                    "ranker": ranker_name, "k": k, "model": m_name,
                    "constraint": True,
                    "pooled_acc": res_con["pooled_acc"],
                    "pooled_macro_f1": res_con["pooled_macro_f1"],
                    "pooled_weighted_f1": res_con["pooled_weighted_f1"],
                    "fold_macro_f1_mean": res_con["fold_macro_f1_mean"],
                    "fold_macro_f1_std": res_con["fold_macro_f1_std"],
                })

                # 收集约束版本的稳定特征
                selected_stats[(ranker_name, k, m_name)] = sel_con

                print(f"  K={k:3d}  {m_name:16s}  "
                      f"unc Acc={res_unc['pooled_acc']:.3f} F1={res_unc['pooled_macro_f1']:.3f}  "
                      f"con Acc={res_con['pooled_acc']:.3f} F1={res_con['pooled_macro_f1']:.3f}  "
                      f"dt={time.time()-t0:.0f}s")

    with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # 稳定性统计（约束版本）
    stability = {}
    for (ranker, k, m_name), selected in selected_stats.items():
        counter = Counter()
        for arr in selected:
            for idx in arr:
                counter[idx] += 1
        stable_5 = [i for i, c in counter.items() if c == 5]
        # 按模态统计
        mod_count = Counter()
        for idx, c in counter.items():
            if c == 5:
                mod_count[mod_of_idx[idx]] += 1
        stability[(ranker, k, m_name)] = {
            "counter": counter, "stable_5_count": len(stable_5),
            "stable_5_indices": stable_5,
            "stable_5_modality_count": dict(mod_count),
        }

    write_markdown(all_rows, stability, feature_names, models, mod_of_idx)
    print(f"\n[exp2_constrained] DONE  dt={time.time()-t0:.0f}s  报告写入 {REPORT_DIR}/report.md")


def write_markdown(rows, stability, feature_names, models, mod_of_idx):
    lines = []
    lines.append("# P2 扩展 v2：NASA 三分类 带模态约束的精细化 K 搜索\n\n")
    lines.append(f"**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject\n\n")
    lines.append(f"**K 集合**（共 {len(TOP_KS)} 点）：{TOP_KS}\n\n")
    lines.append("**模态规则**（翁老师原话：眼动/脑电/心率/行为 4 个维度）：\n\n")
    lines.append("- `eye` (眼动) = `eye_*` ∪ `blink_*`\n")
    lines.append("- `eeg` (脑电) = `eeg_*`\n")
    lines.append("- `hr`  (心率) = `hr_*`\n")
    lines.append("- `behav` (行为) = `log_*`\n\n")
    lines.append("**约束逻辑**：先按 ranker 选 K 个；若 4 模态未全覆盖，"
                 "则用『该模态在 ranker 中排名最高』的特征 **替换** Top-K 中『在 ranker 中最弱』的那一个，"
                 "总特征数严格 = K。\n\n")
    lines.append("**ranker × K × model × {无约束, 带约束}** 全组合扫描。\n\n")

    # 1) 各 (ranker, model) 在"带约束"下的最佳 K
    lines.append("## 1. 各 (ranker, model) 组合 —— 带约束下的最佳 K\n\n")
    lines.append("| ranker | model | best K | constrained Acc | constrained Macro-F1 | (无约束 best F1) |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for m_name, _, _, _ in models:
        for ranker in RANKERS:
            sub_c = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name and r["constraint"]]
            sub_u = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name and not r["constraint"]]
            bc = max(sub_c, key=lambda x: x["pooled_macro_f1"])
            bu = max(sub_u, key=lambda x: x["pooled_macro_f1"])
            lines.append(
                f"| {ranker} | {m_name} | {bc['k']} | "
                f"{bc['pooled_acc']:.3f} | {bc['pooled_macro_f1']:.3f} | "
                f"{bu['pooled_macro_f1']:.3f} |\n"
            )
    lines.append("\n")

    # 2) 全局 Top-10（带约束）
    lines.append("## 2. 全局 Top-10 (ranker, K, model) 组合 —— 带约束\n\n")
    lines.append("| rank | ranker | K | model | pooled Acc | pooled Macro-F1 |\n|---:|---|---:|---|---:|---:|\n")
    rows_con = [r for r in rows if r["constraint"]]
    rows_con_sorted = sorted(rows_con, key=lambda x: -x["pooled_macro_f1"])
    for i, r in enumerate(rows_con_sorted[:10], 1):
        lines.append(
            f"| {i} | {r['ranker']} | {r['k']} | {r['model']} | "
            f"{r['pooled_acc']:.3f} | {r['pooled_macro_f1']:.3f} |\n"
        )
    lines.append("\n")

    # 3) K vs Macro-F1 曲线：无约束 vs 带约束
    lines.append("## 3. K vs pooled Macro-F1 曲线（无约束 vs 带约束）\n\n")
    for m_name, _, _, _ in models:
        for ranker in RANKERS:
            lines.append(f"### {ranker} + {m_name}\n\n")
            lines.append("| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |\n|---:|---:|---:|---:|---:|---:|\n")
            for k in TOP_KS:
                ru = next((r for r in rows if r["ranker"] == ranker and r["k"] == k
                           and r["model"] == m_name and not r["constraint"]), None)
                rc = next((r for r in rows if r["ranker"] == ranker and r["k"] == k
                           and r["model"] == m_name and r["constraint"]), None)
                if ru is None or rc is None:
                    continue
                delta = rc["pooled_macro_f1"] - ru["pooled_macro_f1"]
                lines.append(
                    f"| {k} | {ru['pooled_acc']:.3f} | {ru['pooled_macro_f1']:.3f} | "
                    f"{rc['pooled_acc']:.3f} | {rc['pooled_macro_f1']:.3f} | "
                    f"{delta:+.3f} |\n"
                )
            lines.append("\n")

    # 4) 全局最佳（带约束）的稳定特征 + 模态分布
    best_con = rows_con_sorted[0]
    key = (best_con["ranker"], best_con["k"], best_con["model"])
    stab = stability[key]
    lines.append(f"## 4. 全局最佳（带约束）：{best_con['ranker']} + {best_con['model']} @ K={best_con['k']} (Macro-F1={best_con['pooled_macro_f1']:.3f})\n\n")
    lines.append(f"- pooled Acc = **{best_con['pooled_acc']:.3f}**\n")
    lines.append(f"- stable_5 count: **{stab['stable_5_count']}** / {best_con['k']}\n")
    lines.append(f"- 各模态在 Top-K 中 5/5 折命中的特征数：{stab['stable_5_modality_count']}\n\n")
    lines.append("Top-20 稳定特征（5/5 折命中）：\n\n")
    lines.append("| 特征 | 模态 | 命中折数 |\n|---|---|---:|\n")
    top20 = stab["counter"].most_common(20)
    for idx, cnt in top20:
        lines.append(f"| `{feature_names[idx]}` | {mod_of_idx[idx]} | {cnt} |\n")
    lines.append("\n")

    # 5) 各 (ranker, model) 最佳 K（带约束）的稳定特征
    lines.append("## 5. 各 (ranker, model) 最佳 K（带约束）的稳定特征\n\n")
    for m_name, _, _, _ in models:
        for ranker in RANKERS:
            sub = [r for r in rows if r["ranker"] == ranker and r["model"] == m_name and r["constraint"]]
            best = max(sub, key=lambda x: x["pooled_macro_f1"])
            kk = (ranker, best["k"], m_name)
            stab = stability[kk]
            lines.append(f"### {ranker} + {m_name} @ K={best['k']} (Macro-F1={best['pooled_macro_f1']:.3f})\n\n")
            lines.append(f"- stable_5 count: **{stab['stable_5_count']}** / {best['k']}\n")
            lines.append(f"- 各模态 5/5 折命中的特征数：{stab['stable_5_modality_count']}\n\n")
            top15 = stab["counter"].most_common(15)
            lines.append("| 特征 | 模态 | 命中折数 |\n|---|---|---:|\n")
            for idx, cnt in top15:
                lines.append(f"| `{feature_names[idx]}` | {mod_of_idx[idx]} | {cnt} |\n")
            lines.append("\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
