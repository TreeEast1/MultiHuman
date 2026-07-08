#!/usr/bin/env python3
"""P0 任务级 3 分类 baseline。

评估：
    - 划分：StratifiedGroupKFold(n_splits=5, groups=subject, stratify=y)
      每折都保证 3 类均衡出现在训练与测试
    - 主指标：pooled Accuracy / Macro-F1（合并 5 折预测算总指标，小样本推荐）
    - 参考指标：fold mean±std

模型清单（涵盖不同假设：线性 / 核 / 树集成 / 深度浅）：
    A. DummyClassifier(strategy="stratified")  ← 随机基线
    B. DummyClassifier(strategy="most_frequent") ← 多数类基线
    C. LogisticRegression (L2)
    D. Ridge分类器 (RidgeClassifier)
    E. LinearSVC
    F. SVC (RBF kernel)
    G. GaussianNB
    H. KNeighbors(k=5)
    I. RandomForest_default
    J. RandomForest_shallow
    K. XGBoost_default
    L. XGBoost_shallow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE, median_impute_fold, median_impute_and_scale, pooled_cv_cls,
)


DATASET_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_baseline"
N_SPLITS = 5


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATASET_DIR / "X_cls.npy")
    y = np.load(DATASET_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    groups = np.load(DATASET_DIR / "groups_cls.npy")
    with open(DATASET_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"[baseline_cls] X.shape = {X.shape}")
    print(f"[baseline_cls] class counts:")
    for c in ["低", "中", "高"]:
        print(f"  {c}: {int((y==c).sum())}")
    print()

    experiments = [
        ("Dummy_stratified",
         lambda: DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
         None),
        ("Dummy_most_frequent",
         lambda: DummyClassifier(strategy="most_frequent"),
         None),
        ("LogisticRegression_L2",
         lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("LogisticRegression_L2_strong",
         lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("RidgeClassifier",
         lambda: RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("LinearSVC",
         lambda: LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("SVC_RBF",
         lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
         median_impute_and_scale),
        ("GaussianNB",
         lambda: GaussianNB(),
         median_impute_and_scale),
        ("KNN_k5",
         lambda: KNeighborsClassifier(n_neighbors=5),
         median_impute_and_scale),
        ("RF_default",
         lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold),
        ("RF_shallow",
         lambda: RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                        random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold),
    ]
    from xgboost import XGBClassifier
    experiments.append((
        "XGB_default",
        lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
        None,
    ))
    experiments.append((
        "XGB_shallow",
        lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=3,
                              reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
        None,
    ))

    # XGBoost 需要 int label，其他模型用 str label 都可
    y_int = np.load(DATASET_DIR / "y_cls_int.npy")

    all_results = []
    for name, factory, prep in experiments:
        y_this = y_int if name.startswith("XGB") else y
        res = pooled_cv_cls(factory, X, y_this, groups, N_SPLITS, prep, name=name)
        print(f"  {name:32s}  acc={res.pooled_acc:.3f}  macro-F1={res.pooled_macro_f1:.3f}  weighted-F1={res.pooled_weighted_f1:.3f}")
        all_results.append(res)

    write_report(all_results, X.shape, y, groups)


def write_report(all_results, X_shape, y, groups):
    lines = []
    lines.append("# P0 任务级 3 分类 Baseline 报告（NASA-TLX 三分位分档）\n\n")
    lines.append("**评估**：StratifiedGroupKFold(5) by subject，主指标 pooled Accuracy / Macro-F1\n\n")
    lines.append("**标签**：y_nasa 按 33%/67% 分位数分为 低/中/高（区别于 classification_task_level/ 的 task_difficulty 硬编码）\n\n")

    lines.append("## 数据集\n\n")
    n_low = int((y == "低").sum())
    n_mid = int((y == "中").sum())
    n_high = int((y == "高").sum())
    n_total = len(y)
    _counts = [n_low, n_mid, n_high]
    _most_cls = ["低", "中", "高"][int(np.argmax(_counts))]
    _p_most = max(_counts) / n_total
    _p_strat = sum((c / n_total) ** 2 for c in _counts)
    lines.append(f"- 样本数：{X_shape[0]}\n")
    lines.append(f"- 特征维度：{X_shape[1]}\n")
    lines.append(f"- 独立被试数：{len(np.unique(groups))}\n")
    lines.append(f"- 类别分布：低 {n_low} / 中 {n_mid} / 高 {n_high}\n\n")
    lines.append("**基线参考**：\n")
    lines.append(f"- Dummy_stratified（按类别比例随机猜）：期望 acc ≈ {_p_strat:.3f}\n")
    lines.append(f"- Dummy_most_frequent（永远猜'{_most_cls}'）：期望 acc = {max(_counts)}/{n_total} = {_p_most:.3f}\n")
    lines.append("- 3 均匀分类随机猜：acc = 0.333\n\n")

    # 汇总
    lines.append("## 汇总（按 pooled Macro-F1 排序）\n\n")
    lines.append("| 模型 | pooled Acc | pooled Macro-F1 | pooled Weighted-F1 | fold Acc (μ±σ) | fold Macro-F1 (μ±σ) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for r in sorted(all_results, key=lambda x: -x.pooled_macro_f1):
        lines.append(
            f"| {r.name} | {r.pooled_acc:.3f} | {r.pooled_macro_f1:.3f} | {r.pooled_weighted_f1:.3f} | "
            f"{r.fold_acc_mean:.3f}±{r.fold_acc_std:.3f} | "
            f"{r.fold_macro_f1_mean:.3f}±{r.fold_macro_f1_std:.3f} |\n"
        )

    # 逐类 F1
    lines.append("\n## 各模型逐类 F1（pooled）\n\n")
    lines.append("| 模型 | F1(低) | F1(中) | F1(高) |\n|---|---:|---:|---:|\n")
    for r in sorted(all_results, key=lambda x: -x.pooled_macro_f1):
        pc = r.pooled_per_class_f1
        # 兼容 str 和 int label
        f1_low = pc.get("低", pc.get(0, np.nan))
        f1_mid = pc.get("中", pc.get(1, np.nan))
        f1_high = pc.get("高", pc.get(2, np.nan))
        lines.append(f"| {r.name} | {f1_low:.3f} | {f1_mid:.3f} | {f1_high:.3f} |\n")

    # 最佳模型的混淆矩阵
    best = max(all_results, key=lambda x: x.pooled_macro_f1)
    lines.append(f"\n## 最佳模型 `{best.name}` 的混淆矩阵（pooled）\n\n")
    labels = best.class_labels
    # 转字符串
    labels_disp = [str(l) for l in labels]
    lines.append("| 真值 \\ 预测 | " + " | ".join(labels_disp) + " |\n")
    lines.append("|---|" + "---:|" * len(labels_disp) + "\n")
    for i, row in enumerate(best.confusion):
        lines.append(f"| {labels_disp[i]} | " + " | ".join(str(v) for v in row) + " |\n")

    # 逐折
    lines.append("\n## 各模型逐折详情\n\n")
    for r in all_results:
        lines.append(f"### {r.name}\n\n")
        lines.append("| fold | n_test | n_test_subj | fold Acc | fold Macro-F1 | test 类别分布 (低/中/高) |\n")
        lines.append("|---:|---:|---:|---:|---:|---|\n")
        for f in r.fold_details:
            cc = f["test_class_counts"]
            # 兼容 str/int key
            n_low = cc.get("低", cc.get(0, 0))
            n_mid = cc.get("中", cc.get(1, 0))
            n_high = cc.get("高", cc.get(2, 0))
            lines.append(
                f"| {f['fold']} | {f['n_test']} | {f['n_test_subjects']} | "
                f"{f['fold_acc']:.3f} | {f['fold_macro_f1']:.3f} | {n_low}/{n_mid}/{n_high} |\n"
            )
        lines.append("\n")

    # 与历史 F1=0.807 对齐
    lines.append("## 与历史结果对照\n\n")
    lines.append("| 来源 | 划分 | 严格度 | Macro-F1 |\n|---|---|---|---:|\n")
    lines.append("| 历史随机窗口划分（有泄漏） | 随机 | ❌ | 0.951 |\n")
    lines.append("| 历史任务级 30 次重复 CV | 未强制跨被试 | ⚠️ | 0.807 |\n")
    lines.append(f"| **本次 StratifiedGroupKFold by subject** | **跨被试严格** | **✅** | **{best.pooled_macro_f1:.3f}** |\n")

    (REPORT_DIR / "baseline_report.md").write_text("".join(lines), encoding="utf-8")

    # JSON
    out_json = []
    for r in all_results:
        d = {
            "name": r.name, "n_features": r.n_features, "n_classes": r.n_classes,
            "pooled_acc": r.pooled_acc, "pooled_macro_f1": r.pooled_macro_f1,
            "pooled_weighted_f1": r.pooled_weighted_f1,
            "pooled_per_class_f1": {str(k): v for k, v in r.pooled_per_class_f1.items()},
            "fold_acc_mean": r.fold_acc_mean, "fold_acc_std": r.fold_acc_std,
            "fold_macro_f1_mean": r.fold_macro_f1_mean, "fold_macro_f1_std": r.fold_macro_f1_std,
            "confusion": r.confusion,
            "class_labels": [str(l) for l in r.class_labels],
        }
        out_json.append(d)
    with open(REPORT_DIR / "baseline_results.json", "w", encoding="utf-8") as fp:
        json.dump(out_json, fp, ensure_ascii=False, indent=2)
    print(f"\n[baseline_cls] 报告写入 {REPORT_DIR}/baseline_report.md")


if __name__ == "__main__":
    main()
