#!/usr/bin/env python3
"""任务级 3 分类数据集准备。

复用回归任务级的 84×264 特征矩阵，标签换成 task_difficulty ∈ {低, 中, 高}。

输入：../regression_task_level/dataset/*
输出：./dataset/*
    - X_cls.npy       (84 × 264)  特征矩阵（等同回归版 X_task.npy）
    - y_cls.npy       (84,)       字符串标签 ['低','中','高']
    - y_cls_int.npy   (84,)       整数标签 (低=0, 中=1, 高=2)
    - groups_cls.npy  (84,)       subject 编号
    - sample_cls.npy  (84,)       sample_id
    - feature_names_cls.json      列名
    - dataset_audit_cls.md        审计报告

同时校验：
- 每个 subject 的任务在 3 类中是否均衡（决定 StratifiedGroupKFold 是否可行）
- 是否存在"某折没某一类"的极端情况
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "regression_task_level" / "dataset"
OUT_DIR = HERE / "dataset"

CLASS_LABEL_TO_INT = {"低": 0, "中": 1, "高": 2}
INT_TO_CLASS_LABEL = {v: k for k, v in CLASS_LABEL_TO_INT.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[make_dataset_cls] loading from {SRC_DIR}")
    X = np.load(SRC_DIR / "X_task.npy")
    groups = np.load(SRC_DIR / "groups_task.npy")
    sample_ids = np.load(SRC_DIR / "sample_task.npy", allow_pickle=True)
    task_table = pd.read_csv(SRC_DIR / "task_level_table.csv")
    with open(SRC_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 校验 task_table 与 X/groups 顺序一致（假设一致，检查一下）
    assert len(task_table) == X.shape[0]
    assert (task_table["subject"].to_numpy() == groups).all()
    assert (task_table["sample_id"].to_numpy() == sample_ids).all()

    # 标签
    y_str = task_table["task_difficulty"].to_numpy()
    y_int = np.array([CLASS_LABEL_TO_INT[c] for c in y_str], dtype=np.int64)

    print(f"[make_dataset_cls] X.shape = {X.shape}")
    print(f"[make_dataset_cls] subjects = {len(np.unique(groups))}")
    print(f"[make_dataset_cls] class distribution:")
    for c in ["低", "中", "高"]:
        n = int((y_str == c).sum())
        print(f"  {c}: {n} ({n/len(y_str)*100:.1f}%)")

    # 保存
    np.save(OUT_DIR / "X_cls.npy", X)
    np.save(OUT_DIR / "y_cls.npy", y_str)
    np.save(OUT_DIR / "y_cls_int.npy", y_int)
    np.save(OUT_DIR / "groups_cls.npy", groups)
    np.save(OUT_DIR / "sample_cls.npy", sample_ids)
    with open(OUT_DIR / "feature_names_cls.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # 每被试的类别分布（关键：看 StratifiedGroupKFold 是否可行）
    per_subj = task_table.groupby("subject").agg(
        n_tasks=("sample_id", "count"),
        classes_seen=("task_difficulty", lambda s: sorted(set(s))),
    ).reset_index()

    # 检查 StratifiedGroupKFold(5) 是否成功
    sgkf_ok = True
    sgkf_log = []
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for i, (tr, te) in enumerate(sgkf.split(X, y_int, groups)):
            tr_classes = np.unique(y_int[tr])
            te_classes = np.unique(y_int[te])
            te_class_counts = {int(c): int((y_int[te] == c).sum()) for c in [0, 1, 2]}
            n_te_subj = len(np.unique(groups[te]))
            sgkf_log.append({
                "fold": i,
                "n_test": int(len(te)),
                "n_test_subjects": int(n_te_subj),
                "train_classes": tr_classes.tolist(),
                "test_classes": te_classes.tolist(),
                "test_class_counts": te_class_counts,
            })
            if len(tr_classes) < 3 or len(te_classes) < 3:
                sgkf_ok = False
    except Exception as e:
        sgkf_ok = False
        sgkf_log.append({"error": str(e)})

    # 审计
    lines = []
    lines.append("# 分类数据集审计报告（84 × 264，3 类：低/中/高）\n\n")
    lines.append(f"- 特征矩阵：{X.shape}（等同回归版）\n")
    lines.append(f"- 类别标签：`task_difficulty`\n")
    lines.append(f"- 独立被试数：{len(np.unique(groups))}\n\n")

    lines.append("## 类别分布\n\n")
    lines.append("| 类别 | 编码 | 样本数 | 占比 |\n|---|---:|---:|---:|\n")
    for c in ["低", "中", "高"]:
        n = int((y_str == c).sum())
        lines.append(f"| {c} | {CLASS_LABEL_TO_INT[c]} | {n} | {n/len(y_str)*100:.1f}% |\n")
    lines.append(f"\n**类别均衡度**：max/min = {int(pd.Series(y_str).value_counts().max()) / int(pd.Series(y_str).value_counts().min()):.2f}\n\n")

    lines.append("## 关键隐患：难度与任务类型 100% 绑定\n\n")
    task_diff = task_table.groupby(["task_difficulty", "task"]).size().unstack(fill_value=0)
    lines.append("| 难度 \\ task | " + " | ".join(str(c) for c in task_diff.columns) + " |\n")
    lines.append("|---|" + "---:|" * len(task_diff.columns) + "\n")
    for idx, row in task_diff.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(int(v)) for v in row.values) + " |\n")
    lines.append("\n说明：低难度全部是 task_3 或 task_5，中难度全部是 task_1/2/4，高难度全部是 task_5_6。\n")
    lines.append("**这意味着分类等价于'区分不同 task 类型'**——预测能力有天然上限（如果两个不同 task 都属于'低'难度，模型能否区分它们的可预测性由特征质量决定）\n\n")

    lines.append("## 每被试类别覆盖情况\n\n")
    lines.append("| 被试 | 任务数 | 覆盖类别 |\n|---:|---:|---|\n")
    for _, row in per_subj.iterrows():
        lines.append(f"| {row['subject']} | {row['n_tasks']} | {', '.join(row['classes_seen'])} |\n")
    n_cover_all3 = int(per_subj["classes_seen"].apply(lambda s: len(s) == 3).sum())
    n_cover_2 = int(per_subj["classes_seen"].apply(lambda s: len(s) == 2).sum())
    n_cover_1 = int(per_subj["classes_seen"].apply(lambda s: len(s) == 1).sum())
    lines.append(f"\n- 覆盖全 3 类的被试：{n_cover_all3}\n")
    lines.append(f"- 覆盖 2 类的被试：{n_cover_2}\n")
    lines.append(f"- 只覆盖 1 类的被试：{n_cover_1}\n\n")

    lines.append("## StratifiedGroupKFold(5) 划分验证\n\n")
    lines.append(f"- 是否 5 折都覆盖 3 类：**{'✅ 是' if sgkf_ok else '❌ 否'}**\n\n")
    lines.append("| fold | n_test | n_test_subj | 测试类别分布 (低/中/高) |\n|---:|---:|---:|---|\n")
    for r in sgkf_log:
        cc = r.get("test_class_counts", {})
        cc_str = f"{cc.get(0,0)}/{cc.get(1,0)}/{cc.get(2,0)}"
        lines.append(f"| {r['fold']} | {r.get('n_test','?')} | {r.get('n_test_subjects','?')} | {cc_str} |\n")

    (OUT_DIR / "dataset_audit_cls.md").write_text("".join(lines), encoding="utf-8")

    print(f"[make_dataset_cls] saved to {OUT_DIR}")
    print(f"  StratifiedGroupKFold 可行：{sgkf_ok}")


if __name__ == "__main__":
    main()
