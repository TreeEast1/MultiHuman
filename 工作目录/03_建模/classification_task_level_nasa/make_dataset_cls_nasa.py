#!/usr/bin/env python3
"""任务级 3 分类数据集准备（NASA-TLX 加权分三分位分档）。

与 classification_task_level/ 的核心区别：
- 标签来源：y_nasa（NASA-TLX 加权总分，连续值 1.33–7.80）
  按 33.3% / 66.7% 分位数切成 低/中/高 三档
- 优势：解耦了 task 类型与难度标签——同一 task 在不同被试身上可归入不同档，
  避免了 task_difficulty 与 task 100% 绑定导致"分类等价于区分 task 类型"的问题

输入：../regression_task_level/dataset/*
输出：./dataset/*
    - X_cls.npy       (84 × 264)  特征矩阵（等同回归版 X_task.npy）
    - y_cls.npy       (84,)       字符串标签 ['低','中','高']
    - y_cls_int.npy   (84,)       整数标签 (低=0, 中=1, 高=2)
    - groups_cls.npy  (84,)       subject 编号
    - sample_cls.npy  (84,)       sample_id
    - y_nasa_raw.npy  (84,)       原始 NASA 连续分（留档）
    - feature_names_cls.json      列名
    - dataset_audit_cls.md        审计报告

同时校验：
- 每个 task 在三档中的分布（验证 NASA 是否解耦 task 与难度）
- 与原 task_difficulty 的一致性
- 每个 subject 的类别覆盖
- StratifiedGroupKFold(5) 是否每折都覆盖 3 类
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

    print(f"[make_dataset_cls_nasa] loading from {SRC_DIR}")
    X = np.load(SRC_DIR / "X_task.npy")
    groups = np.load(SRC_DIR / "groups_task.npy")
    sample_ids = np.load(SRC_DIR / "sample_task.npy", allow_pickle=True)
    task_table = pd.read_csv(SRC_DIR / "task_level_table.csv")
    with open(SRC_DIR / "feature_names_task.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 校验 task_table 与 X/groups 顺序一致
    assert len(task_table) == X.shape[0]
    assert (task_table["subject"].to_numpy() == groups).all()
    assert (task_table["sample_id"].to_numpy() == sample_ids).all()

    # ---- NASA-TLX 加权分三分位分档 ----
    y_nasa = task_table["y_nasa"].to_numpy(dtype=float)
    q_lo, q_hi = np.quantile(y_nasa, [1.0 / 3.0, 2.0 / 3.0])
    print(f"[make_dataset_cls_nasa] y_nasa: min={y_nasa.min():.3f} max={y_nasa.max():.3f} "
          f"mean={y_nasa.mean():.3f} std={y_nasa.std():.3f}")
    print(f"[make_dataset_cls_nasa] quantile 1/3={q_lo:.3f}, 2/3={q_hi:.3f}")

    y_str = np.where(y_nasa <= q_lo, "低", np.where(y_nasa <= q_hi, "中", "高"))
    y_int = np.array([CLASS_LABEL_TO_INT[c] for c in y_str], dtype=np.int64)

    print(f"[make_dataset_cls_nasa] X.shape = {X.shape}")
    print(f"[make_dataset_cls_nasa] subjects = {len(np.unique(groups))}")
    print(f"[make_dataset_cls_nasa] class distribution:")
    for c in ["低", "中", "高"]:
        n = int((y_str == c).sum())
        print(f"  {c}: {n} ({n / len(y_str) * 100:.1f}%)")

    # 保存
    np.save(OUT_DIR / "X_cls.npy", X)
    np.save(OUT_DIR / "y_cls.npy", y_str)
    np.save(OUT_DIR / "y_cls_int.npy", y_int)
    np.save(OUT_DIR / "groups_cls.npy", groups)
    np.save(OUT_DIR / "sample_cls.npy", sample_ids)
    np.save(OUT_DIR / "y_nasa_raw.npy", y_nasa)  # 原始连续分留档
    with open(OUT_DIR / "feature_names_cls.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # ============ 审计 ============
    # 1) 每个 task 在三档中的分布（NASA 解耦验证）
    task_vs_bin = pd.crosstab(task_table["task"], pd.Series(y_str, name="nasa_bin"))
    # 2) 与原 task_difficulty 的一致性
    orig_diff = task_table["task_difficulty"].to_numpy().astype(str)
    agree = float((y_str == orig_diff).mean())
    cross = pd.crosstab(
        pd.Series(y_str, name="nasa_bin"),
        pd.Series(orig_diff, name="task_diff"),
    )
    # 3) 每被试类别覆盖
    per_subj_df = pd.DataFrame({"subject": groups, "y": y_str})
    subj_cov = per_subj_df.groupby("subject")["y"].agg(lambda s: sorted(set(s)))

    # 4) StratifiedGroupKFold(5) 可行性
    sgkf_ok = True
    sgkf_log = []
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for i, (tr, te) in enumerate(sgkf.split(X, y_int, groups)):
            tr_classes = np.unique(y_int[tr])
            te_classes = np.unique(y_int[te])
            te_class_counts = {int(c): int((y_int[te] == c).sum()) for c in [0, 1, 2]}
            sgkf_log.append({
                "fold": i,
                "n_test": int(len(te)),
                "n_test_subjects": int(len(np.unique(groups[te]))),
                "train_classes": tr_classes.tolist(),
                "test_classes": te_classes.tolist(),
                "test_class_counts": te_class_counts,
            })
            if len(tr_classes) < 3 or len(te_classes) < 3:
                sgkf_ok = False
    except Exception as e:
        sgkf_ok = False
        sgkf_log.append({"error": str(e)})

    # ---- 写审计 md ----
    lines = []
    lines.append("# 分类数据集审计报告（NASA-TLX 三分位分档，84 × 264，3 类：低/中/高）\n\n")

    lines.append("## 标签来源\n\n")
    lines.append("- **字段**：`y_nasa`（NASA-TLX 加权总分，连续值）\n")
    lines.append(f"- **范围**：{y_nasa.min():.3f} ~ {y_nasa.max():.3f}（均值 {y_nasa.mean():.3f}，标准差 {y_nasa.std():.3f}）\n")
    lines.append(f"- **分档阈值**（33.3% / 66.7% 分位数）：\n")
    lines.append(f"  - 低：y_nasa ≤ {q_lo:.3f}\n")
    lines.append(f"  - 中：{q_lo:.3f} < y_nasa ≤ {q_hi:.3f}\n")
    lines.append(f"  - 高：y_nasa > {q_hi:.3f}\n\n")
    lines.append("> 与 `classification_task_level/`（用 task_difficulty 硬编码查表）的区别：\n")
    lines.append("> NASA 分档基于被试主观负荷评分，同一 task 在不同被试身上可归入不同档，\n")
    lines.append("> 解耦了 task 类型与难度标签。\n\n")

    lines.append("## 类别分布\n\n")
    lines.append("| 类别 | 编码 | 样本数 | 占比 | y_nasa 范围 |\n|---|---:|---:|---:|---|\n")
    for c in ["低", "中", "高"]:
        n = int((y_str == c).sum())
        rng = y_nasa[y_str == c]
        lines.append(f"| {c} | {CLASS_LABEL_TO_INT[c]} | {n} | {n / len(y_str) * 100:.1f}% | [{rng.min():.2f}, {rng.max():.2f}] |\n")
    cnt_series = pd.Series(y_str).value_counts()
    lines.append(f"\n**类别均衡度**：max/min = {cnt_series.max() / cnt_series.min():.2f}\n\n")

    lines.append("## NASA 分档 vs task 类型分布\n\n")
    lines.append("（验证：同一 task 是否横跨多个难度档——这是 NASA 分档相对 task_difficulty 的核心优势）\n\n")
    lines.append("| task | 低 | 中 | 高 | 总 |\n|---|---:|---:|---:|---:|\n")
    for t, row in task_vs_bin.iterrows():
        tot = int(row.sum())
        lines.append(f"| {t} | {int(row.get('低', 0))} | {int(row.get('中', 0))} | {int(row.get('高', 0))} | {tot} |\n")
    n_task_multi = int((task_vs_bin.gt(0).sum(axis=1) > 1).sum())
    lines.append(f"\n- 横跨 ≥2 档的 task 数：**{n_task_multi}** / {len(task_vs_bin)}\n")
    if n_task_multi >= 2:
        lines.append("- ✅ NASA 分档成功解耦了 task 与难度（分类不再等价于区分 task 类型）\n\n")
    else:
        lines.append("- ⚠️ 解耦不明显\n\n")

    lines.append("## NASA 分档 vs 原 task_difficulty 一致性\n\n")
    lines.append(f"- 整体一致率：**{agree * 100:.1f}%**\n\n")
    lines.append("| NASA \\ task_diff | 低 | 中 | 高 |\n|---|---:|---:|---:|\n")
    for idx in ["低", "中", "高"]:
        if idx in cross.index:
            row = cross.loc[idx]
            lines.append(f"| {idx} | {int(row.get('低', 0))} | {int(row.get('中', 0))} | {int(row.get('高', 0))} |\n")
        else:
            lines.append(f"| {idx} | 0 | 0 | 0 |\n")
    lines.append("\n说明：对角线为两标签一致的样本，非对角线为 NASA 主观负荷与任务预设难度不一致的样本\n\n")

    lines.append("## 每被试类别覆盖\n\n")
    lines.append("| 被试 | 覆盖类别 |\n|---:|---|\n")
    for subj in sorted(per_subj_df["subject"].unique()):
        cls = sorted(set(per_subj_df[per_subj_df["subject"] == subj]["y"]))
        lines.append(f"| {subj} | {', '.join(cls)} |\n")
    n_cover_all3 = int(subj_cov.apply(lambda s: len(s) == 3).sum())
    n_cover_2 = int(subj_cov.apply(lambda s: len(s) == 2).sum())
    n_cover_1 = int(subj_cov.apply(lambda s: len(s) == 1).sum())
    lines.append(f"\n- 覆盖全 3 类的被试：{n_cover_all3}\n")
    lines.append(f"- 覆盖 2 类的被试：{n_cover_2}\n")
    lines.append(f"- 只覆盖 1 类的被试：{n_cover_1}\n\n")

    lines.append("## StratifiedGroupKFold(5) 划分验证\n\n")
    lines.append(f"- 是否 5 折都覆盖 3 类：**{'✅ 是' if sgkf_ok else '❌ 否'}**\n\n")
    lines.append("| fold | n_test | n_test_subj | 测试类别分布 (低/中/高) |\n|---:|---:|---:|---|\n")
    for r in sgkf_log:
        cc = r.get("test_class_counts", {})
        lines.append(f"| {r['fold']} | {r.get('n_test', '?')} | {r.get('n_test_subjects', '?')} | {cc.get(0, 0)}/{cc.get(1, 0)}/{cc.get(2, 0)} |\n")

    (OUT_DIR / "dataset_audit_cls.md").write_text("".join(lines), encoding="utf-8")

    print(f"[make_dataset_cls_nasa] saved to {OUT_DIR}")
    print(f"  StratifiedGroupKFold 可行：{sgkf_ok}")
    print(f"  NASA vs task_difficulty 一致率：{agree * 100:.1f}%")


if __name__ == "__main__":
    main()
