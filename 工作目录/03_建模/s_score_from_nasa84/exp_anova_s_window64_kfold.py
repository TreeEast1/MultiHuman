#!/usr/bin/env python3
"""子伟更正：按窗口样本划训练集/测试集，不要按人。

其余与窗口化 64 维圣袁流程相同：
    12624 窗 × 64 维，不做任务级平均/标准差
    被试内标准化 → 折内 ANOVA 四模态定额 → 多算法直接预测 S → 模态组合

划分改成 KFold(n_splits=5, shuffle=True, random_state=0) 直接切窗口。
同一个人、同一次任务的重叠窗可以同时出现在训练和测试里。
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from exp_anova_s_flowchart import MODALITY_ORDER, QUOTA  # noqa: E402
from exp_anova_s_window64 import (  # noqa: E402
    load_window_matrix,
    model_zoo,
    pack_slim,
    run_cv,
)
from run_window64_shengyuan import _enable_xgboost_on_macos  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "reports_anova_s_window64_kfold"
N_SPLITS = 5


def window_kfold_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    dummy = np.zeros(n)
    return list(kf.split(dummy))


def leak_stats(subjects: np.ndarray, sample_ids: np.ndarray, splits) -> list[dict]:
    rows = []
    for i, (tr, te) in enumerate(splits):
        subj_tr, subj_te = set(subjects[tr]), set(subjects[te])
        task_tr, task_te = set(sample_ids[tr]), set(sample_ids[te])
        rows.append({
            "fold": i + 1,
            "n_train_windows": int(len(tr)),
            "n_test_windows": int(len(te)),
            "test_subjects_also_in_train": int(len(subj_te & subj_tr)),
            "n_test_subjects": int(len(subj_te)),
            "test_tasks_also_in_train": int(len(task_te & task_tr)),
            "n_test_tasks": int(len(task_te)),
        })
    return rows


def write_report(split_info, algo_rows, combo_rows, best_name) -> str:
    lines = []
    lines.append("# 窗口化 64 维 + 圣袁流程（按窗口划训练/测试）\n\n")
    lines.append("子伟更正：按**窗口样本**划训练集和测试集，**不要按人**。\n\n")
    lines.append("输入仍是 12624 × 64，中间不做窗平均/标准差。流程仍是被试内标准化 → ANOVA → 四模态定额 → 多算法直接预测 S。\n\n")
    lines.append("`KFold(n_splits=5, shuffle=True, random_state=0)` 直接切窗口。")
    lines.append("同一次任务 30 s 窗、5 s 步重叠 83%，相邻窗很容易一边在训练、一边在测试。\n\n")
    lines.append("| 折 | 训练窗 | 测试窗 | 测试人中也在训练的 | 测试任务中也在训练的 |\n|---|---:|---:|---:|---:|\n")
    for s in split_info:
        lines.append(
            f"| {s['fold']} | {s['n_train_windows']} | {s['n_test_windows']} | "
            f"{s['test_subjects_also_in_train']}/{s['n_test_subjects']} | "
            f"{s['test_tasks_also_in_train']}/{s['n_test_tasks']} |\n"
        )
    lines.append("\n主指标：窗口级 R²（按他的划分，样本就是窗）。任务级是把折外窗口预测按任务取中位数，仅作对照。\n\n")

    lines.append("## 四模态多算法\n\n")
    lines.append("| 算法 | 窗口级 R² | 窗口级 MAE | 任务级 R² |\n|---|---:|---:|---:|\n")
    for r in sorted(algo_rows, key=lambda x: -x["window_r2"]):
        lines.append(
            f"| {r['name']} | {r['window_r2']:+.3f} | {r['window_mae']:.3f} | {r['task_r2']:+.3f} |\n"
        )
    lines.append(f"\n模态组合用窗口级最好的：**{best_name}**。\n\n")

    lines.append("## 单 / 双 / 三 / 四模态（窗口级）\n\n")
    lines.append("| 模态数 | 组合 | n_feat | 窗口级 R² | 任务级 R² |\n|---|---|---:|---:|---:|\n")
    for r in sorted(combo_rows, key=lambda x: (-x["n_modalities"], -x["window_r2"])):
        lines.append(
            f"| {r['n_modalities']} | {'+'.join(r['modalities'])} | {r['n_features']} | "
            f"{r['window_r2']:+.3f} | {r['task_r2']:+.3f} |\n"
        )
    best4 = next(r for r in combo_rows if r["n_modalities"] == 4)
    best1 = max((r for r in combo_rows if r["n_modalities"] == 1), key=lambda x: x["window_r2"])
    lines.append(f"\n- 四模态窗口级直接预测 S：R² = {best4['window_r2']:+.3f}\n")
    lines.append(f"- 最强单模态：`{'+'.join(best1['modalities'])}`，R² = {best1['window_r2']:+.3f}\n")
    lines.append("\n对照：同一套输入按人五折时，四模态任务级 R² 约为 0。")
    lines.append("本版按窗口切，数字会好看，因为重叠窗和同一个人都漏进测试集了。\n")
    return "".join(lines)


def main() -> None:
    _enable_xgboost_on_macos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, subjects, sample_ids, names = load_window_matrix()
    splits = window_kfold_splits(len(y))
    info = leak_stats(subjects, sample_ids, splits)
    print(f"[kfold-win] X={X.shape}")
    for s in info:
        print(
            f"  fold {s['fold']}: train={s['n_train_windows']} test={s['n_test_windows']}  "
            f"人重叠 {s['test_subjects_also_in_train']}/{s['n_test_subjects']}  "
            f"任务重叠 {s['test_tasks_also_in_train']}/{s['n_test_tasks']}"
        )

    print("\n===== 四模态 多算法 =====")
    algo_rows = []
    algo_hats = {}
    for name, factory, prep in model_zoo():
        print(f"  -- {name}")
        row = run_cv(X, y, sample_ids, splits, names, MODALITY_ORDER, factory, prep, name)
        algo_rows.append(pack_slim(row) | {
            "task_r2": row["task_r2"], "task_mae": row["task_mae"],
            "window_r2": row["window_r2"], "window_mae": row["window_mae"],
        })
        algo_hats[name] = row["y_hat"]
        print(f"     窗口 R²={row['window_r2']:+.3f}  任务 R²={row['task_r2']:+.3f}")

    best = max(algo_rows, key=lambda r: r["window_r2"])
    best_name = best["name"]
    best_factory, best_prep = next((f, p) for n, f, p in model_zoo() if n == best_name)
    print(f"\n最佳算法：{best_name}")

    print("\n===== 单双三四模态 =====")
    combo_rows = []
    for k in range(1, 5):
        for combo in combinations(MODALITY_ORDER, k):
            tag = "+".join(combo)
            print(f"  -- {tag}")
            row = run_cv(X, y, sample_ids, splits, names, combo, best_factory, best_prep, tag)
            combo_rows.append(pack_slim(row) | {
                "task_r2": row["task_r2"], "task_mae": row["task_mae"],
                "window_r2": row["window_r2"], "window_mae": row["window_mae"],
            })
            print(f"     窗口 R²={row['window_r2']:+.3f}  任务 R²={row['task_r2']:+.3f}")

    payload = {
        "split": "KFold n_splits=5 shuffle=True random_state=0 on windows; not by subject",
        "n_windows": int(len(y)),
        "folds": info,
        "quota": QUOTA,
        "best_algorithm": best_name,
        "algorithms": algo_rows,
        "modality_combos": combo_rows,
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / "y_s_hat_windows.npy", algo_hats[best_name])
    (OUT_DIR / "report.md").write_text(
        write_report(info, algo_rows, combo_rows, best_name), encoding="utf-8"
    )
    print(f"\n[kfold-win] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
