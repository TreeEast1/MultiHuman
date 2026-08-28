#!/usr/bin/env python3
"""数据诊断：窗口数、切段可行性、早/晚相关、滞后自相关、任务顺序。

先确认用现有聚合函数能还原任务级 264，再决定「下一阶段」怎么切。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_stage import (  # noqa: E402
    REPORTS,
    TASK_ORDER,
    aggregate_windows,
    align_samples_to_task_order,
    eligible_mask,
    load_feature_names,
    load_samples,
    load_task_arrays,
    modality_of,
    safe_r2,
    split_index,
)

OUT = REPORTS / "00_diagnose"


def lag_corr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    a, b = x[:-lag], x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    if np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names_264, raw = load_feature_names()
    task = load_task_arrays()
    samples = align_samples_to_task_order(load_samples(raw), task["samples"])

    n_win = np.array([len(s.W) for s in samples], dtype=int)
    recon = np.vstack([aggregate_windows(s.W) for s in samples])
    # 允许少量 NaN 位置不同
    both = np.isfinite(recon) & np.isfinite(task["X"])
    max_abs = float(np.nanmax(np.abs(recon[both] - task["X"][both])))
    mean_abs = float(np.nanmean(np.abs(recon[both] - task["X"][both])))

    ratios = [0.25, 0.33, 0.50, 0.67, 0.75]
    ratio_cov = {}
    for r in ratios:
        ok = eligible_mask(samples, r, min_each=4)
        ratio_cov[str(r)] = {
            "n_ok": int(ok.sum()),
            "n_drop": int((~ok).sum()),
            "dropped": [s.sample_id for s, flag in zip(samples, ok) if not flag],
            "n_win_ok_min": int(n_win[ok].min()) if ok.any() else None,
            "n_win_ok_median": float(np.median(n_win[ok])) if ok.any() else None,
        }

    # 早段 vs 晚段：各 66 维均值的相关（ratio=0.5，合格样本）
    ok50 = eligible_mask(samples, 0.50, min_each=4)
    early_mean, late_mean = [], []
    for s, flag in zip(samples, ok50):
        if not flag:
            continue
        cut = split_index(len(s.W), 0.50)
        with np.errstate(all="ignore"):
            early_mean.append(np.nanmean(s.W[:cut], axis=0))
            late_mean.append(np.nanmean(s.W[cut:], axis=0))
    early_mean = np.vstack(early_mean)
    late_mean = np.vstack(late_mean)
    stage_rows = []
    for j, name in enumerate(raw):
        stage_rows.append(
            {
                "feature": name,
                "modality": modality_of(name),
                "early_late_r2": safe_r2(late_mean[:, j], early_mean[:, j]),
                "early_late_pearson": float(
                    pd.Series(early_mean[:, j]).corr(pd.Series(late_mean[:, j]))
                ),
            }
        )

    # 滞后相关：步长 1（重叠 25s）和步长 6（约不重叠 30s）
    lag_rows = []
    for lag in (1, 6):
        rs = defaultdict(list)
        for s in samples:
            if len(s.W) < lag + 20:
                continue
            for j, name in enumerate(raw):
                c = lag_corr(s.W[:, j], lag)
                if np.isfinite(c):
                    rs[name].append(c)
        for name in raw:
            vals = rs[name]
            lag_rows.append(
                {
                    "lag_windows": lag,
                    "lag_seconds_approx": lag * 5,
                    "feature": name,
                    "modality": modality_of(name),
                    "median_r": float(np.median(vals)) if vals else None,
                    "mean_r": float(np.mean(vals)) if vals else None,
                    "n_series": int(len(vals)),
                }
            )

    # 被试任务序列
    by_subj = defaultdict(list)
    for s in samples:
        by_subj[s.subject].append(s)
    seq_rows = []
    n_pairs = 0
    for subj, lst in sorted(by_subj.items()):
        lst = sorted(lst, key=lambda z: (TASK_ORDER.get(z.task, 99), z.sample_id))
        seq_rows.append(
            {
                "subject": int(subj),
                "n_tasks": len(lst),
                "tasks": [z.task for z in lst],
                "sample_ids": [z.sample_id for z in lst],
                "n_windows": [int(len(z.W)) for z in lst],
            }
        )
        n_pairs += max(len(lst) - 1, 0)

    # NASA / S 分解
    y = task["y"]
    step = task["step"]
    s_true = task["s_true"]
    nasa_rev = 1.0 - y / 10.0
    var_s = float(np.var(s_true))
    # S = 0.7 step + 0.3 nasa_rev；线性组合方差
    payload = {
        "n_samples": len(samples),
        "n_subjects": int(len(np.unique([s.subject for s in samples]))),
        "n_windows_total": int(n_win.sum()),
        "n_windows": {
            "min": int(n_win.min()),
            "p10": float(np.quantile(n_win, 0.10)),
            "median": float(np.median(n_win)),
            "mean": float(n_win.mean()),
            "p90": float(np.quantile(n_win, 0.90)),
            "max": int(n_win.max()),
            "n_lt_8": int((n_win < 8).sum()),
            "n_lt_16": int((n_win < 16).sum()),
            "n_lt_20": int((n_win < 20).sum()),
        },
        "reconstruct_264_vs_X_task": {
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "ok": bool(max_abs < 1e-8),
        },
        "ratio_coverage": ratio_cov,
        "next_task_pairs": n_pairs,
        "s_components": {
            "S_mean": float(s_true.mean()),
            "S_std": float(s_true.std()),
            "step_mean": float(step.mean()),
            "nasa_mean": float(y.mean()),
            "corr_S_step": float(pd.Series(s_true).corr(pd.Series(step))),
            "corr_S_nasa": float(pd.Series(s_true).corr(pd.Series(y))),
            "corr_S_nasa_rev": float(pd.Series(s_true).corr(pd.Series(nasa_rev))),
            "var_S": var_s,
            "note": "S 的 70% 来自真实步骤分；预报人因指标主要影响 30% 的 NASA 反向分。",
        },
        "stage_early_late_mean_r2_by_modality": {},
        "lag_median_r_by_modality": {},
    }

    stage_df = pd.DataFrame(stage_rows)
    lag_df = pd.DataFrame(lag_rows)
    for mod, g in stage_df.groupby("modality"):
        payload["stage_early_late_mean_r2_by_modality"][mod] = {
            "mean_r2": float(np.nanmean(g["early_late_r2"])),
            "median_pearson": float(np.nanmedian(g["early_late_pearson"])),
        }
    for lag, g in lag_df.groupby("lag_windows"):
        payload["lag_median_r_by_modality"][str(int(lag))] = {
            mod: float(np.nanmedian(sub["median_r"]))
            for mod, sub in g.groupby("modality")
        }

    (OUT / "diagnose.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage_df.to_csv(OUT / "early_vs_late_66mean.csv", index=False, encoding="utf-8-sig")
    lag_df.to_csv(OUT / "lag_autocorr_66.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seq_rows).to_csv(OUT / "subject_task_sequence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples],
            "subject": [s.subject for s in samples],
            "task": [s.task for s in samples],
            "n_windows": n_win,
            "eligible_ratio50": ok50,
        }
    ).to_csv(OUT / "sample_window_counts.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 下一阶段预测：数据诊断\n\n")
    lines.append(f"- 样本 {payload['n_samples']} 条，被试 {payload['n_subjects']} 人，窗口合计 {payload['n_windows_total']}\n")
    nw = payload["n_windows"]
    lines.append(
        f"- 每任务窗口数：min={nw['min']}，P10={nw['p10']:.0f}，中位={nw['median']:.0f}，"
        f"均={nw['mean']:.1f}，P90={nw['p90']:.0f}，max={nw['max']}\n"
    )
    lines.append(
        f"- 窗口过短：<8 窗 {nw['n_lt_8']} 条，<16 窗 {nw['n_lt_16']} 条，<20 窗 {nw['n_lt_20']} 条\n"
    )
    rec = payload["reconstruct_264_vs_X_task"]
    lines.append(
        f"- 用本目录聚合函数还原 264 维 vs 现成 `X_task.npy`：max|Δ|={rec['max_abs_diff']:.2e}，"
        f"{'对齐' if rec['ok'] else '未对齐，检查聚合'}\n\n"
    )
    lines.append("## 按观察比例能切段的样本\n\n")
    lines.append("每段至少 4 窗。\n\n")
    lines.append("| 已观察比例 | 可用条数 | 丢掉 | 可用窗口中位数 |\n|---:|---:|---:|---:|\n")
    for r in ratios:
        d = ratio_cov[str(r)]
        med = d["n_win_ok_median"]
        lines.append(
            f"| {r:.2f} | {d['n_ok']} | {d['n_drop']} | {'' if med is None else f'{med:.0f}'} |\n"
        )
    lines.append("\n## 早段均值 vs 晚段均值（观察 50%）\n\n")
    lines.append("若 R² 高，说明后半段人因水平大致能被前半段「原样沿用」。\n\n")
    lines.append("| 模态 | 66 维均值 R²（平均） | Pearson 中位 |\n|---|---:|---:|\n")
    for mod, d in payload["stage_early_late_mean_r2_by_modality"].items():
        lines.append(f"| {mod} | {d['mean_r2']:+.3f} | {d['median_pearson']:+.3f} |\n")
    lines.append("\n## 窗口滞后相关（中位）\n\n")
    lines.append("步长 1 ≈ 重叠 25 秒，相关会偏高；步长 6 ≈ 30 秒不重叠，更能代表「下一段」。\n\n")
    lines.append("| 模态 | lag-1（5s） | lag-6（30s） |\n|---|---:|---:|\n")
    lag1 = payload["lag_median_r_by_modality"].get("1", {})
    lag6 = payload["lag_median_r_by_modality"].get("6", {})
    for mod in ("眼动", "脑电", "心率", "行为"):
        a = lag1.get(mod)
        b = lag6.get(mod)
        lines.append(
            f"| {mod} | {'' if a is None else f'{a:+.3f}'} | {'' if b is None else f'{b:+.3f}'} |\n"
        )
    lines.append(f"\n## 跨任务「下一任务」\n\n")
    lines.append(f"- 按任务编号 1→2→3→4→5→5_6 排序后，同一被试相邻任务对数：**{n_pairs}**\n")
    lines.append("- 每人通常 3 个不同类型任务，不是同一情景的连续阶段。这条线只作对照，不作主方案。\n")
    sc = payload["s_components"]
    lines.append("\n## S 的结构（为什么必须先报 NASA）\n\n")
    lines.append(
        f"- S 与步骤分相关 {sc['corr_S_step']:+.3f}，与 NASA 相关 {sc['corr_S_nasa']:+.3f}\n"
    )
    lines.append("- 合成公式里步骤占 0.70 且用真值，所以 **S 的 R² 会系统性偏高**；人因预报质量以 NASA R² 为准。\n")
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"[diagnose] wrote {OUT}")


if __name__ == "__main__":
    main()
