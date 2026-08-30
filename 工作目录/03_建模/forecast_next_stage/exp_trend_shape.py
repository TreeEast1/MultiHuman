#!/usr/bin/env python3
"""会画出「趋势」的窗级预报：衰减 Holt、均值回归、阻尼斜率 vs 均值水平线。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    REPORTS,
    eligible_mask,
    load_feature_names,
    load_samples,
    load_task_arrays,
    split_index,
)

OUT = REPORTS / "v11_trend_shape"
FIG = HERE / "figures"
RATIO = 0.50
MIN_EACH = 4

SERIES = [
    ("hr_mean", "心率均值"),
    ("hr_std", "心率波动"),
    ("eye_pupil_filtered_mean", "瞳孔直径"),
    ("eye_aoi_coverage_ratio", "AOI覆盖"),
    ("log_action_density_win", "操作密度"),
    ("log_action_count_win", "操作次数"),
    ("eeg_frontal_theta_alpha_z_within_subject", "额区θ/α"),
    ("blink_rate_per_min", "眨眼频率"),
]


def setup_font() -> None:
    for name in ("PingFang SC", "Heiti SC", "Songti SC"):
        matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
        if matches:
            plt.rcParams["font.family"] = matches[0].name
            break
    plt.rcParams["axes.unicode_minus"] = False


def fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return np.zeros_like(y)
    if ok.sum() < len(y):
        idx = np.arange(len(y))
        y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def persist_mean(early: np.ndarray, n: int) -> np.ndarray:
    y = fill_nan(early)
    return np.full(n, float(y.mean()))


def persist_last(early: np.ndarray, n: int) -> np.ndarray:
    y = fill_nan(early)
    return np.full(n, float(y[-1]))


def linear_trend(early: np.ndarray, n: int) -> np.ndarray:
    y = fill_nan(early)
    x = np.arange(len(y), dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    xt = np.arange(len(y), len(y) + n, dtype=np.float64)
    return a * xt + b


def holt_damped(early: np.ndarray, n: int, alpha=0.28, beta=0.12, phi=0.88) -> np.ndarray:
    y = fill_nan(early)
    if len(y) < 3:
        return persist_last(y, n)
    level, trend = float(y[0]), float(y[1] - y[0])
    for val in y:
        prev = level
        level = alpha * val + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1.0 - beta) * phi * trend
    hat = np.empty(n)
    acc = 0.0
    p = 1.0
    for h in range(n):
        p *= phi
        acc += p
        hat[h] = level + trend * acc
    return hat


def mean_revert(early: np.ndarray, n: int, rho=0.90) -> np.ndarray:
    """末值指数回到前段均值：有曲线，终点接近 persist_mean。"""
    y = fill_nan(early)
    mu, last = float(y.mean()), float(y[-1])
    k = np.arange(1, n + 1, dtype=np.float64)
    return mu + (last - mu) * (rho**k)


def damped_drift(early: np.ndarray, n: int, phi=0.90) -> np.ndarray:
    """末值 + 近期斜率，步长越大斜率越弱。"""
    y = fill_nan(early)
    last = float(y[-1])
    w = min(12, len(y))
    x = np.arange(w, dtype=np.float64)
    sl = float(np.polyfit(x, y[-w:], 1)[0])
    k = np.arange(1, n + 1, dtype=np.float64)
    return last + sl * k * (phi**k)


def ses_level(early: np.ndarray, n: int, alpha=0.35) -> np.ndarray:
    y = fill_nan(early)
    level = float(y[0])
    for val in y:
        level = alpha * val + (1.0 - alpha) * level
    return np.full(n, level)


def safe_r2(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 4:
        return float("nan")
    yt, yh = y[m], yhat[m]
    if np.allclose(yt, yt.mean()):
        return float("nan")
    return float(r2_score(yt, yh))


def safe_mae(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y[m], yhat[m]))


def safe_pearson(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 4:
        return float("nan")
    a, b = y[m], yhat[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


ALGOS = {
    "persist_mean": persist_mean,
    "persist_last": persist_last,
    "ses_level": ses_level,
    "mean_revert": mean_revert,
    "holt_damped": holt_damped,
    "damped_drift": damped_drift,
    "linear_trend": linear_trend,
}


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    _, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    idx = [col[n] for n, _ in SERIES]
    labels = [zh for _, zh in SERIES]
    feat_names = [n for n, _ in SERIES]

    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: k for k, s in enumerate(samples)}
    samples = [samples[by_id[sid]] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, MIN_EACH)

    fold_of = np.full(len(samples), -1, dtype=int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for f, (_, te) in enumerate(gkf.split(task["X"], task["y"], task["groups"])):
        fold_of[te] = f

    meta, truth = [], []
    hats = {k: [] for k in ALGOS}
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        cut = split_index(len(s.W), RATIO)
        early, late = s.W[:cut][:, idx], s.W[cut:][:, idx]
        n2 = len(late)
        truth.append(late)
        meta.append({"sample_id": s.sample_id, "i": i, "fold": int(fold_of[i]), "n_late": n2, "cut": cut})
        for name, fn in ALGOS.items():
            pred = np.column_stack([fn(early[:, j], n2) for j in range(early.shape[1])])
            hats[name].append(pred)

    Y = np.vstack(truth)
    for name in hats:
        hats[name] = np.vstack(hats[name])
    row_fold = np.concatenate([np.full(r["n_late"], r["fold"], dtype=int) for r in meta])
    near = np.concatenate([np.arange(r["n_late"]) < min(12, r["n_late"]) for r in meta])

    table = []
    for name, Yh in hats.items():
        rows, rows1, rowsn = [], [], []
        for j, (fn, lab) in enumerate(zip(feat_names, labels)):
            pack = lambda yt, yh: {
                "feature": fn,
                "label": lab,
                "r2": safe_r2(yt, yh),
                "mae": safe_mae(yt, yh),
                "pearson": safe_pearson(yt, yh),
            }
            rows.append(pack(Y[:, j], Yh[:, j]))
            m1 = row_fold == 0
            rows1.append(pack(Y[m1, j], Yh[m1, j]))
            rowsn.append(pack(Y[near, j], Yh[near, j]))
        rec = {
            "model": name,
            "mean_r2": float(np.nanmean([r["r2"] for r in rows])),
            "mean_pearson": float(np.nanmean([r["pearson"] for r in rows])),
            "fold1_mean_r2": float(np.nanmean([r["r2"] for r in rows1])),
            "near12_mean_r2": float(np.nanmean([r["r2"] for r in rowsn])),
            "per_series": rows,
        }
        table.append(rec)
        print(
            f"  {name:14s}  R²={rec['mean_r2']:+.3f}  "
            f"near12={rec['near12_mean_r2']:+.3f}  fold1={rec['fold1_mean_r2']:+.3f}"
        )

    (OUT / "metrics.json").write_text(
        json.dumps({"n_windows": int(len(Y)), "n_tasks": int(len(meta)), "models": table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat = []
    for r in table:
        for row in r["per_series"]:
            flat.append({"model": r["model"], **row})
    pd.DataFrame(flat).to_csv(OUT / "metrics.csv", index=False)

    plot_ids = [sid for sid in ("subject_02_task_1", "subject_07_task_2", "subject_12_task_2") if any(m["sample_id"] == sid for m in meta)]
    rec_by = {m["sample_id"]: m for m in meta}
    off = np.cumsum([0] + [m["n_late"] for m in meta])
    pos = {m["sample_id"]: (off[k], off[k + 1]) for k, m in enumerate(meta)}
    show = ["persist_mean", "mean_revert", "holt_damped"]
    colors = {"persist_mean": "0.45", "mean_revert": "#1F4E79", "holt_damped": "#C45C26"}
    fig, axes = plt.subplots(len(plot_ids), 2, figsize=(8.6, 7.2))
    series_plot = [0, 4]
    for ri, sid in enumerate(plot_ids):
        rec = rec_by[sid]
        s = samples[rec["i"]]
        cut = rec["cut"]
        a, b = pos[sid]
        for ci, sj in enumerate(series_plot):
            ax = axes[ri, ci]
            t = np.arange(len(s.W))
            ax.plot(t[:cut], s.W[:cut, idx[sj]], color="black", lw=1.1, label="已观察")
            ax.plot(t[cut:], s.W[cut:, idx[sj]], color="0.6", lw=0.9, ls=":", label="未来真值")
            for mn in show:
                ax.plot(t[cut:], hats[mn][a:b, sj], color=colors[mn], lw=1.2, ls="--", label=mn)
            ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
            ax.set_title(f"{sid}  {labels[sj]}", loc="left", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_trend_shape.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "# 趋势形态算法（窗级人因，不是 S）\n\n",
        f"{len(meta)} 场、{len(Y)} 窗，后半段预报。\n\n",
        "| 算法 | 后半段 R² | 近 12 窗 R² | 第 1 折 R² | 形态 |\n|---|---:|---:|---:|---|\n",
    ]
    shape = {
        "persist_mean": "水平线",
        "persist_last": "水平线",
        "ses_level": "水平线（平滑末值）",
        "mean_revert": "从末值弯回均值",
        "holt_damped": "从末值沿衰减趋势走",
        "damped_drift": "从末值带阻尼斜率",
        "linear_trend": "直线外推",
    }
    for r in table:
        lines.append(
            f"| {r['model']} | {r['mean_r2']:+.3f} | {r['near12_mean_r2']:+.3f} | "
            f"{r['fold1_mean_r2']:+.3f} | {shape[r['model']]} |\n"
        )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v11] wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
