#!/usr/bin/env python3
"""软件「趋势预测与预警」界面效果图。

虚线：池化 Transformer 直接预报，再按最后观测点锚定（v16 pool_anchor）。
右侧 S：V8 Ridge 27 → 冻结 XGB，不是窗级虚线积出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIG = HERE / "figures"
DEMO = HERE / "reports" / "v16_tf_anchor" / "selected_demo.npz"
RATIO = 0.50
S_HAT = None  # 从 npz 的 Ridge S 读取
S_THR = 0.51

MAIN = (0, "心率均值", "bpm")
THUMBS = [
    (3, "AOI 覆盖"),
    (4, "操作密度"),
    (2, "瞳孔直径"),
    (1, "心率波动"),
    (5, "操作次数"),
    (6, "额区 θ/α"),
]
CHIPS = ["心率均值", "AOI 覆盖", "操作密度", "瞳孔直径", "操作次数", "…共 27 项"]

BG = "#F2F3F5"
CARD = "#FFFFFF"
LINE = "#2F6FED"
DASH = "#E86B2A"
NOW = "#8A8F99"
INK = "#1A1D23"
MUTED = "#6B7280"
OK = "#1F8A4C"
OK_BG = "#E7F6EC"
RULE = "#E5E7EB"


def setup_font() -> None:
    for name in ("PingFang SC", "Heiti SC", "Songti SC", "STSong"):
        matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
        if matches:
            plt.rcParams["font.family"] = matches[0].name
            break
    plt.rcParams["axes.unicode_minus"] = False


def card(ax, x, y, w, h, fc=CARD):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fc,
        edgecolor="#E2E4E8",
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(p)


def minutes(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * 5.0 / 60.0


def series_xy(W: np.ndarray, col: int):
    y = W[:, col].astype(np.float64)
    t = minutes(len(y))
    return t, y


def _fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return np.zeros_like(y)
    if ok.sum() < len(y):
        idx = np.arange(len(y))
        y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def mean_revert(early: np.ndarray, n_late: int, rho=0.90):
    """从最后一个点指数回到前段均值。评测里比水平均值更好，也更像趋势。"""
    y = _fill_nan(early)
    if n_late < 1:
        return np.array([]), 0.0
    mu, last = float(y.mean()), float(y[-1])
    k = np.arange(1, n_late + 1, dtype=np.float64)
    hat = mu + (last - mu) * (rho**k)
    d = np.diff(y)
    sigma = float(np.std(d)) if len(d) > 2 else float(np.std(y))
    if not np.isfinite(sigma):
        sigma = 0.0
    return hat, sigma


def plot_curve(ax, t, y, cut, hat=None, color=LINE, band=False):
    early = y[:cut]
    n_late = len(y) - cut
    if hat is None:
        hat, sigma = mean_revert(early, n_late)
    else:
        hat = np.asarray(hat, dtype=np.float64)
        d = np.diff(early[np.isfinite(early)]) if np.isfinite(early).any() else np.array([0.0])
        sigma = float(np.std(d)) if len(d) else 0.0
    last = early[np.isfinite(early)][-1] if np.isfinite(early).any() else hat[0]
    t_f = np.concatenate([[t[cut - 1]], t[cut:]])
    y_f = np.concatenate([[last], hat])
    ax.plot(t[:cut], y[:cut], color=color, lw=1.8, solid_capstyle="round", label="已观察")
    if band and sigma > 0:
        k = np.arange(0, n_late + 1, dtype=np.float64)
        half = 1.05 * sigma * np.sqrt(np.maximum(k, 1.0))
        half[0] = 0.25 * sigma
        ax.fill_between(t_f, y_f - half, y_f + half, color=DASH, alpha=0.16, linewidth=0, zorder=1)
    ax.plot(
        t_f,
        y_f,
        color=DASH,
        lw=1.85,
        ls=(0, (4.5, 2.6)),
        label="预测（Transformer 轨迹）",
        zorder=2,
    )
    ax.axvline(t[cut - 1], color=NOW, lw=0.8, ls="--", zorder=1)
    ax.axvspan(t[cut - 1], t[-1], color="#FFF4EC", alpha=0.45, zorder=0)
    ax.set_xlim(t[0], t[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#D0D4DA")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(CARD)


def main() -> None:
    setup_font()
    FIG.mkdir(parents=True, exist_ok=True)
    demo = np.load(DEMO, allow_pickle=True)
    sid = str(demo["sample_id"])
    early, late, yhat = demo["early"], demo["late"], demo["yhat"]
    cut = int(demo["cut"])
    s_hat = float(demo["S_ridge"]) if "S_ridge" in demo.files else (S_HAT or 0.68)
    y_all = np.vstack([early, late])
    t = minutes(len(y_all))
    t_now = t[cut - 1]
    subj, task = sid.replace("subject_", "").split("_task_")

    fig = plt.figure(figsize=(12.6, 7.35), facecolor=BG)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[3.15, 1.55],
        width_ratios=[3.15, 1.0],
        left=0.055,
        right=0.975,
        top=0.88,
        bottom=0.07,
        wspace=0.045,
        hspace=0.16,
    )

    fig.text(0.055, 0.955, "趋势预测与预警", fontsize=16, color=INK, fontweight="medium")
    fig.text(
        0.055,
        0.922,
        f"被试 {int(subj)}  ·  任务 {task}  ·  已观察 "
        f"{t_now:.1f} 分钟  ·  Transformer 轨迹（末值锚定）",
        fontsize=9,
        color=MUTED,
    )

    x0 = 0.40
    for i, name in enumerate(CHIPS):
        on = i == 0
        tw = 0.075 if i < 5 else 0.078
        box = FancyBboxPatch(
            (x0, 0.918),
            tw,
            0.038,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor=LINE if on else "#FFFFFF",
            edgecolor=LINE if on else "#D6DAE1",
            linewidth=0.8,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(box)
        fig.text(
            x0 + tw / 2,
            0.937,
            name,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if on else INK,
            zorder=3,
        )
        x0 += tw + 0.008

    ax = fig.add_subplot(gs[0, 0])
    plot_curve(ax, t, y_all[:, MAIN[0]], cut, hat=yhat[:, MAIN[0]], band=True)
    ax.set_title(f"{MAIN[1]}（{MAIN[2]}）", loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("任务时间（分钟）", fontsize=8.5, color=MUTED)
    ax.set_ylabel(MAIN[2], fontsize=8.5, color=MUTED)
    ax.text(
        t_now,
        ax.get_ylim()[1],
        "  现在",
        color=MUTED,
        fontsize=8,
        va="top",
    )
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )

    axr = fig.add_subplot(gs[0, 1])
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    axr.set_facecolor(BG)
    axr.text(0.06, 0.96, "人员状态", fontsize=11, color=INK, va="top")

    box = FancyBboxPatch(
        (0.06, 0.58),
        0.88,
        0.32,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=OK_BG,
        edgecolor="#B7E0C4",
        linewidth=0.9,
        transform=axr.transAxes,
        clip_on=False,
    )
    axr.add_patch(box)
    axr.text(0.16, 0.82, "正常", fontsize=15, color=OK, fontweight="medium")
    axr.text(0.16, 0.70, f"预测绩效 S  =  {s_hat:.2f}", fontsize=9, color=INK)
    axr.text(0.16, 0.62, f"预警阈值  =  {S_THR:.2f}（低分位）", fontsize=8, color=MUTED)

    axr.text(0.06, 0.50, "安全建议", fontsize=11, color=INK)
    for i, line in enumerate(
        [
            "基于当前预测绩效，未来风险较低。",
            "建议保持现有操作节奏。",
            "若预测 S 低于 0.51，右侧将改为预警。",
        ]
    ):
        axr.text(0.06, 0.42 - i * 0.08, line, fontsize=8.5, color=MUTED)

    gs_b = gs[1, :].subgridspec(1, 6, wspace=0.12)
    for i, (name, label) in enumerate(THUMBS):
        a = fig.add_subplot(gs_b[0, i])
        plot_curve(a, t, y_all[:, name], cut, hat=yhat[:, name], color="#3B6FA0")
        a.set_title(label, fontsize=8, color=INK, loc="left", pad=3)
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("")
        for sp in a.spines.values():
            sp.set_color("#E2E4E8")
    fig.text(
        0.055,
        0.035,
        f"点选缩略图切换主图  ·  {sid}  ·  虚线=Transformer 窗级轨迹，不是 S；S 由 Ridge 27 给出",
        fontsize=8,
        color=MUTED,
    )

    out = FIG / "fig_ui_trend_warning.png"
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
