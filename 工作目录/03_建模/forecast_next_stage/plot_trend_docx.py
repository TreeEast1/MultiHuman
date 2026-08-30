#!/usr/bin/env python3
"""趋势预测 Word 报告配图（与全模态 27 维实验报告同一套黑白体例）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
PRED = HERE / "reports" / "v8_quota27_space" / "models" / "ridge_scaled" / "predictions.csv"

for name in ("Songti SC", "STSong", "Heiti SC", "PingFang SC"):
    matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
    if matches:
        plt.rcParams["font.family"] = matches[0].name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
BLACK = "black"


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig", p)
    return p


def fig_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.10, 1.05, 1.85, 1.25, "已观察阶段\n聚合为 264 维"),
        (2.10, 1.05, 1.80, 1.25, "分模态定额\n互信息\n→ 27 维"),
        (4.05, 1.05, 1.90, 1.25, "标准化 Ridge\n预报整场 27 维"),
        (6.10, 1.05, 1.80, 1.25, "冻结浅树\nXGBoost\n→ 负荷 L"),
        (8.05, 1.05, 1.95, 1.25, "公式合成\n预测绩效 S"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="white",
                edgecolor=BLACK,
                linewidth=1.1,
            )
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.4, color=BLACK)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.annotate(
            "",
            xy=(x2 - 0.02, y),
            xytext=(x1 + 0.02, y),
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0),
        )
    save(fig, "fig1_pipeline.png")
    return FIG / "fig1_pipeline.png"


def load_fold1() -> pd.DataFrame:
    pred = pd.read_csv(PRED)
    gkf = GroupKFold(n_splits=5)
    fold = np.full(len(pred), -1)
    for i, (_, te) in enumerate(gkf.split(pred["y_nasa"], pred["y_nasa"], pred["subject"])):
        fold[te] = i
    pred = pred.assign(fold=fold)
    return pred[(pred["fold"] == 0) & (pred["eligible"] == 1)].copy()


def fig_scatter() -> Path:
    df = load_fold1()
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(
        df["S_true"], df["S_hat"], s=28, c="white",
        edgecolors=BLACK, linewidths=0.85, zorder=3,
    )
    slim = [0.38, 0.90]
    ax.plot(slim, slim, color=BLACK, lw=0.9, ls="--")
    ax.set_xlim(slim)
    ax.set_ylim(slim)
    ax.set_xlabel("真值 S")
    ax.set_ylabel("预测 S")
    ax.set_title(
        f"真值与预测绩效 S（17 条）：R²＝{r2_score(df['S_true'], df['S_hat']):.3f}，"
        f"MAE＝{mean_absolute_error(df['S_true'], df['S_hat']):.3f}",
        fontsize=10.5,
    )
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig2_scatter.png")
    return FIG / "fig2_scatter.png"


def main() -> None:
    fig_pipeline()
    fig_scatter()


if __name__ == "__main__":
    main()
