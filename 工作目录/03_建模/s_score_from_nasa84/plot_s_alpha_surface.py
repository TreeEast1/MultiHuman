#!/usr/bin/env python3
"""合成绩效 S 的 α 曲面图 + 本实验 84 条样本的实际取值。

S(α) = α × 步骤分（客观）+ (1 − α) × NASA 反向分（主观）
正式口径 α = 0.70。

产出（figures/）：
    fig1_s_alpha_surface.png   三块 α 曲面，红点为 84 条真实样本
    fig2_s_results.png         α=0.70 下 S 的分布 / 分任务 / 随 α 变化 / 预测效果

运行：
    uv run --with pandas --with numpy --with matplotlib --with scikit-learn \
        python plot_s_alpha_surface.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
S_TABLE = HERE / "output" / "s_score_84samples.csv"
PRED_DIR = HERE / "reports_s_fullmodal"
FIG_DIR = HERE / "figures"

ALPHA_MAIN = 0.70
ALPHA_PANELS = (0.30, 0.50, 0.70)
TASK_ORDER = ["1", "2", "3", "4", "5", "5_6"]

C_POINT = "#D6202A"
C_STEP = "#3A7CA5"
C_S = "#E07B39"


CJK_FONT_FILES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def setup_font() -> None:
    from matplotlib import font_manager

    for path in CJK_FONT_FILES:
        if not Path(path).exists():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
        except RuntimeError:
            continue
        plt.rcParams["font.family"] = name
        break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["savefig.dpi"] = 300


def mix_s(step: np.ndarray, nasa_reverse: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * step + (1.0 - alpha) * nasa_reverse


def load_table() -> pd.DataFrame:
    df = pd.read_csv(S_TABLE)
    df["task"] = df["task"].astype(str)
    return df


def plot_surfaces(df: pd.DataFrame, figsize=(14.5, 5.4), out_name="fig1_s_alpha_surface.png") -> Path:
    step = df["weighted_step_score"].to_numpy(dtype=float)
    nasa_rev = df["nasa_reverse"].to_numpy(dtype=float)

    grid = np.linspace(0.0, 1.0, 41)
    gx, gy = np.meshgrid(grid, grid)

    fig = plt.figure(figsize=figsize)
    for i, alpha in enumerate(ALPHA_PANELS):
        ax = fig.add_subplot(1, len(ALPHA_PANELS), i + 1, projection="3d")
        # 样本点与曲面严格共面，关掉自动深度排序才能保证红点画在曲面之上
        ax.computed_zorder = False
        zz = mix_s(gx, gy, alpha)
        ax.plot_surface(
            gx, gy, zz, cmap="viridis", vmin=0.0, vmax=1.0,
            rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.55, zorder=1,
        )

        s_val = mix_s(step, nasa_rev, alpha)
        ax.scatter(
            step, nasa_rev, s_val, s=20, color=C_POINT,
            edgecolors="white", linewidths=0.4, depthshade=False, zorder=6,
        )

        ax.set_xlabel("客观值 · 步骤分", labelpad=6)
        ax.set_ylabel("主观值 · NASA 反向分", labelpad=6)
        ax.set_zlabel("S", labelpad=0)
        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_zticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(labelsize=8, pad=-1)
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect((1, 1, 0.85), zoom=1.02)

        formal = abs(alpha - ALPHA_MAIN) < 1e-9
        head = f"α = {alpha:.2f}" + ("（正式口径）" if formal else "")
        ax.set_title(head, fontsize=12,
                     fontweight="bold" if formal else "normal", pad=2)
        ax.text2D(
            0.0, 0.99,
            f"84 条样本 S：{s_val.min():.2f} – {s_val.max():.2f}\n"
            f"均值 {s_val.mean():.3f} ± {s_val.std(ddof=1):.3f}",
            transform=ax.transAxes, fontsize=9, color="#333333",
            va="top", linespacing=1.4,
        )

    fig.suptitle(
        "不同 α 下的合成绩效 S 曲面（α 越大，客观步骤分的贡献越大）　"
        "S = α × 步骤分 + (1 − α) × NASA 反向分",
        fontsize=13.5, y=0.995,
    )
    fig.legend(
        handles=[Line2D([], [], marker="o", linestyle="", color=C_POINT,
                        markeredgecolor="white", markersize=7,
                        label="本实验 84 条被试–任务样本的实际取值（按定义严格落在曲面上）")],
        loc="lower center", frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.13, wspace=0.10)

    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_results(df: pd.DataFrame, figsize=(12.0, 8.4), out_name="fig2_s_results.png") -> Path:
    step = df["weighted_step_score"].to_numpy(dtype=float)
    nasa_rev = df["nasa_reverse"].to_numpy(dtype=float)
    s07 = mix_s(step, nasa_rev, ALPHA_MAIN)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    ax.hist(s07, bins=16, range=(0.1, 1.0), color=C_S, edgecolor="white", alpha=0.9)
    ax.axvline(s07.mean(), color=C_POINT, linestyle="--", linewidth=1.4)
    ax.text(
        s07.mean() + 0.012, ax.get_ylim()[1] * 0.92,
        f"均值 {s07.mean():.3f}", color=C_POINT, fontsize=9.5,
    )
    ax.set_xlabel("合成绩效 S（α = 0.70）")
    ax.set_ylabel("样本数")
    ax.set_title(
        f"(a) 真值 S 的分布：{s07.min():.2f} – {s07.max():.2f}，"
        f"{s07.mean():.3f} ± {s07.std(ddof=1):.3f}",
        loc="left", fontsize=11,
    )
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    g = (
        df.assign(S07=s07)
        .groupby("task")
        .agg(n=("S07", "size"), step=("weighted_step_score", "mean"), S=("S07", "mean"))
        .reindex(TASK_ORDER)
    )
    x = np.arange(len(g))
    w = 0.38
    ax.bar(x - w / 2, g["step"], w, label="步骤分均值", color=C_STEP)
    ax.bar(x + w / 2, g["S"], w, label="合成 S 均值", color=C_S)
    for xi, (sv, ss) in enumerate(zip(g["step"], g["S"])):
        ax.text(xi - w / 2, sv + 0.012, f"{sv:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, ss + 0.012, f"{ss:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"任务 {t}\n(n={int(n)})" for t, n in zip(g.index, g["n"])], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分值")
    ax.set_title("(b) 按任务类型的步骤分与合成 S（α = 0.70）", loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    alphas = np.linspace(0.0, 1.0, 101)
    curves = np.outer(alphas, step) + np.outer(1.0 - alphas, nasa_rev)
    for j in range(curves.shape[1]):
        ax.plot(alphas, curves[:, j], color="0.72", linewidth=0.5, alpha=0.6)
    ax.plot(alphas, curves.mean(axis=1), color=C_POINT, linewidth=2.2, label="84 条均值")
    ax.axvline(ALPHA_MAIN, color="#2B2B2B", linestyle="--", linewidth=1.2)
    ax.plot([ALPHA_MAIN], [s07.mean()], marker="o", color=C_POINT, markersize=7,
            markeredgecolor="white")
    ax.annotate(
        f"α = 0.70\n均值 {s07.mean():.3f}",
        xy=(ALPHA_MAIN, s07.mean()), xytext=(ALPHA_MAIN - 0.30, s07.mean() + 0.20),
        fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#2B2B2B", lw=1.0),
    )
    ax.set_xlabel("α（客观步骤分的权重）")
    ax.set_ylabel("S")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) 每条样本的 S 随 α 的变化（灰线 84 条，红线为均值）", loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    y_true = np.load(PRED_DIR / "y_s07.npy")
    y_hat = np.load(PRED_DIR / "yhat_s07_xgb.npy")
    ax.scatter(y_true, y_hat, s=26, color=C_POINT, alpha=0.75,
               edgecolors="white", linewidths=0.4)
    lim = (0.12, 1.0)
    ax.plot(lim, lim, color="0.35", linestyle="--", linewidth=1.1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("真值 S（α = 0.70）")
    ax.set_ylabel("预测 S（27 维 · NASA 公式法）")
    ax.set_title(
        f"(d) 五折交叉验证：R² = {r2_score(y_true, y_hat):.3f}，"
        f"MAE = {mean_absolute_error(y_true, y_hat):.3f}",
        loc="left", fontsize=11,
    )
    ax.grid(alpha=0.25)

    fig.suptitle(
        "本实验计算出的合成绩效 S（26 被试 / 84 次任务，α = 0.70）",
        fontsize=13.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()
    print(f"[plot] 样本数 {len(df)}")
    print("[plot] 写出", plot_surfaces(df))
    print("[plot] 写出", plot_results(df))
    # _docx 版：画布更小、字号不变 → 插进 A4 页宽后字仍然看得清
    print("[plot] 写出", plot_surfaces(df, (10.0, 4.3), "fig1_s_alpha_surface_docx.png"))
    print("[plot] 写出", plot_results(df, (9.2, 6.6), "fig2_s_results_docx.png"))


if __name__ == "__main__":
    main()
