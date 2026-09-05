#!/usr/bin/env python3
"""Transparent-background, monochrome flowcharts for the S pathway."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_flow"
FIG.mkdir(parents=True, exist_ok=True)

BG = "none"
INK = "#1A1A1A"
LINE = "#1A1A1A"
FILL = "#FFFFFF"
MUTED = "#4A4A4A"

for name in ("Heiti SC", "PingFang SC", "Songti SC", "STHeiti", "STSong"):
    matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
    if matches:
        plt.rcParams["font.family"] = matches[0].name
        FONT = matches[0].name
        break
else:
    FONT = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def save(fig, name: str):
    png = FIG / f"{name}.png"
    fig.savefig(
        png, dpi=260, facecolor="none", edgecolor="none",
        transparent=True, bbox_inches="tight", pad_inches=0.18,
    )
    plt.close(fig)
    print("fig", png)
    return png


class Canvas:
    def __init__(self, w, h, xlim, ylim):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(ylim[1], ylim[0])  # y increases downward
        self.ax.set_facecolor("none")
        self.fig.patch.set_facecolor("none")
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.nodes = {}

    def _text(self, x, y, text, size=8.3, color=INK, weight="medium"):
        self.ax.text(
            x, y, text, ha="center", va="center", fontsize=size, color=color,
            fontfamily=FONT, linespacing=1.28, zorder=5,
            fontweight=weight if weight != "medium" else "normal",
        )

    def box(self, key, x, y, w, h, text, kind="process", size=8.2):
        self.nodes[key] = (x, y, w, h, kind)
        if kind == "start":
            patch = FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.28",
                facecolor=FILL, edgecolor=LINE, linewidth=1.15, zorder=3,
            )
        elif kind == "decision":
            cx, cy = x + w / 2, y + h / 2
            patch = Polygon(
                [(cx, y), (x + w, cy), (cx, y + h), (x, cy)],
                closed=True, facecolor=FILL, edgecolor=LINE, linewidth=1.15, zorder=3,
            )
        elif kind == "data":
            s = min(0.22, w * 0.12)
            patch = Polygon(
                [(x + s, y), (x + w, y), (x + w - s, y + h), (x, y + h)],
                closed=True, facecolor=FILL, edgecolor=LINE, linewidth=1.15, zorder=3,
            )
        else:
            patch = FancyBboxPatch(
                (x, y), w, h, boxstyle="square,pad=0",
                facecolor=FILL, edgecolor=LINE, linewidth=1.15, zorder=3,
            )
        self.ax.add_patch(patch)
        self._text(x + w / 2, y + h / 2, text, size=size)
        return key

    def port(self, key, side: str):
        x, y, w, h, _ = self.nodes[key]
        if side == "n":
            return x + w / 2, y
        if side == "s":
            return x + w / 2, y + h
        if side == "w":
            return x, y + h / 2
        if side == "e":
            return x + w, y + h / 2
        raise ValueError(side)

    def arrow(self, x1, y1, x2, y2, text=None, text_off=(0.0, 0.0), size=7.2):
        self.ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=LINE, lw=1.12,
                mutation_scale=11, shrinkA=0.4, shrinkB=0.4,
            ),
            zorder=2,
        )
        if text:
            self.ax.text(
                (x1 + x2) / 2 + text_off[0], (y1 + y2) / 2 + text_off[1],
                text, ha="center", va="center", fontsize=size, color=MUTED,
                fontfamily=FONT, zorder=6,
            )

    def v(self, a, b, text=None, text_off=(0.28, 0.0)):
        x1, y1 = self.port(a, "s")
        x2, y2 = self.port(b, "n")
        self.arrow(x1, y1, x2, y2, text, text_off)

    def h(self, a, b, text=None, text_off=(0.0, -0.18)):
        x1, y1 = self.port(a, "e")
        x2, y2 = self.port(b, "w")
        self.arrow(x1, y1, x2, y2, text, text_off)

    def polyline(self, pts, text=None, text_at=None, text_off=(0.0, 0.0)):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:-1]):
            self.ax.plot([x1, x2], [y1, y2], color=LINE, lw=1.12, zorder=2, solid_capstyle="butt")
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
        self.arrow(x1, y1, x2, y2)
        if text:
            tx, ty = text_at if text_at is not None else pts[0]
            self.ax.text(
                tx + text_off[0], ty + text_off[1], text, ha="center", va="center",
                fontsize=7.2, color=MUTED, fontfamily=FONT, zorder=6,
            )

    def elbow_loop(self, src, dst, x_left, yes_no="否"):
        x1, y1 = self.port(src, "w")
        x2, y2 = self.port(dst, "w")
        self.polyline([(x1, y1), (x_left, y1), (x_left, y2), (x2, y2)],
                      text=yes_no, text_at=(x_left, y1), text_off=(0.0, -0.22))

    def label(self, x, y, text, size=8.6, color=MUTED, ha="left"):
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, fontfamily=FONT)

    def badge(self, x, y, text):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), 0.36, 0.36, boxstyle="square,pad=0",
            facecolor=FILL, edgecolor=LINE, linewidth=1.0, zorder=3,
        ))
        self._text(x + 0.18, y + 0.18, text, size=7.6)

    def legend(self, x, y, items):
        for i, (kind, name) in enumerate(items):
            xx = x + i * 2.35
            self.box(f"leg{i}", xx, y, 0.42, 0.28, "", kind=kind, size=1)
            self.ax.text(xx + 0.52, y + 0.14, name, ha="left", va="center",
                         fontsize=7.4, color=MUTED, fontfamily=FONT)


def fig_method():
    C = Canvas(8.9, 13.4, (0.0, 10.0), (0.0, 16.15))
    C.label(5.0, 0.38, "方法流程图  ·  全模态绩效 S 预测", size=12.2, color=LINE, ha="center")

    cx, w = 2.55, 4.90
    items = [
        ("s1", 0.82, 1.8, 0.62, "开始", "start", 8.6),
        ("s2", 1.62, 5.4, 0.78, "输入原始数据\n四模态时序 · NASA-TLX 问卷 · 步骤完成表", "data", 8.0),
        ("s3", 2.58, 5.4, 0.78, "时间对齐并切窗\n窗长 30 s，步长 5 s", "process", 8.1),
        ("s4", 3.54, 5.4, 0.78, "窗内提取 66 项指标\n脑电 / 心率 / 眼动 / 行为", "process", 8.1),
        ("s5", 4.50, 5.4, 0.78, "任务内汇总四统计量\n得到 84 × 264 特征表", "process", 8.1),
        ("s6", 5.46, 5.4, 0.78, "计算步骤分 S_step\n并由问卷得到真值 L 与真值 S", "process", 8.1),
        ("s7", 6.42, 5.4, 0.78, "按被试 GroupKFold\n划分为 5 折，同一人不同时入训考", "process", 8.0),
        ("s8", 7.38, 5.4, 0.86, "训练折：中位数填充\n分模态互信息定额 → 27 维", "process", 8.1),
        ("s9", 8.42, 5.4, 0.78, "浅树 XGBoost 回归\n拟合目标为 NASA-TLX，不是 S", "process", 8.1),
        ("s10", 9.38, 5.4, 0.78, "考试折沿用同一变换\n输出折外预测 L̂", "process", 8.1),
        ("s11", 10.38, 4.4, 1.12, "五折是否\n全部结束？", "decision", 8.0),
        ("s12", 11.72, 5.4, 0.78, "拼合 84 条折外 L̂\nL̂_rev = 1 − L̂ / 10", "process", 8.1),
        ("s13", 12.68, 5.4, 0.86, "公式合成预测绩效\nŜ = 0.70 S_step + 0.30 L̂_rev", "process", 8.1),
        ("s14", 13.72, 5.4, 0.78, "与真值 S 对照\n输出 R²、MAE", "process", 8.1),
        ("s15", 14.68, 1.8, 0.62, "结束", "start", 8.6),
    ]
    for key, y, ww, hh, text, kind, sz in items:
        x = 5.0 - ww / 2
        if key == "s1":
            x = 5.0 - 1.8 / 2
            ww, hh = 1.8, 0.62
        if key == "s15":
            x = 5.0 - 1.8 / 2
            ww, hh = 1.8, 0.62
        if key == "s11":
            x = 5.0 - ww / 2
        C.box(key, x, y, ww, hh, text, kind=kind, size=sz)

    for a, b in [
        ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5"),
        ("s5", "s6"), ("s6", "s7"), ("s7", "s8"), ("s8", "s9"),
        ("s9", "s10"), ("s10", "s11"), ("s12", "s13"), ("s13", "s14"),
        ("s14", "s15"),
    ]:
        C.v(a, b)

    C.v("s11", "s12", "是", text_off=(0.32, 0.0))
    C.elbow_loop("s11", "s8", x_left=1.15, yes_no="否")

    C.ax.text(7.85, 7.81, "眼动 6  脑电 5\n心率 4  行为 12", ha="left", va="center",
              fontsize=7.1, color=MUTED, fontfamily=FONT, linespacing=1.25)
    C.ax.text(7.85, 8.81, "目标为 NASA\n不是 S", ha="left", va="center",
              fontsize=7.1, color=MUTED, fontfamily=FONT, linespacing=1.25)

    C.legend(0.55, 15.55, [
        ("start", "起止"),
        ("data", "数据"),
        ("process", "处理"),
        ("decision", "判断"),
    ])
    return save(C.fig, "fig1_s_method_flow")


def fig_dataflow():
    C = Canvas(12.8, 10.4, (0.0, 16.4), (0.0, 12.35))
    C.label(8.2, 0.38, "数据流图  ·  全模态绩效 S 预测", size=12.2, color=LINE, ha="center")

    layers = [
        (1.02, "原始数据"),
        (2.28, "窗  级"),
        (4.55, "任务级"),
        (6.55, "折  内"),
        (8.35, "模  型"),
        (10.05, "合  成"),
        (11.25, "评  价"),
    ]
    for y, t in layers:
        C.label(0.22, y, t, size=7.4, color=MUTED, ha="left")

    srcs = [
        ("eeg", 1.15, "脑电\n.set  256 Hz"),
        ("hr", 2.90, "心率\n逐点时序"),
        ("eye", 4.65, "眼动\n.tsv 注视/瞳孔"),
        ("log", 6.40, "操作日志\n动作序列"),
    ]
    for k, x, t in srcs:
        C.box(k, x, 0.78, 1.55, 0.72, t, kind="data", size=7.2)
    C.box("nasa", 10.35, 0.78, 2.10, 0.72, "NASA-TLX 问卷\n六维加权总分", kind="data", size=7.2)
    C.box("step", 13.15, 0.78, 2.10, 0.72, "步骤完成表\n关键 / 非关键", kind="data", size=7.2)

    C.box("win", 1.15, 2.00, 6.80, 0.64, "切窗并对齐    30 s / 5 s", kind="process", size=8.0)
    C.box("readL", 10.35, 2.00, 2.10, 0.64, "读取加权总分", kind="process", size=7.5)
    C.box("readS", 13.15, 2.00, 2.10, 0.64, "统计完成率", kind="process", size=7.5)

    C.box("wtab", 1.15, 2.92, 6.80, 0.64, "窗级特征表     n_win × 66", kind="data", size=8.0)
    C.box("L", 10.35, 2.92, 2.10, 0.64, "L    (84,)", kind="data", size=8.0)
    C.box("Sstep", 13.15, 2.92, 2.10, 0.64, "S_step    (84,)", kind="data", size=7.8)

    C.box("agg", 1.15, 3.84, 6.80, 0.64, "任务内汇总    mean / std / median / slope", kind="process", size=7.8)
    C.box("x264", 1.15, 4.76, 6.80, 0.64, "任务级特征表     84 × 264", kind="data", size=8.0)
    C.box("Strue", 10.35, 4.76, 4.90, 0.64, "真值 S ＝ 0.70 S_step ＋ 0.30 (1 − L / 10)", kind="data", size=7.2)

    C.box("sel", 1.15, 5.88, 6.80, 0.64, "折内填充 · 分模态定额互信息", kind="process", size=7.8)
    C.box("x27", 1.15, 6.80, 6.80, 0.64, "紧凑特征表     84 × 27    （眼动 6 / 脑电 5 / 心率 4 / 行为 12）", kind="data", size=7.4)

    C.box("xgb", 1.15, 7.80, 6.80, 0.64, "浅树 XGBoost（回归目标为 NASA，不是 S）", kind="process", size=7.8)
    C.box("Lhat", 1.15, 8.80, 6.80, 0.64, "折外预测  L̂    (84,)", kind="data", size=8.0)

    C.box("mix", 1.15, 9.88, 6.80, 0.64, "公式合成   Ŝ ＝ 0.70 S_step ＋ 0.30 (1 − L̂ / 10)", kind="process", size=7.6)
    C.box("Shat", 3.20, 10.88, 2.70, 0.64, "预测  Ŝ    (84,)", kind="data", size=8.0)
    C.box("met", 6.20, 10.88, 2.20, 0.64, "R² 、 MAE", kind="data", size=8.0)
    C.box("Sout", 10.35, 10.88, 4.90, 0.64, "真值  S    (84,)", kind="data", size=8.0)

    bus_y = 1.68
    for k in ("eeg", "hr", "eye", "log"):
        x1, y1 = C.port(k, "s")
        C.ax.plot([x1, x1], [y1, bus_y], color=LINE, lw=1.12, zorder=2)
    C.ax.plot([C.port("eeg", "s")[0], C.port("log", "s")[0]], [bus_y, bus_y], color=LINE, lw=1.12, zorder=2)
    wx, wy = C.port("win", "n")
    C.arrow(wx, bus_y, wx, wy)

    C.v("win", "wtab")
    C.v("wtab", "agg")
    C.v("agg", "x264")
    C.v("x264", "sel")
    C.v("sel", "x27")
    C.v("x27", "xgb")
    C.v("xgb", "Lhat")
    C.v("Lhat", "mix")
    C.v("mix", "Shat")

    C.v("nasa", "readL")
    C.v("readL", "L")
    C.v("step", "readS")
    C.v("readS", "Sstep")

    lx, ly = C.port("L", "s")
    sx, sy = C.port("Sstep", "s")
    tn, ty = C.port("Strue", "n")
    mid_y = (ly + ty) / 2
    C.polyline([(lx, ly), (lx, mid_y), (tn, mid_y), (tn, ty)])
    C.polyline([(sx, sy), (sx, mid_y), (tn, mid_y)])

    rail = 15.70
    se_x, se_y = C.port("Sstep", "e")
    mx, my = C.port("mix", "e")
    C.polyline(
        [(se_x, se_y), (rail, se_y), (rail, my), (mx, my)],
        text="真实步骤分", text_at=(rail, 6.55), text_off=(-0.62, 0.0),
    )

    C.v("Strue", "Sout")
    C.h("Shat", "met", text="对照", text_off=(0.0, -0.20))
    x1, y1 = C.port("Sout", "w")
    x2, y2 = C.port("met", "e")
    C.polyline([(x1, y1), (x2, y2)], text="对照", text_at=((x1 + x2) / 2, y1), text_off=(0.0, -0.20))

    C.legend(1.15, 11.80, [
        ("data", "数据 / 表"),
        ("process", "处理过程"),
    ])
    C.label(6.4, 11.94, "步骤分始终来自真实操作记录，不经过模型", size=7.4, color=MUTED, ha="left")
    return save(C.fig, "fig2_s_data_flow")


if __name__ == "__main__":
    fig_method()
    fig_dataflow()
