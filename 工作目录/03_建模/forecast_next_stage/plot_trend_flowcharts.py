#!/usr/bin/env python3
"""Transparent monochrome flowcharts for next-stage trend forecast → S."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_flow"
FIG.mkdir(parents=True, exist_ok=True)
S_PLOT = HERE.parent / "s_score_from_nasa84" / "plot_s_method_flowcharts.py"

spec = importlib.util.spec_from_file_location("s_flow", S_PLOT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Canvas = mod.Canvas
LINE = mod.LINE
MUTED = mod.MUTED
FONT = mod.FONT


def save(fig, name: str):
    png = FIG / f"{name}.png"
    fig.savefig(
        png, dpi=260, facecolor="none", edgecolor="none",
        transparent=True, bbox_inches="tight", pad_inches=0.18,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("fig", png)
    return png


def fig_method():
    C = Canvas(8.9, 13.0, (0.0, 10.0, ), (0.0, 15.55))
    C.label(5.0, 0.36, "方法流程图  ·  跨被试未来阶段人因与绩效 S", size=11.8, color=LINE, ha="center")

    items = [
        ("s1", 0.78, 1.8, 0.58, "开始", "start", 8.6),
        ("s2", 1.54, 5.5, 0.76, "输入原始数据\n四模态时序 · NASA-TLX 问卷 · 步骤完成表", "data", 8.0),
        ("s3", 2.48, 5.5, 0.76, "时间对齐并切窗\n窗长 30 s，步长 5 s，窗内 66 项指标", "process", 8.0),
        ("s4", 3.42, 5.5, 0.76, "按窗口时间顺序切段\n已观察阶段 / 未来阶段（默认 1/2）", "process", 8.0),
        ("s5", 4.36, 5.5, 0.80, "已观察段汇总四统计量\n得到前段 264 维（未来窗不进入）", "process", 8.0),
        ("s6", 5.34, 5.5, 0.76, "计算步骤分 S_step\n并由问卷得到真值 L 与真值 S", "process", 8.0),
        ("s7", 6.28, 5.5, 0.80, "按被试划分：训练 21 人 / 验证 5 人\n同一人不同时入训考", "process", 8.0),
        ("s8", 7.26, 5.5, 0.86, "训练堆：全任务 264 定额 27 维\n拟合浅树 XGBoost（之后冻结）", "process", 8.0),
        ("s9", 8.30, 5.5, 0.80, "验证新人：取出前段同一 27 列\n标准化 Ridge（α＝10）预报整场 27 列", "process", 7.9),
        ("s10", 9.28, 5.5, 0.76, "冻结 XGBoost 估计负荷 L̂\n拟合目标为 NASA-TLX，不是 S", "process", 8.0),
        ("s11", 10.22, 5.5, 0.80, "公式合成预测绩效\nŜ ＝ 0.70 S_step ＋ 0.30 (1 − L̂ / 10)", "process", 8.0),
        ("s12", 11.20, 5.5, 0.76, "验证组 17 条与真值 S 对照\n输出 R²、MAE", "process", 8.0),
        ("s13", 12.14, 1.8, 0.58, "结束", "start", 8.6),
    ]
    for key, y, ww, hh, text, kind, sz in items:
        if key in ("s1", "s13"):
            x, ww, hh = 5.0 - 1.8 / 2, 1.8, 0.58
        else:
            x = 5.0 - ww / 2
        C.box(key, x, y, ww, hh, text, kind=kind, size=sz)

    for a, b in [
        ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5"),
        ("s5", "s6"), ("s6", "s7"), ("s7", "s8"), ("s8", "s9"),
        ("s9", "s10"), ("s10", "s11"), ("s11", "s12"), ("s12", "s13"),
    ]:
        C.v(a, b)

    C.ax.text(7.90, 7.69, "眼动 6  脑电 5\n心率 4  行为 12", ha="left", va="center",
              fontsize=7.1, color=MUTED, fontfamily=FONT, linespacing=1.25)
    C.ax.text(7.90, 8.70, "未来窗不进 Ridge", ha="left", va="center",
              fontsize=7.1, color=MUTED, fontfamily=FONT)

    C.legend(0.55, 14.95, [
        ("start", "起止"),
        ("data", "数据"),
        ("process", "处理"),
    ])
    return save(C.fig, "fig1_trend_method_flow")


def fig_dataflow():
    C = Canvas(12.8, 10.8, (0.0, 16.4), (0.0, 12.85))
    C.label(8.2, 0.34, "数据流图  ·  跨被试未来阶段人因与绩效 S", size=11.8, color=LINE, ha="center")

    layers = [
        (0.98, "原始数据"),
        (2.18, "窗  级"),
        (3.85, "阶  段"),
        (5.55, "预  报"),
        (7.85, "下  游"),
        (9.75, "合  成"),
        (11.15, "评  价"),
    ]
    for y, t in layers:
        C.label(0.20, y, t, size=7.3, color=MUTED, ha="left")

    srcs = [
        ("eeg", 1.15, "脑电\n.set  256 Hz"),
        ("hr", 2.90, "心率\n逐点时序"),
        ("eye", 4.65, "眼动\n.tsv 注视/瞳孔"),
        ("log", 6.40, "操作日志\n动作序列"),
    ]
    for k, x, t in srcs:
        C.box(k, x, 0.72, 1.55, 0.68, t, kind="data", size=7.1)
    C.box("nasa", 10.35, 0.72, 2.10, 0.68, "NASA-TLX 问卷\n六维加权总分", kind="data", size=7.1)
    C.box("step", 13.15, 0.72, 2.10, 0.68, "步骤完成表\n关键 / 非关键", kind="data", size=7.1)

    C.box("win", 1.15, 1.88, 6.80, 0.60, "切窗并对齐    30 s / 5 s", kind="process", size=7.9)
    C.box("readL", 10.35, 1.88, 2.10, 0.60, "读取加权总分", kind="process", size=7.4)
    C.box("readS", 13.15, 1.88, 2.10, 0.60, "统计完成率", kind="process", size=7.4)

    C.box("wtab", 1.15, 2.72, 6.80, 0.58, "窗级特征表     n_win × 66", kind="data", size=7.9)
    C.box("L", 10.35, 2.72, 2.10, 0.58, "L    (84,)", kind="data", size=7.9)
    C.box("Sstep", 13.15, 2.72, 2.10, 0.58, "S_step    (84,)", kind="data", size=7.6)

    C.box("cut", 1.15, 3.54, 6.80, 0.58, "按时间切段：已观察窗进入，未来窗不进入预报器", kind="process", size=7.5)
    C.box("early", 1.15, 4.36, 6.80, 0.58, "已观察段 264 维    mean / std / median / slope", kind="data", size=7.6)
    C.box("Strue", 10.35, 4.36, 4.90, 0.58, "真值 S ＝ 0.70 S_step ＋ 0.30 (1 − L / 10)", kind="data", size=7.1)

    C.box("pick", 1.15, 5.18, 6.80, 0.58, "取出训练堆定额的 27 列（眼动 6 / 脑电 5 / 心率 4 / 行为 12）", kind="process", size=7.3)
    C.box("e27", 1.15, 6.00, 6.80, 0.58, "已观察 27 维    （验证新人）", kind="data", size=7.8)

    C.box("ridge", 1.15, 6.82, 6.80, 0.58, "标准化 Ridge（α＝10）预报整场同一 27 列", kind="process", size=7.7)
    C.box("f27", 1.15, 7.64, 6.80, 0.58, "预报整场 27 维  27̂", kind="data", size=7.8)

    C.box("xgb", 1.15, 8.46, 6.80, 0.58, "冻结浅树 XGBoost（只估 NASA，不重训）", kind="process", size=7.6)
    C.box("Lhat", 1.15, 9.28, 6.80, 0.58, "预测  L̂    （验证组 17 条）", kind="data", size=7.8)

    C.box("mix", 1.15, 10.10, 6.80, 0.58, "公式合成   Ŝ ＝ 0.70 S_step ＋ 0.30 (1 − L̂ / 10)", kind="process", size=7.5)
    C.box("Shat", 3.20, 11.00, 2.70, 0.58, "预测  Ŝ    (17,)", kind="data", size=7.8)
    C.box("met", 6.20, 11.00, 2.20, 0.58, "R² 、 MAE", kind="data", size=7.8)
    C.box("Sout", 10.35, 11.00, 4.90, 0.58, "真值  S    (17,)", kind="data", size=7.8)

    # 训练堆：定额与冻结 XGB
    C.box("tr264", 10.35, 5.18, 4.90, 0.58, "训练堆全任务 264 维（21 人）", kind="data", size=7.3)
    C.box("trfit", 10.35, 6.00, 4.90, 0.58, "互信息定额＋拟合 XGB 后冻结", kind="process", size=7.3)

    bus_y = 1.58
    for k in ("eeg", "hr", "eye", "log"):
        x1, y1 = C.port(k, "s")
        C.ax.plot([x1, x1], [y1, bus_y], color=LINE, lw=1.12, zorder=2)
    C.ax.plot([C.port("eeg", "s")[0], C.port("log", "s")[0]], [bus_y, bus_y], color=LINE, lw=1.12, zorder=2)
    wx, wy = C.port("win", "n")
    C.arrow(wx, bus_y, wx, wy)

    C.v("win", "wtab")
    C.v("wtab", "cut")
    C.v("cut", "early")
    C.v("early", "pick")
    C.v("pick", "e27")
    C.v("e27", "ridge")
    C.v("ridge", "f27")
    C.v("f27", "xgb")
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

    C.v("Strue", "Sout")
    C.v("tr264", "trfit")
    mid_x = 8.85
    px, py = C.port("pick", "e")
    tx, ty = C.port("trfit", "w")
    C.polyline(
        [(tx, ty), (mid_x, ty), (mid_x, py), (px, py)],
        text="列名", text_at=(mid_x, (ty + py) / 2), text_off=(0.32, 0.0),
    )
    xx, xy = C.port("xgb", "e")
    C.polyline(
        [(tx, ty), (mid_x, ty), (mid_x, xy), (xx, xy)],
        text="冻结模型", text_at=(mid_x, (ty + xy) / 2 + 0.55), text_off=(0.48, 0.0),
    )

    rail = 15.72
    se_x, se_y = C.port("Sstep", "e")
    mx, my = C.port("mix", "e")
    C.polyline(
        [(se_x, se_y), (rail, se_y), (rail, my), (mx, my)],
        text="真实步骤分", text_at=(rail, 7.20), text_off=(-0.62, 0.0),
    )

    C.h("Shat", "met", text="对照", text_off=(0.0, -0.18))
    x1, y1 = C.port("Sout", "w")
    x2, y2 = C.port("met", "e")
    C.polyline([(x1, y1), (x2, y2)], text="对照", text_at=((x1 + x2) / 2, y1), text_off=(0.0, -0.18))

    C.legend(1.15, 12.20, [
        ("data", "数据 / 表"),
        ("process", "处理过程"),
    ])
    C.label(6.4, 12.34, "步骤分始终来自真实操作记录；未来窗不进入 Ridge", size=7.3, color=MUTED, ha="left")
    return save(C.fig, "fig2_trend_data_flow")


if __name__ == "__main__":
    fig_method()
    fig_dataflow()
