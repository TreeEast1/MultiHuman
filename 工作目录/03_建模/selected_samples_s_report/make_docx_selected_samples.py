#!/usr/bin/env python3
"""指定 15 条样本：评估 S、趋势预警 S 与被试 2 任务 5_6 预测曲线。体例同趋势预测报告。

先跑 export_selected_samples.py，再：
    uv run --with python-docx python make_docx_selected_samples.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forecast_next_stage"))

from report_fmt import (  # noqa: E402
    add_body,
    add_caption,
    add_equation,
    add_figure,
    add_h,
    add_note,
    add_page_break,
    add_title,
    add_toc_heading,
    add_toc_line,
    eq_l_rev,
    eq_s,
    eq_s_hat,
    eq_s_step,
    make_table,
    new_doc,
)

FIG = HERE / "figures"
REP = HERE / "reports"
CASE = HERE.parent / "forecast_next_stage" / "case_subject_02_task_5_6" / "figures"
UI = HERE.parent / "forecast_next_stage" / "figures" / "fig_ui_trend_warning.png"
DESK = Path("/Users/licochen/Desktop")
OUT = HERE / "指定样本_评估S与趋势预警.docx"


def f3(x) -> str:
    return f"{float(x):.3f}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def build() -> Path:
    data = json.loads((REP / "selected_samples.json").read_text(encoding="utf-8"))
    stats = data["stats"]
    rows = data["rows"]
    n_warn = stats["n_warn"]
    n_ok = stats["n_ok"]

    doc = new_doc("指定样本  评估 S 与趋势预警")
    add_title(doc, "指定样本的绩效评估 S 与趋势预警", "15 条被试–任务　·　正式口径 α＝0.70")

    add_toc_heading(doc)
    for text, lv in [
        ("1  口径与公式", 1),
        ("2  本批样本", 1),
        ("3  评估 S（完整观测）", 1),
        ("4  趋势预警 S", 1),
        ("5  趋势预警曲线", 1),
        ("6  被试 2 · 任务 5_6 预测曲线", 1),
        ("7  小结", 1),
    ]:
        add_toc_line(doc, text, lv)
    add_page_break(doc)

    add_h(doc, "1  口径与公式", 1)
    add_body(
        doc,
        "本文件只抽出给定的 15 条被试–任务，报告两套正式口径下的绩效 S，并给出趋势预警主图。"
        "表中样本编号来自清单；清单里 subject_05_task_1 出现两次，去重后共 15 条。"
        "评估 S 是整场任务做完后的完整观测路径：264 维按模态定额互信息降至 27 维，浅树 XGBoost 预测 NASA-TLX，再与真实步骤分合成。"
        "趋势预警 S 是任务做到一半时的路径：只用已观察阶段的 27 维，经标准化 Ridge 补成整场 27 维，再送入同一套冻结 XGBoost。"
        "两套路径都不直接以 S 为回归目标。S 一场任务只有一个数，不能画成随时间起伏的曲线；图上的曲线是窗级人因（默认心率均值）。",
    )
    add_body(
        doc,
        "记步骤表现为 S_step，主观负荷问卷加权总分为 L，负荷反向分记为 L_rev。子任务格子取值不小于 0.5 记为做成。"
        "仅有关键或仅有非关键子任务时，该侧权重改为 1。正式配比为步骤 0.70、负荷反向 0.30。",
    )
    add_equation(doc, eq_s_step())
    add_equation(doc, eq_l_rev())
    add_equation(doc, eq_s())
    add_equation(doc, eq_s_hat())
    add_note(doc, "式中 L̂ 为模型给出的负荷估计，S_step 两侧均取真实操作记录。预警阈值取低分位 0.51：预测 S ≥ 0.51 为正常，否则预警。")

    add_h(doc, "2  本批样本", 1)
    add_body(
        doc,
        "15 条覆盖任务 1、2、3、4、5、5_6，难度含低、中、高。窗口数从 67 到 416。"
        f"真值 S 范围为 {stats['S_true_min']:.2f}—{stats['S_true_max']:.2f}，均值 {stats['S_true_mean']:.2f}。"
        "数字均来自仓库内已复现的五折折外预测，不是另训一套模型。",
    )
    add_caption(doc, "表 1  本批 15 条样本")
    make_table(
        doc,
        ["样本", "被试", "任务", "难度", "窗口数", "步骤分", "真值 NASA", "真值 S"],
        [
            [
                r["sample_id"],
                str(int(r["subject"])),
                str(r["task"]),
                str(r["difficulty"]),
                str(int(r["n_windows"])),
                f3(r["step"]),
                f2(r["nasa_true"]),
                f3(r["S_true"]),
            ]
            for r in rows
        ],
        [4.2, 1.3, 1.3, 1.2, 1.6, 1.5, 1.8, 1.5],
        left_cols={0},
        size=9,
    )

    add_h(doc, "3  评估 S（完整观测）", 1)
    add_body(
        doc,
        "评估路径用整场 27 维人因预测 NASA，再按公式合成 S。"
        f"在这 15 条上，评估预测 S 的 R² 为 {stats['eval_r2']:.3f}，MAE 为 {stats['eval_mae']:.3f}。"
        "全样本 84 条的正式成绩仍是 S 的 pooled R²＝0.979、MAE＝0.025；本表只是清单子集。",
    )
    add_caption(doc, "表 2  评估 S（完整观测 27 维公式法，五折折外）")
    make_table(
        doc,
        ["样本", "真值 NASA", "预测 NASA", "真值 S", "评估预测 S", "差值"],
        [
            [
                r["sample_id"],
                f2(r["nasa_true"]),
                f2(r["nasa_eval"]),
                f3(r["S_true"]),
                f3(r["S_eval"]),
                f"{r['dS_eval']:+.3f}",
            ]
            for r in rows
        ],
        [4.4, 2.0, 2.0, 1.8, 2.2, 1.6],
        left_cols={0},
        size=9,
    )
    add_figure(doc, FIG / "fig1_eval_scatter.png", "图 1  本批 15 条的真值 S 与评估预测 S", 11.2)
    add_figure(doc, FIG / "fig4_s_alpha.png", "图 2  本批 15 条的 S 随 α 变化（圆点为正式口径 α＝0.70）", 12.0)
    add_body(
        doc,
        "图 2 是各条样本自己的 S–α 直线。α 越大，客观步骤分的权重越大。"
        "正式口径取 α＝0.70，圆点即表 2 中的真值 S。负荷很高而步骤分很低的任务（如被试 5 任务 4、被试 26 任务 5_6）整条线都压在下方。",
    )

    add_h(doc, "4  趋势预警 S", 1)
    add_body(
        doc,
        "趋势预警只用本场已观察的前 50% 窗口。标准化 Ridge（α＝10）把已观察 27 列映射为整场 27 列，再经冻结浅树 XGBoost 得负荷，按同一公式合成预测 S。"
        f"在这 15 条上，趋势预测 S 的 R² 为 {stats['trend_r2']:.3f}，MAE 为 {stats['trend_mae']:.3f}。"
        f"按阈值 0.51 判定：{n_ok} 条正常，{n_warn} 条预警。"
        "验证组 17 条的正式成绩仍是 S 的 R²＝0.948、MAE＝0.025。",
    )
    add_caption(doc, "表 3  趋势预警 S（半场 Ridge → 冻结 XGB）")
    make_table(
        doc,
        ["样本", "真值 S", "趋势预测 S", "差值", "判定"],
        [
            [
                r["sample_id"],
                f3(r["S_true"]),
                f3(r["S_trend"]),
                f"{r['dS_trend']:+.3f}",
                r["status"],
            ]
            for r in rows
        ],
        [4.8, 2.2, 2.6, 2.0, 1.8],
        left_cols={0},
        size=9,
    )
    add_figure(doc, FIG / "fig2_trend_scatter.png", "图 3  本批 15 条的真值 S 与趋势预测 S（红边为预警）", 11.2)
    add_figure(doc, FIG / "fig5_warning.png", "图 4  趋势预警判定：预测 S 相对阈值 0.51", 12.2)
    add_figure(doc, FIG / "fig3_s_compare.png", "图 5  真值 S、评估预测 S 与趋势预测 S 对照", 14.6)
    add_body(
        doc,
        "预警的 6 条是：被试 20 任务 4、被试 15 任务 5、被试 5 任务 4、被试 26 任务 5_6、被试 15 任务 5_6、被试 4 任务 5_6。"
        "这 6 条的真值 S 本身都低于或接近 0.51，趋势路径没有把低绩效判成正常。"
        "被试 12 任务 5_6 的趋势预测 S＝0.535，刚过阈值，记为正常。",
    )

    add_h(doc, "5  趋势预警曲线", 1)
    add_body(
        doc,
        "软件主图画的是窗级人因，不是 S。下图对本批 15 条都给出心率均值："
        "左段黑实线为已观察，竖虚线为“现在”，右段浅底为预测段；橙虚线为从末观测点出发的瞬时走势，蓝点线为已观察段斜率外推的整体走势。"
        "右侧标题写出该条的趋势预测 S 与正常／预警。S 仍由 Ridge 27 维路径给出，不由这条心率虚线积分。",
    )
    add_figure(doc, FIG / "fig6_trend_hr_grid.png", "图 6  本批 15 条趋势预警主图（心率均值）", 15.2)

    add_h(doc, "6  被试 2 · 任务 5_6 预测曲线", 1)
    add_body(
        doc,
        "被试 2 任务 5_6 是趋势预警的固定示范案例（官方验证折）。已观察约 5.3 分钟，预测段约 5.5 分钟。"
        "五折折外表中该条趋势预测 S＝0.696；发给软件公司的静态包按官方验证折单独重训，预测 S＝0.679。两者都未见过被试 2，判定均为正常。"
        "界面右侧只写预测 S，不写真值 S。真值 S＝0.662，仅作对照。",
    )
    add_caption(doc, "表 4  被试 2 · 任务 5_6")
    make_table(
        doc,
        ["项", "取值"],
        [
            ["样本编号", "subject_02_task_5_6"],
            ["已观察 / 预测", "前 50% 窗口 / 后 50% 窗口"],
            ["真值 S", "0.662"],
            ["评估预测 S", "0.695"],
            ["趋势预测 S（五折折外）", "0.696"],
            ["静态示范预测 S（官方折重训）", "0.679"],
            ["预警阈值", "0.51"],
            ["人员状态", "正常"],
        ],
        [6.4, 8.4],
        left_cols={0, 1},
    )
    if UI.exists():
        add_figure(doc, UI, "图 7  趋势预测与预警界面（被试 2 · 任务 5_6，主图为心率均值）", 15.4)
    add_figure(doc, CASE / "01_hr_mean.png", "图 8  被试 2 · 任务 5_6 心率均值：已观察、Transformer 瞬时细节与 Ridge 整体走势", 14.8)
    add_figure(doc, CASE / "00_人员状态_S.png", "图 9  人员状态卡片：正常，预测 S＝0.679", 12.0)
    add_figure(doc, CASE / "03_eye_pupil_filtered_mean.png", "图 10  被试 2 · 任务 5_6 瞳孔直径预测曲线", 14.8)
    add_figure(doc, CASE / "04_eye_aoi_coverage_ratio.png", "图 11  被试 2 · 任务 5_6 兴趣区覆盖比例预测曲线", 14.8)
    add_figure(doc, CASE / "05_log_action_density_win.png", "图 12  被试 2 · 任务 5_6 操作密度预测曲线", 14.8)
    add_body(
        doc,
        "图 8 至图 12 的画法已定：黑实线是已观察的窗级人因；橙虚线是池化 Transformer 对后半段瞬时细节的预报，并按最后观测点锚定；"
        "蓝点线是 Ridge 折出的整体走势。切换指标只换主图，右侧预测 S 整场只有一套。",
    )

    add_h(doc, "7  小结", 1)
    add_body(
        doc,
        f"在给定的 15 条被试–任务上，完整观测的评估 S 与真值接近（R²＝{stats['eval_r2']:.3f}，MAE＝{stats['eval_mae']:.3f}）；"
        f"半场观察的趋势预警 S 同样可用（R²＝{stats['trend_r2']:.3f}，MAE＝{stats['trend_mae']:.3f}）。"
        f"按阈值 0.51，{n_ok} 条正常、{n_warn} 条预警，预警条的真值 S 本身偏低。"
        "被试 2 任务 5_6 的示范曲线给出心率、瞳孔、兴趣区与操作密度的已观察段与预测段，人员状态为正常。"
        "S 不随时间画成曲线；预警看的是这一场的预测 S，主图看的是窗级人因怎么走。",
    )

    doc.save(OUT)
    desk = DESK / OUT.name
    shutil.copy2(OUT, desk)
    print("wrote", OUT)
    print("copy ", desk)
    return OUT


if __name__ == "__main__":
    build()
