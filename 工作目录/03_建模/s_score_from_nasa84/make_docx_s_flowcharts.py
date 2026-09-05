#!/usr/bin/env python3
"""Word 呈现：绩效 S 方法流程图 + 数据流图。"""
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")
from report_fmt import (
    add_body,
    add_caption,
    add_equation,
    add_figure,
    add_h,
    add_note,
    add_title,
    eq_l_rev,
    eq_s,
    eq_s_hat,
    eq_s_step,
    make_table,
    new_doc,
)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_flow"
DESK = Path("/Users/licochen/Desktop")


def build():
    doc = new_doc("全模态 27 维绩效 S  方法流程与数据流")
    add_title(doc, "全模态绩效 S 预测：方法流程图与数据流图")

    add_h(doc, "1  图的用途", 1)
    add_body(doc, "本文件只画绩效 S 这一条路径，不涉及 NASA 三分位分类。目标是把“从原始数据到预测版 S”说清楚。图 1 是方法流程图，画算法控制流：先做什么、后做什么、五折在何处循环。图 2 是数据流图，画对象及其维数：哪一张表进哪一步、步骤分从哪里来、模型究竟改了公式里的哪一项。")
    add_body(doc, "两图均为透明底、单色线稿。圆角矩形表示起止，平行四边形表示数据，直角矩形表示处理，菱形表示判断。箭头为正交走向。正式输入为四模态 27 维（眼动 6、脑电 5、心率 4、行为 12）。模型只回归 NASA-TLX 加权总分，不直接回归 S。")

    add_h(doc, "2  方法流程图", 1)
    add_body(doc, "图 1 从“开始”读入三类原始材料：四模态时序、任务后 NASA-TLX 问卷、步骤完成表。连续信号按 30 秒窗、5 秒步切分，窗内得到 66 项指标，任务内再取均值、标准差、中位数与斜率，收成 84×264 的任务级表。步骤分由操作记录直接算出，真值 S 按官方权重合成。学习阶段按被试做 GroupKFold 五折：训练折内中位数填充、分模态互信息定额到 27 维，再用浅树 XGBoost 拟合 NASA；考试折只沿用已定的填充器与列名。五折未结束则回到训练折；全部结束后拼合 84 条折外负荷预测，代入公式得到预测绩效 Ŝ，再与真值 S 比较。")
    add_figure(doc, FIG / "fig1_s_method_flow.png", "图 1  全模态绩效 S 预测方法流程图", 14.6)
    add_body(doc, "读图时抓住三处。第一，菱形是唯一判断：五折是否考完；“否”从左侧回到定额与训练，保证筛选不看见考试被试。第二，浅树框写明拟合目标是 NASA-TLX，不是 S。第三，公式合成发生在循环之外，步骤分不进入模型。")

    add_h(doc, "3  正式公式", 1)
    add_body(doc, "步骤分、NASA 反向分与绩效 S 的定义如下。预测版只把公式中的 L 换成折外预测 L̂，S_step 两侧都用真实操作记录。")
    add_equation(doc, eq_s_step())
    add_equation(doc, eq_l_rev())
    add_equation(doc, eq_s())
    add_equation(doc, eq_s_hat())
    add_note(doc, "注：仅关键或仅非关键时，该侧权重归一为 1。正式合成权重固定为 0.70 / 0.30。")

    add_h(doc, "4  数据流图", 1)
    add_body(doc, "图 2 按层排列数据对象。左侧是生理与行为支路：四路原始记录经切窗成为 n_win×66，再汇总为 84×264，折内定额后成为 84×27，浅树输出折外 L̂。右侧是标签支路：问卷得到 L，步骤表得到 S_step，二者合成真值 S。两支在公式合成处汇合：L̂ 与真实 S_step 生成 Ŝ，再与真值 S 对照得到 R² 与 MAE。")
    add_figure(doc, FIG / "fig2_s_data_flow.png", "图 2  全模态绩效 S 预测数据流图", 15.4)
    add_body(doc, "数据流图不画循环，只画对象如何变窄、在何处汇合。264 到 27 的收缩发生在折内，列名随训练被试变化。步骤分始终走右侧实线，不经过 XGBoost。因此预测 S 与真值 S 的差别只来自负荷估计误差，再乘以 0.30。")

    add_h(doc, "5  符号与口径", 1)
    add_caption(doc, "表 1  框图符号")
    make_table(doc, ["符号", "含义", "本路径中的例子"], [
        ["圆角矩形", "起止", "开始、结束"],
        ["平行四边形", "数据或表", "84×264 特征表、L̂、S_step、Ŝ"],
        ["直角矩形", "处理", "切窗、定额互信息、浅树回归、公式合成"],
        ["菱形", "判断", "五折是否全部结束"],
    ], [3.2, 3.6, 8.0], left_cols={1, 2})
    add_caption(doc, "表 2  主数据对象")
    make_table(doc, ["对象", "形状", "说明"], [
        ["窗级特征", "n_win × 66", "一次任务约数十至数百窗，66 为窗内指标"],
        ["任务级特征", "84 × 264", "66 项 × 4 个统计量；26 人、84 条被试–任务"],
        ["定额后特征", "84 × 27", "眼动 6、脑电 5、心率 4、行为 12；折内录取"],
        ["L / L̂", "(84,)", "真值负荷与折外预测负荷"],
        ["S_step", "(84,)", "真实步骤分，不经模型"],
        ["S / Ŝ", "(84,)", "真值绩效与公式法预测绩效"],
    ], [3.2, 3.4, 8.2], left_cols={0, 2})
    add_body(doc, "图 1 回答“算法按什么顺序做”，图 2 回答“数据变成了什么”。两图合在一起，就是以 S 为目标的全模态技巧评估路径：多模态信号负责估计负荷，步骤记录负责绩效中的完成率，二者按固定公式合成，而不是另训一个直接猜 S 的模型。")

    out = DESK / "全模态27维_绩效S_方法流程与数据流.docx"
    doc.save(out)
    print("wrote", out, out.stat().st_size)
    return out


if __name__ == "__main__":
    build()
