#!/usr/bin/env python3
"""趋势预测正式 Word：与《全模态27维绩效S预测实验报告》同一套论文体例。

运行：
    uv run --with pandas --with numpy --with scikit-learn --with matplotlib --with python-docx \
        python plot_trend_docx.py
    uv run --with python-docx python make_docx_trend.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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
FLOW = HERE / "figures_flow"
DESK = Path("/Users/licochen/Desktop")
OUT_LONG = HERE / "趋势预测_跨被试未来阶段人因与S_实验报告.docx"
OUT_SHORT = HERE / "趋势预测_跨被试未来阶段人因与S_结果摘要.docx"


def toc_block(doc, items):
    add_toc_heading(doc)
    for text, lv in items:
        add_toc_line(doc, text, lv)
    add_page_break(doc)


def save_both(doc, project_path: Path) -> Path:
    doc.save(project_path)
    desk = DESK / project_path.name
    shutil.copy2(project_path, desk)
    print("wrote", project_path)
    print("copy ", desk)
    return project_path


def build_short():
    doc = new_doc("趋势预测  跨被试未来阶段人因与绩效 S  结果摘要")
    add_title(doc, "趋势预测：跨被试预报未来阶段人因并合成绩效 S", "结果摘要")

    toc_block(doc, [
        ("1  口径与公式", 1),
        ("2  实验设置", 1),
        ("3  验证结果", 1),
        ("4  小结", 1),
    ])

    add_h(doc, "1  口径与公式", 1)
    add_body(
        doc,
        "本摘要给出趋势预测的验证结果。功能定义为：已有 n 名被试的完整任务阶段数据；对第 n＋1 个未见过的人，依据其本场试验已观察阶段的脑电、心率、眼动与行为，预报本场未来阶段所对应的人因指标，再送入与正式口径相同的 27 维定额浅树 XGBoost 得到负荷，按公式合成绩效 S。模型不直接以 S 为回归目标。",
    )
    add_body(
        doc,
        "记步骤表现为 S_step，主观负荷问卷加权总分为 L，负荷反向分记为 L_rev。子任务格子取值不小于 0.5 记为做成。仅有关键或仅有非关键子任务时，该侧权重改为 1。正式配比为步骤 0.70、负荷反向 0.30。",
    )
    add_equation(doc, eq_s_step())
    add_equation(doc, eq_l_rev())
    add_equation(doc, eq_s())
    add_equation(doc, eq_s_hat())
    add_note(doc, "式中 L̂ 为趋势预测给出的负荷估计，S_step 两侧均取真实操作记录。预测 S 与真值 S 的差别仅来自负荷估计偏差，再乘以 0.30。")

    add_h(doc, "2  实验设置", 1)
    add_caption(doc, "表 1  实验设置")
    make_table(doc, ["项", "内容"], [
        ["样本", "26 名被试、84 条被试–任务"],
        ["验证组", "5 名新被试、17 条任务（被试 2、7、12、16、23）"],
        ["训练组", "21 名被试、67 条任务"],
        ["切窗", "窗长 30 秒，步长 5 秒"],
        ["任务级特征", "66 个窗口指标×4 个统计量＝264 维"],
        ["阶段", "一次任务按窗口时间顺序切成已观察阶段与未来阶段"],
        ["预报器", "标准化 Ridge（α＝10），只预报下游实际使用的 27 列"],
        ["降维", "分模态定额互信息：眼动 6、脑电 5、心率 4、行为 12，合计 27 维"],
        ["下游模型", "XGBoost：max_depth＝2，n_estimators＝500，learning_rate＝0.02，reg_λ＝2.0"],
        ["验证", "按被试 GroupKFold 划分；预报器与 XGBoost 只在训练被试上拟合"],
        ["真值 S（验证组）", "范围 0.42—0.86，均值 0.59"],
    ], [3.6, 11.2], left_cols={0, 1})

    add_h(doc, "3  验证结果", 1)
    add_body(
        doc,
        "验证组为按被试划出的 5 名新被试、17 条任务。合成绩效 S 的决定系数 R² 为 0.948，平均绝对误差 MAE 为 0.025。验证组真值 S 范围为 0.42—0.86，均值 0.59。",
    )
    add_caption(doc, "表 2  验证组主指标（5 人、17 条，目标为合成 S）")
    make_table(doc, ["指标", "绩效 S"], [
        ["R²", "0.948"],
        ["MAE", "0.025"],
    ], [7.4, 7.4], left_cols={0})
    add_figure(doc, FIG / "fig2_scatter.png", "图 1  验证组真值 S 与预测 S（17 条）", 11.5)

    add_h(doc, "4  小结", 1)
    add_body(
        doc,
        "在 26 名被试、84 次任务上，将趋势预测定义为：用已有被试的完整任务阶段数据训练，对未见过的新被试，依据其本场试验已观察阶段的脑电、心率、眼动与行为，预报本场未来阶段所对应的人因指标。264 维按模态定额互信息降至 27 维（眼动 6、脑电 5、心率 4、行为 12），以标准化 Ridge 补全整场 27 维，再采用与正式口径相同的浅树 XGBoost 估计负荷分量，按步骤分 0.70、负荷反向分 0.30 合成绩效 S。在按被试划出的验证组（5 人、17 条任务）上，合成 S 的 R² 为 0.948（MAE＝0.025）。模型不直接以 S 为回归目标；人因趋势先落到 27 维指标，再经公式得到预测版绩效。",
    )
    return save_both(doc, OUT_SHORT)


def build_long():
    doc = new_doc("趋势预测  跨被试未来阶段人因与绩效 S  实验报告")
    add_title(doc, "趋势预测：跨被试预报未来阶段人因并合成绩效 S 实验报告")

    toc_block(doc, [
        ("1  引言", 1),
        ("1.1  研究目标", 2),
        ("1.2  技术路线概要", 2),
        ("2  绩效指标 S 的定义", 1),
        ("2.1  步骤表现分", 2),
        ("2.2  负荷反向分与合成公式", 2),
        ("2.3  预测版 S 的构造", 2),
        ("3  实验数据", 1),
        ("3.1  被试与任务", 2),
        ("3.2  信号与标签来源", 2),
        ("3.3  训练组与验证组", 2),
        ("4  特征工程", 1),
        ("4.1  切窗与窗口指标", 2),
        ("4.2  任务级 264 维", 2),
        ("4.3  已观察阶段与未来阶段", 2),
        ("4.4  分模态定额至 27 维", 2),
        ("5  两段式学习算法", 1),
        ("5.1  人因预报器", 2),
        ("5.2  下游负荷模型", 2),
        ("5.3  评价协议", 2),
        ("6  验证结果", 1),
        ("6.1  主指标", 2),
        ("6.2  验证组明细", 2),
        ("7  本验证组录取的 27 列", 1),
        ("8  结论", 1),
        ("附录 A  66 个窗口指标", 1),
        ("附录 B  本验证组 27 维列名", 1),
    ])

    add_h(doc, "1  引言", 1)
    add_h(doc, "1.1  研究目标", 2)
    add_body(
        doc,
        "核电模拟机人因试验中，操纵员的主观负荷与综合绩效通常要等整次任务测完才能计算。本实验要做的是趋势预测：已有 n 名被试在相同场景下的全部任务阶段数据；来了第 n＋1 个新人后，用他本场已经发生的脑电、心率、眼动与行为，预报本场尚未发生阶段所对应的人因状态，再落到负荷与绩效 S。",
    )
    add_body(
        doc,
        "预测目标仍是自定义综合绩效 S，而不是把问卷负荷本身当作最终报告指标。S 由真实操作步骤与主观负荷反向分按固定权重合成。模型部分只估计负荷分量，步骤分取真实记录。人因预报也不直接以 S 为回归目标：先补全下游模型真正使用的 27 维人因指标，再经冻结的浅树 XGBoost 得到负荷。",
    )

    add_h(doc, "1.2  技术路线概要", 2)
    add_body(
        doc,
        "技术路线为：四模态连续信号按 30 秒窗、5 秒步切分，提取 66 个窗口指标；把一次任务按时间切成已观察阶段与未来阶段，只对已观察窗口汇总为均值、标准差、中位数与斜率，得到已观察段的 264 维；每一折仅在训练被试上按模态定额做互信息筛选，得到 27 维；用标准化 Ridge 把已观察 27 列映射为整场任务的同一 27 列；再用与正式完整观测口径相同的浅树 XGBoost 估计负荷；最后与真实步骤分合成 S。划分方式为按被试的 GroupKFold，同一人不同时出现在训练与验证。",
    )
    add_figure(doc, FLOW / "fig1_trend_method_flow.png", "图 1  跨被试未来阶段人因与绩效 S 方法流程图", 14.2)
    add_body(
        doc,
        "图 1 给出从原始数据到预测绩效 Ŝ 的计算顺序：四模态信号切窗后按时间分成已观察阶段与未来阶段，只把已观察段收成 264 维；训练堆用全任务 264 定额 27 维并拟合浅树 XGBoost 后冻结；验证新人用标准化 Ridge 由已观察 27 列预报整场 27 列，再经冻结模型估计负荷，与真实步骤分按 0.70／0.30 合成 Ŝ。",
    )
    add_figure(doc, FLOW / "fig2_trend_data_flow.png", "图 2  跨被试未来阶段人因与绩效 S 数据流图", 15.2)
    add_body(
        doc,
        "图 2 给出同一路径上的数据对象：左侧由已观察窗得到前段 264 与定额 27 维，经 Ridge 得到整场 27̂ 与负荷 L̂；右侧由问卷与步骤表得到 L、S_step 与真值 S；训练堆全任务表只用于确定列名并冻结 XGBoost。L̂ 与真实 S_step 合成 Ŝ，在验证组 17 条上与真值对照。",
    )

    add_h(doc, "2  绩效指标 S 的定义", 1)
    add_h(doc, "2.1  步骤表现分", 2)
    add_body(
        doc,
        "步骤表现来自任务序列完成统计表。黄表头列定义为关键子任务。格子取值不小于 0.5 记为做成（1），否则记为未做成（0）。关键完成率记为 r_key，非关键完成率记为 r_nkey。若某次任务只有关键列或只有非关键列，则该侧权重改为 1。任务 5 与 5_6 的列全部为关键。",
    )
    add_equation(doc, eq_s_step())
    add_body(doc, "上式给出步骤表现分 S_step。验证组 17 条上，真值 S 的范围为 0.42 至 0.86，均值 0.59。")

    add_h(doc, "2.2  负荷反向分与合成公式", 2)
    add_body(
        doc,
        "主观负荷取任务后问卷的加权总分 L（NASA-TLX 六维加权，原始表直接给出，实验代码不重算权重）。为使负荷与绩效同向（越高越好），定义负荷反向分：",
    )
    add_equation(doc, eq_l_rev())
    add_body(doc, "正式合成采用步骤 0.70、负荷反向 0.30。该配比为项目对外口径，全文只报告这一组权重。")
    add_equation(doc, eq_s())

    add_h(doc, "2.3  预测版 S 的构造", 2)
    add_body(doc, "预测时，S_step 仍用真实操作记录，仅将 L 替换为趋势预测给出的负荷估计 L̂：")
    add_equation(doc, eq_s_hat())
    add_body(doc, "因此，预测 S 与真值 S 的差等于 0.30×(L−L̂)/10。模型的作用体现在负荷分量上。")

    add_h(doc, "3  实验数据", 1)
    add_h(doc, "3.1  被试与任务", 2)
    add_body(
        doc,
        "对齐后的建模样本为 26 名被试、84 条被试–任务，含重复测量。重复任务共用步骤分，问卷负荷各自独立，故 S 可以不同。任务类型包括 1、2、3、4、5、5_6。验证组 17 条的窗口数从 17 到 413，覆盖短任务与长任务。",
    )
    add_caption(doc, "表 1  样本规模")
    make_table(doc, ["项目", "取值"], [
        ["被试人数", "26"],
        ["被试–任务条数", "84"],
        ["训练组", "21 人、67 条"],
        ["验证组", "5 人、17 条"],
        ["任务类型", "1、2、3、4、5、5_6"],
        ["窗口长度 / 步长", "30 s / 5 s"],
        ["验证组窗口数（最小 / 最大）", "17 / 413"],
    ], [7.4, 7.4], left_cols={0, 1})

    add_h(doc, "3.2  信号与标签来源", 2)
    add_caption(doc, "表 2  数据来源")
    make_table(doc, ["内容", "来源"], [
        ["主观负荷问卷", "任务后六维加权总分，不在代码中重算权重"],
        ["脑电", "预处理后 EEGLAB 数据，分析采样率 256 Hz"],
        ["心率", "手环逐点心率"],
        ["眼动", "Tobii 导出的注视、扫视与瞳孔"],
        ["操作日志与步骤", "模拟机操作记录及任务序列完成统计表"],
    ], [4.6, 10.2], left_cols={0, 1})
    add_body(doc, "四路生理与行为信号按眼动时间轴对齐。特征中不纳入任务编号，兴趣区也不使用具体按钮名称，以避免模型凭借任务类型取巧。")

    add_h(doc, "3.3  训练组与验证组", 2)
    add_body(
        doc,
        "划分采用 sklearn 的 GroupKFold，折数 5，不打乱，分组变量为被试。本报告采用其中一组验证划分，考试人与正式完整观测口径同一划分。同一人不会既出现在训练又出现在验证。Ridge 与 XGBoost 都只在训练堆上拟合。验证组的每一条 S 预测，都来自“没见过这个人”时的推理。",
    )
    add_caption(doc, "表 3  本报告采用的训练组与验证组")
    make_table(doc, ["堆", "被试", "条数"], [
        ["训练", "1、3、4、5、6、8、9、10、11、13、14、15、17、18、19、20、21、22、24、25、26", "67"],
        ["验证（新人）", "2、7、12、16、23", "17"],
    ], [2.6, 10.0, 2.2], left_cols={1})
    add_body(doc, "主指标在这 17 对（真值，预测）上一次算完 R² 与 MAE。这是在检验：换一批没见过的人，这套趋势预测是否有效。")

    add_h(doc, "4  特征工程", 1)
    add_h(doc, "4.1  切窗与窗口指标", 2)
    add_body(
        doc,
        "一次任务的连续信号切成窗长 30 秒、步长 5 秒的片段，相邻窗重叠 25 秒。每一段计算 66 个窗口指标。脑电功率按该被试自身的均值与标准差做成 z 分数。脑区划分为额、中央、顶、枕；频段为 δ（1–4 Hz）、θ（4–8 Hz）、α（8–13 Hz）、β（13–30 Hz）、γ（30–45 Hz），另有 θ/α 与 β/α。眨眼以持续 50–500 ms 的“找不到眼睛”片段作为代理。操作日志只用本窗口统计，不用从任务开始累计的列。",
    )
    add_caption(doc, "表 4  66 个窗口指标的模态构成")
    make_table(doc, ["模态", "窗口指标个数", "说明"], [
        ["脑电", "28", "四脑区×（五频段功率＋两个比值）"],
        ["心率", "5", "均值、标准差、最低、最高、窗内斜率"],
        ["眼动（瞳孔/类型）", "6", "瞳孔、有效采样、注视/扫视/丢眼比例"],
        ["眼动（兴趣区）", "9", "区间条数、覆盖、集中度、熵等"],
        ["眨眼", "6", "次数、频率、时长"],
        ["行为（操作日志）", "12", "次数、密度、对错、多余/重复、设备与步骤种数"],
        ["合计", "66", "见附录 A"],
    ], [4.2, 3.2, 7.4], left_cols={0, 2})

    add_h(doc, "4.2  任务级 264 维", 2)
    add_body(
        doc,
        "每个窗口指标在指定时间范围内再取四个统计量：均值（平均水平，nanmean）、标准差（过程起伏，nanstd 且 ddof＝0）、中位数（稳健水平，nanmedian）、斜率（过程走势，窗口序号对指标的一元最小二乘）。66×4＝264。斜率只在至少 2 个有限窗时计算。列名与正式完整观测口径同一套。",
    )
    add_caption(doc, "表 5  任务级 264 维按模态的列数")
    make_table(doc, ["模态", "窗口指标", "任务级列数"], [
        ["脑电", "28", "112"],
        ["心率", "5", "20"],
        ["眼动（含兴趣区与眨眼）", "21", "84"],
        ["行为", "12", "48"],
        ["合计", "66", "264"],
    ], [5.4, 4.7, 4.7])
    add_body(doc, "建模时把瞳孔、兴趣区与眨眼都算作眼动，与正式口径一致。")

    add_h(doc, "4.3  已观察阶段与未来阶段", 2)
    add_body(
        doc,
        "一次任务的窗口按时间顺序排列，切成已观察阶段与未来阶段。切点为窗口数的一半向下取整，并保证两段都至少有窗口。已观察段是前一段窗口，未来段是后一段窗口。只对已观察窗口做 66→264 聚合，得到“新人走到现在”的人因表。未来段窗口不进入预报器的输入。",
    )
    add_body(
        doc,
        "预报器要输出的是整场任务那 27 个人因指标，与下游 XGBoost 的输入同构，相当于把尚未发生阶段的信息补进全任务表征。两段都至少 4 个窗，均值、波动与走势能算。验证组 17 条均满足。对一次 118 窗的任务，已观察段约 59 窗、未来段约 59 窗；一次 413 窗的长任务则各约 206 窗。Ridge 看见的是已观察段收成的 27 个数，不是逐窗原始曲线。",
    )
    add_caption(doc, "表 6  阶段切分规则")
    make_table(doc, ["项", "设置"], [
        ["切点", "cut ＝ floor（窗口数 × 1/2）"],
        ["已观察阶段", "windows[:cut]，单独聚合为 264 维"],
        ["未来阶段", "windows[cut:]，不进入预报器输入"],
        ["预报目标", "整场任务同一套 27 维人因指标"],
        ["最短切段", "两段各不少于 4 个窗口"],
    ], [4.2, 10.6], left_cols={0, 1})

    add_h(doc, "4.4  分模态定额至 27 维", 2)
    add_body(
        doc,
        "不宜直接使用 264 列。采用与正式口径相同的分模态定额互信息：在各模态内部按与负荷连续值的互信息排序，再按规定个数录取，以保证四模态始终同时进入模型。",
    )
    add_caption(doc, "表 7  27 维定额")
    make_table(doc, ["模态", "264 维中的列数", "录取个数"], [
        ["眼动", "84", "6"],
        ["脑电", "112", "5"],
        ["心率", "20", "4"],
        ["行为", "48", "12"],
        ["合计", "264", "27"],
    ], [4.9, 4.9, 5.0])
    add_body(
        doc,
        "互信息用 sklearn.feature_selection.mutual_info_regression，目标是训练堆的真实负荷连续分，随机种子为 0。只在训练的 21 人上计算互信息、做录取；5 名验证被试只用已经定好的 27 个列名。缺失值用训练堆中位数填充。本验证组录取的 27 列与正式完整观测实验同一划分下的名单一致，因为定额都是在同一批训练被试的真实全任务 264 与真实负荷上计算的。完整列名见第 7 节与附录 B。",
    )

    add_h(doc, "5  两段式学习算法", 1)
    add_body(
        doc,
        "趋势预测不另训一个端到端“人因→S”模型。考试时只替换送进下游 XGBoost 的 27 维输入，不重训树、不重选列。",
    )

    add_h(doc, "5.1  人因预报器", 2)
    add_body(
        doc,
        "预报器把已观察阶段的 27 列映射为整场任务的同一 27 列，属于多输出回归：每列一条岭回归，共享同一设计矩阵。",
    )
    add_caption(doc, "表 8  人因预报器设置")
    make_table(doc, ["项", "设置"], [
        ["输入", "已观察阶段 264 维里、本折定额的 27 列"],
        ["输出", "整场任务同一 27 列"],
        ["预处理", "中位数填充＋StandardScaler，均在训练被试上拟合"],
        ["模型", "Ridge，L2 正则系数 10"],
        ["训练对", "训练堆每条任务：已观察 27 列 → 该条真实全任务 27 列"],
    ], [3.4, 11.4], left_cols={0, 1})
    add_body(doc, "只预报下游 XGBoost 真正用到的 27 列。标准化后的线性多输出在小样本下稳定、可复现。")

    add_h(doc, "5.2  下游负荷模型", 2)
    add_body(
        doc,
        "回归器与正式完整观测口径完全相同，只在训练被试的真实全任务 27 维到真实负荷上拟合。验证时把 Ridge 预报出的 27 维送进去。",
    )
    add_caption(doc, "表 9  下游浅树 XGBoost 超参数")
    make_table(doc, ["超参数", "值", "作用"], [
        ["max_depth", "2", "树很浅，限制过拟合"],
        ["n_estimators", "500", "树的棵数"],
        ["learning_rate", "0.02", "学得慢、稳"],
        ["reg_lambda", "2.0", "L2 正则"],
        ["subsample", "0.8", "每棵树用 80% 样本"],
        ["colsample_bytree", "0.8", "每棵树用 80% 特征"],
        ["tree_method", "hist", "直方图加速"],
        ["random_state", "0", "可复现"],
    ], [4.4, 3.2, 7.2], left_cols={0, 2})

    add_h(doc, "5.3  评价协议", 2)
    add_body(
        doc,
        "主指标为验证组 17 条上合成绩效 S 的决定系数 R² 与平均绝对误差 MAE。合成 S 按第 2 节公式计算，步骤分取真实记录。",
    )

    add_h(doc, "6  验证结果", 1)
    add_h(doc, "6.1  主指标", 2)
    add_body(
        doc,
        "在 5 名新被试、17 条任务上，趋势预报 27 维后再经冻结 XGBoost 估计负荷分量，按正式公式合成绩效 S。合成 S 的 R² 为 0.948，MAE 为 0.025。验证组真值 S 范围为 0.42—0.86，均值 0.59。图 3 给出 17 条真值与预测的散点，点沿对角线聚集。",
    )
    add_caption(doc, "表 10  验证组主指标（5 人、17 条，目标为合成 S）")
    make_table(doc, ["指标", "绩效 S"], [
        ["R²", "0.948"],
        ["MAE", "0.025"],
    ], [7.4, 7.4], left_cols={0})
    add_body(
        doc,
        "S 平均每条相差约 0.025（S 在 0—1 附近）。说明在没见过的人上，这套趋势预测可以给出有效的预测版绩效。",
    )
    add_figure(doc, FIG / "fig2_scatter.png", "图 3  验证组真值 S 与预测 S（17 条）", 11.5)

    add_h(doc, "6.2  验证组明细", 2)
    add_body(doc, "验证组覆盖低、中、高三类预设难度，以及任务 1、2、3、5、5_6。")
    add_caption(doc, "表 11  验证组 17 条明细（合成 S）")
    make_table(
        doc,
        ["被试", "任务", "难度", "窗口", "真 S", "预测 S"],
        [
            ["2", "1", "中", "278", "0.568", "0.581"],
            ["2", "3", "低", "65", "0.628", "0.630"],
            ["2", "5_6", "高", "131", "0.662", "0.696"],
            ["7", "2", "中", "198", "0.819", "0.796"],
            ["7", "3", "低", "79", "0.528", "0.523"],
            ["7", "5_6", "高", "118", "0.514", "0.535"],
            ["12", "2", "中", "413", "0.805", "0.784"],
            ["12", "3", "低", "60", "0.522", "0.580"],
            ["12", "5_6", "高", "132", "0.490", "0.535"],
            ["16", "3", "低", "67", "0.860", "0.881"],
            ["16", "5", "低", "17", "0.425", "0.437"],
            ["16", "5_6", "高", "50", "0.418", "0.434"],
            ["16", "5_6 重复", "高", "136", "0.418", "0.411"],
            ["23", "3", "低", "69", "0.666", "0.621"],
            ["23", "5", "低", "27", "0.589", "0.624"],
            ["23", "5_6", "高", "102", "0.594", "0.608"],
            ["23", "5_6 重复", "高", "79", "0.604", "0.657"],
        ],
        [2.0, 2.4, 2.0, 2.2, 3.1, 3.1],
    )

    add_h(doc, "7  本验证组录取的 27 列", 1)
    add_body(
        doc,
        "下列为训练堆上互信息定额结果。5 名验证被试只用这些列做 Ridge 预报与 XGBoost 推理。眼动 6 席全部是兴趣区（覆盖、切换、集中度及其波动与走势）；脑电以额、顶、中央的 α、θ、γ 及 θ/α 为主；心率以均值和波动的走势为主；行为以操作次数、密度、正确与多余操作、设备与步骤种数为主。这与正式完整观测实验里稳定入选的核心列一致。",
    )
    add_caption(doc, "表 12  本验证组 27 维列名与含义")
    make_table(doc, ["模态", "列名", "含义"], [
        ["眼动", "eye_aoi_unique_hit_n__std", "点到几个不同兴趣区 · 波动"],
        ["眼动", "eye_aoi_interval_n__mean", "兴趣区区间条数 · 平均"],
        ["眼动", "eye_aoi_coverage_ratio__median", "兴趣区注视覆盖比例 · 中位数"],
        ["眼动", "eye_aoi_interval_n__std", "兴趣区区间条数 · 波动"],
        ["眼动", "eye_aoi_max_share__median", "最主要兴趣区占比 · 中位数"],
        ["眼动", "eye_aoi_coverage_ratio__slope", "兴趣区注视覆盖比例 · 走势"],
        ["脑电", "eeg_frontal_alpha_power_z_within_subject__median", "额区 α 功率 · 中位数"],
        ["脑电", "eeg_parietal_theta_alpha_z_within_subject__std", "顶区 θ/α · 波动"],
        ["脑电", "eeg_frontal_gamma_power_z_within_subject__std", "额区 γ 功率 · 波动"],
        ["脑电", "eeg_central_alpha_power_z_within_subject__mean", "中央区 α 功率 · 平均"],
        ["脑电", "eeg_parietal_theta_power_z_within_subject__median", "顶区 θ 功率 · 中位数"],
        ["心率", "hr_max__slope", "最高心率 · 走势"],
        ["心率", "hr_std__slope", "心率波动 · 走势"],
        ["心率", "hr_std__std", "心率波动 · 波动"],
        ["心率", "hr_mean__slope", "心率均值 · 走势"],
        ["行为", "log_correct_action_count_win__std", "正确操作次数 · 波动"],
        ["行为", "log_unique_step_count_win__mean", "步骤种数 · 平均"],
        ["行为", "log_extra_action_count_win__slope", "多余操作次数 · 走势"],
        ["行为", "log_action_density_win__median", "操作密度 · 中位数"],
        ["行为", "log_action_count_win__mean", "操作次数 · 平均"],
        ["行为", "log_action_density_win__mean", "操作密度 · 平均"],
        ["行为", "log_action_count_win__median", "操作次数 · 中位数"],
        ["行为", "log_extra_rate_win__slope", "多余操作比例 · 走势"],
        ["行为", "log_correct_action_count_win__mean", "正确操作次数 · 平均"],
        ["行为", "log_unique_device_count_win__mean", "设备种数 · 平均"],
        ["行为", "log_extra_rate_win__mean", "多余操作比例 · 平均"],
        ["行为", "log_unique_device_count_win__slope", "设备种数 · 走势"],
    ], [1.8, 7.0, 6.0], left_cols={1, 2}, size=9)

    add_h(doc, "8  结论", 1)
    add_body(
        doc,
        "在 26 名被试、84 次任务上，采用已观察阶段聚合、四模态 27 维定额、标准化 Ridge 预报整场 27 维、冻结浅树 XGBoost 估计负荷分量，再按步骤分 0.70、负荷反向分 0.30 合成绩效 S。验证组为 5 名新被试、17 条任务。正式结果为：合成 S 的 R²＝0.948，MAE＝0.025。该两段式趋势预测可以在未见过的人上给出有效的预测版绩效。",
    )

    add_h(doc, "附录 A  66 个窗口指标", 1)
    add_body(doc, "下列为切窗后、尚未做阶段内四统计量之前的 66 个窗口指标。进入模型的是各指标的 mean、std、median、slope 四列。")
    add_h(doc, "A.1  脑电（28）", 2)
    add_caption(doc, "表 A1  脑电窗口指标")
    eeg_rows = []
    for region, rname in [("frontal", "额区"), ("central", "中央区"), ("parietal", "顶区"), ("occipital", "枕区")]:
        for band, bname in [
            ("delta", "δ 功率"),
            ("theta", "θ 功率"),
            ("alpha", "α 功率"),
            ("beta", "β 功率"),
            ("gamma", "γ 功率"),
        ]:
            eeg_rows.append([rname, f"eeg_{region}_{band}_power_z_within_subject", bname + "（被试内 z）"])
        eeg_rows.append([rname, f"eeg_{region}_theta_alpha_z_within_subject", "θ/α（被试内 z）"])
        eeg_rows.append([rname, f"eeg_{region}_beta_alpha_z_within_subject", "β/α（被试内 z）"])
    make_table(doc, ["脑区", "列名", "含义"], eeg_rows, [2.4, 7.2, 5.2], left_cols={1, 2}, size=9)

    add_h(doc, "A.2  心率、眼动、眨眼与行为", 2)
    add_caption(doc, "表 A2  心率窗口指标")
    make_table(doc, ["列名", "含义"], [
        ["hr_mean", "窗内心率均值"],
        ["hr_std", "窗内心率标准差"],
        ["hr_min", "窗内最低心率"],
        ["hr_max", "窗内最高心率"],
        ["hr_slope_bpm_per_min", "窗内心率斜率"],
    ], [7.4, 7.4], left_cols={0, 1})
    add_caption(doc, "表 A3  眼动与眨眼窗口指标")
    make_table(doc, ["列名", "含义"], [
        ["eye_pupil_filtered_mean / std", "滤波瞳孔直径的均值与波动"],
        ["eye_valid_ratio", "双眼有效采样比例"],
        ["eye_fixation_ratio / eye_saccade_ratio", "注视、扫视时间比例"],
        ["eye_eyes_not_found_ratio", "找不到眼睛的时间比例"],
        ["eye_aoi_interval_n", "兴趣区注视区间条数"],
        ["eye_aoi_unique_hit_n", "命中的不同兴趣区个数"],
        ["eye_aoi_total_fix_ms / eye_aoi_fixation_n", "兴趣区注视总时长与次数"],
        ["eye_aoi_fixation_density_per_sec", "每秒兴趣区注视密度"],
        ["eye_aoi_coverage_ratio / eye_aoi_max_share", "覆盖比例与最大份额"],
        ["eye_aoi_entropy", "兴趣区注视份额熵"],
        ["eye_aoi_pupil_weighted_mean", "兴趣区加权瞳孔直径"],
        ["blink_count / blink_rate_per_min", "疑似眨眼次数与每分钟频率"],
        ["blink_duration_mean/std/median_ms", "疑似眨眼时长"],
        ["blink_total_duration_ratio", "疑似眨眼总时长占比"],
    ], [7.4, 7.4], left_cols={0, 1})
    add_caption(doc, "表 A4  行为（本窗口操作日志）")
    make_table(doc, ["列名", "含义"], [
        ["log_action_count_win / log_action_density_win", "操作次数与密度"],
        ["log_unique_device_count_win / log_unique_step_count_win", "设备种数与步骤种数"],
        ["log_correct_action_count_win", "正确操作次数"],
        ["log_error/duplicate/extra_action_count_win", "错误、重复、多余次数"],
        ["log_disallowed_action_count_win", "三类不合规合计"],
        ["log_error/duplicate/extra_rate_win", "相应比例"],
    ], [8.2, 6.6], left_cols={0, 1})

    add_h(doc, "附录 B  本验证组 27 维列名", 1)
    add_body(doc, "下列为训练被试上互信息定额的结果。验证组只用这些列做推理。")
    add_caption(doc, "表 B1  验证组入选列（被试 2、7、12、16、23）")
    make_table(doc, ["模态", "列名"], [
        ["眼动", "eye_aoi_unique_hit_n__std"],
        ["眼动", "eye_aoi_interval_n__mean"],
        ["眼动", "eye_aoi_coverage_ratio__median"],
        ["眼动", "eye_aoi_interval_n__std"],
        ["眼动", "eye_aoi_max_share__median"],
        ["眼动", "eye_aoi_coverage_ratio__slope"],
        ["脑电", "eeg_frontal_alpha_power_z_within_subject__median"],
        ["脑电", "eeg_parietal_theta_alpha_z_within_subject__std"],
        ["脑电", "eeg_frontal_gamma_power_z_within_subject__std"],
        ["脑电", "eeg_central_alpha_power_z_within_subject__mean"],
        ["脑电", "eeg_parietal_theta_power_z_within_subject__median"],
        ["心率", "hr_max__slope"],
        ["心率", "hr_std__slope"],
        ["心率", "hr_std__std"],
        ["心率", "hr_mean__slope"],
        ["行为", "log_correct_action_count_win__std"],
        ["行为", "log_unique_step_count_win__mean"],
        ["行为", "log_extra_action_count_win__slope"],
        ["行为", "log_action_density_win__median"],
        ["行为", "log_action_count_win__mean"],
        ["行为", "log_action_density_win__mean"],
        ["行为", "log_action_count_win__median"],
        ["行为", "log_extra_rate_win__slope"],
        ["行为", "log_correct_action_count_win__mean"],
        ["行为", "log_unique_device_count_win__mean"],
        ["行为", "log_extra_rate_win__mean"],
        ["行为", "log_unique_device_count_win__slope"],
    ], [3.2, 11.6], left_cols={1}, size=9)

    return save_both(doc, OUT_LONG)


if __name__ == "__main__":
    build_short()
    build_long()
