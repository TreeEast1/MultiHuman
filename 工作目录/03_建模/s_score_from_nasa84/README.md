# 当前 NASA 84 样本上的综合绩效 S

在最新 NASA 实验（84 条被试–任务，与 `regression_task_level` / `classification_task_level_nasa` 同一套标签）上，按历史最终公式回算自定义绩效 **S**。

S 是描述性合成指标，**不是**新的官方量表，也没有改 NASA 建模主线。

## 公式

```
nasa_reverse   = 1 - y_nasa / 10
weighted_step  = 0.75 × key_completion + 0.25 × nonkey_completion
S              = 0.40 × weighted_step + 0.60 × nasa_reverse
```

- NASA：`工作目录/03_建模/regression_task_level/dataset/task_level_table.csv` 的 `y_nasa`
- 步骤：`data/06_任务表现与操作日志/任务序列完成统计.xlsx`（黄表头 = 关键子任务）

## 复现

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with pandas --with openpyxl --with numpy python compute_s.py
```

## 结果摘要

| 项 | 值 |
|---|---|
| 覆盖 | 84 / 84 |
| S 范围 | 0.217 – 0.920 |
| S 均值 | 0.567 |
| 与 NASA Spearman ρ | −0.695 |
| 按预设难度 S 均值 | 低 0.668 > 中 0.601 > 高 0.432 |

明细见 `output/report.md` 与 `output/s_score_84samples.csv`。
