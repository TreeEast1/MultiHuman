# 案例：被试 2 · 任务 5_6 的趋势预测图件

本文件夹是给软件「趋势预测与预警」用的**固定示范案例**。图的画法已定：左边已观察，右边预测；预测里同时画 **Transformer 瞬时人因** 和 **Ridge 整体走势**。

## 1. 这个人、这场任务

- 样本编号：`subject_02_task_5_6`
- 验证组：被试 2、7、12、16、23 中的一条（被试 2，任务 5_6）
- 已观察：前 50% 窗口（约 5.3 分钟）
- 预测段：后 50% 窗口（约 5.5 分钟）
- 窗口：30 秒窗长、5 秒一步，共 131 窗

## 2. 右侧人员状态怎么写

界面右侧**只写预测 S**，不要把 Transformer 虚线积成 S。

| 项 | 写法 |
|---|---|
| 人员状态 | **正常** |
| 预测绩效 S | **0.679** |
| 真值 S（报告对照，可不进界面） | 0.662 |
| 预警阈值 | 0.51（低分位） |
| 判定 | 预测 S 0.679 ≥ 0.51 → 正常 |

S 的算法：已观察 27 维 → 标准化 Ridge(α=10) 补成整场 27 维 → 冻结浅树 XGB 得 NASA → S = 0.70 × 真实步骤 + 0.30 × (1 − NASA/10)。本条真值步骤与 NASA 见 `data/s.json`。

## 3. 每张图怎么读

- **实线（黑）**：已观察的窗级人因，即 27 维之前的原料（每 5 秒一个点）。
- **橙色虚线**：Transformer 对后半段**瞬时细节**的预报。
- **蓝色点线**：Ridge 折出来的**整体走势**（有斜率列就用预报斜率，从「现在」连出去；没有斜率列就画已观察均值的水平线）。这不是逐窗细节。
- 竖虚线 = 现在。右侧浅底 = 预测段。
- 尺寸：宽∶高 = **2∶1**，PNG，300 dpi。

这些曲线**不是**表 B1 的 27 个汇总数字。27 维是整段的 mean / std / median / slope，一场各一个数，见 `data/ridge27.csv`。

## 4. 图清单

| 文件 | 指标 | 纵轴 | 说明 |
|---|---|---|---|
| `figures/00_人员状态_S.png` | 绩效 S | — | 右侧卡片：正常，预测 S=0.679 |
| `figures/01_hr_mean.png` | 心率均值 | bpm | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/02_hr_std.png` | 心率波动 | bpm | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/03_eye_pupil_filtered_mean.png` | 瞳孔直径 | mm | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/04_eye_aoi_coverage_ratio.png` | AOI覆盖比例 | 比例 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/05_log_action_density_win.png` | 操作密度 | 密度 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/06_log_action_count_win.png` | 操作次数 | 次 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/07_eeg_frontal_theta_alpha_z_within_subject.png` | 额区θ/α | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/08_blink_rate_per_min.png` | 眨眼频率 | 次/分 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/09_eye_aoi_unique_hit_n.png` | 点到不同AOI数 | 个 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/10_eye_aoi_interval_n.png` | AOI区间条数 | 条 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/11_eye_aoi_max_share.png` | 最主要AOI占比 | 比例 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/12_eeg_frontal_alpha_power_z_within_subject.png` | 额区α功率 | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/13_eeg_parietal_theta_alpha_z_within_subject.png` | 顶区θ/α | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/14_eeg_frontal_gamma_power_z_within_subject.png` | 额区γ功率 | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/15_eeg_central_alpha_power_z_within_subject.png` | 中央区α功率 | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/16_eeg_parietal_theta_power_z_within_subject.png` | 顶区θ功率 | z | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/17_hr_max.png` | 最高心率 | bpm | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/18_log_correct_action_count_win.png` | 正确操作次数 | 次 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/19_log_unique_step_count_win.png` | 步骤种数 | 种 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/20_log_extra_action_count_win.png` | 多余操作次数 | 次 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/21_log_extra_rate_win.png` | 多余操作比例 | 比例 | 窗级人因；橙=Transformer，蓝=Ridge |
| `figures/22_log_unique_device_count_win.png` | 设备种数 | 种 | 窗级人因；橙=Transformer，蓝=Ridge |

## 5. 数据子文件夹 `data/`

- `s.json`：真值 / 预测 S、NASA、步骤、阈值、状态。
- `windows.csv`：每个窗口、每个指标的已观察、未来真值、Transformer、Ridge 走势。
- `ridge27.csv`：本折定额 27 列的已观察、Ridge 预报整场、真值整场。
- `meta.json`：样本、切分、指标列表。
- `series_long.csv`：与 `windows.csv` 相同，长表便于画图复查。

## 6. 复现

```bash
cd 工作目录/03_建模/forecast_next_stage
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \
    python export_case_pack.py
```
