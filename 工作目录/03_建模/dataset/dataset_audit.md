# 建模数据集审计报告
- 数据源：`工作目录/01_预处理/output_30s_step5s_final`
- 总窗口数：12624
- 样本数（sample_id）：84
- 特征列数：66
- 标签列：`nasa_tlx_weighted_task_label`
- 标签范围：[1.333, 7.800]，均值 5.370

## 特征缺失率（前 20 高）

| 特征 | 缺失率 |
|---|---:|
| `blink_duration_std_ms` | 15.11% |
| `eye_aoi_pupil_weighted_mean` | 7.67% |
| `eye_aoi_entropy` | 7.67% |
| `eye_aoi_max_share` | 7.67% |
| `hr_slope_bpm_per_min` | 7.42% |
| `hr_std` | 7.42% |
| `hr_min` | 6.31% |
| `hr_mean` | 6.31% |
| `hr_max` | 6.31% |
| `blink_duration_median_ms` | 5.96% |
| `blink_duration_mean_ms` | 5.96% |
| `eye_aoi_fixation_density_per_sec` | 2.54% |
| `eye_aoi_coverage_ratio` | 2.54% |
| `eye_saccade_ratio` | 0.26% |
| `eye_eyes_not_found_ratio` | 0.26% |
| `eye_pupil_filtered_std` | 0.26% |
| `eye_valid_ratio` | 0.26% |
| `eye_pupil_filtered_mean` | 0.26% |
| `eye_fixation_ratio` | 0.26% |
| `blink_rate_per_min` | 0.00% |

总体缺失率：1.371%

## 特征分类

| 类别 | 列数 |
|---|---:|
| EEG（z-score） | 28 |
| 心率 | 5 |
| 眼动（瞳孔+注视+AOI） | 15 |
| 眨眼（过滤后） | 6 |
| 日志（仅 win） | 12 |
| **合计** | **66** |
