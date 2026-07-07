# 任务级建模数据集审计报告

- 数据源：`工作目录/01_预处理/output_30s_step5s_final`
- 样本数（sample_id）：84
- 独立被试数：26
- 原始特征列数：66
- 聚合统计量：mean, std, median, slope
- 任务级特征列数：264（= 66 × 4）
- 标签范围：[1.333, 7.800]，均值 4.971，std 1.581

## 样本分布

### 按被试

- 被试数：26
- 每被试样本数：min=3, max=4, mean=3.23

### 按任务难度

| 难度 | 样本数 | NASA 均值 | NASA std |
|---|---:|---:|---:|
| 中 | 24 | 5.233 | 0.642 |
| 低 | 31 | 3.344 | 1.067 |
| 高 | 29 | 6.492 | 0.748 |

### 每个样本的窗口数

- min=3, max=463, mean=150.3, median=116

## 特征缺失率（聚合后，前 20 高）

| 特征 | 缺失率 |
|---|---:|
| `eye_aoi_entropy__mean` | 2.38% |
| `eye_aoi_entropy__median` | 2.38% |
| `eye_aoi_fixation_density_per_sec__median` | 2.38% |
| `eye_aoi_fixation_density_per_sec__slope` | 2.38% |
| `eye_aoi_coverage_ratio__mean` | 2.38% |
| `eye_aoi_coverage_ratio__std` | 2.38% |
| `eye_aoi_coverage_ratio__median` | 2.38% |
| `eye_aoi_coverage_ratio__slope` | 2.38% |
| `eye_aoi_max_share__mean` | 2.38% |
| `eye_aoi_max_share__std` | 2.38% |
| `eye_aoi_max_share__median` | 2.38% |
| `eye_aoi_max_share__slope` | 2.38% |
| `eye_aoi_fixation_density_per_sec__mean` | 2.38% |
| `eye_aoi_entropy__std` | 2.38% |
| `eye_aoi_fixation_density_per_sec__std` | 2.38% |
| `eye_aoi_entropy__slope` | 2.38% |
| `eye_aoi_pupil_weighted_mean__std` | 2.38% |
| `eye_aoi_pupil_weighted_mean__mean` | 2.38% |
| `eye_aoi_pupil_weighted_mean__median` | 2.38% |
| `eye_aoi_pupil_weighted_mean__slope` | 2.38% |

整体缺失率：0.216%

备注：`__slope` 列整体缺失率 0.22%（当某特征在整任务内全 NaN 或仅 1 个窗口有效时会置 NaN）
