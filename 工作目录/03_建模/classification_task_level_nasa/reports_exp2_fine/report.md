# P2 扩展：NASA 三分类 精细化 K 搜索

**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject

**K 集合**（共 16 点）：[15, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55, 60, 70, 80]

**ranker × K × model** 全组合扫描；下表汇总每个 (ranker, model) 在所有 K 上的最佳 K 与对应精度。

## 1. 各 (排序, 模型) 组合的最佳 K

| 排序 | 模型 | best K | pooled Acc | pooled Macro-F1 |
|---|---|---:|---:|---:|
| MI | LR_L2_strong | 25 | 0.690 | 0.692 |
| RF_importance | LR_L2_strong | 70 | 0.738 | 0.738 |
| Permutation | LR_L2_strong | 40 | 0.631 | 0.632 |
| MI | RF_shallow | 30 | 0.738 | 0.737 |
| RF_importance | RF_shallow | 60 | 0.738 | 0.738 |
| Permutation | RF_shallow | 80 | 0.667 | 0.667 |
| MI | XGB_shallow | 40 | 0.786 | 0.786 |
| RF_importance | XGB_shallow | 30 | 0.774 | 0.776 |
| Permutation | XGB_shallow | 70 | 0.714 | 0.715 |

## 2. 全局 Top-10 (ranker, K, model) 组合

| rank | ranker | K | model | pooled Acc | pooled Macro-F1 |
|---:|---|---:|---|---:|---:|
| 1 | MI | 40 | XGB_shallow | 0.786 | 0.786 |
| 2 | RF_importance | 30 | XGB_shallow | 0.774 | 0.776 |
| 3 | RF_importance | 32 | XGB_shallow | 0.774 | 0.774 |
| 4 | RF_importance | 35 | XGB_shallow | 0.762 | 0.762 |
| 5 | MI | 45 | XGB_shallow | 0.762 | 0.762 |
| 6 | RF_importance | 25 | XGB_shallow | 0.762 | 0.762 |
| 7 | RF_importance | 55 | XGB_shallow | 0.762 | 0.761 |
| 8 | MI | 38 | XGB_shallow | 0.762 | 0.761 |
| 9 | MI | 32 | XGB_shallow | 0.762 | 0.758 |
| 10 | MI | 50 | XGB_shallow | 0.750 | 0.751 |

## 3. K vs pooled Macro-F1 曲线

### LR_L2_strong

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 15 | 0.645 | 0.632 | 0.629 |
| 20 | 0.656 | 0.666 | 0.607 |
| 22 | 0.678 | 0.702 | 0.595 |
| 25 | 0.692 | 0.702 | 0.620 |
| 28 | 0.644 | 0.691 | 0.583 |
| 30 | 0.664 | 0.715 | 0.573 |
| 32 | 0.643 | 0.715 | 0.585 |
| 35 | 0.619 | 0.679 | 0.597 |
| 38 | 0.631 | 0.691 | 0.632 |
| 40 | 0.644 | 0.679 | 0.632 |
| 45 | 0.668 | 0.737 | 0.596 |
| 50 | 0.668 | 0.703 | 0.620 |
| 55 | 0.679 | 0.714 | 0.608 |
| 60 | 0.677 | 0.703 | 0.608 |
| 70 | 0.691 | 0.738 | 0.611 |
| 80 | 0.680 | 0.702 | 0.611 |

### RF_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 15 | 0.728 | 0.701 | 0.607 |
| 20 | 0.680 | 0.736 | 0.584 |
| 22 | 0.692 | 0.737 | 0.572 |
| 25 | 0.705 | 0.725 | 0.559 |
| 28 | 0.727 | 0.725 | 0.560 |
| 30 | 0.737 | 0.702 | 0.548 |
| 32 | 0.714 | 0.690 | 0.584 |
| 35 | 0.702 | 0.702 | 0.583 |
| 38 | 0.691 | 0.690 | 0.606 |
| 40 | 0.702 | 0.679 | 0.573 |
| 45 | 0.691 | 0.715 | 0.596 |
| 50 | 0.703 | 0.727 | 0.559 |
| 55 | 0.703 | 0.727 | 0.595 |
| 60 | 0.716 | 0.738 | 0.644 |
| 70 | 0.691 | 0.714 | 0.631 |
| 80 | 0.703 | 0.726 | 0.667 |

### XGB_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 15 | 0.738 | 0.737 | 0.644 |
| 20 | 0.738 | 0.702 | 0.560 |
| 22 | 0.725 | 0.750 | 0.571 |
| 25 | 0.727 | 0.762 | 0.535 |
| 28 | 0.739 | 0.726 | 0.621 |
| 30 | 0.733 | 0.776 | 0.606 |
| 32 | 0.758 | 0.774 | 0.619 |
| 35 | 0.748 | 0.762 | 0.595 |
| 38 | 0.761 | 0.749 | 0.618 |
| 40 | 0.786 | 0.750 | 0.608 |
| 45 | 0.762 | 0.750 | 0.595 |
| 50 | 0.751 | 0.750 | 0.561 |
| 55 | 0.726 | 0.761 | 0.573 |
| 60 | 0.726 | 0.739 | 0.678 |
| 70 | 0.749 | 0.727 | 0.715 |
| 80 | 0.713 | 0.738 | 0.715 |

## 4. 全局最佳组合：MI + XGB_shallow @ K=40 (Macro-F1=0.786)

- stable_5 count: **7** / 40

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `log_unique_step_count_win__mean` | 4 |
| `eye_pupil_filtered_std__slope` | 4 |
| `log_unique_step_count_win__std` | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 4 |
| `eye_aoi_entropy__median` | 4 |
| `eye_aoi_max_share__slope` | 4 |
| `eye_aoi_total_fix_ms__slope` | 4 |
| `eye_aoi_coverage_ratio__mean` | 3 |
| `log_action_density_win__slope` | 3 |
| `log_action_count_win__slope` | 3 |
| `eeg_parietal_beta_power_z_within_subject__slope` | 3 |
| `log_action_count_win__mean` | 3 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 3 |

## 5. 各 (排序, 模型) 最佳 K 的稳定特征（命中 5/5 折）

### MI + LR_L2_strong @ K=25 (Macro-F1=0.692)

- stable_5 count: **4** / 25

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `eye_aoi_interval_n__std` | 4 |
| `log_unique_step_count_win__mean` | 4 |
| `eye_aoi_interval_n__median` | 4 |
| `eye_aoi_interval_n__mean` | 4 |
| `eye_pupil_filtered_std__slope` | 4 |
| `log_unique_step_count_win__std` | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 4 |
| `eye_aoi_coverage_ratio__mean` | 3 |
| `log_action_density_win__slope` | 3 |
| `log_action_count_win__slope` | 3 |
| `eye_aoi_total_fix_ms__slope` | 3 |

### RF_importance + LR_L2_strong @ K=70 (Macro-F1=0.738)

- stable_5 count: **20** / 70

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `log_unique_device_count_win__mean` | 5 |
| `eye_aoi_unique_hit_n__mean` | 5 |
| `log_action_count_win__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `log_action_density_win__mean` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `eye_aoi_max_share__mean` | 5 |
| `log_error_rate_win__std` | 5 |
| `eye_aoi_interval_n__median` | 5 |

### Permutation + LR_L2_strong @ K=40 (Macro-F1=0.632)

- stable_5 count: **40** / 40

| 特征 | 命中折数 |
|---|---:|
| `eeg_frontal_delta_power_z_within_subject__mean` | 5 |
| `eye_aoi_total_fix_ms__slope` | 5 |
| `eye_aoi_fixation_n__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__median` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eye_aoi_fixation_density_per_sec__mean` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `eye_aoi_fixation_density_per_sec__median` | 5 |
| `eye_aoi_fixation_density_per_sec__slope` | 5 |
| `eye_aoi_coverage_ratio__mean` | 5 |
| `eye_aoi_coverage_ratio__std` | 5 |
| `eye_aoi_coverage_ratio__median` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `eye_aoi_max_share__mean` | 5 |

### MI + RF_shallow @ K=30 (Macro-F1=0.737)

- stable_5 count: **6** / 30

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `log_unique_step_count_win__mean` | 4 |
| `eye_aoi_interval_n__median` | 4 |
| `eye_pupil_filtered_std__slope` | 4 |
| `log_unique_step_count_win__std` | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 4 |
| `eye_aoi_coverage_ratio__mean` | 3 |
| `log_action_density_win__slope` | 3 |
| `log_action_count_win__slope` | 3 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 3 |

### RF_importance + RF_shallow @ K=60 (Macro-F1=0.738)

- stable_5 count: **17** / 60

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_unique_hit_n__mean` | 5 |
| `log_action_count_win__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `log_action_density_win__mean` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `eye_aoi_max_share__mean` | 5 |
| `log_error_rate_win__std` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 5 |

### Permutation + RF_shallow @ K=80 (Macro-F1=0.667)

- stable_5 count: **80** / 80

| 特征 | 命中折数 |
|---|---:|
| `eeg_frontal_delta_power_z_within_subject__mean` | 5 |
| `eye_aoi_total_fix_ms__slope` | 5 |
| `eye_aoi_fixation_n__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__median` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eye_aoi_fixation_density_per_sec__mean` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `eye_aoi_fixation_density_per_sec__median` | 5 |
| `eye_aoi_fixation_density_per_sec__slope` | 5 |
| `eye_aoi_coverage_ratio__mean` | 5 |
| `eye_aoi_coverage_ratio__std` | 5 |
| `eye_aoi_coverage_ratio__median` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `eye_aoi_max_share__mean` | 5 |

### MI + XGB_shallow @ K=40 (Macro-F1=0.786)

- stable_5 count: **7** / 40

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `log_unique_step_count_win__mean` | 4 |
| `eye_pupil_filtered_std__slope` | 4 |
| `log_unique_step_count_win__std` | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 4 |
| `eye_aoi_entropy__median` | 4 |
| `eye_aoi_max_share__slope` | 4 |
| `eye_aoi_total_fix_ms__slope` | 4 |
| `eye_aoi_coverage_ratio__mean` | 3 |

### RF_importance + XGB_shallow @ K=30 (Macro-F1=0.776)

- stable_5 count: **10** / 30

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_unique_hit_n__mean` | 5 |
| `log_action_count_win__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eye_aoi_max_share__mean` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 4 |
| `log_action_density_win__mean` | 4 |
| `eye_aoi_coverage_ratio__slope` | 4 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 4 |
| `eye_aoi_coverage_ratio__median` | 4 |

### Permutation + XGB_shallow @ K=70 (Macro-F1=0.715)

- stable_5 count: **70** / 70

| 特征 | 命中折数 |
|---|---:|
| `eeg_frontal_delta_power_z_within_subject__mean` | 5 |
| `eye_aoi_total_fix_ms__slope` | 5 |
| `eye_aoi_fixation_n__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_n__median` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eye_aoi_fixation_density_per_sec__mean` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `eye_aoi_fixation_density_per_sec__median` | 5 |
| `eye_aoi_fixation_density_per_sec__slope` | 5 |
| `eye_aoi_coverage_ratio__mean` | 5 |
| `eye_aoi_coverage_ratio__std` | 5 |
| `eye_aoi_coverage_ratio__median` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `eye_aoi_max_share__mean` | 5 |

