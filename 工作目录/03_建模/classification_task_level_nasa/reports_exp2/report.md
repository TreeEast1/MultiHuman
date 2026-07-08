# P2 分类特征筛选实验报告

**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject

## 各 (排序方法, 模型) 组合的最佳 K

| 排序 | 模型 | best K | pooled Acc | pooled Macro-F1 | (Full F1) |
|---|---|---:|---:|---:|---:|
| MI | LR_L2_strong | 80 | 0.679 | 0.680 | 0.632 |
| RF_importance | LR_L2_strong | 30 | 0.714 | 0.715 | 0.632 |
| Permutation | LR_L2_strong | 130 | 0.643 | 0.651 | 0.632 |
| MI | RF_shallow | 30 | 0.738 | 0.737 | 0.715 |
| RF_importance | RF_shallow | 20 | 0.738 | 0.736 | 0.715 |
| Permutation | RF_shallow | 130 | 0.690 | 0.691 | 0.715 |
| MI | XGB_shallow | 50 | 0.750 | 0.751 | 0.750 |
| RF_importance | XGB_shallow | 30 | 0.774 | 0.776 | 0.750 |
| Permutation | XGB_shallow | 130 | 0.762 | 0.762 | 0.750 |

## K vs pooled Macro-F1

### LR_L2_strong

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.556 | 0.587 | 0.571 |
| 10 | 0.669 | 0.606 | 0.571 |
| 15 | 0.645 | 0.632 | 0.629 |
| 20 | 0.656 | 0.666 | 0.607 |
| 30 | 0.664 | 0.715 | 0.573 |
| 50 | 0.668 | 0.703 | 0.620 |
| 80 | 0.680 | 0.702 | 0.611 |
| 130 | 0.679 | 0.656 | 0.651 |
| 264 | 0.632 | 0.632 | 0.632 |

### RF_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.620 | 0.679 | 0.535 |
| 10 | 0.692 | 0.701 | 0.548 |
| 15 | 0.728 | 0.701 | 0.607 |
| 20 | 0.680 | 0.736 | 0.584 |
| 30 | 0.737 | 0.702 | 0.548 |
| 50 | 0.703 | 0.727 | 0.559 |
| 80 | 0.703 | 0.726 | 0.667 |
| 130 | 0.691 | 0.667 | 0.691 |
| 264 | 0.715 | 0.715 | 0.715 |

### XGB_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.613 | 0.662 | 0.482 |
| 10 | 0.714 | 0.714 | 0.535 |
| 15 | 0.738 | 0.737 | 0.644 |
| 20 | 0.738 | 0.702 | 0.560 |
| 30 | 0.733 | 0.776 | 0.606 |
| 50 | 0.751 | 0.750 | 0.561 |
| 80 | 0.713 | 0.738 | 0.715 |
| 130 | 0.750 | 0.727 | 0.762 |
| 264 | 0.750 | 0.750 | 0.750 |

## 稳定选中的特征（每 (ranker, model) 组合取最佳 K）

### MI + LR_L2_strong @ K=80 (Macro-F1=0.680)

- stable_5 count: **21** / 80

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `log_unique_step_count_win__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_pupil_filtered_std__slope` | 5 |
| `log_unique_step_count_win__std` | 5 |
| `eye_aoi_unique_hit_n__slope` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__mean` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `log_action_density_win__slope` | 5 |
| `log_action_count_win__slope` | 5 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `log_unique_step_count_win__slope` | 5 |
| `log_unique_device_count_win__mean` | 5 |
| `eye_aoi_total_fix_ms__slope` | 5 |

### RF_importance + LR_L2_strong @ K=30 (Macro-F1=0.715)

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
| `log_error_rate_win__std` | 4 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 4 |
| `eye_aoi_total_fix_ms__median` | 3 |
| `log_unique_device_count_win__mean` | 3 |
| `eye_aoi_max_share__slope` | 3 |

### Permutation + LR_L2_strong @ K=130 (Macro-F1=0.651)

- stable_5 count: **130** / 130

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
| `eye_aoi_max_share__std` | 5 |
| `eye_aoi_max_share__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |

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
| `eye_aoi_total_fix_ms__slope` | 3 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 3 |
| `eye_aoi_entropy__mean` | 3 |
| `eye_aoi_max_share__slope` | 3 |
| `eeg_parietal_theta_alpha_z_within_subject__slope` | 3 |

### RF_importance + RF_shallow @ K=20 (Macro-F1=0.736)

- stable_5 count: **6** / 20

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 4 |
| `eye_aoi_unique_hit_n__mean` | 4 |
| `log_action_count_win__mean` | 4 |
| `log_action_density_win__mean` | 4 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 4 |
| `eye_aoi_coverage_ratio__median` | 4 |
| `eye_aoi_max_share__mean` | 4 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 4 |
| `eye_aoi_fixation_n__slope` | 3 |
| `eye_aoi_coverage_ratio__slope` | 3 |
| `eye_aoi_total_fix_ms__median` | 2 |
| `eye_aoi_entropy__slope` | 2 |
| `eye_aoi_max_share__slope` | 2 |
| `eye_aoi_interval_n__median` | 2 |

### Permutation + RF_shallow @ K=130 (Macro-F1=0.691)

- stable_5 count: **130** / 130

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
| `eye_aoi_max_share__std` | 5 |
| `eye_aoi_max_share__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |

### MI + XGB_shallow @ K=50 (Macro-F1=0.751)

- stable_5 count: **10** / 50

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_pupil_filtered_std__slope` | 5 |
| `hr_slope_bpm_per_min__slope` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |
| `log_action_count_win__slope` | 5 |
| `eeg_central_theta_alpha_z_within_subject__slope` | 5 |
| `log_unique_step_count_win__mean` | 4 |
| `log_unique_step_count_win__std` | 4 |
| `log_action_density_win__slope` | 4 |
| `eye_aoi_entropy__median` | 4 |
| `eeg_central_beta_power_z_within_subject__slope` | 4 |
| `eye_aoi_max_share__slope` | 4 |
| `log_correct_action_count_win__std` | 4 |
| `eye_aoi_total_fix_ms__slope` | 4 |
| `eye_aoi_coverage_ratio__mean` | 3 |
| `eeg_parietal_beta_power_z_within_subject__slope` | 3 |

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
| `log_error_rate_win__std` | 4 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 4 |
| `eye_aoi_total_fix_ms__median` | 3 |
| `log_unique_device_count_win__mean` | 3 |
| `eye_aoi_max_share__slope` | 3 |

### Permutation + XGB_shallow @ K=130 (Macro-F1=0.762)

- stable_5 count: **130** / 130

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
| `eye_aoi_max_share__std` | 5 |
| `eye_aoi_max_share__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |

