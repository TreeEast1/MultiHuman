# P2 分类特征筛选实验报告

**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject

## 各 (排序方法, 模型) 组合的最佳 K

| 排序 | 模型 | best K | pooled Acc | pooled Macro-F1 | (Full F1) |
|---|---|---:|---:|---:|---:|
| MI | LR_L2_strong | 130 | 0.786 | 0.787 | 0.787 |
| RF_importance | LR_L2_strong | 130 | 0.845 | 0.848 | 0.787 |
| Permutation | LR_L2_strong | 130 | 0.762 | 0.763 | 0.787 |
| MI | RF_shallow | 50 | 0.857 | 0.861 | 0.779 |
| RF_importance | RF_shallow | 15 | 0.798 | 0.801 | 0.779 |
| Permutation | RF_shallow | 80 | 0.810 | 0.815 | 0.779 |
| MI | XGB_shallow | 15 | 0.810 | 0.809 | 0.776 |
| RF_importance | XGB_shallow | 50 | 0.774 | 0.776 | 0.776 |
| Permutation | XGB_shallow | 80 | 0.786 | 0.789 | 0.776 |

## K vs pooled Macro-F1

### LR_L2_strong

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.663 | 0.637 | 0.607 |
| 10 | 0.651 | 0.680 | 0.649 |
| 15 | 0.677 | 0.727 | 0.663 |
| 20 | 0.701 | 0.776 | 0.600 |
| 30 | 0.748 | 0.762 | 0.595 |
| 50 | 0.762 | 0.789 | 0.631 |
| 80 | 0.763 | 0.811 | 0.702 |
| 130 | 0.787 | 0.848 | 0.763 |
| 264 | 0.787 | 0.787 | 0.787 |

### RF_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.789 | 0.718 | 0.612 |
| 10 | 0.797 | 0.775 | 0.627 |
| 15 | 0.775 | 0.801 | 0.593 |
| 20 | 0.777 | 0.792 | 0.608 |
| 30 | 0.754 | 0.775 | 0.693 |
| 50 | 0.861 | 0.767 | 0.671 |
| 80 | 0.791 | 0.790 | 0.815 |
| 130 | 0.803 | 0.791 | 0.804 |
| 264 | 0.779 | 0.779 | 0.779 |

### XGB_shallow

| K | MI | RF_importance | Permutation |
|---:|---:|---:|---:|
| 5 | 0.801 | 0.764 | 0.611 |
| 10 | 0.799 | 0.757 | 0.642 |
| 15 | 0.809 | 0.750 | 0.631 |
| 20 | 0.732 | 0.717 | 0.656 |
| 30 | 0.786 | 0.749 | 0.711 |
| 50 | 0.775 | 0.776 | 0.699 |
| 80 | 0.766 | 0.776 | 0.789 |
| 130 | 0.775 | 0.765 | 0.773 |
| 264 | 0.776 | 0.776 | 0.776 |

## 稳定选中的特征（每 (ranker, model) 组合取最佳 K）

### MI + LR_L2_strong @ K=130 (Macro-F1=0.787)

- stable_5 count: **57** / 130

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__median` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `log_action_density_win__slope` | 5 |
| `log_action_count_win__slope` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 5 |
| `log_correct_action_count_win__std` | 5 |
| `log_disallowed_action_count_win__slope` | 5 |
| `log_error_rate_win__slope` | 5 |
| `log_unique_device_count_win__slope` | 5 |
| `log_unique_step_count_win__slope` | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | 5 |
| `eye_aoi_interval_n__slope` | 5 |
| `log_extra_action_count_win__slope` | 5 |
| `eye_aoi_unique_hit_n__slope` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `log_extra_rate_win__slope` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 5 |
| `eeg_occipital_theta_power_z_within_subject__slope` | 5 |

### RF_importance + LR_L2_strong @ K=130 (Macro-F1=0.848)

- stable_5 count: **65** / 130

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_central_alpha_power_z_within_subject__std` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `log_unique_device_count_win__slope` | 5 |
| `log_extra_action_count_win__slope` | 5 |
| `log_action_count_win__slope` | 5 |
| `eeg_parietal_alpha_power_z_within_subject__std` | 5 |
| `eye_aoi_coverage_ratio__median` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `log_action_count_win__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |
| `eye_aoi_unique_hit_n__mean` | 5 |
| `eye_aoi_max_share__mean` | 5 |
| `log_disallowed_action_count_win__std` | 5 |
| `log_disallowed_action_count_win__mean` | 5 |
| `eye_aoi_total_fix_ms__std` | 5 |

### Permutation + LR_L2_strong @ K=130 (Macro-F1=0.763)

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

### MI + RF_shallow @ K=50 (Macro-F1=0.861)

- stable_5 count: **13** / 50

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__median` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `log_action_density_win__slope` | 5 |
| `log_action_count_win__slope` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `log_correct_action_count_win__std` | 5 |
| `log_unique_device_count_win__slope` | 5 |
| `eye_aoi_interval_n__slope` | 5 |
| `eye_aoi_unique_hit_n__slope` | 5 |
| `log_extra_rate_win__mean` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 4 |
| `log_disallowed_action_count_win__slope` | 4 |
| `log_error_rate_win__slope` | 4 |
| `log_extra_action_count_win__slope` | 4 |
| `eye_aoi_entropy__median` | 4 |
| `log_extra_rate_win__slope` | 4 |
| `eeg_occipital_theta_power_z_within_subject__slope` | 4 |

### RF_importance + RF_shallow @ K=15 (Macro-F1=0.801)

- stable_5 count: **5** / 15

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `eeg_central_alpha_power_z_within_subject__std` | 4 |
| `eye_aoi_fixation_density_per_sec__std` | 4 |
| `eeg_central_gamma_power_z_within_subject__std` | 3 |
| `eye_aoi_pupil_weighted_mean__slope` | 3 |
| `eye_aoi_entropy__median` | 2 |
| `eye_aoi_coverage_ratio__median` | 2 |
| `log_action_count_win__mean` | 2 |
| `eye_aoi_entropy__std` | 2 |
| `eye_aoi_coverage_ratio__std` | 2 |
| `log_error_rate_win__std` | 2 |
| `log_extra_rate_win__mean` | 2 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 2 |
| `log_action_density_win__mean` | 2 |
| `log_extra_rate_win__std` | 2 |
| `blink_duration_median_ms__std` | 2 |

### Permutation + RF_shallow @ K=80 (Macro-F1=0.815)

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
| `eye_aoi_max_share__std` | 5 |
| `eye_aoi_max_share__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |

### MI + XGB_shallow @ K=15 (Macro-F1=0.809)

- stable_5 count: **4** / 15

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__median` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `log_correct_action_count_win__std` | 4 |
| `log_error_rate_win__slope` | 4 |
| `log_action_density_win__slope` | 2 |
| `log_action_count_win__slope` | 2 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 2 |
| `log_disallowed_action_count_win__slope` | 2 |
| `log_unique_device_count_win__slope` | 2 |
| `eye_aoi_unique_hit_n__slope` | 2 |
| `eye_aoi_interval_n__slope` | 2 |
| `log_correct_action_count_win__slope` | 2 |
| `eye_aoi_fixation_n__std` | 2 |
| `eeg_occipital_gamma_power_z_within_subject__slope` | 2 |
| `log_extra_action_count_win__slope` | 2 |
| `log_duplicate_rate_win__slope` | 1 |
| `log_error_action_count_win__std` | 1 |
| `hr_min__slope` | 1 |

### RF_importance + XGB_shallow @ K=50 (Macro-F1=0.776)

- stable_5 count: **19** / 50

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `eeg_central_alpha_power_z_within_subject__std` | 5 |
| `eye_aoi_fixation_density_per_sec__std` | 5 |
| `log_extra_action_count_win__slope` | 5 |
| `eeg_parietal_alpha_power_z_within_subject__std` | 5 |
| `eye_aoi_fixation_n__std` | 5 |
| `log_action_count_win__mean` | 5 |
| `log_disallowed_action_count_win__mean` | 5 |
| `eye_aoi_total_fix_ms__std` | 5 |
| `eye_aoi_pupil_weighted_mean__slope` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | 5 |
| `eye_aoi_coverage_ratio__std` | 5 |
| `eye_aoi_interval_n__slope` | 5 |
| `log_unique_step_count_win__slope` | 5 |
| `log_action_density_win__mean` | 5 |
| `log_error_rate_win__std` | 5 |
| `eye_aoi_entropy__median` | 4 |

### Permutation + XGB_shallow @ K=80 (Macro-F1=0.789)

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
| `eye_aoi_max_share__std` | 5 |
| `eye_aoi_max_share__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `eye_aoi_entropy__mean` | 5 |
| `eye_aoi_entropy__std` | 5 |

