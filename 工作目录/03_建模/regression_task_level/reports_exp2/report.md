# P2 特征筛选实验报告

**设置**：84×264 任务级表，折内筛选（fit only on train fold），5×GroupKFold by subject

## 各 (排序方法, 模型) 组合的最佳 K

| 排序 | 模型 | best K | pooled R² | pooled MAE | (Full 264 R²) |
|---|---|---:|---:|---:|---:|
| MI | Ridge_alpha10 | 30 | +0.382 | 1.012 | -0.948 |
| RF_importance | Ridge_alpha10 | 5 | +0.371 | 1.003 | -0.948 |
| Permutation | Ridge_alpha10 | 5 | +0.376 | 0.991 | -0.948 |
| MI | RF_shallow | 15 | +0.483 | 0.915 | +0.451 |
| RF_importance | RF_shallow | 130 | +0.448 | 0.921 | +0.451 |
| Permutation | RF_shallow | 80 | +0.447 | 0.915 | +0.451 |
| MI | XGB_shallow | 30 | +0.470 | 0.959 | +0.465 |
| RF_importance | XGB_shallow | 80 | +0.486 | 0.895 | +0.465 |
| Permutation | XGB_shallow | 50 | +0.489 | 0.883 | +0.465 |

## K vs pooled R² 曲线

### Ridge_alpha10

| K | MI R² | RF_importance R² | Permutation R² |
|---:|---:|---:|---:|
| 5 | +0.302 | +0.371 | +0.376 |
| 10 | +0.340 | -0.367 | -0.015 |
| 15 | +0.354 | -0.297 | +0.186 |
| 20 | +0.348 | -0.131 | +0.219 |
| 30 | +0.382 | -0.366 | +0.090 |
| 50 | +0.047 | -0.560 | -0.602 |
| 80 | -0.048 | -0.831 | -0.789 |
| 130 | -0.148 | -0.812 | -0.917 |
| 264 | -0.948 | -0.948 | -0.948 |

### RF_shallow

| K | MI R² | RF_importance R² | Permutation R² |
|---:|---:|---:|---:|
| 5 | +0.389 | +0.385 | +0.382 |
| 10 | +0.471 | +0.408 | +0.404 |
| 15 | +0.483 | +0.395 | +0.413 |
| 20 | +0.468 | +0.429 | +0.412 |
| 30 | +0.460 | +0.424 | +0.444 |
| 50 | +0.443 | +0.441 | +0.443 |
| 80 | +0.434 | +0.446 | +0.447 |
| 130 | +0.434 | +0.448 | +0.447 |
| 264 | +0.451 | +0.451 | +0.451 |

### XGB_shallow

| K | MI R² | RF_importance R² | Permutation R² |
|---:|---:|---:|---:|
| 5 | +0.331 | +0.260 | +0.260 |
| 10 | +0.395 | +0.395 | +0.392 |
| 15 | +0.419 | +0.409 | +0.403 |
| 20 | +0.403 | +0.421 | +0.435 |
| 30 | +0.470 | +0.436 | +0.442 |
| 50 | +0.430 | +0.472 | +0.489 |
| 80 | +0.432 | +0.486 | +0.478 |
| 130 | +0.457 | +0.476 | +0.480 |
| 264 | +0.465 | +0.465 | +0.465 |

## 稳定选中的特征（每 (ranker, model) 组合，取最佳 K）

*stable_5 = 在 5 折训练中都被选中；对应"极稳健"信号*

### MI + Ridge_alpha10 @ K=30 (R²=+0.382)

- stable_5 count: **5** / 30

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `log_action_density_win__median` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `log_action_density_win__mean` | 4 |
| `log_action_count_win__mean` | 4 |
| `log_correct_action_count_win__std` | 4 |
| `log_unique_device_count_win__slope` | 4 |
| `log_action_count_win__median` | 4 |
| `eeg_parietal_gamma_power_z_within_subject__std` | 4 |
| `log_unique_step_count_win__mean` | 3 |
| `log_unique_device_count_win__mean` | 3 |
| `eeg_occipital_gamma_power_z_within_subject__mean` | 3 |
| `eye_pupil_filtered_std__slope` | 3 |
| `log_extra_rate_win__mean` | 3 |
| `eeg_frontal_theta_power_z_within_subject__mean` | 3 |
| `log_unique_step_count_win__std` | 3 |
| `eye_aoi_max_share__slope` | 3 |
| `log_action_count_win__slope` | 3 |

### RF_importance + Ridge_alpha10 @ K=5 (R²=+0.371)

- stable_5 count: **2** / 5

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 3 |
| `eye_aoi_interval_n__mean` | 3 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 2 |
| `log_disallowed_action_count_win__slope` | 1 |
| `eye_pupil_filtered_std__slope` | 1 |
| `eye_fixation_ratio__slope` | 1 |
| `eye_aoi_max_share__slope` | 1 |
| `blink_duration_median_ms__slope` | 1 |
| `eye_aoi_interval_n__slope` | 1 |
| `log_unique_step_count_win__std` | 1 |

### Permutation + Ridge_alpha10 @ K=5 (R²=+0.376)

- stable_5 count: **2** / 5

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 3 |
| `blink_duration_median_ms__slope` | 2 |
| `eye_aoi_interval_n__mean` | 2 |
| `eye_aoi_interval_n__slope` | 2 |
| `log_disallowed_action_count_win__slope` | 1 |
| `eye_fixation_ratio__slope` | 1 |
| `eye_pupil_filtered_std__slope` | 1 |
| `eye_aoi_max_share__slope` | 1 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 1 |
| `log_unique_step_count_win__std` | 1 |

### MI + RF_shallow @ K=15 (R²=+0.483)

- stable_5 count: **3** / 15

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `log_unique_device_count_win__slope` | 4 |
| `eye_aoi_interval_n__median` | 4 |
| `log_action_density_win__mean` | 3 |
| `log_correct_action_count_win__std` | 3 |
| `eeg_parietal_gamma_power_z_within_subject__std` | 3 |
| `log_unique_step_count_win__std` | 3 |
| `log_unique_step_count_win__mean` | 2 |
| `log_action_count_win__mean` | 2 |
| `log_unique_device_count_win__mean` | 2 |
| `eye_aoi_coverage_ratio__median` | 2 |
| `log_action_density_win__median` | 2 |
| `eeg_frontal_theta_power_z_within_subject__mean` | 2 |
| `log_action_count_win__slope` | 2 |
| `log_action_density_win__slope` | 2 |
| `eye_aoi_fixation_density_per_sec__median` | 1 |
| `log_extra_action_count_win__mean` | 1 |
| `eeg_parietal_theta_alpha_z_within_subject__slope` | 1 |

### RF_importance + RF_shallow @ K=130 (R²=+0.448)

- stable_5 count: **46** / 130

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 5 |
| `log_unique_step_count_win__std` | 5 |
| `eye_aoi_interval_n__slope` | 5 |
| `eye_aoi_entropy__slope` | 5 |
| `blink_duration_median_ms__slope` | 5 |
| `log_error_action_count_win__slope` | 5 |
| `log_extra_action_count_win__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `blink_duration_mean_ms__slope` | 5 |
| `log_extra_action_count_win__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eeg_frontal_beta_alpha_z_within_subject__std` | 5 |
| `eeg_parietal_alpha_power_z_within_subject__slope` | 5 |
| `eeg_central_theta_alpha_z_within_subject__std` | 5 |
| `eeg_parietal_theta_alpha_z_within_subject__std` | 5 |
| `hr_slope_bpm_per_min__mean` | 5 |
| `hr_max__std` | 5 |

### Permutation + RF_shallow @ K=80 (R²=+0.447)

- stable_5 count: **22** / 80

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `blink_duration_median_ms__slope` | 5 |
| `eye_aoi_interval_n__slope` | 5 |
| `log_unique_step_count_win__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 5 |
| `blink_duration_mean_ms__slope` | 5 |
| `log_error_action_count_win__slope` | 5 |
| `log_action_count_win__mean` | 5 |
| `eeg_parietal_alpha_power_z_within_subject__slope` | 5 |
| `log_action_density_win__mean` | 5 |
| `eeg_frontal_alpha_power_z_within_subject__slope` | 5 |
| `eeg_frontal_beta_alpha_z_within_subject__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eeg_parietal_beta_power_z_within_subject__slope` | 5 |
| `eeg_central_alpha_power_z_within_subject__std` | 5 |
| `eeg_occipital_beta_alpha_z_within_subject__median` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eye_aoi_coverage_ratio__slope` | 5 |

### MI + XGB_shallow @ K=30 (R²=+0.470)

- stable_5 count: **5** / 30

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `log_action_density_win__median` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eye_aoi_interval_n__median` | 5 |
| `log_action_density_win__mean` | 4 |
| `log_action_count_win__mean` | 4 |
| `log_correct_action_count_win__std` | 4 |
| `log_unique_device_count_win__slope` | 4 |
| `log_action_count_win__median` | 4 |
| `eeg_parietal_gamma_power_z_within_subject__std` | 4 |
| `log_unique_step_count_win__mean` | 3 |
| `log_unique_device_count_win__mean` | 3 |
| `eeg_occipital_gamma_power_z_within_subject__mean` | 3 |
| `eye_pupil_filtered_std__slope` | 3 |
| `log_extra_rate_win__mean` | 3 |
| `eeg_frontal_theta_power_z_within_subject__mean` | 3 |
| `log_unique_step_count_win__std` | 3 |
| `eye_aoi_max_share__slope` | 3 |
| `log_action_count_win__slope` | 3 |

### RF_importance + XGB_shallow @ K=80 (R²=+0.486)

- stable_5 count: **20** / 80

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 5 |
| `log_unique_step_count_win__std` | 5 |
| `log_error_action_count_win__slope` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `blink_duration_mean_ms__slope` | 5 |
| `eye_aoi_interval_n__mean` | 5 |
| `eeg_frontal_beta_alpha_z_within_subject__std` | 5 |
| `eeg_parietal_alpha_power_z_within_subject__slope` | 5 |
| `log_action_count_win__mean` | 5 |
| `log_action_density_win__mean` | 5 |
| `eeg_frontal_alpha_power_z_within_subject__slope` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `eeg_occipital_beta_alpha_z_within_subject__median` | 5 |
| `eye_aoi_max_share__slope` | 5 |
| `log_correct_action_count_win__std` | 5 |
| `eeg_parietal_beta_power_z_within_subject__slope` | 5 |
| `eye_aoi_coverage_ratio__median` | 5 |

### Permutation + XGB_shallow @ K=50 (R²=+0.489)

- stable_5 count: **11** / 50

| 特征 | 命中折数 |
|---|---:|
| `eye_aoi_interval_n__std` | 5 |
| `eye_aoi_unique_hit_n__std` | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | 5 |
| `log_unique_step_count_win__std` | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | 5 |
| `blink_duration_mean_ms__slope` | 5 |
| `log_action_density_win__mean` | 5 |
| `eeg_frontal_alpha_power_z_within_subject__slope` | 5 |
| `eeg_frontal_beta_alpha_z_within_subject__std` | 5 |
| `eye_aoi_fixation_n__slope` | 5 |
| `eye_aoi_entropy__median` | 5 |
| `blink_duration_median_ms__slope` | 4 |
| `eye_aoi_interval_n__slope` | 4 |
| `hr_std__std` | 4 |
| `log_error_action_count_win__slope` | 4 |
| `log_action_count_win__mean` | 4 |
| `eeg_parietal_alpha_power_z_within_subject__slope` | 4 |
| `blink_duration_median_ms__std` | 4 |
| `eeg_central_alpha_power_z_within_subject__std` | 4 |
| `eeg_occipital_beta_alpha_z_within_subject__median` | 4 |

