# P2 扩展 v2：NASA 三分类 带模态约束的精细化 K 搜索

**设置**：84×264，折内筛选（防泄漏），StratifiedGroupKFold(5) by subject

**K 集合**（共 9 点）：[15, 20, 25, 30, 35, 40, 50, 60, 80]

**模态规则**（翁老师原话：眼动/脑电/心率/行为 4 个维度）：

- `eye` (眼动) = `eye_*` ∪ `blink_*`
- `eeg` (脑电) = `eeg_*`
- `hr`  (心率) = `hr_*`
- `behav` (行为) = `log_*`

**约束逻辑**：先按 ranker 选 K 个；若 4 模态未全覆盖，则用『该模态在 ranker 中排名最高』的特征 **替换** Top-K 中『在 ranker 中最弱』的那一个，总特征数严格 = K。

**ranker × K × model × {无约束, 带约束}** 全组合扫描。

## 1. 各 (ranker, model) 组合 —— 带约束下的最佳 K

| ranker | model | best K | constrained Acc | constrained Macro-F1 | (无约束 best F1) |
|---|---|---:|---:|---:|---:|
| MI | LR_L2_strong | 25 | 0.690 | 0.692 | 0.692 |
| RF_importance | LR_L2_strong | 30 | 0.714 | 0.715 | 0.715 |
| MI | RF_shallow | 15 | 0.738 | 0.739 | 0.737 |
| RF_importance | RF_shallow | 60 | 0.738 | 0.738 | 0.738 |
| MI | XGB_shallow | 40 | 0.786 | 0.786 | 0.786 |
| RF_importance | XGB_shallow | 30 | 0.774 | 0.776 | 0.776 |

## 2. 全局 Top-10 (ranker, K, model) 组合 —— 带约束

| rank | ranker | K | model | pooled Acc | pooled Macro-F1 |
|---:|---|---:|---|---:|---:|
| 1 | MI | 40 | XGB_shallow | 0.786 | 0.786 |
| 2 | RF_importance | 30 | XGB_shallow | 0.774 | 0.776 |
| 3 | RF_importance | 25 | XGB_shallow | 0.762 | 0.763 |
| 4 | RF_importance | 35 | XGB_shallow | 0.762 | 0.762 |
| 5 | MI | 50 | XGB_shallow | 0.750 | 0.751 |
| 6 | RF_importance | 40 | XGB_shallow | 0.750 | 0.750 |
| 7 | RF_importance | 50 | XGB_shallow | 0.750 | 0.750 |
| 8 | RF_importance | 15 | XGB_shallow | 0.750 | 0.749 |
| 9 | MI | 35 | XGB_shallow | 0.750 | 0.748 |
| 10 | RF_importance | 20 | XGB_shallow | 0.738 | 0.739 |

## 3. K vs pooled Macro-F1 曲线（无约束 vs 带约束）

### MI + LR_L2_strong

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.643 | 0.645 | 0.643 | 0.645 | +0.000 |
| 20 | 0.655 | 0.656 | 0.643 | 0.644 | -0.011 |
| 25 | 0.690 | 0.692 | 0.690 | 0.692 | +0.000 |
| 30 | 0.667 | 0.664 | 0.667 | 0.664 | +0.000 |
| 35 | 0.619 | 0.619 | 0.619 | 0.619 | +0.000 |
| 40 | 0.643 | 0.644 | 0.643 | 0.644 | +0.000 |
| 50 | 0.667 | 0.668 | 0.667 | 0.668 | +0.000 |
| 60 | 0.679 | 0.677 | 0.679 | 0.677 | +0.000 |
| 80 | 0.679 | 0.680 | 0.679 | 0.680 | +0.000 |

### RF_importance + LR_L2_strong

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.631 | 0.632 | 0.690 | 0.690 | +0.059 |
| 20 | 0.667 | 0.666 | 0.690 | 0.691 | +0.024 |
| 25 | 0.702 | 0.702 | 0.702 | 0.703 | +0.001 |
| 30 | 0.714 | 0.715 | 0.714 | 0.715 | +0.000 |
| 35 | 0.679 | 0.679 | 0.679 | 0.679 | +0.000 |
| 40 | 0.679 | 0.679 | 0.679 | 0.679 | +0.000 |
| 50 | 0.702 | 0.703 | 0.702 | 0.703 | +0.000 |
| 60 | 0.702 | 0.703 | 0.702 | 0.703 | +0.000 |
| 80 | 0.702 | 0.702 | 0.702 | 0.702 | +0.000 |

### MI + RF_shallow

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.726 | 0.728 | 0.738 | 0.739 | +0.011 |
| 20 | 0.679 | 0.680 | 0.667 | 0.668 | -0.012 |
| 25 | 0.702 | 0.705 | 0.702 | 0.705 | +0.000 |
| 30 | 0.738 | 0.737 | 0.738 | 0.737 | +0.000 |
| 35 | 0.702 | 0.702 | 0.702 | 0.702 | +0.000 |
| 40 | 0.702 | 0.702 | 0.702 | 0.702 | +0.000 |
| 50 | 0.702 | 0.703 | 0.702 | 0.703 | +0.000 |
| 60 | 0.714 | 0.716 | 0.714 | 0.716 | +0.000 |
| 80 | 0.702 | 0.703 | 0.702 | 0.703 | +0.000 |

### RF_importance + RF_shallow

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.702 | 0.701 | 0.702 | 0.701 | +0.000 |
| 20 | 0.738 | 0.736 | 0.738 | 0.737 | +0.001 |
| 25 | 0.726 | 0.725 | 0.714 | 0.714 | -0.011 |
| 30 | 0.702 | 0.702 | 0.702 | 0.702 | +0.000 |
| 35 | 0.702 | 0.702 | 0.702 | 0.702 | +0.000 |
| 40 | 0.679 | 0.679 | 0.679 | 0.679 | +0.000 |
| 50 | 0.726 | 0.727 | 0.726 | 0.727 | +0.000 |
| 60 | 0.738 | 0.738 | 0.738 | 0.738 | +0.000 |
| 80 | 0.726 | 0.726 | 0.726 | 0.726 | +0.000 |

### MI + XGB_shallow

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.738 | 0.738 | 0.738 | 0.738 | -0.001 |
| 20 | 0.738 | 0.738 | 0.738 | 0.739 | +0.000 |
| 25 | 0.726 | 0.727 | 0.726 | 0.727 | +0.000 |
| 30 | 0.738 | 0.733 | 0.738 | 0.733 | +0.000 |
| 35 | 0.750 | 0.748 | 0.750 | 0.748 | +0.000 |
| 40 | 0.786 | 0.786 | 0.786 | 0.786 | +0.000 |
| 50 | 0.750 | 0.751 | 0.750 | 0.751 | +0.000 |
| 60 | 0.726 | 0.726 | 0.726 | 0.726 | +0.000 |
| 80 | 0.714 | 0.713 | 0.714 | 0.713 | +0.000 |

### RF_importance + XGB_shallow

| K | 无约束 Acc | 无约束 F1 | 带约束 Acc | 带约束 F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.738 | 0.737 | 0.750 | 0.749 | +0.012 |
| 20 | 0.702 | 0.702 | 0.738 | 0.739 | +0.038 |
| 25 | 0.762 | 0.762 | 0.762 | 0.763 | +0.001 |
| 30 | 0.774 | 0.776 | 0.774 | 0.776 | +0.000 |
| 35 | 0.762 | 0.762 | 0.762 | 0.762 | +0.000 |
| 40 | 0.750 | 0.750 | 0.750 | 0.750 | +0.000 |
| 50 | 0.750 | 0.750 | 0.750 | 0.750 | +0.000 |
| 60 | 0.738 | 0.739 | 0.738 | 0.739 | +0.000 |
| 80 | 0.738 | 0.738 | 0.738 | 0.738 | +0.000 |

## 4. 全局最佳（带约束）：MI + XGB_shallow @ K=40 (Macro-F1=0.786)

- pooled Acc = **0.786**
- stable_5 count: **7** / 40
- 各模态在 Top-K 中 5/5 折命中的特征数：{np.str_('eye'): 5, np.str_('eeg'): 1, np.str_('hr'): 1}

Top-20 稳定特征（5/5 折命中）：

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 5 |
| `eye_aoi_interval_n__median` | eye | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | eeg | 5 |
| `eye_aoi_interval_n__mean` | eye | 5 |
| `hr_slope_bpm_per_min__slope` | hr | 5 |
| `eye_aoi_coverage_ratio__slope` | eye | 5 |
| `log_unique_step_count_win__mean` | behav | 4 |
| `eye_pupil_filtered_std__slope` | eye | 4 |
| `log_unique_step_count_win__std` | behav | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | eeg | 4 |
| `eye_aoi_entropy__median` | eye | 4 |
| `eye_aoi_max_share__slope` | eye | 4 |
| `eye_aoi_total_fix_ms__slope` | eye | 4 |
| `eye_aoi_coverage_ratio__mean` | eye | 3 |
| `log_action_density_win__slope` | behav | 3 |
| `log_action_count_win__slope` | behav | 3 |
| `eeg_parietal_beta_power_z_within_subject__slope` | eeg | 3 |
| `log_action_count_win__mean` | behav | 3 |
| `eeg_frontal_gamma_power_z_within_subject__std` | eeg | 3 |

## 5. 各 (ranker, model) 最佳 K（带约束）的稳定特征

### MI + LR_L2_strong @ K=25 (Macro-F1=0.692)

- stable_5 count: **4** / 25
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 2, np.str_('eeg'): 1, np.str_('hr'): 1}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | eeg | 5 |
| `hr_slope_bpm_per_min__slope` | hr | 5 |
| `eye_aoi_coverage_ratio__slope` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 4 |
| `log_unique_step_count_win__mean` | behav | 4 |
| `eye_aoi_interval_n__median` | eye | 4 |
| `eye_aoi_interval_n__mean` | eye | 4 |
| `eye_pupil_filtered_std__slope` | eye | 4 |
| `log_unique_step_count_win__std` | behav | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | eeg | 4 |
| `eye_aoi_coverage_ratio__mean` | eye | 3 |
| `log_action_density_win__slope` | behav | 3 |
| `log_action_count_win__slope` | behav | 3 |
| `eye_aoi_total_fix_ms__slope` | eye | 3 |

### RF_importance + LR_L2_strong @ K=30 (Macro-F1=0.715)

- stable_5 count: **10** / 30
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 9, np.str_('behav'): 1}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eye_aoi_entropy__median` | eye | 5 |
| `eye_aoi_entropy__mean` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 5 |
| `eye_aoi_interval_n__mean` | eye | 5 |
| `eye_aoi_unique_hit_n__mean` | eye | 5 |
| `log_action_count_win__mean` | behav | 5 |
| `eye_aoi_fixation_n__std` | eye | 5 |
| `eye_aoi_fixation_n__slope` | eye | 5 |
| `eye_aoi_max_share__mean` | eye | 5 |
| `eye_aoi_fixation_density_per_sec__std` | eye | 4 |
| `log_action_density_win__mean` | behav | 4 |
| `eye_aoi_coverage_ratio__slope` | eye | 4 |
| `eeg_frontal_gamma_power_z_within_subject__std` | eeg | 4 |
| `eye_aoi_coverage_ratio__median` | eye | 4 |

### MI + RF_shallow @ K=15 (Macro-F1=0.739)

- stable_5 count: **3** / 15
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 1, np.str_('eeg'): 1, np.str_('hr'): 1}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | eeg | 5 |
| `hr_slope_bpm_per_min__slope` | hr | 5 |
| `eye_aoi_interval_n__std` | eye | 4 |
| `eye_aoi_interval_n__mean` | eye | 4 |
| `log_unique_step_count_win__std` | behav | 4 |
| `log_unique_step_count_win__mean` | behav | 3 |
| `eye_aoi_interval_n__median` | eye | 3 |
| `eye_aoi_coverage_ratio__slope` | eye | 3 |
| `log_error_action_count_win__std` | behav | 2 |
| `eye_pupil_filtered_std__slope` | eye | 2 |
| `eye_aoi_max_share__median` | eye | 2 |
| `log_action_density_win__std` | behav | 2 |
| `log_action_count_win__std` | behav | 2 |
| `eeg_parietal_beta_alpha_z_within_subject__slope` | eeg | 2 |

### RF_importance + RF_shallow @ K=60 (Macro-F1=0.738)

- stable_5 count: **17** / 60
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 10, np.str_('behav'): 5, np.str_('eeg'): 2}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eye_aoi_entropy__median` | eye | 5 |
| `eye_aoi_entropy__mean` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 5 |
| `eye_aoi_interval_n__mean` | eye | 5 |
| `eye_aoi_unique_hit_n__mean` | eye | 5 |
| `log_action_count_win__mean` | behav | 5 |
| `eye_aoi_fixation_n__std` | eye | 5 |
| `eye_aoi_fixation_n__slope` | eye | 5 |
| `log_action_density_win__mean` | behav | 5 |
| `eeg_frontal_gamma_power_z_within_subject__std` | eeg | 5 |
| `eye_aoi_max_share__mean` | eye | 5 |
| `log_error_rate_win__std` | behav | 5 |
| `eye_aoi_interval_n__median` | eye | 5 |
| `eeg_parietal_beta_alpha_z_within_subject__std` | eeg | 5 |

### MI + XGB_shallow @ K=40 (Macro-F1=0.786)

- stable_5 count: **7** / 40
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 5, np.str_('eeg'): 1, np.str_('hr'): 1}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 5 |
| `eye_aoi_interval_n__median` | eye | 5 |
| `eeg_parietal_theta_power_z_within_subject__slope` | eeg | 5 |
| `eye_aoi_interval_n__mean` | eye | 5 |
| `hr_slope_bpm_per_min__slope` | hr | 5 |
| `eye_aoi_coverage_ratio__slope` | eye | 5 |
| `log_unique_step_count_win__mean` | behav | 4 |
| `eye_pupil_filtered_std__slope` | eye | 4 |
| `log_unique_step_count_win__std` | behav | 4 |
| `eeg_central_theta_alpha_z_within_subject__slope` | eeg | 4 |
| `eye_aoi_entropy__median` | eye | 4 |
| `eye_aoi_max_share__slope` | eye | 4 |
| `eye_aoi_total_fix_ms__slope` | eye | 4 |
| `eye_aoi_coverage_ratio__mean` | eye | 3 |

### RF_importance + XGB_shallow @ K=30 (Macro-F1=0.776)

- stable_5 count: **10** / 30
- 各模态 5/5 折命中的特征数：{np.str_('eye'): 9, np.str_('behav'): 1}

| 特征 | 模态 | 命中折数 |
|---|---|---:|
| `eye_aoi_unique_hit_n__std` | eye | 5 |
| `eye_aoi_entropy__median` | eye | 5 |
| `eye_aoi_entropy__mean` | eye | 5 |
| `eye_aoi_interval_n__std` | eye | 5 |
| `eye_aoi_interval_n__mean` | eye | 5 |
| `eye_aoi_unique_hit_n__mean` | eye | 5 |
| `log_action_count_win__mean` | behav | 5 |
| `eye_aoi_fixation_n__std` | eye | 5 |
| `eye_aoi_fixation_n__slope` | eye | 5 |
| `eye_aoi_max_share__mean` | eye | 5 |
| `eye_aoi_fixation_density_per_sec__std` | eye | 4 |
| `log_action_density_win__mean` | behav | 4 |
| `eye_aoi_coverage_ratio__slope` | eye | 4 |
| `eeg_frontal_gamma_power_z_within_subject__std` | eeg | 4 |
| `eye_aoi_coverage_ratio__median` | eye | 4 |

