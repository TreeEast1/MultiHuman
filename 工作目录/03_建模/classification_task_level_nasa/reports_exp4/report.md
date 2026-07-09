# P4 NASA 三分类：最优特征筛选 + 模型优化

**目标**：从 264 指标中筛选最佳输入子集，突破 P1 最佳 0.809

**设置**：84×264，StratifiedGroupKFold(5) by subject，pooled 指标，折内筛选防泄漏

**参考基线**：
- P1 minus_EEG + XGB = **0.809**（当前最高）
- P2 RF_importance + XGB @ K=30（全264上选）= 0.776
- P0 Full + XGB = 0.750

## 1. 全局 Top-20 组合

| rank | 实验 | 子集 | 排序 | K | 模型 | Acc | Macro-F1 | fold F1 μ±σ |
|---:|---|---|---|---:|---|---:|---:|---|
| 1 | D_tuning | minus_EEG | MI | 30 | XGB_d3_lr0.02_l5.0_n300 | 0.810 | **0.809** | 0.810±0.096 |
| 2 | C_ensemble_fixed | stable_aoi_log_no_eeg | fixed | 15 | XGB+RF | 0.798 | **0.798** | 0.791±0.061 |
| 3 | B_stable | stable_aoi_log_no_eeg | fixed | 15 | XGB | 0.798 | **0.798** | 0.792±0.050 |
| 4 | A_selection | minus_EEG | RF_imp | 15 | XGB | 0.798 | **0.798** | 0.798±0.080 |
| 5 | D_tuning | minus_EEG | MI | 30 | XGB_d2_lr0.05_l2.0_n300 | 0.798 | **0.798** | 0.796±0.082 |
| 6 | A_selection | minus_EEG | MI | 30 | XGB | 0.798 | **0.797** | 0.798±0.082 |
| 7 | D_tuning | minus_EEG | MI | 30 | XGB_d2_lr0.02_l2.0_n300 | 0.798 | **0.797** | 0.798±0.082 |
| 8 | D_tuning | minus_EEG | MI | 30 | XGB_d3_lr0.05_l2.0_n300 | 0.798 | **0.797** | 0.793±0.091 |
| 9 | A_baseline | minus_EEG | Full | 152 | XGB_shallow | 0.798 | **0.796** | 0.796±0.124 |
| 10 | A_selection | AOI_Log_Blink | MI | 30 | XGB | 0.786 | **0.786** | 0.783±0.073 |
| 11 | A_selection | minus_EEG | RF_imp | 40 | XGB | 0.786 | **0.786** | 0.785±0.111 |
| 12 | D_tuning | minus_EEG | MI | 30 | XGB_d2_lr0.02_l5.0_n300 | 0.786 | **0.785** | 0.784±0.073 |
| 13 | D_tuning | minus_EEG | MI | 30 | XGB_d2_lr0.05_l5.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 14 | D_tuning | minus_EEG | MI | 30 | XGB_d3_lr0.02_l2.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 15 | D_tuning | minus_EEG | MI | 30 | XGB_d3_lr0.05_l5.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 16 | A_selection | minus_EEG | MI | 40 | XGB | 0.774 | **0.771** | 0.760±0.139 |
| 17 | A_selection | AOI_Log | MI | 25 | XGB | 0.774 | **0.771** | 0.774±0.074 |
| 18 | A_selection | AOI_Log_Blink | MI | 20 | XGB | 0.774 | **0.771** | 0.772±0.096 |
| 19 | A_selection | minus_EEG | RF_imp | 30 | XGB | 0.762 | **0.765** | 0.755±0.072 |
| 20 | A_selection | minus_EEG_HR | MI | 15 | RF | 0.762 | **0.765** | 0.762±0.072 |

## 2. 实验 A：模态子集 × 折内特征选择

### 各子集最佳组合

| 子集 | n_feat | 排序 | K | 模型 | Acc | Macro-F1 |
|---|---:|---|---:|---|---:|---:|
| minus_EEG | 15 | RF_imp | 15 | XGB | 0.798 | **0.798** |
| minus_EEG (Full) | 152 | — | 152 | XGB_shallow | 0.798 | 0.796 |
| minus_EEG (Full) | 152 | — | 152 | RF_shallow | 0.690 | 0.690 |
| minus_EEG_HR | 15 | MI | 15 | RF | 0.762 | **0.765** |
| minus_EEG_HR (Full) | 132 | — | 132 | XGB_shallow | 0.762 | 0.761 |
| minus_EEG_HR (Full) | 132 | — | 132 | RF_shallow | 0.714 | 0.715 |
| AOI_Log | 25 | MI | 25 | XGB | 0.774 | **0.771** |
| AOI_Log (Full) | 84 | — | 84 | XGB_shallow | 0.738 | 0.739 |
| AOI_Log (Full) | 84 | — | 84 | RF_shallow | 0.714 | 0.713 |
| AOI_Log_Blink | 30 | MI | 30 | XGB | 0.786 | **0.786** |
| AOI_Log_Blink (Full) | 108 | — | 108 | XGB_shallow | 0.738 | 0.738 |
| AOI_Log_Blink (Full) | 108 | — | 108 | RF_shallow | 0.738 | 0.739 |

### minus_EEG 上 K vs Macro-F1

| K | MI+XGB | MI+RF | RF_imp+XGB | RF_imp+RF |
|---:|---:|---:|---:|---:|
| 15 | 0.726 | 0.726 | 0.798 | 0.727 |
| 20 | 0.749 | 0.751 | 0.750 | 0.727 |
| 25 | 0.760 | 0.738 | 0.764 | 0.739 |
| 30 | 0.797 | 0.727 | 0.765 | 0.714 |
| 40 | 0.771 | 0.728 | 0.786 | 0.750 |

## 3. 实验 B：固定稳定特征集

### stable_rf_xgb_k30（10 特征）

| 模型 | Acc | Macro-F1 | fold F1 μ±σ |
|---|---:|---:|---|
| XGB | 0.714 | 0.717 | 0.704±0.074 |
| RF | 0.714 | 0.713 | 0.706±0.088 |
| LR | 0.667 | 0.669 | 0.656±0.076 |

特征列表：
- `eye_aoi_unique_hit_n__std`
- `eye_aoi_entropy__median`
- `eye_aoi_entropy__mean`
- `eye_aoi_interval_n__std`
- `eye_aoi_interval_n__mean`
- `eye_aoi_unique_hit_n__mean`
- `log_action_count_win__mean`
- `eye_aoi_fixation_n__std`
- `eye_aoi_fixation_n__slope`
- `eye_aoi_max_share__mean`

### stable_core_aoi（11 特征）

| 模型 | Acc | Macro-F1 | fold F1 μ±σ |
|---|---:|---:|---|
| XGB | 0.690 | 0.690 | 0.660±0.090 |
| RF | 0.679 | 0.680 | 0.663±0.110 |
| LR | 0.655 | 0.659 | 0.638±0.117 |

特征列表：
- `eye_aoi_unique_hit_n__std`
- `eye_aoi_interval_n__std`
- `eye_aoi_interval_n__mean`
- `eye_aoi_coverage_ratio__slope`
- `eye_aoi_entropy__median`
- `eye_aoi_entropy__mean`
- `eye_aoi_fixation_n__std`
- `eye_aoi_fixation_n__slope`
- `eye_aoi_max_share__mean`
- `eye_aoi_unique_hit_n__mean`
- `eye_aoi_interval_n__median`

### stable_aoi_log_no_eeg（15 特征）

| 模型 | Acc | Macro-F1 | fold F1 μ±σ |
|---|---:|---:|---|
| XGB | 0.798 | 0.798 | 0.792±0.050 |
| RF | 0.762 | 0.762 | 0.758±0.085 |
| LR | 0.690 | 0.693 | 0.682±0.073 |

特征列表：
- `eye_aoi_unique_hit_n__std`
- `eye_aoi_interval_n__std`
- `eye_aoi_interval_n__mean`
- `eye_aoi_entropy__median`
- `eye_aoi_entropy__mean`
- `eye_aoi_unique_hit_n__mean`
- `eye_aoi_fixation_n__std`
- `eye_aoi_fixation_n__slope`
- `eye_aoi_max_share__mean`
- `eye_aoi_coverage_ratio__slope`
- `log_action_count_win__mean`
- `log_action_density_win__mean`
- `log_error_rate_win__std`
- `eye_aoi_coverage_ratio__median`
- `eye_aoi_total_fix_ms__median`

## 4. 实验 C：集成投票

### 固定特征集 + 投票

| 特征集 | 集成 | Acc | Macro-F1 |
|---|---|---:|---:|
| stable_rf_xgb_k30 | XGB+RF | 0.714 | **0.715** |
| stable_rf_xgb_k30 | XGB+RF+LR | 0.726 | **0.725** |
| stable_core_aoi | XGB+RF | 0.679 | **0.679** |
| stable_core_aoi | XGB+RF+LR | 0.714 | **0.716** |
| stable_aoi_log_no_eeg | XGB+RF | 0.798 | **0.798** |
| stable_aoi_log_no_eeg | XGB+RF+LR | 0.750 | **0.751** |

### minus_EEG 折内选择 + 投票

| 排序 | K | 集成 | Acc | Macro-F1 |
|---|---:|---|---:|---:|
| MI | 20 | XGB+RF | 0.738 | **0.740** |
| MI | 20 | XGB+RF+LR | 0.738 | **0.738** |
| MI | 30 | XGB+RF | 0.762 | **0.762** |
| MI | 30 | XGB+RF+LR | 0.738 | **0.739** |
| RF_imp | 20 | XGB+RF | 0.738 | **0.738** |
| RF_imp | 20 | XGB+RF+LR | 0.726 | **0.726** |
| RF_imp | 30 | XGB+RF | 0.750 | **0.752** |
| RF_imp | 30 | XGB+RF+LR | 0.738 | **0.739** |

## 5. 实验 D：精细调参（minus_EEG + MI 选择）

### XGB Top-10

| K | 配置 | Acc | Macro-F1 | fold F1 μ±σ |
|---:|---|---:|---:|---|
| 30 | XGB_d3_lr0.02_l5.0_n300 | 0.810 | **0.809** | 0.810±0.096 |
| 30 | XGB_d2_lr0.05_l2.0_n300 | 0.798 | **0.798** | 0.796±0.082 |
| 30 | XGB_d2_lr0.02_l2.0_n300 | 0.798 | **0.797** | 0.798±0.082 |
| 30 | XGB_d3_lr0.05_l2.0_n300 | 0.798 | **0.797** | 0.793±0.091 |
| 30 | XGB_d2_lr0.02_l5.0_n300 | 0.786 | **0.785** | 0.784±0.073 |
| 30 | XGB_d2_lr0.05_l5.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 30 | XGB_d3_lr0.02_l2.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 30 | XGB_d3_lr0.05_l5.0_n300 | 0.786 | **0.784** | 0.783±0.087 |
| 20 | XGB_d3_lr0.05_l2.0_n300 | 0.762 | **0.761** | 0.765±0.064 |
| 20 | XGB_d2_lr0.02_l5.0_n300 | 0.762 | **0.760** | 0.760±0.065 |

### RF Top-10

| K | 配置 | Acc | Macro-F1 | fold F1 μ±σ |
|---:|---|---:|---:|---|
| 20 | RF_d5_msl2_n300 | 0.762 | **0.763** | 0.758±0.079 |
| 20 | RF_d4_msl2_n300 | 0.750 | **0.752** | 0.744±0.095 |
| 20 | RF_d4_msl3_n300 | 0.750 | **0.751** | 0.744±0.063 |
| 20 | RF_d5_msl3_n300 | 0.750 | **0.751** | 0.744±0.063 |
| 30 | RF_d5_msl2_n300 | 0.738 | **0.739** | 0.736±0.110 |
| 30 | RF_d5_msl3_n300 | 0.738 | **0.739** | 0.736±0.110 |
| 20 | RF_d3_msl2_n300 | 0.726 | **0.728** | 0.719±0.077 |
| 30 | RF_d4_msl3_n300 | 0.726 | **0.727** | 0.724±0.098 |
| 30 | RF_d4_msl2_n300 | 0.726 | **0.727** | 0.725±0.100 |
| 30 | RF_d3_msl2_n300 | 0.714 | **0.715** | 0.711±0.103 |

## 6. 总结

**全局最佳**：minus_EEG + MI K=30 + XGB_d3_lr0.02_l5.0_n300

- pooled Accuracy = **0.810**
- pooled Macro-F1 = **0.809**
- fold F1 = 0.810 ± 0.096

- vs P1 最佳(0.809)：↑ +0.000
- vs P0 baseline(0.750)：+0.059

