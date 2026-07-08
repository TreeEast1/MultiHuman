# P1 模态消融实验报告

**设置**：84 样本 × 264 特征，5×GroupKFold by subject，主指标 pooled MAE / R²

## 模态定义（264 列覆盖情况）

| 模态 | 特征列数 | 覆盖范围 |
|---|---:|---|
| EEG | 112 | 4 统计量 × N 原始列 |
| HR | 20 | 4 统计量 × N 原始列 |
| EyePupil | 24 | 4 统计量 × N 原始列 |
| AOI | 36 | 4 统计量 × N 原始列 |
| Blink | 24 | 4 统计量 × N 原始列 |
| Log | 48 | 4 统计量 × N 原始列 |

## Full baseline

| subset | n_feat | RF pooled R² | Ridge pooled R² | XGB pooled R² | XGB MAE |
|---|---:|---:|---:|---:|---:|
| Full_264 | 264 | +0.451 | -0.038 | +0.465 | 0.910 |

## Leave-One-Modality-Out（去除某一模态后的表现）

| subset | n_feat | RF pooled R² | Ridge pooled R² | XGB pooled R² | XGB MAE |
|---|---:|---:|---:|---:|---:|
| minus_EEG | 152 | +0.429 | +0.042 | +0.447 | 0.915 |
| minus_HR | 244 | +0.452 | +0.021 | +0.470 | 0.901 |
| minus_EyePupil | 240 | +0.459 | +0.025 | +0.457 | 0.912 |
| minus_AOI | 228 | +0.066 | -0.548 | +0.042 | 1.256 |
| minus_Blink | 240 | +0.452 | -0.078 | +0.454 | 0.927 |
| minus_Log | 216 | +0.439 | -0.011 | +0.394 | 0.944 |

解读：与 Full 相比 pooled R² 下降越多 → 该模态贡献越大

## Only-One-Modality（仅使用某一模态）

| subset | n_feat | RF pooled R² | Ridge pooled R² | XGB pooled R² | XGB MAE |
|---|---:|---:|---:|---:|---:|
| only_EEG | 112 | +0.055 | -0.492 | +0.000 | 1.312 |
| only_HR | 20 | +0.060 | -0.843 | -0.032 | 1.326 |
| only_EyePupil | 24 | -0.225 | -0.566 | -0.317 | 1.416 |
| only_AOI | 36 | +0.453 | +0.158 | +0.344 | 1.029 |
| only_Blink | 24 | +0.003 | -0.012 | -0.112 | 1.335 |
| only_Log | 48 | +0.047 | -0.178 | +0.001 | 1.228 |

解读：单模态 pooled R² 越高 → 该模态独立预测能力越强

## Only-One-Statistic（仅使用某一统计量的所有 66 列）

| subset | n_feat | RF pooled R² | Ridge pooled R² | XGB pooled R² | XGB MAE |
|---|---:|---:|---:|---:|---:|
| only_mean | 66 | +0.276 | -0.007 | +0.185 | 1.112 |
| only_std | 66 | +0.489 | +0.358 | +0.470 | 0.937 |
| only_median | 66 | +0.214 | -0.078 | +0.039 | 1.214 |
| only_slope | 66 | +0.134 | -0.646 | +0.186 | 1.109 |

解读：判断 mean/std/median/slope 四种聚合方式的相对价值

## 全部实验按 XGBoost pooled R² 排序

| 排名 | subset | n_features | pooled MAE | pooled R² |
|---:|---|---:|---:|---:|
| 1 | minus_HR | 244 | 0.901 | +0.470 |
| 2 | only_std | 66 | 0.937 | +0.470 |
| 3 | Full_264 | 264 | 0.910 | +0.465 |
| 4 | minus_EyePupil | 240 | 0.912 | +0.457 |
| 5 | minus_Blink | 240 | 0.927 | +0.454 |
| 6 | minus_EEG | 152 | 0.915 | +0.447 |
| 7 | minus_Log | 216 | 0.944 | +0.394 |
| 8 | only_AOI | 36 | 1.029 | +0.344 |
| 9 | only_slope | 66 | 1.109 | +0.186 |
| 10 | only_mean | 66 | 1.112 | +0.185 |
| 11 | minus_AOI | 228 | 1.256 | +0.042 |
| 12 | only_median | 66 | 1.214 | +0.039 |
| 13 | only_Log | 48 | 1.228 | +0.001 |
| 14 | only_EEG | 112 | 1.312 | +0.000 |
| 15 | only_HR | 20 | 1.326 | -0.032 |
| 16 | only_Blink | 24 | 1.335 | -0.112 |
| 17 | only_EyePupil | 24 | 1.416 | -0.317 |
