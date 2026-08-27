# S 预测实验报告（回归 + 三分位分类）

与现行 NASA 84×264 实验同一特征矩阵、同一被试分组交叉验证。本轮补的是**预测结果**，不是只算出 S 标量。

## 标签

- 连续 S：范围 [0.217, 0.920]，均值 0.567，std 0.163
- 三分位：低 ≤ 0.510（28），中 ≤ 0.640（29），高（27）
- 档含义：低/中/高 = **绩效**低/中/高（S 越高越好，与 NASA 负荷方向相反）

## 评估协议

| 线 | 划分 | 主指标 |
|---|---|---|
| 回归 | GroupKFold(5) by subject | pooled MAE / R² |
| 分类 | StratifiedGroupKFold(5) by subject | pooled Accuracy / Macro-F1 |

- Dummy 均值回归下限：MAE = 0.135（等于 |S − mean(S)| 的均值）
- 去 Log：去掉与步骤分同源的操作日志特征，避免把 S 的 40% 成分从特征里直接读回来
- 提升树：本机无 libomp，`HistGB` = sklearn HistGradientBoosting（深度/学习率对齐 NASA 浅树 XGB）

## 回归：预测连续 S

| 方案 | n_feat | pooled MAE | pooled R² | fold R² (μ±σ) |
|---|---:|---:|---:|---:|
| Full + HistGB_shallow | 264 | 0.106 | +0.370 | +0.168±0.392 |
| minus_EEG + HistGB_shallow | 152 | 0.107 | +0.312 | +0.081±0.506 |
| Full + RF_shallow | 264 | 0.113 | +0.295 | +0.112±0.464 |
| minus_AOI + HistGB_shallow | 228 | 0.110 | +0.283 | -0.059±0.791 |
| minus_Log + HistGB_shallow | 216 | 0.118 | +0.195 | -0.133±0.572 |
| only_AOI + HistGB_shallow | 36 | 0.123 | +0.147 | -0.206±0.599 |
| MI Top-30 + HistGB_nasa_best | 30 | 0.123 | +0.145 | -0.125±0.517 |
| Full + Dummy_mean | 264 | 0.138 | -0.046 | -0.384±0.675 |
| only_Log + HistGB_shallow | 48 | 0.144 | -0.112 | -0.646±0.973 |
| Full + Ridge | 264 | 0.159 | -0.615 | -2.345±3.694 |

- 全特征最佳：`Full + HistGB_shallow`，pooled R² = +0.370，MAE = 0.106
- 去 Log（更诚实）：`minus_Log + HistGB_shallow`，pooled R² = +0.195，MAE = 0.118

## 分类：S 三分位（低/中/高绩效）

| 方案 | n_feat | pooled Acc | Macro-F1 | F1低 | F1中 | F1高 | fold F1 (μ±σ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| minus_EEG + HistGB_shallow | 152 | 0.464 | 0.463 | 0.567 | 0.321 | 0.500 | 0.457±0.055 |
| minus_AOI + HistGB_shallow | 228 | 0.452 | 0.453 | 0.517 | 0.379 | 0.462 | 0.454±0.038 |
| Full + HistGB_shallow | 264 | 0.440 | 0.444 | 0.414 | 0.316 | 0.604 | 0.433±0.117 |
| only_Log + HistGB_shallow | 48 | 0.440 | 0.444 | 0.475 | 0.367 | 0.490 | 0.428±0.062 |
| minus_Log + HistGB_shallow | 216 | 0.417 | 0.420 | 0.415 | 0.328 | 0.519 | 0.409±0.115 |
| MI Top-30 + HistGB_shallow | 30 | 0.381 | 0.383 | 0.526 | 0.237 | 0.385 | 0.365±0.124 |
| Full + LR_L2_strong | 264 | 0.369 | 0.371 | 0.436 | 0.300 | 0.377 | 0.362±0.067 |
| Full + RF_shallow | 264 | 0.345 | 0.349 | 0.467 | 0.197 | 0.383 | 0.339±0.072 |
| Full + Dummy_stratified | 264 | 0.333 | 0.323 | 0.233 | 0.373 | 0.364 | 0.286±0.103 |
| only_AOI + HistGB_shallow | 36 | 0.298 | 0.302 | 0.377 | 0.197 | 0.333 | 0.290±0.086 |
| Full + Dummy_most_frequent | 264 | 0.321 | 0.227 | 0.222 | 0.458 | 0.000 | 0.161±0.009 |

- 全特征最佳：`minus_EEG + HistGB_shallow`，Macro-F1 = 0.463，Acc = 0.464
- 去 Log（更诚实）：`minus_Log + HistGB_shallow`，Macro-F1 = 0.420，Acc = 0.417

### 最佳分类模型 `minus_EEG + HistGB_shallow` 混淆矩阵（pooled）

| 真 \ 预 | 低 | 中 | 高 |
|---|---:|---:|---:|
| 低 | 17 | 9 | 2 |
| 中 | 10 | 9 | 10 |
| 高 | 5 | 9 | 13 |

## 和 NASA 主线对照（同一 84×264、同一 group CV）

| 目标 | 回归最佳 R² | 回归 MAE | 分类最佳 Macro-F1 |
|---|---:|---:|---:|
| NASA-TLX（已有主线） | +0.519 | 0.911 | 0.809 |
| S 绩效（本轮） | +0.370 | 0.106 | 0.463 |
