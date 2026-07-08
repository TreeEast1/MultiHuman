# P3 调参实验报告（MI + Top-30）

**设置**：折内 MI 筛选 top-30；84 样本 5×GroupKFold by subject；pooled 指标

## XGBoost Top-10 配置

| rank | cfg | pooled MAE | pooled R² | fold R² (mean±std) |
|---:|---|---:|---:|---:|
| 1 | max_depth=2, learning_rate=0.02, reg_lambda=2.0, n_estimators=500 | 0.911 | +0.519 | +0.501±0.121 |
| 2 | max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=300 | 0.904 | +0.515 | +0.489±0.105 |
| 3 | max_depth=2, learning_rate=0.02, reg_lambda=2.0, n_estimators=300 | 0.921 | +0.515 | +0.500±0.118 |
| 4 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=500 | 0.918 | +0.515 | +0.490±0.097 |
| 5 | max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=500 | 0.903 | +0.513 | +0.484±0.116 |
| 6 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=800 | 0.914 | +0.513 | +0.484±0.102 |
| 7 | max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=800 | 0.904 | +0.511 | +0.482±0.121 |
| 8 | max_depth=2, learning_rate=0.02, reg_lambda=2.0, n_estimators=800 | 0.914 | +0.509 | +0.487±0.128 |
| 9 | max_depth=2, learning_rate=0.02, reg_lambda=1.0, n_estimators=300 | 0.930 | +0.509 | +0.492±0.123 |
| 10 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=300 | 0.929 | +0.507 | +0.485±0.099 |

## XGBoost Bottom-3 配置（做对照）

| cfg | pooled MAE | pooled R² |
|---|---:|---:|
| max_depth=4, learning_rate=0.1, reg_lambda=2.0, n_estimators=500 | 0.974 | +0.426 |
| max_depth=4, learning_rate=0.1, reg_lambda=2.0, n_estimators=300 | 0.974 | +0.426 |
| max_depth=4, learning_rate=0.1, reg_lambda=2.0, n_estimators=800 | 0.974 | +0.426 |

## RandomForest Top-5

| rank | cfg | pooled MAE | pooled R² |
|---:|---|---:|---:|
| 1 | max_depth=4, min_samples_leaf=2, n_estimators=300 | 0.933 | +0.468 |
| 2 | max_depth=6, min_samples_leaf=2, n_estimators=300 | 0.934 | +0.468 |
| 3 | max_depth=None, min_samples_leaf=2, n_estimators=300 | 0.934 | +0.467 |
| 4 | max_depth=4, min_samples_leaf=5, n_estimators=300 | 0.930 | +0.465 |
| 5 | max_depth=6, min_samples_leaf=5, n_estimators=300 | 0.930 | +0.465 |
