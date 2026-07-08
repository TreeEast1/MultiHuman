# P3 分类调参报告（MI 折内筛选）

**筛选**：XGB K=50, RF K=30, LR/SVC K=80
**评估**：StratifiedGroupKFold(5) by subject，pooled Macro-F1

## XGBoost Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=300 | 0.774 | 0.775 | 0.771±0.072 |
| 2 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=500 | 0.762 | 0.763 | 0.762±0.097 |
| 3 | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=800 | 0.762 | 0.763 | 0.762±0.097 |
| 4 | max_depth=3, learning_rate=0.1, reg_lambda=1.0, n_estimators=500 | 0.762 | 0.763 | 0.763±0.115 |
| 5 | max_depth=4, learning_rate=0.02, reg_lambda=2.0, n_estimators=300 | 0.762 | 0.762 | 0.757±0.109 |
| 6 | max_depth=4, learning_rate=0.1, reg_lambda=2.0, n_estimators=500 | 0.762 | 0.762 | 0.757±0.120 |
| 7 | max_depth=4, learning_rate=0.1, reg_lambda=2.0, n_estimators=800 | 0.762 | 0.762 | 0.757±0.120 |
| 8 | max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=300 | 0.750 | 0.752 | 0.753±0.116 |
| 9 | max_depth=2, learning_rate=0.02, reg_lambda=2.0, n_estimators=800 | 0.750 | 0.751 | 0.748±0.128 |
| 10 | max_depth=2, learning_rate=0.1, reg_lambda=5.0, n_estimators=300 | 0.750 | 0.751 | 0.747±0.112 |

## RandomForest Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | max_depth=4, min_samples_leaf=3, n_estimators=500 | 0.738 | 0.737 | 0.732±0.076 |
| 2 | max_depth=6, min_samples_leaf=3, n_estimators=500 | 0.738 | 0.737 | 0.732±0.076 |
| 3 | max_depth=None, min_samples_leaf=3, n_estimators=500 | 0.738 | 0.737 | 0.732±0.076 |
| 4 | max_depth=4, min_samples_leaf=2, n_estimators=300 | 0.726 | 0.726 | 0.719±0.078 |
| 5 | max_depth=6, min_samples_leaf=2, n_estimators=300 | 0.726 | 0.726 | 0.719±0.078 |
| 6 | max_depth=6, min_samples_leaf=2, n_estimators=500 | 0.726 | 0.726 | 0.719±0.078 |
| 7 | max_depth=None, min_samples_leaf=2, n_estimators=300 | 0.726 | 0.726 | 0.719±0.078 |
| 8 | max_depth=None, min_samples_leaf=2, n_estimators=500 | 0.726 | 0.726 | 0.719±0.078 |
| 9 | max_depth=3, min_samples_leaf=3, n_estimators=300 | 0.726 | 0.725 | 0.719±0.078 |
| 10 | max_depth=3, min_samples_leaf=5, n_estimators=300 | 0.726 | 0.725 | 0.719±0.078 |

## LogisticRegression Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | C=0.03, penalty=l2, solver=lbfgs | 0.714 | 0.715 | 0.700±0.078 |
| 2 | C=0.03, penalty=l2, solver=liblinear | 0.702 | 0.703 | 0.689±0.071 |
| 3 | C=0.01, penalty=l2, solver=lbfgs | 0.702 | 0.702 | 0.694±0.099 |
| 4 | C=0.01, penalty=l2, solver=liblinear | 0.702 | 0.702 | 0.696±0.108 |
| 5 | C=0.1, penalty=l2, solver=lbfgs | 0.679 | 0.680 | 0.653±0.068 |
| 6 | C=0.1, penalty=l2, solver=liblinear | 0.679 | 0.680 | 0.666±0.065 |
| 7 | C=0.3, penalty=l2, solver=liblinear | 0.655 | 0.655 | 0.633±0.057 |
| 8 | C=0.3, penalty=l1, solver=liblinear | 0.655 | 0.655 | 0.634±0.101 |
| 9 | C=0.1, penalty=l1, solver=liblinear | 0.655 | 0.643 | 0.638±0.077 |
| 10 | C=0.3, penalty=l2, solver=lbfgs | 0.643 | 0.643 | 0.618±0.079 |

## SVC-RBF Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | C=1.0, gamma=0.01 | 0.655 | 0.655 | 0.646±0.120 |
| 2 | C=3.0, gamma=0.01 | 0.655 | 0.654 | 0.640±0.094 |
| 3 | C=1.0, gamma=scale | 0.643 | 0.643 | 0.641±0.114 |
| 4 | C=3.0, gamma=scale | 0.643 | 0.641 | 0.628±0.089 |
| 5 | C=10.0, gamma=scale | 0.631 | 0.629 | 0.622±0.060 |
| 6 | C=0.5, gamma=0.01 | 0.619 | 0.620 | 0.615±0.120 |
| 7 | C=10.0, gamma=0.01 | 0.619 | 0.619 | 0.611±0.070 |
| 8 | C=0.5, gamma=scale | 0.607 | 0.607 | 0.601±0.117 |
| 9 | C=3.0, gamma=0.05 | 0.571 | 0.566 | 0.561±0.075 |
| 10 | C=10.0, gamma=0.05 | 0.571 | 0.566 | 0.561±0.075 |

## 四类模型 Top-1 对比

| 模型 | 最佳 cfg | pooled Acc | pooled Macro-F1 |
|---|---|---:|---:|
| XGBoost | max_depth=2, learning_rate=0.02, reg_lambda=5.0, n_estimators=300 | 0.774 | 0.775 |
| RandomForest | max_depth=4, min_samples_leaf=3, n_estimators=500 | 0.738 | 0.737 |
| LogisticRegression | C=0.03, penalty=l2, solver=lbfgs | 0.714 | 0.715 |
| SVC-RBF | C=1.0, gamma=0.01 | 0.655 | 0.655 |
