# P3 分类调参报告（MI 折内筛选）

**筛选**：XGB K=15, RF K=50, LR/SVC K=130
**评估**：StratifiedGroupKFold(5) by subject，pooled Macro-F1

## XGBoost Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | max_depth=2, learning_rate=0.05, reg_lambda=2.0, n_estimators=500 | 0.833 | 0.833 | 0.829±0.130 |
| 2 | max_depth=2, learning_rate=0.05, reg_lambda=2.0, n_estimators=800 | 0.833 | 0.833 | 0.829±0.130 |
| 3 | max_depth=3, learning_rate=0.02, reg_lambda=1.0, n_estimators=800 | 0.833 | 0.833 | 0.829±0.130 |
| 4 | max_depth=3, learning_rate=0.05, reg_lambda=1.0, n_estimators=500 | 0.833 | 0.832 | 0.827±0.128 |
| 5 | max_depth=3, learning_rate=0.1, reg_lambda=1.0, n_estimators=800 | 0.821 | 0.823 | 0.818±0.122 |
| 6 | max_depth=2, learning_rate=0.02, reg_lambda=1.0, n_estimators=800 | 0.821 | 0.822 | 0.817±0.135 |
| 7 | max_depth=2, learning_rate=0.05, reg_lambda=2.0, n_estimators=300 | 0.821 | 0.822 | 0.817±0.135 |
| 8 | max_depth=2, learning_rate=0.1, reg_lambda=1.0, n_estimators=300 | 0.821 | 0.822 | 0.817±0.135 |
| 9 | max_depth=2, learning_rate=0.1, reg_lambda=2.0, n_estimators=300 | 0.821 | 0.822 | 0.817±0.135 |
| 10 | max_depth=2, learning_rate=0.1, reg_lambda=2.0, n_estimators=500 | 0.821 | 0.822 | 0.817±0.135 |

## RandomForest Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | max_depth=4, min_samples_leaf=3, n_estimators=300 | 0.857 | 0.861 | 0.858±0.075 |
| 2 | max_depth=4, min_samples_leaf=3, n_estimators=500 | 0.857 | 0.861 | 0.857±0.064 |
| 3 | max_depth=6, min_samples_leaf=3, n_estimators=500 | 0.857 | 0.861 | 0.857±0.064 |
| 4 | max_depth=None, min_samples_leaf=3, n_estimators=500 | 0.857 | 0.861 | 0.857±0.064 |
| 5 | max_depth=6, min_samples_leaf=3, n_estimators=300 | 0.845 | 0.850 | 0.847±0.064 |
| 6 | max_depth=None, min_samples_leaf=3, n_estimators=300 | 0.845 | 0.850 | 0.847±0.064 |
| 7 | max_depth=4, min_samples_leaf=2, n_estimators=500 | 0.845 | 0.849 | 0.845±0.050 |
| 8 | max_depth=6, min_samples_leaf=2, n_estimators=500 | 0.845 | 0.849 | 0.845±0.050 |
| 9 | max_depth=None, min_samples_leaf=2, n_estimators=500 | 0.845 | 0.849 | 0.845±0.050 |
| 10 | max_depth=4, min_samples_leaf=2, n_estimators=300 | 0.833 | 0.838 | 0.836±0.027 |

## LogisticRegression Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | C=10.0, penalty=l2, solver=lbfgs | 0.833 | 0.834 | 0.829±0.131 |
| 2 | C=1.0, penalty=l2, solver=lbfgs | 0.821 | 0.823 | 0.813±0.151 |
| 3 | C=3.0, penalty=l2, solver=lbfgs | 0.821 | 0.822 | 0.817±0.133 |
| 4 | C=0.3, penalty=l2, solver=lbfgs | 0.798 | 0.799 | 0.790±0.152 |
| 5 | C=0.1, penalty=l2, solver=lbfgs | 0.786 | 0.787 | 0.778±0.150 |
| 6 | C=0.01, penalty=l2, solver=lbfgs | 0.762 | 0.764 | 0.753±0.164 |
| 7 | C=0.03, penalty=l2, solver=lbfgs | 0.762 | 0.764 | 0.753±0.164 |

## SVC-RBF Top-10

| rank | cfg | pooled Acc | pooled Macro-F1 | fold F1 (μ±σ) |
|---:|---|---:|---:|---:|
| 1 | C=10.0, gamma=scale | 0.821 | 0.818 | 0.815±0.109 |
| 2 | C=3.0, gamma=scale | 0.810 | 0.805 | 0.799±0.104 |
| 3 | C=3.0, gamma=0.01 | 0.798 | 0.794 | 0.790±0.101 |
| 4 | C=10.0, gamma=0.01 | 0.786 | 0.781 | 0.775±0.110 |
| 5 | C=0.5, gamma=0.01 | 0.774 | 0.774 | 0.772±0.091 |
| 6 | C=1.0, gamma=scale | 0.774 | 0.772 | 0.768±0.124 |
| 7 | C=0.5, gamma=scale | 0.762 | 0.763 | 0.761±0.126 |
| 8 | C=1.0, gamma=0.01 | 0.762 | 0.757 | 0.753±0.082 |
| 9 | C=3.0, gamma=0.05 | 0.583 | 0.549 | 0.531±0.062 |
| 10 | C=10.0, gamma=0.05 | 0.583 | 0.549 | 0.531±0.062 |

## 四类模型 Top-1 对比

| 模型 | 最佳 cfg | pooled Acc | pooled Macro-F1 |
|---|---|---:|---:|
| XGBoost | max_depth=2, learning_rate=0.05, reg_lambda=2.0, n_estimators=500 | 0.833 | 0.833 |
| RandomForest | max_depth=4, min_samples_leaf=3, n_estimators=300 | 0.857 | 0.861 |
| LogisticRegression | C=10.0, penalty=l2, solver=lbfgs | 0.833 | 0.834 |
| SVC-RBF | C=10.0, gamma=scale | 0.821 | 0.818 |
