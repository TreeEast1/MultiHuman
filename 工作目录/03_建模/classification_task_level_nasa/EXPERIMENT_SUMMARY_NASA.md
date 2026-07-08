# NASA-TLX 三分位分档 · 任务级 3 分类 · 实验总结

> 工作区：`classification_task_level_nasa/`
> 标签：`y_nasa`（NASA-TLX 加权总分）按 33.3% / 66.7% 分位数切为 低/中/高
> 区别于 `classification_task_level/`（用 `task_difficulty` 硬编码查表）

---

## 1. 标签构建

| 项 | 值 |
|---|---|
| 原始字段 | `y_nasa`（连续值，1.333 ~ 7.800） |
| 分档阈值 | 低 ≤ 4.267，中 (4.267, 5.733]，高 > 5.733 |
| 类别分布 | 低 29 / 中 28 / 高 27（均衡度 max/min = 1.07） |
| 独立被试 | 26 |
| 特征矩阵 | 84 × 264（等同回归版） |

**核心优势——解耦 task 与难度**：6 个 task 中有 5 个横跨 ≥2 档（仅 task_2 完全在中档），分类不再等价于"区分 task 类型"。

**与原 task_difficulty 一致率**：84.5%（13 个样本被 NASA 重新分档，这些是主观负荷与预设难度不一致的修正）。

---

## 2. 实验总览

| 实验 | 内容 | 组数 | 最佳 Macro-F1 |
|---|---|---:|---:|
| P0 Baseline | 12 模型对比 | 12 | **0.750** (XGB_shallow) |
| P1 模态消融 | LOMO / OOM / OOS × 4 模型 | 68 | **0.809** (minus_EEG + XGB) |
| P2 特征选择 | 3 排序法 × 8 个 K × 3 模型 + baseline | 75 | **0.776** (RF_imp + XGB @ K=30) |
| P3 调参 | XGB 81 + RF 24 + LR 21 + SVC 16 | 142 | **0.775** (XGB depth=2,lr=0.02,λ=5,n=300) |

---

## 3. P0 Baseline（12 模型，按 Macro-F1 排序）

| 模型 | pooled Acc | Macro-F1 | fold F1 (μ±σ) |
|---|---:|---:|---:|
| **XGB_shallow** | **0.750** | **0.750** | 0.749±0.098 |
| XGB_default | 0.726 | 0.729 | 0.723±0.102 |
| RF_shallow | 0.714 | 0.715 | 0.713±0.168 |
| RF_default | 0.702 | 0.703 | 0.704±0.116 |
| LR_L2_strong | 0.631 | 0.632 | 0.603±0.087 |
| SVC_RBF | 0.619 | 0.611 | 0.594±0.098 |
| Dummy_stratified（随机基线） | 0.321 | 0.308 | — |
| Dummy_most_frequent（多数类） | 0.286 | 0.190 | — |

最佳模型 XGB_shallow 混淆矩阵（pooled）：

| 真值\预测 | 低 | 中 | 高 |
|---|---:|---:|---:|
| 低 | 22 | 5 | 2 |
| 中 | 6 | 19 | 3 |
| 高 | 2 | 3 | 22 |

→ 高档 F1=0.815 最好认，中档 F1=0.691 最易混淆。

---

## 4. P1 模态消融关键发现

### Leave-One-Modality-Out（去掉某模态后 XGB 的 F1）

| 去掉的模态 | n_feat | XGB F1 | vs Full(0.750) |
|---|---:|---:|---:|
| EEG | 152 | 0.809 | **+0.059** ↑ |
| EyePupil | 240 | 0.763 | +0.013 |
| Blink | 240 | 0.751 | +0.001 |
| —（Full） | 264 | 0.750 | — |
| HR | 244 | 0.750 | 0.000 |
| Log | 216 | 0.725 | -0.025 |
| **AOI** | 228 | **0.593** | **-0.157** ↓↓ |

### Only-One-Modality（单模态 XGB F1）

| 模态 | n_feat | XGB F1 |
|---|---:|---:|
| **AOI** | 36 | **0.677** |
| Log | 48 | 0.595 |
| HR | 20 | 0.508 |
| Blink | 24 | 0.473 |
| EyePupil | 24 | 0.430 |
| EEG | 112 | 0.419 |

**解读**：
- **AOI（眼动注视区域）是最关键模态**——去掉后 F1 暴跌 0.157，单独使用就有 0.677
- **EEG 在 NASA 标签下反而是噪声**——去掉 EEG 后 F1 反升到 0.809（全实验最高！）
  - 与原 task_difficulty 版不同，说明 NASA 主观负荷更多由行为/眼动指标驱动，而非脑电
- 统计量中 **std 最佳**（单用 0.691），mean/median/slope 接近

---

## 5. P2 特征选择最佳组合

| 排序方法 | 模型 | best K | Macro-F1 |
|---|---|---:|---:|
| RF_importance | XGB_shallow | 30 | **0.776** |
| MI | XGB_shallow | 50 | 0.751 |
| MI | RF_shallow | 30 | 0.737 |
| RF_importance | RF_shallow | 20 | 0.736 |
| Permutation | XGB_shallow | 130 | 0.762 |

→ RF_importance + XGB @ K=30 达到 0.776，比 Full(0.750) 提升 0.026，用 30 个特征代替 264 个。

---

## 6. P3 调参四类模型 Top-1

| 模型 | 最佳配置 | pooled Acc | Macro-F1 | fold F1 (μ±σ) |
|---|---|---:|---:|---:|
| **XGBoost** | depth=2, lr=0.02, λ=5.0, n=300, MI K=50 | 0.774 | **0.775** | 0.771±0.072 |
| RandomForest | depth=4, msl=3, n=500, MI K=30 | 0.738 | 0.737 | 0.732±0.076 |
| LogisticRegression | C=0.03, l2, lbfgs, MI K=80 | 0.714 | 0.715 | 0.700±0.078 |
| SVC-RBF | C=1.0, gamma=0.01, MI K=80 | 0.655 | 0.655 | 0.646±0.120 |

→ XGBoost 调参后 0.775，且方差最小（±0.072），是跨被试泛化最稳的配置。
→ 最佳 XGB 偏好浅树(depth=2)+慢学习(lr=0.02)+强正则(λ=5)，防过拟合。

---

## 7. 与原 task_difficulty 版对比

| 指标 | task_difficulty 版 | NASA 三分位版 |
|---|---:|---:|
| 类别分布 | 低31/中24/高29 | 低29/中28/高27（更均衡） |
| Baseline 最佳 Macro-F1 | 0.787 (LR_L2_strong) | 0.750 (XGB_shallow) |
| 调参后最佳 Macro-F1 | — | 0.775 (XGB 调参) |
| task 与难度是否绑定 | 100% 绑定（泄漏隐患） | 解耦（5/6 task 横跨多档） |
| 最关键模态 | — | AOI（去掉 -0.157） |
| EEG 作用 | — | 去掉反升（噪声） |
| 最高单次实验 | — | 0.809 (minus_EEG + XGB) |

**结论**：
1. NASA 版整体 F1 比原版低约 0.03–0.04，这是**预期内的**——NASA 标签解耦了 task 类型，
   难度更高、更"真实"，不再有"分类=区分 task"的捷径。
2. NASA 版的结论更可信：不存在 task 与难度 100% 绑定的泄漏隐患。
3. 最优配置：**去掉 EEG + XGB_shallow**，pooled Macro-F1 = **0.809**，是所有 297 组实验中的最高值。
4. 模态贡献排序：AOI >> Log > HR ≈ Blink > EyePupil > EEG（EEG 为负贡献）。

---

## 8. 文件清单

```
classification_task_level_nasa/
├── make_dataset_cls_nasa.py          ← NASA 三分位分档数据集
├── baseline_cls.py                   ← 12 模型对比
├── exp1_modality_ablation_cls.py     ← 模态消融
├── exp2_feature_selection_cls.py     ← 特征选择
├── exp3_tuning_cls.py                ← 调参 142 组
├── dataset/
│   ├── X_cls.npy, y_cls.npy, y_cls_int.npy, groups_cls.npy, sample_cls.npy
│   ├── y_nasa_raw.npy                ← 原始 NASA 连续分留档
│   ├── feature_names_cls.json
│   └── dataset_audit_cls.md          ← 数据集审计报告
├── reports_baseline/                 ← baseline_report.md + baseline_results.json
├── reports_exp1/                     ← report.md + results.json
├── reports_exp2/                     ← report.md + results.json
└── reports_exp3/                     ← report.md + results.json
```
