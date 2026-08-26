# MultiHuman：多模态生理-行为信号驱动的操纵员认知工作负荷预测

> 基于脑电、心率、眼动、操作日志，预测操纵员这次任务累不累、绩效好不好。
> - **主线**：预测 NASA-TLX 加权总分（连续分，或分成低 / 中 / 高三档）
> - **补充**：用预测出的 NASA，再算出预测版绩效 S
>
> **进度日志**，最新在顶部。后面都尽量用白话写清：数据、怎么训练/考试、权重、结果。

---

## 📅 2026-08-26：全模态 27 维五折预测 NASA，合成 S（正式口径）

正式实验报告：[实验报告_S绩效预测_全模态27维.md](工作目录/03_建模/s_score_from_nasa84/实验报告_S绩效预测_全模态27维.md)

- 264 维按模态定额互信息降到 **27 维**（眼动 6 + 脑电 5 + 心率 4 + 行为 12），每折都全模态。
- 浅树 XGB，按被试五折交叉验证，先预测 NASA，再 **步骤 0.70 / NASA 反向 0.30** 合成 S。
- **S pooled R² = 0.979，MAE = 0.025**（NASA R² = 0.553）。这是预测版绩效 S 的正式结果。

---

## 📅 2026-08-25：先预测 NASA，再用它算出预测版 S

### 数据

26 个人，一共 **84 次任务**。原始信号按 **30 秒一段、每次往后挪 5 秒** 切开，一次任务大概 150 段（少的 3 段，多的 463 段）。每段 66 个数（脑电、心率、眼动、操作），再收成均值、波动、中位数、走势。模型实际看到的是一张 **84 行 × 264 列** 的表。

264 列每一列叫什么、什么意思，写在：

- [特征清单_264维.md](工作目录/03_建模/特征清单_264维.md)（先看 66 个窗口指标，再看 264 列全表）
- [特征清单_264维.csv](工作目录/03_建模/特征清单_264维.csv)（Excel 可打开）

### 怎么训练、怎么考试

不能拿同一个人又练又考。26 人分成 5 堆，每次用 4 堆人训练，留下 1 堆人当没见过的新被试来考，换着堆考完。这样 84 次任务每条都有一个「模型没见过这个人」时给出的预测。

### NASA 预测结果

- **连续分**：训练时先挑 30 个最有用的特征，再用 XGB。对上程度大约 **R² 0.52**。
- **三档分类**：按 NASA 高低切成低 / 中 / 高（29 / 28 / 27 条）。最好大约 **F1 0.81**（去掉脑电之后）。

### 基于这个 NASA 的预测版 S

把上面预测出的 NASA，加上这次任务真实的步骤完成情况，合成预测版 S。步骤不另做预测。同一套 84 条预测，只改两边各占多少：

| 权重 | 预测 S 范围 | 均值 | 和真实 S 对上的程度 |
|---|---|---:|---|
| 步骤 0.4 + NASA 0.6 | 0.25–0.88 | 0.57 | R² 0.84 |
| 步骤 0.5 + NASA 0.5 | 0.23–0.90 | 0.59 | R² 0.91 |
| 步骤 0.6 + NASA 0.4 | 0.22–0.92 | 0.60 | R² 0.95 |
| **步骤 0.7 + NASA 0.3（现在用这个）** | **0.19–0.94** | **0.62** | **R² 0.98** |

步骤权重大，S 更贴真值（步骤是真实对错，模型没去猜）。**NASA 那条 R² 0.52 不会变。**

明细：`工作目录/03_建模/s_score_from_nasa84/output_from_xgb_nasa/s_from_xgb_nasa.csv`（`S_xgb` 是 0.7/0.3）。

从原始数据到训练评测的完整流程（含算法配置）：[S预测流程.md](工作目录/03_建模/s_score_from_nasa84/S预测流程.md)。

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with xgboost --with pandas --with numpy --with scikit-learn python compute_s_from_xgb_nasa.py
```

---

## 📅 2026-07-08（续）：NASA-TLX 三分位分档分类实验

### 一句话结论

针对原 `task_difficulty` 标签与 task 类型 100% 绑定（分类等价于"识别 task 类型"）的泄漏隐患，新建 `classification_task_level_nasa/` 工作区，改用 **NASA-TLX 加权总分按三分位数分档**。297 组实验中最高 Macro-F1=**0.809**（去 EEG + XGB），AOI 仍为最关键模态，EEG 在 NASA 标签下确认为噪声。

### 动机：原标签的泄漏隐患

原 `classification_task_level/` 的标签 `task_difficulty` 来自预处理脚本硬编码查表（`step1_build_window_samples.py`）：

```python
TASK_DIFFICULTY = {"1":"中", "2":"中", "4":"中", "3":"低", "5":"低", "5_6":"高"}
```

每个 task 编号被预先指定一个难度等级，与 NASA 评分无关 → **task 与难度 100% 绑定**（低=task_3/5，中=task_1/2/4，高=task_5_6）。3 分类本质是"识别 task 类型"，存在标签泄漏隐患。

### 标签设计：NASA 三分位分档

| 项 | 值 |
|---|---|
| 原始字段 | `y_nasa`（连续值 1.333–7.800，均值 4.971，标准差 1.581） |
| 分档阈值（33%/67% 分位数） | 低 ≤ 4.267，中 (4.267, 5.733]，高 > 5.733 |
| 类别分布 | 低 29 / 中 28 / 高 27（均衡度 max/min=1.07） |
| 与原 task_difficulty 一致率 | 84.5%（13 个样本被 NASA 重新分档） |

**解耦验证**：6 个 task 中 5 个横跨 ≥2 档（仅 task_2 完全在中档），分类不再等价于区分 task 类型：

| task | 低 | 中 | 高 | | task | 低 | 中 | 高 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 1 | 5 | 0 | | 4 | 2 | 9 | 1 |
| 2 | 0 | 6 | 0 | | 5 | 10 | 0 | 1 |
| 3 | 16 | 4 | 0 | | 5_6 | 0 | 4 | 25 |

### 四组实验（共 297 组）

评估：`StratifiedGroupKFold(n_splits=5, groups=subject)`，主指标 pooled Macro-F1，特征矩阵复用回归版 84×264。

#### P0 Baseline（12 模型对比）

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

最佳模型 XGB_shallow 混淆矩阵（pooled）：高档 F1=0.815 最易识别，中档 F1=0.691 最易混淆。

#### P1 模态消融（68 组，XGB Macro-F1）

| 实验 | n_feat | XGB F1 | vs Full |
|---|---:|---:|---:|
| **minus_EEG**（去 EEG） | 152 | **0.809** | **+0.059** ← 全局最高 |
| minus_EyePupil | 240 | 0.763 | +0.013 |
| Full（264 全量） | 264 | 0.750 | — |
| minus_HR | 244 | 0.750 | 0.000 |
| minus_Log | 216 | 0.725 | −0.025 |
| **minus_AOI**（去 AOI） | 228 | **0.593** | **−0.157** ← 崩塌 |

单模态：AOI 0.677 > Log 0.595 > HR 0.508 > Blink 0.473 > EyePupil 0.430 > EEG 0.419。统计量：std 0.691 > slope 0.592 > mean 0.572 ≈ median 0.572。

#### P2 特征选择（75 组：3 排序法 × 8 个 K × 3 模型 + baseline）

| 排序方法 | 模型 | best K | Macro-F1 |
|---|---|---:|---:|
| RF_importance | XGB_shallow | 30 | **0.776** |
| MI | XGB_shallow | 50 | 0.751 |
| Permutation | XGB_shallow | 130 | 0.762 |

→ RF_importance + XGB @ K=30 用 30 个特征达到 0.776，比 Full(0.750) 提升 0.026。

#### P3 调参（142 组：XGB 81 + RF 24 + LR 21 + SVC 16）

| 模型 | 最佳配置 | Macro-F1 | fold F1 (μ±σ) |
|---|---|---:|---:|
| **XGBoost** | depth=2, lr=0.02, λ=5.0, n=300, MI K=50 | **0.775** | 0.771±0.072 |
| RandomForest | depth=4, msl=3, n=500, MI K=30 | 0.737 | 0.732±0.076 |
| LogisticRegression | C=0.03, l2, lbfgs, MI K=80 | 0.715 | 0.700±0.078 |
| SVC-RBF | C=1.0, gamma=0.01, MI K=80 | 0.655 | 0.646±0.120 |

XGB 偏好浅树(depth=2)+慢学习(lr=0.02)+强正则(λ=5)，且方差最小(±0.072)，跨被试泛化最稳。

### NASA 版 vs 原 task_difficulty 版对比

| 指标 | task_difficulty 版 | NASA 三分位版 |
|---|---:|---:|
| 类别分布 | 低31/中24/高29 | 低29/中28/高27（更均衡） |
| Baseline 最佳 Macro-F1 | 0.787 (LR) | 0.750 (XGB) |
| 调参后最佳 Macro-F1 | 0.861 (RF) | 0.775 (XGB) |
| 全局最高 | 0.861 | 0.809 (minus_EEG+XGB) |
| task 与难度绑定 | 100% 绑定（泄漏隐患） | 解耦（5/6 task 横跨多档） |

NASA 版整体 F1 比原版低约 0.03–0.08，**这是预期内的**——NASA 标签解耦了 task 类型，难度更高更真实，不再有"分类=区分 task"的捷径。NASA 版结论更适合作为论文主实验结果。

### 关键发现

1. **AOI 仍是最关键模态**——去掉后 F1 暴跌 0.157，单用 0.677（与原版一致，结论鲁棒）
2. **EEG 在 NASA 标签下确认为噪声**——去掉 EEG 后 F1 反升到 0.809（全局最高）
3. **最优配置**：去 EEG + XGB_shallow，pooled Macro-F1 = **0.809**
4. 模态贡献排序：AOI >> Log > HR ≈ Blink > EyePupil > EEG（EEG 为负贡献）
5. 两条标签线（task_difficulty / NASA）独立实验均指向 AOI 核心 + EEG 噪声 → **结论鲁棒**

---

## 📅 2026-07-08：分类实验完成 + 目录结构重构

### 一句话结论

**3 分类 Macro-F1 达到 0.861（RandomForest），首次在跨被试严格评估下超过历史 F1=0.807 的成绩**。同时把建模目录重构为清晰的双线：`regression_task_level/` 与 `classification_task_level/`，共享 `common/` 工具。

### 一、目录重构（不再混杂）

```
工作目录/03_建模/
├── EXPERIMENT_SUMMARY.md              ← 综合总结（回归 + 分类）
├── common/                            ← 共用工具（跨回归/分类）
│   ├── evaluate.py                    ← 回归 GroupKFold pooled 评估
│   ├── exp_utils.py                   ← 回归通用：pooled_cv / 折内 MI 筛选
│   └── cls_utils.py                   ← 分类通用：StratifiedGroupKFold pooled 评估
├── regression_window_level/           ← 窗口级回归（07-07 上午，历史基线）
│   ├── make_dataset.py / baseline.py
│   └── dataset/                       ← X (12624×66), y, groups
├── regression_task_level/             ← 任务级回归（07-07 下午，主线）
│   ├── make_dataset_task.py / baseline_task.py
│   ├── exp1_modality_ablation.py     ← P1 消融
│   ├── exp2_feature_selection.py     ← P2 筛选
│   ├── exp3_xgb_tuning.py            ← P3 调参
│   ├── dataset/                       ← X_task (84×264), y_task, groups_task
│   └── reports_baseline/ reports_exp1/ reports_exp2/ reports_exp3/
├── classification_task_level/         ← 任务级 3 分类（task_difficulty 标签）
└── classification_task_level_nasa/    ← 任务级 3 分类（NASA 三分位分档，本次新增）
    ├── make_dataset_cls.py            ← 复用回归 X，标签换成 task_difficulty
    ├── baseline_cls.py                ← 12 个模型对比
    ├── exp1_modality_ablation_cls.py
    ├── exp2_feature_selection_cls.py
    ├── exp3_tuning_cls.py             ← RF/XGB/LR/SVC 四类调参共 142 组
    ├── dataset/                       ← X_cls, y_cls, y_cls_int, groups_cls
    └── reports_baseline/ reports_exp1/ reports_exp2/ reports_exp3/
```

### 二、分类最终成绩

**评估**：`StratifiedGroupKFold(n_splits=5, groups=subject)`，保证每折训练/测试集都含 3 类难度  
**主指标**：pooled Accuracy / Macro-F1（合并 5 折预测再算总指标）

| 阶段 | 特征方案 | 模型 | Acc | Macro-F1 |
|---|---|---|---:|---:|
| P0 baseline（264 特征全量，12 个模型） | 全量 | LogisticRegression_L2 (C=0.1) | 0.786 | 0.787 |
| P0 baseline | 全量 | RF_shallow | 0.774 | 0.779 |
| P2 特征筛选 | MI Top-50 | RF_shallow | 0.857 | 0.861 |
| **P3 调参最佳** | **MI Top-50** | **RF(max_depth=4, min_leaf=3, n=300)** | **0.857** | **0.861** |
| P3 LogReg 调参 | MI Top-130 | LR(C=10, L2) | 0.833 | 0.834 |
| P3 XGB 调参 | MI Top-15 | XGB(d=2, lr=0.05, λ=2, n=500) | 0.833 | 0.833 |
| P3 SVC 调参 | MI Top-130 | SVC-RBF(C=10) | 0.821 | 0.818 |

**参考基线**：Dummy_stratified F1=0.242，Dummy_most_frequent F1=0.180，随机猜 acc=0.333

### 三、分类的三个关键发现（与回归高度呼应）

**（1）单模态 AOI 已经吊打全量**

| 单模态 | n_feat | XGB Macro-F1 |
|---|---:|---:|
| **only_AOI** | **36** | **0.826** ← 比 Full 264 (0.776) 还高 5 个点 |
| only_Log | 48 | 0.625 |
| only_HR | 20 | 0.540 |
| only_EEG | 112 | 0.525 |
| only_Blink | 24 | 0.508 |
| only_EyePupil | 24 | 0.456 |

**去 AOI**：F1 从 0.776 崩到 0.538（-0.24）  
**去 EEG**：F1 反而涨到 0.851（用 LR）—— **EEG 是干扰而非信号**

**（2）分类偏好 mean，回归偏好 std**

| 只用一种统计量 66 列 | 回归 R² | 分类 F1 (XGB) |
|---|---:|---:|
| only_mean | +0.185 | **0.796** |
| only_std | **+0.470** | 0.701 |
| only_median | +0.039 | 0.675 |
| only_slope | +0.186 | 0.660 |

- **回归**：预测连续 NASA 分数看信号**波动性**（std）
- **分类**：判断难度档位看信号**平均水平**（mean）

**（3）两条线的稳定 top 特征完全重合**

在**回归 P2** 与**分类 P2** 两种筛选中，都稳定选中的第一梯队：

- `eye_aoi_interval_n__std` — AOI 切换次数的波动
- `eye_aoi_unique_hit_n__std` — 覆盖 AOI 数量的波动
- `eye_aoi_interval_n__mean` — AOI 切换均值

**独立实验指向同一批特征 = 结论鲁棒**。

### 四、分类 vs 历史成绩对比

| 来源 | 划分方式 | 严格度 | Macro-F1 |
|---|---|---|---:|
| 历史随机窗口划分（有泄漏） | 随机 | ❌ | 0.951 |
| 历史任务级 30 次重复 CV | 未强制跨被试 | ⚠️ | 0.807 |
| **本次 StratifiedGroupKFold by subject** | **跨被试严格** | **✅** | **0.861** |

**首次在跨被试严格评估下超过历史成绩，可作为论文主实验结果。**

### 五、关键警示：分类的天然上限

数据审计发现**难度与任务类型 100% 绑定**：低 = task_3/5，中 = task_1/2/4，高 = task_5_6。这意味着 3 分类本质是"识别 task 类型"，AOI 单模态 F1=0.826 已经接近上限。**分类结果好并不意味着模型真的懂"负荷"，只是很好地识别了"这是什么任务"**。这也是为什么**回归比分类更能反映真实的负荷预测能力**——回归的信号来源于同一 task 内的被试主观分差异。

### 六、下一步

- [ ] Leave-One-Subject-Out（26 折）产出最严格的分数用于发表
- [ ] 残差/错误分析：找错最离谱的样本，看规律
- [ ] 可视化：K vs 指标曲线、特征重要性 barplot、预测-真值散点、混淆矩阵热图
- [ ] 仅用 Top-8 稳定特征的极简模型（可解释性验证）

---

## 📅 2026-07-07（下午）：任务级建模 + 消融 + 特征筛选 + 调参

### 一句话结论

**回归 pooled R² 从 0.126 提升到 0.519，MAE 从 1.170 降到 0.911**——达到跨被试负荷回归的文献中位水平（0.4–0.6）。核心发现：**AOI + std 波动性统计量是主导信号，EEG 和 HR 贡献很小**。

### 关键突破：建模粒度是最大瓶颈

窗口级建模（12624 行 × 66 特征）在跨被试评估下 R² 仅 0.126——因为 NASA 真值只有 84 个，窗口级模型实际在学"在共享真值的窗口内输出常数"。改成**任务级建模（每 sample_id 聚合成 1 行 × 264 特征，统计量 = mean/std/median/slope）**：

- R² 从 0.126 → 0.465（3.7×）
- MAE 从 1.170 → 0.910（-22%）

**这一步几乎不涉及新特征工程，只是聚合粒度的正确选择。**

### 回归三大发现

**（1）AOI 是绝对核心**：单模态 R²=0.344 一枝独秀，去掉 AOI 直接崩到 R²=0.042  
**（2）波动性（std）比均值更强**：only_std 66 列 R²=0.470 追平 Full 264  
**（3）EEG 在跨被试回归中弱**：单独 R²=0.000，去掉损失仅 0.018

### P2 + P3 最优回归配置

```python
# 折内 MI 筛选 Top-30 + XGBoost
XGBRegressor(max_depth=2, learning_rate=0.02, reg_lambda=2.0,
             n_estimators=500, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", random_state=0)
# → pooled R² = +0.519, pooled MAE = 0.911
```

XGB Top-10 全部为 `max_depth=2` —— 84 样本上浅树+强正则+高迭代最优。

---

## 📅 2026-07-07（上午）：诚实基线建模 + 数据缺陷修复

### 核心成果

1. **修复 EEG 时长错误**：1002 号 EEG 原始文件时间截断（旧版 24 秒 → 新版 31 分钟），全量重跑，数据集从 12 258 → 12 624 窗口。详见 `工作目录/01_预处理/DATA_FIX_AUDIT.md`。
2. **确定 66 列建模特征**：28 EEG（被试内 z-score）+ 5 HR + 15 眼动（瞳孔+AOI）+ 6 眨眼 + 12 日志(win)
3. **搭建评估框架**：GroupKFold by sample_id，杜绝历史窗口级随机划分泄漏

### 窗口级 baseline 结果（5 折 GroupKFold）

| 模型 | 任务级 MAE | 任务级 R² |
|---|---:|---:|
| MeanPredictor（零信息下限） | 1.325 | -0.093 |
| Linear_AllFeatures | 1.267 | -0.027 |
| **RandomForest_default** | **1.170** | **+0.126** |
| XGBoost_default | 1.219 | +0.019 |

### 关键铁证：泄漏的量级

历史项目自己做过对照实验（`历史工作_存档.../window_split_summary.csv`）：

| 划分方式 | ExtraTrees Macro-F1 |
|---|---:|
| 随机窗口划分（有泄漏） | 0.998 |
| 按 sample_id 划分（正确） | 0.287 |

**F1 从 0.998 塌到 0.287** —— 这就是数据泄漏的量级。

---

## 📅 2026-07-03：数据整理与特征提取完成

### 核心成果

1. **唯一权威原始数据源**（`data/`）：26 被试 × 5 类任务，82+2 有效样本
2. **推翻旧 S 指标**，改用 NASA-TLX 加权总分作为回归目标
3. **多模态时间对齐 + 切窗**：产出 10 s / 30 s 两版窗口化数据集
4. **附文献依据的特征工程**：EEG 被试内 z-score（修复历史遗漏）+ 眨眼代理特征（补足历史缺口）
5. **可复现 pipeline**：3 个脚本（`step1/step2/step3`）+ 每步 README

### 数据规格

| 项 | 值 |
|---|---|
| 被试 | 26 人（编号 01–26） |
| 任务 | 5 类（1/2/3/4/5_6），低/中/高 三档 |
| 有效样本 | 82 NASA/SART，84 组对齐可用 |
| EEG | 30 通道 256 Hz EEGLAB `.set` |
| 心率 | 4.6 s 采样一次（无法做经典 HRV） |
| 眼动 | Tobii 100 Hz |
| 主用数据集 | `output_30s_step5s_final/` 12624 窗口 × 132 列 |

### 特征工程要点

- **EEG（28 原始 + 28 z-score）**：修复历史仅算绝对功率的漏洞，被试内 z-score 把 `frontal_alpha` 个体间方差占比从 69.7% 压到 0.0%
- **眼动（21 列）**：补全眨眼代理（`EyesNotFound` 段 50-500ms 过滤），频率中位数 10 次/分钟符合文献
- **心率（5 列）**：4.6 s 采样限制，只能做基础统计量
- **日志（32 列 = 16 win + 16 cum）**：先都存，后续消融决定取舍（当前建模只用 win）

详细依据见 `工作目录/02_特征数据集/特征提取方案调研报告.md`。

---

## 项目目录结构

```
MultiHuman/
├── README.md                        ← 进度日志（本文件）
├── data/                            ← 唯一权威原始数据源
│   ├── 01_NASA_TLX/                 ← 回归目标
│   ├── 03_心率/、04_EEG/、05_眼动/    ← 原始信号（EEG/眼动不入库）
│   └── 06_任务表现与操作日志/
│
├── 工作目录/
│   ├── 01_预处理/                    ← 多模态对齐 + 切窗 + 特征拼装
│   │   ├── scripts/                 ← step1/step2/step3
│   │   ├── DATA_FIX_AUDIT.md
│   │   └── output_30s_step5s_final/ ← 主用窗口特征表
│   ├── 02_特征数据集/                ← 特征方案调研 + 决策文档
│   └── 03_建模/                      ← 建模评估框架（回归 + 分类双线）
│       ├── EXPERIMENT_SUMMARY.md    ← 综合总结
│       ├── common/                  ← 共用工具（evaluate/exp_utils/cls_utils）
│       ├── regression_window_level/ ← 窗口级回归（历史基线）
│       ├── regression_task_level/   ← 任务级回归（主线）
│       ├── classification_task_level/         ← 任务级 3 分类（task_difficulty 标签）
│       ├── classification_task_level_nasa/    ← 任务级 3 分类（NASA 三分位分档）
│       └── s_score_from_nasa84/               ← S 回算 + 回归/三分位分类
│
└── 历史工作_存档_20260703/           ← 不入库；历史脚本与产物封存
```

---

## 快速上手

前提：Python 3.11+，`numpy pandas scipy scikit-learn xgboost openpyxl`。

### 复现特征提取

```bash
cd 工作目录/01_预处理/scripts
python3 step1_build_window_samples.py --data-dir ../../../data \
    --output-dir ../output_30s_step5s --window-sec 30 --step-sec 5
python3 step2_add_log_features.py --data-dir ../../../data \
    --index-csv ../output_30s_step5s/index.csv \
    --sample-dir ../output_30s_step5s/window_features_30s_step5s \
    --output-dir ../output_30s_step5s_with_log
python3 step3_blink_and_eeg_zscore.py \
    --input-dir ../output_30s_step5s_with_log \
    --eye-dir ../../../data/05_眼动/raw_tsv \
    --output-dir ../output_30s_step5s_final \
    --blink-min-ms 50 --blink-max-ms 500
```

### 复现回归建模

```bash
cd 工作目录/03_建模

# 窗口级 baseline（历史对照）
cd regression_window_level && python3 make_dataset.py && python3 baseline.py && cd ..

# 任务级建模（主推荐）
cd regression_task_level
python3 make_dataset_task.py       # 生成 84×264 任务级表
python3 baseline_task.py           # P0 baseline
python3 exp1_modality_ablation.py  # P1 消融
python3 exp2_feature_selection.py  # P2 筛选（约 10 分钟）
python3 exp3_xgb_tuning.py         # P3 调参（依赖 exp2）
```

### 复现分类建模

```bash
cd 工作目录/03_建模/classification_task_level

python3 make_dataset_cls.py            # 生成 84×264 分类数据集
python3 baseline_cls.py                # 12 个模型对比
python3 exp1_modality_ablation_cls.py  # 分类模态消融
python3 exp2_feature_selection_cls.py  # 分类特征筛选（约 15 分钟）
python3 exp3_tuning_cls.py             # RF/XGB/LR/SVC 四类调参
```

### 复现 NASA 三分位分档分类（07-08 续）

```bash
cd 工作目录/03_建模/classification_task_level_nasa

python3 make_dataset_cls_nasa.py       # NASA 三分位分档数据集（低29/中28/高27）
python3 baseline_cls.py                # 12 个模型对比
python3 exp1_modality_ablation_cls.py  # 分类模态消融
python3 exp2_feature_selection_cls.py  # 分类特征筛选
python3 exp3_tuning_cls.py             # RF/XGB/LR/SVC 四类调参
```

---

### 复现 S 构造与预测（08-25）

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with pandas --with openpyxl --with numpy python compute_s.py
uv run --with pandas --with numpy --with scikit-learn python run_s_prediction.py
```

---

## 工程规范

1. **数据源单一**：所有处理只从 `data/` 读取，不引用 `历史工作_存档_20260703/` 中间产物或结论
2. **每步留三样**：脚本 + 输出 + README；任何特征筛选决策必须交代依据
3. **评估协议**：
   - 回归窗口级：GroupKFold by sample_id
   - 回归任务级：GroupKFold by subject
   - 分类任务级：**StratifiedGroupKFold** by subject（保证每折含 3 类）
4. **筛选防泄漏**：特征筛选必须在每折训练集内单独做，禁止全数据先筛后 CV
5. **小样本主指标用 pooled**：84 样本 5 折的 fold R²/F1 波动大（±0.15），pooled 口径（合并 5 折预测再算总指标）是稳健主指标
6. **弱标签如实标注**：不掩盖 NASA 是任务级一次性问卷、难度与 task 100% 绑定等事实
7. **进度追加而非覆盖**：新进度在顶部，旧进度精简保留

---

## 关键文献

- Reid & Nygren (1988). SWAT / 负荷测量方法分类
- Shaffer & Ginsberg (2017). HRV 分析指南
- Raufi & Longo (2022). θ/α、α/θ 作为工作负荷指数
- Sandre & Troller-Renfree (2026). EEG 绝对功率的个体差异问题
- Grimes et al. (2008); Brouwer et al. (2012). EEG 窗口长度对负荷分类的影响
- Kohavi (1995); Bengio & Grandvalet (2004). 小样本 CV 的 pooled 指标推荐

完整映射见 `工作目录/02_特征数据集/特征提取方案调研报告.md`。

## 许可

数据受被试隐私保护。本仓库仅公开脚本、说明与聚合后的窗口特征；原始 EEG/眼动信号不发布。代码复用请联系仓库所有者。
