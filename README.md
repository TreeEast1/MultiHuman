# MultiHuman：多模态生理-行为信号驱动的操纵员认知工作负荷预测

> 基于 EEG、心率、眼动、任务操作日志四模态数据，两种预测目标并存：
> - **回归**：NASA-TLX 加权总分（连续值 1.33–7.80）
> - **分类**：低 / 中 / 高 3 档难度
>
> **进度日志格式**，最新在顶部。

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
└── classification_task_level/         ← 任务级 3 分类（本次新增）
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
│       └── classification_task_level/ ← 任务级 3 分类
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
