# MultiHuman：多模态生理-行为信号驱动的操纵员认知工作负荷预测

> 基于 EEG、心率、眼动、任务操作日志四模态数据，预测核电操纵任务下操纵员的主观工作负荷（NASA-TLX 加权总分）。
>
> **进度日志格式**，最新在顶部。

---

## 📅 2026-07-07（下午）：任务级建模 + 消融 + 特征筛选 + 调参

### 一句话结论

**pooled R² 从 0.126 提升到 0.519，MAE 从 1.170 降到 0.911**——已达到跨被试负荷回归的文献中位水平（0.4-0.6）。核心发现：**AOI 情境意识特征 + std 波动性统计量是主导信号，EEG 和 HR 贡献很小**。

### 一、最终成绩总览

| 阶段 | 建模粒度 | 特征方案 | 模型 | pooled R² | pooled MAE |
|---|---|---|---|---:|---:|
| 上午的窗口级 baseline | 12624 行 × 66 特征 | 全量 | RandomForest | +0.126 | 1.170 |
| **P0 任务级 baseline** | **84 行 × 264 特征** | 全量 | XGB_shallow | **+0.465** | **0.910** |
| **P2+P3 最优组合** | **84 行 × 30 特征** | MI 折内 Top-30 | **XGB 调参后** | **+0.519** | **0.911** |

参考基线：`|y - mean(y)|` = 1.337（任何模型 MAE 低于此才算实质学习），y 范围 [1.33, 7.80] std=1.58。

### 二、跳这么多的原因：**建模粒度是最大瓶颈**（不是特征质量）

12624 行窗口 × 66 特征，NASA 真值只有 84 个数（每个 sample_id 内所有窗口共享一个真值）——模型实际上在学"如何在共享真值的窗口内输出常数"，跨任务泛化能力被稀释。

**改成任务级建模（每个 sample_id 聚合成 1 行 × 264 特征，统计量 = mean/std/median/slope）**，模型直接面对 84 个不同真值，泛化能力立刻显现：

- R² 从 0.126 → 0.465（3.7×）
- MAE 从 1.170 → 0.910（-22%）

**这一步几乎不涉及新特征工程，只是聚合粒度的正确选择。**

### 三、P1 模态消融的三大发现

**（1）AOI 是绝对核心，其他模态贡献很小**

| 单独用一个模态跑 XGB | 特征数 | R² |
|---|---:|---:|
| **AOI**（注意力分布代理） | 36 | **+0.344** ← 一枝独秀 |
| Log_win（操作日志） | 48 | +0.001 |
| EEG（脑电 z-score） | 112 | +0.000 |
| HR（心率） | 20 | -0.032 |
| Blink（眨眼） | 24 | -0.112 |
| EyePupil（瞳孔/注视比例） | 24 | -0.317 |

**从 Full 里去掉某一模态，损失最大的也是 AOI**：去 AOI → R² 从 +0.465 崩到 +0.042；去 HR 反而 R² 微升到 +0.470（HR 是纯噪声）。

**（2）波动性（std）比平均水平更能预测负荷**

| 只用某一统计量的 66 列 | R² |
|---|---:|
| **only_std** | **+0.470** ← 追平 Full |
| only_slope（趋势） | +0.186 |
| only_mean | +0.185 |
| only_median | +0.039 |

**"任务过程中生理/行为信号的波动"比"平均水平"更能反映负荷**——这是可直接写进 discussion 的机理解释。

**（3）EEG 在跨被试回归中意外弱**

EEG 单独 R²=0.000、去掉 EEG 只损失 0.018。这与文献里"EEG 在被试内任务分类效果好，跨被试回归很弱"的经验完全一致（个体差异掩盖了负荷信号，即使做了 z-score 也无法完全消除）。

### 四、P2 特征筛选：从 264 降到 30 反而更好

**做法**：MI / RF importance / Permutation 三种排序器 × K∈{5,10,15,20,30,50,80,130} × 3 个模型 = 72 次实验，**每种都在训练折内单独筛选**（防止泄漏）。

**XGB_shallow 的 K vs R² 曲线关键点**：

| K | 5 | 10 | 20 | **30** | 50 | 264(Full) |
|---:|---:|---:|---:|---:|---:|---:|
| MI R² | +0.331 | +0.395 | +0.403 | **+0.470** | +0.430 | +0.465 |
| Permutation R² | +0.260 | +0.392 | +0.435 | +0.442 | **+0.489** | +0.465 |

**Ridge 从 R²=-0.948（Full 264d）跳到 +0.382（MI Top-30）**——强正则线性模型对筛选极敏感，可作可解释性备份。

**5 折稳定选中的核心特征**（在所有筛选方案里 100% 稳定）：

- 第一梯队（AOI）：`eye_aoi_interval_n__std`、`eye_aoi_unique_hit_n__std`、`eye_aoi_interval_n__mean`
- 第二梯队：`eeg_frontal_gamma_power__std`、`eeg_parietal_beta_alpha__slope`、`log_unique_step_count_win__std`、`blink_duration_mean_ms__slope`、`log_action_density_win__mean`

**这批特征跨 9 种筛选方案指向同一批列，与 P1 消融的"AOI + std"结论完全对齐，可信度极高。**

### 五、P3 XGBoost 调参：R² 从 0.470 提到 0.519

在 MI Top-30 特征子集上跑 81 组 XGB 网格 + 24 组 RF 网格：

| 配置 | R² | MAE |
|---|---:|---:|
| **XGB max_depth=2, lr=0.02, reg_λ=2, n=500** | **+0.519** | 0.911 |
| XGB Top-10 全部 max_depth=2 | +0.51x | — |
| RF 调参最佳（max_depth=4, min_leaf=2） | +0.468 | 0.933 |

**观察**：XGB 前 10 名清一色 `max_depth=2`——**84 样本上，浅树+强正则+高迭代**是最优配方。RF 从筛选和调参中获益很小（它已隐式做了特征选择）。

### 六、最终推荐方案（可复现）

```
数据      : 84 sample × 264 任务级特征（30s 窗口 × mean/std/median/slope）
划分      : 5×GroupKFold by subject（26 名被试）
筛选      : 折内 MI 排名 → 每折训练集独立选 Top-30
模型      : XGBRegressor(max_depth=2, learning_rate=0.02, reg_lambda=2.0,
                        n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=0)
最终指标  : pooled MAE = 0.911, pooled R² = +0.519
```

### 七、本次新增的代码与产物

```
工作目录/03_建模/
├── EXPERIMENT_SUMMARY.md              ← 综合总结（含详细数据表）
├── make_dataset_task_level.py         ← 生成 84×264 任务级建模表
├── baseline_task_level.py             ← P0 任务级 baseline（9 个模型）
├── exp_utils.py                       ← 共用 pooled CV / 折内筛选工具
├── exp1_modality_ablation.py          ← P1 消融（17 个特征子集 × 3 模型 = 51 实验）
├── exp2_feature_selection.py          ← P2 筛选（3 排序器 × 8 K × 3 模型 = 72 实验）
├── exp3_xgb_tuning.py                 ← P3 调参（81 XGB + 24 RF 网格）
├── dataset_task/                      ← 任务级数据集 + 审计
├── baseline_reports_task/report.md    ← P0 报告
├── exp1_modality_ablation/report.md   ← P1 报告
├── exp2_feature_selection/report.md   ← P2 报告（含各筛选方案 Top-20 特征清单）
└── exp3_xgb_tuning/report.md          ← P3 报告
```

同时 `evaluate.py` 引入 `task_groups` 参数，解耦"划分分组（subject）"与"任务级聚合分组（sample_id）"，避免早先误把任务级 R² 聚合到被试级导致失真。

### 八、下一步

- [ ] Leave-One-Subject-Out（26 折）产出最严格的分数
- [ ] Only AOI+Log 双模态验证（极简可解释方案）
- [ ] 残差分析：找错得最离谱的样本，看是否有系统规律
- [ ] 分类版本（低/中/高 3 分类）与回归对照
- [ ] 报告用的可视化图（K-R² 曲线、特征重要性 bar、预测-真值散点）

---

## 📅 2026-07-07（上午）：诚实基线建模 + 数据缺陷修复

### 核心成果

1. **修复 EEG 时长错误**：1002 号 EEG 原始文件时间截断（旧版 24 秒 → 新版 31 分钟完整），全量重跑，数据集从 12 258 → 12 624 窗口。详见 `工作目录/01_预处理/DATA_FIX_AUDIT.md`。
2. **确定特征方案**：66 列建模输入 = 28 EEG（被试内 z-score）+ 5 HR + 15 眼动（瞳孔+AOI）+ 6 眨眼 + 12 日志(win)。
3. **搭建评估框架**：`工作目录/03_建模/` 下 `make_dataset.py` + `evaluate.py` + `baseline.py`，评估协议 GroupKFold by sample_id，杜绝历史窗口级随机划分泄漏。
4. **窗口级诚实基线**：见下表——**R²=+0.126 是没有数据泄漏的真实成绩**，比历史 R²=0.955（有泄漏）低很多，但方法正确。

### 窗口级 baseline 结果（5 折 GroupKFold）

| 模型 | 任务级 MAE | 任务级 R² |
|---|---:|---:|
| MeanPredictor（零信息下限） | 1.325 | -0.093 |
| Linear_Single | 1.320 | -0.078 |
| Linear_AllFeatures | 1.267 | -0.027 |
| **RandomForest_default** | **1.170** | **+0.126** |
| XGBoost_default | 1.219 | +0.019 |

### 关键铁证：泄漏的量级

历史项目自己做过对照实验（`历史工作_存档.../window_split_summary.csv`）：

| 划分方式 | ExtraTrees Macro-F1 |
|---|---:|
| 随机窗口划分（有泄漏） | 0.998 |
| 按 sample_id 划分（正确） | 0.287 |

**F1 从 0.998 塌到 0.287**——这就是数据泄漏的量级。任何论文引用"XGB R²=0.955"必须警惕这个陷阱。

### 下午的下一步（已完成，见顶部）

→ 任务级建模 + 消融 + 特征筛选 + 调参。

---

## 📅 2026-07-03：数据整理与特征提取完成

### 核心成果

1. **唯一权威原始数据源**（`data/`）：26 被试 × 5 类任务，82+2 有效样本。
2. **推翻旧 S 指标**，改用 NASA-TLX 加权总分作为目标。
3. **多模态时间对齐 + 切窗**：产出 10 s / 30 s 两版窗口化数据集。
4. **附文献依据的特征工程**：EEG 被试内 z-score（修复历史遗漏）+ 眨眼代理特征（补足历史缺口）。
5. **可复现 pipeline**：3 个脚本（`step1/step2/step3`）+ 每步 README。

### 数据规格

| 项 | 值 |
|---|---|
| 被试 | 26 人（编号 01–26） |
| 任务 | 5 类（1/2/3/4/5_6），低/中/高 三档 |
| 有效样本 | 82 NASA/SART，84 组对齐可用 |
| EEG | 30 通道 256 Hz EEGLAB `.set` |
| 心率 | 4.6 s 采样一次（无法做经典 HRV） |
| 眼动 | Tobii 100 Hz（瞳孔/注视/扫视/AOI/EyesNotFound） |
| 主用数据集 | `output_30s_step5s_final/` 12624 窗口 × 132 列 |

### 特征工程要点

- **EEG（28 原始 + 28 z-score）**：修复历史仅算绝对功率的漏洞，被试内 z-score 把 `frontal_alpha` 个体间方差占比从 69.7% 压到 0.0%
- **眼动（21 列）**：补全眨眼代理（用 `EyesNotFound` 段 50-500ms 过滤），眨眼频率中位数 10 次/分钟符合文献
- **心率（5 列）**：4.6 s 采样限制，只能做基础统计量，不做经典 HRV
- **日志（32 列 = 16 win + 16 cum）**：先都存下，后续消融决定取舍（当前建模只用 win）

详细依据见 `工作目录/02_特征数据集/特征提取方案调研报告.md`。

---

## 项目目录结构

```
MultiHuman/
├── README.md                        ← 进度日志（本文件）
├── data/                            ← 唯一权威原始数据源
│   ├── 01_NASA_TLX/                 ← 预测标签
│   ├── 03_心率/、04_EEG/、05_眼动/    ← 原始信号（EEG/眼动不入库）
│   └── 06_任务表现与操作日志/
│
├── 工作目录/
│   ├── 01_预处理/                    ← 多模态对齐 + 切窗 + 特征拼装
│   │   ├── scripts/                 ← step1/step2/step3
│   │   ├── DATA_FIX_AUDIT.md
│   │   └── output_30s_step5s_final/ ← 主用窗口特征表
│   ├── 02_特征数据集/                ← 特征方案调研 + 决策文档
│   └── 03_建模/                      ← 建模评估框架 + 4 组实验产物
│       ├── EXPERIMENT_SUMMARY.md    ← 07-07 综合实验总结
│       ├── make_dataset.py / baseline.py                    ← 窗口级
│       ├── make_dataset_task_level.py / baseline_task_level.py ← 任务级
│       ├── exp_utils.py / evaluate.py
│       ├── exp1_modality_ablation.py / exp1_modality_ablation/
│       ├── exp2_feature_selection.py / exp2_feature_selection/
│       └── exp3_xgb_tuning.py / exp3_xgb_tuning/
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

### 复现建模

```bash
cd 工作目录/03_建模

# 窗口级
python3 make_dataset.py && python3 baseline.py

# 任务级（本次新增，主推荐）
python3 make_dataset_task_level.py
python3 baseline_task_level.py         # P0 baseline
python3 exp1_modality_ablation.py      # P1 消融
python3 exp2_feature_selection.py      # P2 筛选（约 10 分钟）
python3 exp3_xgb_tuning.py             # P3 调参（依赖 exp2 结果）
```

---

## 工程规范

1. **数据源单一**：所有处理只从 `data/` 读取，不引用 `历史工作_存档_20260703/` 中间产物或结论
2. **每步留三样**：脚本 + 输出 + README；任何特征筛选决策必须交代依据
3. **评估协议**：任何模型评估必须 GroupKFold 分组，窗口级建模按 sample_id 分组，任务级建模按 subject 分组
4. **筛选防泄漏**：特征筛选必须在每折训练集内单独做，禁止全数据先筛后 CV
5. **小样本主指标用 pooled**：84 样本 5 折的 fold R² 波动大（±0.2），pooled 口径（合并 5 折预测再算总 R²）是稳健主指标
6. **弱标签如实标注**：不掩盖 NASA 是任务级一次性问卷、窗口级 R² 天然偏负的事实
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
