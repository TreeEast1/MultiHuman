# 最终特征数据集（30s窗口 / 5s步长 / 含眨眼与EEG个体归一）

**生成日期**：2026-07-06
**生成脚本**：`工作目录/01_预处理/scripts/step3_blink_and_eeg_zscore.py`
**上游输入**：`工作目录/01_预处理/output_30s_step5s_with_log/` (step2产物，含日志特征)

## 一、本轮新增/修复的两件事

### 1. EEG 被试内 z-score 标准化（修复历史遗漏）

对全部 28 个EEG特征（4脑区 × 5频段功率 + 4脑区 × 2比值）按被试做组内标准化：

```
x_z = (x - subject_mean(x)) / subject_std(x)
```

新增列：`{原列名}_z_within_subject`，共28列。原始列保留不动，方便对比。

**效果实测**：

| 特征 | 原始个体间方差占比 | z-score后 |
|---|---|---|
| eeg_frontal_alpha_power | 69.7% | **0.0%** |
| eeg_frontal_beta_power | 84.0% | **0.0%** |
| eeg_central_alpha_power | 70.8% | **0.0%** |
| eeg_frontal_theta_alpha | 52.4% | **0.0%** |

个体差异全部被剥离，剩下的才是"负荷差异"和"任务差异"信号。

### 2. 眼动新增眨眼特征（补足遗漏）

眼动原始tsv没有直接的blink标签，但`Eye movement type == EyesNotFound` 表示追踪器暂时看不到眼睛，主要成因就是眨眼。100Hz采样率下每个EyesNotFound连续段就是一次疑似眨眼事件。

**过滤规则**：只有段时长 `50ms ≤ dur ≤ 500ms` 才算作疑似眨眼（<50ms多为追踪抖动，>500ms多为长时间遮挡/低头/离屏，都不是自发眨眼）。

新增列（每窗8个）：

| 列名 | 含义 |
|---|---|
| `blink_count_raw` | 所有 EyesNotFound 段数（未过滤，供审计） |
| `blink_total_ms_raw` | 所有 EyesNotFound 段的总毫秒数（未过滤） |
| `blink_count` | **过滤后**疑似眨眼次数（50-500ms） |
| `blink_rate_per_min` | 眨眼频率（次/分钟） |
| `blink_duration_mean_ms` | 眨眼平均持续时长 |
| `blink_duration_std_ms` | 眨眼时长标准差 |
| `blink_duration_median_ms` | 眨眼时长中位数 |
| `blink_total_duration_ratio` | 眨眼总时长 / 窗口时长 |

**实测统计**：眨眼频率中位数10次/分钟（成人静息15-20次/分钟，任务态下降到10左右属正常范围）；眨眼平均时长85.5ms（100Hz采样率下的时间分辨率10ms，与文献自发眨眼时长100-400ms的下沿一致）。

**归属规则**：一次眨眼归属于包含它中点的那个窗口，防止跨窗口重复计数。

## 二、最终数据集规格

| 项目 | 值 |
|---|---|
| 样本文件数 | 84 |
| 总窗口数 | 12258 |
| 每窗特征列数 | 132 |
| 元信息列 | 14（sample_id/subject/task/window_id等） |
| 心率特征 | 5 |
| EEG原始特征 | 28 |
| EEG被试内z-score | 28 |
| 眼动瞳孔/注视 | 7 |
| 眼动AOI | 9 |
| **眨眼（新）** | **8** |
| 日志特征 | 32（16 win + 16 cum，两套并列） |
| NASA标签（任务级） | 1 |

## 三、目录结构

```
output_30s_step5s_final/
├── subject_XX_task_Y.csv   ← 84个样本文件，每文件=一被试一任务的所有窗口
├── index.csv               ← 索引：sample_id / subject / task / 窗口数 / NASA标签
├── blink_audit.csv         ← 眨眼审计：每(subject,task) 的段数/中位时长
└── README.md               ← 本文件
```

## 四、下一步：建模数据准备

数据整理已完成。下一步建模需要：

1. **合并成建模表**：把84个CSV按需要拼接成宽表（如需按sample_id分组用GroupKFold防泄漏）
2. **NASA标签核对**：任务级标签，同一sample_id内的所有窗口共享同一NASA值
3. **特征选择实验**：
   - 主用z-score版EEG特征，原始EEG作为对照
   - blink 8个特征全部纳入（若相关性冗余，PCA或先验筛掉部分）
   - 日志 win/cum 通过消融决定用哪个
4. **建模基线**：GroupKFold按sample_id分组 → RF/XGB/GPR baseline → 三篇文献泄漏对比

## 五、脚本可复现命令

```bash
cd 工作目录/01_预处理/scripts
python3 step3_blink_and_eeg_zscore.py \
  --input-dir ../output_30s_step5s_with_log \
  --eye-dir ../../../data/05_眼动/raw_tsv \
  --output-dir ../output_30s_step5s_final \
  --blink-min-ms 50 --blink-max-ms 500
```
