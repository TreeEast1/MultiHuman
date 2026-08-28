# 下一阶段人因原始指标预报，再走定额 XGB 合成 S

当前主报 **V8 定额 27 列 + Ridge（标准化）**：观察任务前 50% 时，NASA pooled R² = **+0.264**（Early-only +0.129，完整观测 Oracle +0.528），合成 S R² = 0.966。S 含 70% 真实步骤，人因质量以 NASA 为准。

老师需求对应的实现：**先预报下一阶段的人因原始指标，再送进与正式口径相同的 27 维定额 XGB 算 NASA，最后按 0.70/0.30 合成 S**。不直接把 S 当回归目标。

- 方法与泄漏约定：[PROTOCOL.md](PROTOCOL.md)
- 结果解读（验收用）：[reports/RESULTS.md](reports/RESULTS.md)
- 数据诊断：[reports/00_diagnose/report.md](reports/00_diagnose/report.md)
- 全量对照表：[reports/COMPARISON.md](reports/COMPARISON.md)
- 选用一页纸：[reports/SELECTED.md](reports/SELECTED.md)
- 每个版本的预测明细：`reports/<version>/models/<model>/predictions.csv`

## 版本一览

| 版本 | 数据处理 | 有效？ | 目录 |
|---|---|---|---|
| V8 | 只预报下游 XGB 的 27 列 | **主报** NASA 0.264 | `reports/v8_quota27_space/` |
| V2 | 前段 264 → 全任务 264 | 并列，xgb_means 0.219 | `reports/v2_direct_full/` |
| V1 | 前段 264 → 后段 264，矩合并 | 部分有效 0.201 | `reports/v1_stage_late_pool/` |
| V6 | 观察比例 25–75% | 部署曲线 | `reports/v6_observe_ratio/` |
| V4 | 逐窗条件预报后段 | 仅 KNN 0.175 | `reports/v4_horizon_windows/` |
| V3 | 预报后段 66 均值再铺窗 | 否（毁掉 std） | `reports/v3_tile_late_mean/` |
| V5 | 窗级自回归滚动 | 否 | `reports/v5_ar_rollout_*/` |
| V7 | 下一任务 | 否（任务类型不同） | `reports/v7_next_task/` |

部署要点：观察不足一半必须预报；看到约 2/3 之后应停止预报、直接用已观察聚合（NASA R² 可到 0.36–0.40）。

## 复现

```bash
cd 工作目录/03_建模/forecast_next_stage
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \
    python diagnose_data.py
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \
    python run_matrix.py
```

窗口缓存写在 `cache/`（可删，脚本会重建）。对照结果全部留在 `reports/`，不互相覆盖。
