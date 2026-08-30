# 趋势预测：跨被试预报未来阶段人因，再走定额 XGB 合成 S

用已有被试的完整任务阶段数据训练，对未见过的新被试，依据其本场**已观察阶段**的脑电 / 心率 / 眼动 / 行为，预报本场**未来阶段**对应的 27 维人因指标，再送入与正式口径相同的定额浅树 XGB 得 NASA，按 0.70 / 0.30 合成 S。不直接把 S 当回归目标。

验证组为按被试划出的 **5 人、17 条任务**（被试 2、7、12、16、23）。合成 S 的 R² = **0.948**，MAE = **0.025**。

- 正式 Word 实验报告：[趋势预测_跨被试未来阶段人因与S_实验报告.docx](趋势预测_跨被试未来阶段人因与S_实验报告.docx)
- 正式 Word 结果摘要：[趋势预测_跨被试未来阶段人因与S_结果摘要.docx](趋势预测_跨被试未来阶段人因与S_结果摘要.docx)
- Markdown 底稿：[实验报告_趋势预测_跨被试未来阶段人因与S.md](实验报告_趋势预测_跨被试未来阶段人因与S.md)
- Transformer 轨迹 + Ridge S 双模块：[实验报告_Transformer轨迹与绩效双模块.md](实验报告_Transformer轨迹与绩效双模块.md)
- 方法协议：[PROTOCOL.md](PROTOCOL.md)

脚本：`run_matrix.py`（主路径 `v8_quota27_space` + `ridge_scaled`）。验证组 17 条明细：`reports/v8_quota27_space/models/ridge_scaled/predictions.csv`。

## 复现

```bash
cd 工作目录/03_建模/forecast_next_stage
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \
    python diagnose_data.py
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \
    python run_matrix.py
```

窗口缓存写在 `cache/`（可删，脚本会重建）。
