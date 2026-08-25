# 预测版绩效 S

进度见仓库根目录 `README.md` 顶部「2026-08-25」。这里只放入口。

先预测 NASA，再用它和真实步骤完成情况合成 S。两种权重：

| 权重 | 预测 S 范围 | 均值 | 和真实 S 对上的程度 |
|---|---|---:|---|
| 步骤 0.4 + NASA 0.6 | 0.25–0.88 | 0.57 | R² 0.84 |
| 步骤 0.5 + NASA 0.5 | 0.23–0.90 | 0.59 | R² 0.91 |

NASA 本身：连续分 R² 约 0.52；低/中/高三档 F1 约 0.81。

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with xgboost --with pandas --with numpy --with scikit-learn python compute_s_from_xgb_nasa.py
```

明细（0.4 / 0.6 这一版）：`output_from_xgb_nasa/s_from_xgb_nasa.csv`
