# 预测版绩效 S

进度见仓库根目录 `README.md` 顶部「2026-08-25」。

先预测 NASA，再和真实步骤合成 S。现在对外用 **步骤 0.7 + NASA 0.3**（R² 0.98）。

| 步骤 : NASA | 预测 S 范围 | 均值 | R² |
|---|---|---:|---|
| 0.4 : 0.6 | 0.25–0.88 | 0.57 | 0.84 |
| 0.5 : 0.5 | 0.23–0.90 | 0.59 | 0.91 |
| 0.6 : 0.4 | 0.22–0.92 | 0.60 | 0.95 |
| **0.7 : 0.3** | **0.19–0.94** | **0.62** | **0.98** |

NASA 连续分 R² 约 0.52；低/中/高三档 F1 约 0.81。改 S 权重不会改这两行。

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with xgboost --with pandas --with numpy --with scikit-learn python compute_s_from_xgb_nasa.py
```

明细：`output_from_xgb_nasa/s_from_xgb_nasa.csv`
