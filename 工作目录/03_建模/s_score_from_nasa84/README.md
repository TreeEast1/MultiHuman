# 预测版绩效 S（XGB 折外 NASA）

现行最佳 NASA 回归（MI Top-30 + XGB）按被试 5 折折外预测，再得到 `S_xgb`。进度叙述见仓库根目录 `README.md` 顶部「2026-08-25」。

## 折外精度

| 目标 | MAE | R² | Spearman |
|---|---:|---:|---:|
| NASA | 0.868 | +0.521 | 0.745 |
| 预测版 S | 0.052 | +0.838 | 0.911 |

`S_xgb` 范围 0.251–0.876，均值 0.571。

## 复现

```bash
cd 工作目录/03_建模/s_score_from_nasa84
uv run --with xgboost --with pandas --with numpy --with scikit-learn python compute_s_from_xgb_nasa.py
```

明细：`output_from_xgb_nasa/s_from_xgb_nasa.csv`
