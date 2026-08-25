# XGB 折外预测 NASA → 预测版 S

协议：84 条、264 维、5 折 GroupKFold by subject；折内 MI Top-30 + XGB（depth=2, lr=0.02, λ=2, n=500）。

## 精度

| 目标 | MAE | R² | Spearman |
|---|---:|---:|---:|
| NASA（折外） | 0.868 | +0.521 | 0.745 |
| S_xgb vs 真值 S | 0.052 | +0.838 | 0.911 |

- `S_xgb` 范围 [0.251, 0.876]，均值 0.571
- `ΔS = 0.06 × (NASA_XGB − NASA 真值)`

明细：`s_from_xgb_nasa.csv`
