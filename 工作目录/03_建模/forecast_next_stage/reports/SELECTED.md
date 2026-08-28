# 验收选用

选择规则：任务内切段、观察 50% 时，NASA pooled R² 最高，且必须超过 Early-only（0.129）。S R² 只并列，不单独当选用依据。

## 主报

- **路径**：`v8_quota27_space` + `ridge_scaled`
- **NASA R² = +0.264**（Early-only +0.129，Oracle +0.528）
- **S R² = +0.966**（步骤 0.70 为真值；正式完整观测口径是 0.979）
- 明细：`reports/v8_quota27_space/models/ridge_scaled/`
- 解释：只预报下游 XGB 真正用到的 27 列，比预报全部 264 维更稳。

并列可报：`v2_direct_full` + `xgb_means`（NASA R² +0.219），实现更简单（预报 66 个均值，其余沿用前段）。

## 部署时不要用错版本

| 场景 | 用什么 |
|---|---|
| 任务约一半、要补全人因再算 S | **V8 Ridge** |
| 已经看到 ≥2/3 任务 | **不要预报**，用已观察段聚合（V6 persist，NASA R² 0.36–0.40） |
| 铺后段均值 / 窗级自回归 / 把上一任务当下一任务 | 无效，已保留对照但不选用 |

其余路径全部留在 `reports/`，不删除。解读见 [RESULTS.md](RESULTS.md)。
