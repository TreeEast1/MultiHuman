# 下一阶段人因预报对照总表

生成时间：2026-08-29 03:28:28

主指标是 **NASA pooled R²**（人因预报是否有用）。S 的 R² 含 70% 真实步骤，只作并列。

## 当前有效路径（按 NASA R²）

**v8_quota27_space / ridge_scaled**：NASA R² = +0.264，MAE = 1.056；S R² = +0.966。同折 Oracle NASA R² = +0.528，Early-only = +0.129。

该路径超过 Early-only 且 NASA R²>0，可作为验收主结果。

## 全量对照

| 版本 | 模型 | n | NASA R² | NASA MAE | S R² | Oracle NASA | Early NASA |
|---|---|---:|---:|---:|---:|---:|---:|
| v1_stage_late_pool | pls6 | 82 | +0.201 | 1.103 | +0.963 | +0.528 | +0.129 |
| v1_stage_late_pool | extra_trees | 82 | +0.201 | 1.070 | +0.963 | +0.528 | +0.129 |
| v1_stage_late_pool | knn5 | 82 | +0.196 | 1.074 | +0.963 | +0.528 | +0.129 |
| v1_stage_late_pool | ridge_scaled | 82 | +0.173 | 1.168 | +0.962 | +0.528 | +0.129 |
| v1_stage_late_pool | dummy_mean | 82 | +0.151 | 1.107 | +0.961 | +0.528 | +0.129 |
| v1_stage_late_pool | persist_early | 82 | +0.150 | 1.116 | +0.961 | +0.528 | +0.129 |
| v1_stage_late_pool | ridge | 82 | +0.124 | 1.133 | +0.959 | +0.528 | +0.129 |
| v1_stage_late_pool | persist_lastwin_tile | 82 | -0.148 | 1.379 | +0.947 | +0.528 | +0.129 |
| v2_direct_full | xgb_means | 82 | +0.219 | 1.120 | +0.964 | +0.528 | +0.129 |
| v2_direct_full | ridge_scaled | 82 | +0.206 | 1.119 | +0.963 | +0.528 | +0.129 |
| v2_direct_full | ridge_residual | 82 | +0.197 | 1.124 | +0.963 | +0.528 | +0.129 |
| v2_direct_full | persist_early | 82 | +0.129 | 1.163 | +0.960 | +0.528 | +0.129 |
| v2_direct_full | pls6 | 82 | -0.044 | 1.281 | +0.952 | +0.528 | +0.129 |
| v2_direct_full | ridge | 82 | -0.055 | 1.265 | +0.951 | +0.528 | +0.129 |
| v2_direct_full | persist_lastwin_tile | 82 | -0.148 | 1.379 | +0.947 | +0.528 | +0.129 |
| v2_direct_full | knn5 | 82 | -0.159 | 1.277 | +0.946 | +0.528 | +0.129 |
| v2_direct_full | dummy_mean | 82 | -0.213 | 1.323 | +0.944 | +0.528 | +0.129 |
| v2_direct_full | extra_trees | 82 | -0.215 | 1.331 | +0.944 | +0.528 | +0.129 |
| v3_tile_late_mean | persist_lastwin_tile | 82 | -0.148 | 1.379 | +0.947 | +0.528 | +0.129 |
| v3_tile_late_mean | knn5 | 82 | -0.262 | 1.421 | +0.941 | +0.528 | +0.129 |
| v3_tile_late_mean | ridge | 82 | -0.264 | 1.397 | +0.941 | +0.528 | +0.129 |
| v3_tile_late_mean | ridge_scaled | 82 | -0.287 | 1.446 | +0.940 | +0.528 | +0.129 |
| v3_tile_late_mean | dummy_mean | 82 | -0.293 | 1.453 | +0.940 | +0.528 | +0.129 |
| v3_tile_late_mean | dummy_mean_tile | 82 | -0.293 | 1.453 | +0.940 | +0.528 | +0.129 |
| v3_tile_late_mean | pls6 | 82 | -0.354 | 1.491 | +0.937 | +0.528 | +0.129 |
| v3_tile_late_mean | extra_trees | 82 | -0.376 | 1.507 | +0.936 | +0.528 | +0.129 |
| v4_horizon_windows | knn5 | 82 | +0.175 | 1.113 | +0.962 | +0.528 | +0.129 |
| v4_horizon_windows | ridge_scaled | 82 | -0.300 | 1.427 | +0.940 | +0.528 | +0.129 |
| v4_horizon_windows | dummy_mean | 82 | -0.358 | 1.466 | +0.937 | +0.528 | +0.129 |
| v4_horizon_windows | extra_trees | 82 | -0.412 | 1.525 | +0.934 | +0.528 | +0.129 |
| v4_horizon_windows | pls6 | 82 | -0.436 | 1.520 | +0.933 | +0.528 | +0.129 |
| v5_ar_rollout_hop1_overlap | ridge_scaled | 82 | -0.334 | 1.451 | +0.938 | +0.528 | +0.129 |
| v5_ar_rollout_hop1_overlap | extra_trees | 82 | -0.368 | 1.477 | +0.936 | +0.528 | +0.129 |
| v5_ar_rollout_hop6_30s | ridge_scaled | 82 | -0.334 | 1.463 | +0.938 | +0.528 | +0.129 |
| v5_ar_rollout_hop6_30s | extra_trees | 82 | -0.358 | 1.472 | +0.937 | +0.528 | +0.129 |
| v6_observe_ratio | r75_persist_early | 82 | +0.400 | 0.957 | +0.972 | +0.528 | +0.400 |
| v6_observe_ratio | r67_persist_early | 82 | +0.356 | 1.011 | +0.970 | +0.528 | +0.356 |
| v6_observe_ratio | r75_ridge_scaled | 82 | +0.336 | 1.051 | +0.969 | +0.528 | +0.400 |
| v6_observe_ratio | r67_ridge_scaled | 82 | +0.279 | 1.062 | +0.967 | +0.528 | +0.356 |
| v6_observe_ratio | r50_ridge_scaled | 82 | +0.206 | 1.119 | +0.963 | +0.528 | +0.129 |
| v6_observe_ratio | r33_ridge_scaled | 82 | +0.139 | 1.114 | +0.960 | +0.528 | -0.087 |
| v6_observe_ratio | r50_persist_early | 82 | +0.129 | 1.163 | +0.960 | +0.528 | +0.129 |
| v6_observe_ratio | r25_extra_trees | 82 | +0.072 | 1.140 | +0.957 | +0.528 | -0.165 |
| v6_observe_ratio | r25_ridge_scaled | 82 | +0.042 | 1.210 | +0.956 | +0.528 | -0.165 |
| v6_observe_ratio | r75_extra_trees | 82 | +0.018 | 1.210 | +0.954 | +0.528 | +0.400 |
| v6_observe_ratio | r33_extra_trees | 82 | -0.029 | 1.198 | +0.952 | +0.528 | -0.087 |
| v6_observe_ratio | r33_persist_early | 82 | -0.087 | 1.275 | +0.950 | +0.528 | -0.087 |
| v6_observe_ratio | r67_extra_trees | 82 | -0.114 | 1.299 | +0.948 | +0.528 | +0.356 |
| v6_observe_ratio | r25_persist_early | 82 | -0.165 | 1.347 | +0.946 | +0.528 | -0.165 |
| v6_observe_ratio | r50_extra_trees | 82 | -0.215 | 1.331 | +0.944 | +0.528 | +0.129 |
| v7_next_task | dummy_mean | 58 | +0.039 | 1.194 | +0.956 | +0.426 | -0.630 |
| v7_next_task | extra_trees | 58 | -0.050 | 1.214 | +0.952 | +0.426 | -0.630 |
| v7_next_task | ridge_scaled | 58 | -0.175 | 1.305 | +0.946 | +0.426 | -0.630 |
| v7_next_task | pls6 | 58 | -0.296 | 1.379 | +0.940 | +0.426 | -0.630 |
| v7_next_task | knn5 | 58 | -0.386 | 1.335 | +0.936 | +0.426 | -0.630 |
| v7_next_task | persist_prev_task | 58 | -0.630 | 1.488 | +0.925 | +0.426 | -0.630 |
| v8_quota27_space | ridge_scaled | 82 | +0.264 | 1.056 | +0.966 | +0.528 | +0.129 |
| v8_quota27_space | persist_early | 82 | +0.129 | 1.163 | +0.960 | +0.528 | +0.129 |
| v8_quota27_space | pls6 | 82 | +0.117 | 1.138 | +0.959 | +0.528 | +0.129 |
| v8_quota27_space | extra_trees | 82 | +0.018 | 1.229 | +0.954 | +0.528 | +0.129 |
