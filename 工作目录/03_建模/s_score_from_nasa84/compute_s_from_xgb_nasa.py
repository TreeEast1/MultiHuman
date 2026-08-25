#!/usr/bin/env python3
"""用 XGB 折外预测的 NASA 再算 S。

不是把问卷 NASA 换成模型分再训练一遍，而是：
    1. 复现现行最佳 NASA 回归（MI Top-30 + XGB，exp3 第 1 名）
    2. 取按被试 GroupKFold 的折外预测 y_nasa_xgb
    3. 步骤分仍用问卷实验那张序列表（与 compute_s.py 同一列）
    4. 同一条公式：
           S_xgb = 0.40 × weighted_step + 0.60 × (1 − y_nasa_xgb / 10)

折外预测：每个样本的 NASA 预测来自没见过该被试的折，避免把训练集拟合值当成新标签。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS, pooled_cv_with_selection  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
S_TABLE = HERE / "output" / "s_score_84samples.csv"
OUT_DIR = HERE / "output_from_xgb_nasa"
N_SPLITS = 5
TOP_K = 30
STEP_WEIGHT_IN_S = 0.40
XGB_CFG = dict(
    max_depth=2,
    learning_rate=0.02,
    reg_lambda=2.0,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=0,
)


def _enable_xgboost_on_macos() -> None:
    """本机没有 brew libomp 时，先把 sklearn 自带的 libomp 载入进程。"""
    import ctypes
    from pathlib import Path

    import sklearn

    omp_lib = Path(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp_lib.exists():
        ctypes.CDLL(str(omp_lib), mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    _enable_xgboost_on_macos()
    from xgboost import XGBRegressor

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)

    if list(s_table["sample_id"]) != list(samples):
        # 允许顺序不同，按 sample_id 对齐
        s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    if not np.allclose(s_table["y_nasa"].to_numpy(), y, atol=1e-8):
        raise RuntimeError("s_score 表的 y_nasa 与回归标签对不齐")
    if not np.array_equal(s_table["subject"].to_numpy(), groups):
        raise RuntimeError("s_score 表的 subject 与 groups_task 对不齐")

    print(f"[xgb→S] X={X.shape}  复现 MI Top-{TOP_K} + XGB {XGB_CFG}")
    res, selected = pooled_cv_with_selection(
        lambda: XGBRegressor(**XGB_CFG),
        X,
        y,
        groups,
        N_SPLITS,
        top_k=TOP_K,
        ranker=RANKERS["MI"],
        preprocessor=None,
        name="XGB_MI30_oof",
    )
    print(f"[xgb→S] NASA 折外  MAE={res.pooled_mae:.3f}  R²={res.pooled_r2:+.3f}  "
          f"（对齐目标：MAE≈0.911  R²≈+0.519）")

    y_hat = res.y_pred_pooled
    step = s_table["weighted_step_score"].to_numpy(dtype=float)
    nasa_rev_true = 1.0 - y / 10.0
    nasa_rev_hat = 1.0 - y_hat / 10.0
    s_true = STEP_WEIGHT_IN_S * step + (1.0 - STEP_WEIGHT_IN_S) * nasa_rev_true
    s_hat = STEP_WEIGHT_IN_S * step + (1.0 - STEP_WEIGHT_IN_S) * nasa_rev_hat

    out = s_table.copy()
    out["y_nasa_true"] = y
    out["y_nasa_xgb"] = y_hat
    out["nasa_reverse_true"] = nasa_rev_true
    out["nasa_reverse_xgb"] = nasa_rev_hat
    out["S_true"] = s_true
    out["S_xgb"] = s_hat
    out["S_xgb_minus_S_true"] = s_hat - s_true
    # 0.60 × (true_rev − pred_rev) = 0.06 × (y_hat − y_true)
    out["delta_from_nasa_error"] = 0.06 * (y_hat - y)

    keep = [
        "sample_id", "subject", "task", "task_difficulty",
        "y_nasa_true", "y_nasa_xgb",
        "weighted_step_score",
        "nasa_reverse_true", "nasa_reverse_xgb",
        "S_true", "S_xgb", "S_xgb_minus_S_true",
    ]
    out[keep].to_csv(OUT_DIR / "s_from_xgb_nasa.csv", index=False, encoding="utf-8-sig")
    np.save(OUT_DIR / "y_nasa_xgb_oof.npy", y_hat)
    np.save(OUT_DIR / "y_s_from_xgb.npy", s_hat)

    rho_nasa = float(pd.Series(y).rank().corr(pd.Series(y_hat).rank()))
    rho_s = float(pd.Series(s_true).rank().corr(pd.Series(s_hat).rank()))
    mae_s = float(np.mean(np.abs(s_hat - s_true)))
    r2_s = float(1.0 - np.sum((s_hat - s_true) ** 2) / np.sum((s_true - s_true.mean()) ** 2))

    payload = {
        "nasa_model": "MI Top-30 + XGB (exp3 best)",
        "xgb_cfg": XGB_CFG,
        "cv": "GroupKFold by subject, 5 folds, OOF",
        "formula": "S = 0.40 * weighted_step + 0.60 * (1 - y_nasa / 10)",
        "nasa_oof_mae": res.pooled_mae,
        "nasa_oof_r2": res.pooled_r2,
        "nasa_oof_spearman": rho_nasa,
        "S_xgb_vs_S_true_mae": mae_s,
        "S_xgb_vs_S_true_r2": r2_s,
        "S_xgb_vs_S_true_spearman": rho_s,
        "S_true_range": [float(s_true.min()), float(s_true.max())],
        "S_xgb_range": [float(s_hat.min()), float(s_hat.max())],
        "S_true_mean": float(s_true.mean()),
        "S_xgb_mean": float(s_hat.mean()),
        "n": int(len(y)),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 用 XGB 预测 NASA 再算的 S\n\n",
        "问卷 NASA → `S_true`；XGB 折外预测 NASA → `S_xgb`。步骤分两边相同。\n\n",
        "```\n",
        "S = 0.40 × 步骤分 + 0.60 × (1 − NASA / 10)\n",
        "```\n\n",
        f"- NASA 模型：MI Top-30 + XGB（`max_depth=2, lr=0.02, λ=2, n=500`）\n",
        f"- 验证：5 折 GroupKFold by subject，折外预测\n",
        f"- NASA 折外：MAE={res.pooled_mae:.3f}，R²={res.pooled_r2:+.3f}，Spearman={rho_nasa:.3f}\n",
        f"- `S_xgb` 相对 `S_true`：MAE={mae_s:.3f}，R²={r2_s:+.3f}，Spearman={rho_s:.3f}\n",
        f"- `S_true` 范围 [{s_true.min():.3f}, {s_true.max():.3f}]，均值 {s_true.mean():.3f}\n",
        f"- `S_xgb` 范围 [{s_hat.min():.3f}, {s_hat.max():.3f}]，均值 {s_hat.mean():.3f}\n\n",
        "S 的差只来自 NASA 预测误差：`S_xgb − S_true = 0.06 × (y_nasa_xgb − y_nasa)`。\n",
        f"问卷 NASA 的 MAE=0.911 时，S 大约差 {0.06 * res.pooled_mae:.3f}。\n\n",
        "明细：`s_from_xgb_nasa.csv`\n",
    ]
    (OUT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")
    print(f"[xgb→S] S_xgb vs S_true  MAE={mae_s:.3f}  R²={r2_s:+.3f}  ρ={rho_s:.3f}")
    print(f"[xgb→S] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
