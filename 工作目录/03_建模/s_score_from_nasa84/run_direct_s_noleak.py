#!/usr/bin/env python3
"""直接预测 S、不泄漏（按被试 GroupKFold）。给子伟的数字。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

from run_window64_shengyuan import (  # noqa: E402
    FEATURES_64,
    N_SPLITS,
    OUT_DIR,
    S_TABLE,
    STEP_W,
    TOP_K,
    _enable_xgboost_on_macos,
    aggregate_task,
    load_windows,
    make_splits,
    metrics,
    mix_s,
    run_cv,
)

NASA_DS = HERE.parent / "regression_task_level" / "dataset"


def main() -> None:
    _enable_xgboost_on_macos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_true_map = {}
    for _, row in s_table.iterrows():
        s_true_map[row["sample_id"]] = float(
            mix_s(
                np.array([row["weighted_step_score"]], dtype=float),
                np.array([row["y_nasa"]], dtype=float),
                STEP_W,
            )[0]
        )

    print("[direct S] 64 维窗口，按被试五折，直接预测 S")
    win = load_windows()
    X = win[FEATURES_64].to_numpy(dtype=np.float64)
    sample_ids = win["sample_id"].to_numpy()
    subjects = win["subject"].to_numpy(dtype=np.int64)
    y_s_win = np.array([s_true_map[str(s)] for s in sample_ids], dtype=np.float64)
    uniq = pd.unique(sample_ids)
    sid_to_int = {s: i for i, s in enumerate(uniq)}
    sample_int = np.array([sid_to_int[s] for s in sample_ids], dtype=np.int64)
    splits = make_splits("group_subject", len(y_s_win), subjects, sample_int)

    rows = {}
    for name, top_k in [("all64", None), ("mi30", TOP_K)]:
        y_hat_win, fold_rows, _ = run_cv(X, y_s_win, splits, FEATURES_64, top_k, f"directS_{name}")
        task = aggregate_task(sample_ids, y_s_win, y_hat_win)
        y_true = task["y_nasa_true"].to_numpy()  # 这里存的是 S 真值
        y_hat = task["y_nasa_hat"].to_numpy()
        m = metrics(y_true, y_hat)
        rows[f"window64_subject_{name}"] = {
            "n_features": 64 if top_k is None else TOP_K,
            "protocol": "GroupKFold by subject, window-level X, target=S",
            **m,
            "folds": fold_rows,
        }
        print(f"  {name:6s}  任务级直接 S  R²={m['r2']:+.3f}  MAE={m['mae']:.3f}")

    print("\n[direct S] 84×264 任务级，按被试五折，直接预测 S（圣袁同一套 XGB）")
    from exp_utils import RANKERS, pooled_cv, pooled_cv_with_selection  # noqa: E402
    from xgboost import XGBRegressor

    X_task = np.load(NASA_DS / "X_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    y_s = np.array([s_true_map[s] for s in samples], dtype=np.float64)

    def xgb():
        from run_window64_shengyuan import XGB_CFG

        return XGBRegressor(**XGB_CFG)

    res_all = pooled_cv(xgb, X_task, y_s, groups, N_SPLITS, None, "S_all264")
    res_mi, _ = pooled_cv_with_selection(
        xgb, X_task, y_s, groups, N_SPLITS, TOP_K, RANKERS["MI"], None, "S_MI30"
    )
    rows["task264_subject_all264"] = {
        "n_features": 264,
        "protocol": "GroupKFold by subject, 84×264, target=S",
        "mae": res_all.pooled_mae,
        "r2": res_all.pooled_r2,
        "fold_r2_mean": res_all.fold_r2_mean,
        "fold_r2_std": res_all.fold_r2_std,
    }
    rows["task264_subject_mi30"] = {
        "n_features": 30,
        "protocol": "GroupKFold by subject, 84×264 MI Top-30 vs S, target=S",
        "mae": res_mi.pooled_mae,
        "r2": res_mi.pooled_r2,
        "fold_r2_mean": res_mi.fold_r2_mean,
        "fold_r2_std": res_mi.fold_r2_std,
    }
    print(f"  all264  R²={res_all.pooled_r2:+.3f}  MAE={res_all.pooled_mae:.3f}  "
          f"折间 {res_all.fold_r2_mean:+.3f}±{res_all.fold_r2_std:.3f}")
    print(f"  MI30    R²={res_mi.pooled_r2:+.3f}  MAE={res_mi.pooled_mae:.3f}  "
          f"折间 {res_mi.fold_r2_mean:+.3f}±{res_mi.fold_r2_std:.3f}")

    out = {
        "question": "直接预测 S，不泄露的 R² 有多少？",
        "answer_for_ziwei": "大约 0.05（按被试五折、XGB 直接预测 S；全模态 MI Top-30 是 0.05，不强制模态是 0.08）",
        "results": rows,
    }
    path = OUT_DIR / "direct_s_noleak.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
