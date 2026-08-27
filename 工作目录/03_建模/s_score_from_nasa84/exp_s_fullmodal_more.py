#!/usr/bin/env python3
"""补实验：全模态约束接到「先 NASA 再合成 S」；以及直接猜 S 换模型/K。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS, pooled_cv, pooled_cv_with_selection  # noqa: E402

from exp_s_fullmodal_mi30 import (  # noqa: E402
    MODALITY_ORDER,
    NASA_DS,
    N_SPLITS,
    OUT_DIR,
    S_TABLE,
    STEP_W,
    TOP_K,
    XGB_CFG,
    _enable_xgboost,
    counts_of,
    fold_summary,
    make_fullmodal_ranker,
    pack_cv,
)


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor

    X = np.load(NASA_DS / "X_task.npy")
    y_nasa = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    step = s_table["weighted_step_score"].to_numpy(dtype=float)
    y_s = STEP_W * step + (1.0 - STEP_W) * (1.0 - y_nasa / 10.0)

    def xgb():
        return XGBRegressor(**XGB_CFG)

    rows = []

    def add(name, target, res, selected=None, extra=None):
        item = {
            "name": name,
            "target": target,
            "pooled_r2": float(res.pooled_r2),
            "pooled_mae": float(res.pooled_mae),
            "fold_r2_mean": float(res.fold_r2_mean),
            "fold_r2_std": float(res.fold_r2_std),
        }
        if selected is not None:
            item["avg_counts"] = fold_summary(names, selected)["avg_counts"]
            item["per_fold_counts"] = [counts_of(names, idx) for idx in selected]
        if extra:
            item.update(extra)
        rows.append(item)
        extra_s = ""
        if extra and "composed_S_r2" in extra:
            extra_s = f"  合成S R²={extra['composed_S_r2']:+.3f}"
        print(f"{name:42s} {target:5s} R²={res.pooled_r2:+.3f} MAE={res.pooled_mae:.3f}{extra_s}")

    # 1) 全模态约束，互信息仍对 NASA，预测 NASA，再合成 S
    for min_per, tag in ((1, "min1"), (2, "min2")):
        res, sel = pooled_cv_with_selection(
            xgb, X, y_nasa, groups, N_SPLITS, TOP_K,
            make_fullmodal_ranker(names, min_per), None, f"NASA_fullmodal_{tag}",
        )
        hat_s = STEP_W * step + (1.0 - STEP_W) * (1.0 - res.y_pred_pooled / 10.0)
        add(
            f"NASA_MI30_fullmodal_{tag}_then_S",
            "NASA",
            res,
            sel,
            extra={
                "composed_S_r2": float(r2_score(y_s, hat_s)),
                "composed_S_mae": float(mean_absolute_error(y_s, hat_s)),
                "note": f"MI对NASA；四模态各≥{min_per}；再合成S",
            },
        )
        np.save(OUT_DIR / f"yhat_nasa_fullmodal_{tag}.npy", res.y_pred_pooled)

    # 2) 互信息对 S（全模态），模型仍预测 NASA，再合成 S
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error as mae_fn

    gkf = GroupKFold(n_splits=N_SPLITS)
    rank_s = make_fullmodal_ranker(names, 1)
    y_nasa_hat_srank = np.full(len(y_nasa), np.nan)
    sel_srank = []
    for tr, te in gkf.split(X, y_nasa, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        top = rank_s(Xtr, y_s[tr])[:TOP_K]
        sel_srank.append(top)
        m = xgb()
        m.fit(Xtr[:, top], y_nasa[tr])
        y_nasa_hat_srank[te] = m.predict(Xte[:, top])
    hat_s = STEP_W * step + (1.0 - STEP_W) * (1.0 - y_nasa_hat_srank / 10.0)

    class _Tmp:
        pooled_r2 = float(r2_score(y_nasa, y_nasa_hat_srank))
        pooled_mae = float(mae_fn(y_nasa, y_nasa_hat_srank))
        fold_r2_mean = float("nan")
        fold_r2_std = float("nan")

    add(
        "MI_vs_S_fullmodal_min1_predict_NASA_then_S",
        "NASA",
        _Tmp(),
        sel_srank,
        extra={
            "composed_S_r2": float(r2_score(y_s, hat_s)),
            "composed_S_mae": float(mean_absolute_error(y_s, hat_s)),
            "note": "MI对S且全模态；XGB仍预测NASA；再合成S",
        },
    )
    np.save(OUT_DIR / "yhat_nasa_rank_on_S.npy", y_nasa_hat_srank)

    # 3) 直接猜 S：换模型 / K / 全量
    rf = lambda: RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)
    et = lambda: ExtraTreesRegressor(n_estimators=400, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)
    hgb = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)

    for k in (15, 30, 50, 80):
        res, sel = pooled_cv_with_selection(
            xgb, X, y_s, groups, N_SPLITS, k,
            make_fullmodal_ranker(names, 1), None, f"S_xgb_full_k{k}",
        )
        add(f"S_XGB_fullmodal_min1_k{k}", "S", res, sel)

    res, sel = pooled_cv_with_selection(
        rf, X, y_s, groups, N_SPLITS, 30, make_fullmodal_ranker(names, 1), None, "S_RF_full_k30",
    )
    add("S_RF_fullmodal_min1_k30", "S", res, sel)
    res, sel = pooled_cv_with_selection(
        rf, X, y_s, groups, N_SPLITS, 15, RANKERS["MI"], None, "S_RF_MI15",
    )
    add("S_RF_MI15_free", "S", res, sel)
    res, sel = pooled_cv_with_selection(
        et, X, y_s, groups, N_SPLITS, 30, make_fullmodal_ranker(names, 1), None, "S_ET_full_k30",
    )
    add("S_ET_fullmodal_min1_k30", "S", res, sel)
    res, sel = pooled_cv_with_selection(
        hgb, X, y_s, groups, N_SPLITS, 30, make_fullmodal_ranker(names, 1), None, "S_HGB_full_k30",
    )
    add("S_HGB_fullmodal_min1_k30", "S", res, sel)

    res = pooled_cv(xgb, X, y_s, groups, N_SPLITS, None, "S_XGB_full264")
    add("S_XGB_all264", "S", res)
    res = pooled_cv(rf, X, y_s, groups, N_SPLITS, None, "S_RF_full264")
    add("S_RF_all264", "S", res)

    # 只行为 / 只眼动 对照
    idx_log = np.array([i for i, n in enumerate(names) if n.startswith("log_")])
    idx_eye = np.array([i for i, n in enumerate(names) if n.startswith(("eye_", "blink_"))])
    res = pooled_cv(xgb, X[:, idx_log], y_s, groups, N_SPLITS, None, "S_XGB_log")
    add("S_XGB_only_log48", "S", res)
    res = pooled_cv(rf, X[:, idx_log], y_s, groups, N_SPLITS, None, "S_RF_log")
    add("S_RF_only_log48", "S", res)
    res = pooled_cv(xgb, X[:, idx_eye], y_s, groups, N_SPLITS, None, "S_XGB_eye")
    add("S_XGB_only_eye84", "S", res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results_more.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("写完", OUT_DIR / "results_more.json")


if __name__ == "__main__":
    main()
