#!/usr/bin/env python3
"""S 作为预测目标：回归 + 三分位分类（与现行 NASA 实验同一套协议）。

针对「只算了一个 S 值、没有分类/预测结果」：
- 回归：预测连续 S（GroupKFold by subject，pooled MAE / R²）
- 分类：S 按 33%/67% 切低/中/高绩效（StratifiedGroupKFold，pooled Acc / Macro-F1）
- 对照：Full / 去 EEG（NASA 最佳迁移）/ 去 Log（去掉与步骤分同源的特征）/ 单模态
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import (  # noqa: E402
    RANDOM_STATE,
    RANKERS_CLS,
    median_impute_and_scale,
    median_impute_fold,
    pooled_cv_cls,
    pooled_cv_cls_with_selection,
)
from exp_utils import RANKERS, pooled_cv, pooled_cv_with_selection  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
S_OUT = HERE / "output"
REPORT_DIR = HERE / "reports_prediction"
N_SPLITS = 5


def modality_mask(names: list[str], kind: str) -> np.ndarray:
    if kind == "EEG":
        return np.array([c.startswith("eeg_") for c in names])
    if kind == "HR":
        return np.array([c.startswith("hr_") for c in names])
    if kind == "EyePupil":
        return np.array([c.startswith("eye_") and "aoi" not in c for c in names])
    if kind == "AOI":
        return np.array([c.startswith("eye_aoi_") for c in names])
    if kind == "Blink":
        return np.array([c.startswith("blink_") for c in names])
    if kind == "Log":
        return np.array([c.startswith("log_") for c in names])
    raise ValueError(kind)


def subset(X: np.ndarray, names: list[str], mode: str) -> tuple[np.ndarray, int]:
    if mode == "Full":
        return X, X.shape[1]
    if mode.startswith("minus_"):
        mask = ~modality_mask(names, mode.replace("minus_", ""))
        return X[:, mask], int(mask.sum())
    if mode.startswith("only_"):
        mask = modality_mask(names, mode.replace("only_", ""))
        return X[:, mask], int(mask.sum())
    raise ValueError(mode)


def _xgboost_available() -> bool:
    try:
        from xgboost import XGBRegressor  # noqa: F401
        XGBRegressor(n_estimators=1, tree_method="hist").fit(
            np.zeros((4, 2)), np.arange(4, dtype=float)
        )
        return True
    except Exception:
        return False


USE_XGB = _xgboost_available()
BOOST_TAG = "XGB" if USE_XGB else "HistGB"


def xgb_reg_shallow():
    if USE_XGB:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_depth=3, l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )


def xgb_reg_nasa_best():
    if USE_XGB:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500, learning_rate=0.02, max_depth=2, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.02, max_depth=2, l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )


def xgb_cls_shallow():
    if USE_XGB:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=3, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, max_depth=3, l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )


def pack_reg(name, n_feat, res) -> dict:
    return {
        "task": "regression",
        "name": name,
        "n_features": n_feat,
        "pooled_mae": res.pooled_mae,
        "pooled_r2": res.pooled_r2,
        "fold_mae_mean": res.fold_mae_mean,
        "fold_mae_std": res.fold_mae_std,
        "fold_r2_mean": res.fold_r2_mean,
        "fold_r2_std": res.fold_r2_std,
    }


def pack_cls(name, n_feat, res) -> dict:
    pc = res.pooled_per_class_f1
    f1_low = pc.get("低", pc.get(0, np.nan))
    f1_mid = pc.get("中", pc.get(1, np.nan))
    f1_high = pc.get("高", pc.get(2, np.nan))
    return {
        "task": "classification",
        "name": name,
        "n_features": n_feat,
        "pooled_acc": res.pooled_acc,
        "pooled_macro_f1": res.pooled_macro_f1,
        "pooled_weighted_f1": res.pooled_weighted_f1,
        "f1_low": float(f1_low),
        "f1_mid": float(f1_mid),
        "f1_high": float(f1_high),
        "fold_f1_mean": res.fold_macro_f1_mean,
        "fold_f1_std": res.fold_macro_f1_std,
        "confusion": res.confusion,
        "class_labels": [str(c) for c in res.class_labels],
    }


def write_report(meta: dict, reg_rows: list[dict], cls_rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# S 预测实验报告（回归 + 三分位分类）\n\n")
    lines.append("与现行 NASA 84×264 实验同一特征矩阵、同一被试分组交叉验证。")
    lines.append("本轮补的是**预测结果**，不是只算出 S 标量。\n\n")

    lines.append("## 标签\n\n")
    lines.append(f"- 连续 S：范围 [{meta['s_min']:.3f}, {meta['s_max']:.3f}]，均值 {meta['s_mean']:.3f}，std {meta['s_std']:.3f}\n")
    lines.append(f"- 三分位：低 ≤ {meta['s_q_lo']:.3f}（{meta['n_low']}），中 ≤ {meta['s_q_hi']:.3f}（{meta['n_mid']}），高（{meta['n_high']}）\n")
    lines.append("- 档含义：低/中/高 = **绩效**低/中/高（S 越高越好，与 NASA 负荷方向相反）\n\n")

    lines.append("## 评估协议\n\n")
    lines.append("| 线 | 划分 | 主指标 |\n|---|---|---|\n")
    lines.append("| 回归 | GroupKFold(5) by subject | pooled MAE / R² |\n")
    lines.append("| 分类 | StratifiedGroupKFold(5) by subject | pooled Accuracy / Macro-F1 |\n\n")
    lines.append(f"- Dummy 均值回归下限：MAE = {meta['dummy_mae']:.3f}（等于 |S − mean(S)| 的均值）\n")
    lines.append("- 去 Log：去掉与步骤分同源的操作日志特征，避免把 S 的 40% 成分从特征里直接读回来\n")
    if not meta.get("xgboost_available", True):
        lines.append(f"- 提升树：本机无 libomp，`{meta.get('booster')}` = sklearn HistGradientBoosting（深度/学习率对齐 NASA 浅树 XGB）\n")
    lines.append("\n")

    lines.append("## 回归：预测连续 S\n\n")
    lines.append("| 方案 | n_feat | pooled MAE | pooled R² | fold R² (μ±σ) |\n|---|---:|---:|---:|---:|\n")
    for r in sorted(reg_rows, key=lambda x: -x["pooled_r2"]):
        lines.append(
            f"| {r['name']} | {r['n_features']} | {r['pooled_mae']:.3f} | "
            f"{r['pooled_r2']:+.3f} | {r['fold_r2_mean']:+.3f}±{r['fold_r2_std']:.3f} |\n"
        )

    best_reg = max(reg_rows, key=lambda x: x["pooled_r2"])
    hon_reg = [r for r in reg_rows if r["name"].startswith("minus_Log")]
    lines.append(f"\n- 全特征最佳：`{best_reg['name']}`，pooled R² = {best_reg['pooled_r2']:+.3f}，MAE = {best_reg['pooled_mae']:.3f}\n")
    if hon_reg:
        h = max(hon_reg, key=lambda x: x["pooled_r2"])
        lines.append(f"- 去 Log（更诚实）：`{h['name']}`，pooled R² = {h['pooled_r2']:+.3f}，MAE = {h['pooled_mae']:.3f}\n")

    lines.append("\n## 分类：S 三分位（低/中/高绩效）\n\n")
    lines.append("| 方案 | n_feat | pooled Acc | Macro-F1 | F1低 | F1中 | F1高 | fold F1 (μ±σ) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(cls_rows, key=lambda x: -x["pooled_macro_f1"]):
        lines.append(
            f"| {r['name']} | {r['n_features']} | {r['pooled_acc']:.3f} | {r['pooled_macro_f1']:.3f} | "
            f"{r['f1_low']:.3f} | {r['f1_mid']:.3f} | {r['f1_high']:.3f} | "
            f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n"
        )

    best_cls = max(cls_rows, key=lambda x: x["pooled_macro_f1"])
    hon_cls = [r for r in cls_rows if r["name"].startswith("minus_Log")]
    lines.append(f"\n- 全特征最佳：`{best_cls['name']}`，Macro-F1 = {best_cls['pooled_macro_f1']:.3f}，Acc = {best_cls['pooled_acc']:.3f}\n")
    if hon_cls:
        h = max(hon_cls, key=lambda x: x["pooled_macro_f1"])
        lines.append(f"- 去 Log（更诚实）：`{h['name']}`，Macro-F1 = {h['pooled_macro_f1']:.3f}，Acc = {h['pooled_acc']:.3f}\n")

    lines.append(f"\n### 最佳分类模型 `{best_cls['name']}` 混淆矩阵（pooled）\n\n")
    labels = best_cls["class_labels"]
    pretty = [{"0": "低", "1": "中", "2": "高"}.get(str(c), str(c)) for c in labels]
    lines.append("| 真 \\ 预 | " + " | ".join(pretty) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(pretty)) + "|\n")
    for i, lab in enumerate(pretty):
        row = best_cls["confusion"][i]
        lines.append("| " + lab + " | " + " | ".join(str(int(v)) for v in row) + " |\n")

    lines.append("\n## 和 NASA 主线对照（同一 84×264、同一 group CV）\n\n")
    lines.append("| 目标 | 回归最佳 R² | 回归 MAE | 分类最佳 Macro-F1 |\n|---|---:|---:|---:|\n")
    lines.append(f"| NASA-TLX（已有主线） | +0.519 | 0.911 | 0.809 |\n")
    lines.append(
        f"| S 绩效（本轮） | {best_reg['pooled_r2']:+.3f} | {best_reg['pooled_mae']:.3f} | "
        f"{best_cls['pooled_macro_f1']:.3f} |\n"
    )
    return "".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(NASA_DS / "X_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    sample_ids = np.load(NASA_DS / "sample_task.npy", allow_pickle=True)
    with open(NASA_DS / "feature_names_task.json", encoding="utf-8") as f:
        names = json.load(f)
    y_s = np.load(S_OUT / "y_s.npy")
    sid_s = np.load(S_OUT / "sample_s.npy", allow_pickle=True)
    assert np.array_equal(sample_ids, sid_s), "S 与 NASA 样本顺序不一致"

    table = pd.read_csv(S_OUT / "s_score_84samples.csv")
    y_str = table["S_bin"].to_numpy().astype(str)
    y_int = np.array([{"低": 0, "中": 1, "高": 2}[c] for c in y_str], dtype=np.int64)
    q_lo, q_hi = np.quantile(y_s, [1.0 / 3.0, 2.0 / 3.0])

    dummy_mae = float(mean_absolute_error(y_s, np.full_like(y_s, y_s.mean())))
    dummy_r2 = float(r2_score(y_s, np.full_like(y_s, y_s.mean())))
    meta = {
        "n": int(len(y_s)),
        "s_min": float(y_s.min()),
        "s_max": float(y_s.max()),
        "s_mean": float(y_s.mean()),
        "s_std": float(y_s.std()),
        "s_q_lo": float(q_lo),
        "s_q_hi": float(q_hi),
        "n_low": int((y_str == "低").sum()),
        "n_mid": int((y_str == "中").sum()),
        "n_high": int((y_str == "高").sum()),
        "dummy_mae": dummy_mae,
        "dummy_r2": dummy_r2,
        "booster": BOOST_TAG,
        "xgboost_available": bool(USE_XGB),
    }
    print(f"[s_pred] X={X.shape}  S=[{y_s.min():.3f},{y_s.max():.3f}]  "
          f"bin 低/中/高={meta['n_low']}/{meta['n_mid']}/{meta['n_high']}")
    print(f"[s_pred] booster = {BOOST_TAG}"
          + ("" if USE_XGB else "（本机无 libomp，XGBoost 不可用，改用 sklearn HistGradientBoosting）"))

    reg_rows: list[dict] = []
    print("\n=== regression ===")
    reg_models = [
        ("Dummy_mean", lambda: DummyRegressor(strategy="mean"), None),
        ("Ridge", lambda: Ridge(alpha=10.0), median_impute_and_scale),
        ("RF_shallow", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1), median_impute_fold),
        (f"{BOOST_TAG}_shallow", xgb_reg_shallow, None),
    ]
    for name, factory, prep in reg_models:
        res = pooled_cv(factory, X, y_s, groups, N_SPLITS, prep, name=name)
        print(f"  {name:28s} MAE={res.pooled_mae:.3f}  R2={res.pooled_r2:+.3f}")
        reg_rows.append(pack_reg(f"Full + {name}", X.shape[1], res))

    res, _ = pooled_cv_with_selection(
        xgb_reg_nasa_best, X, y_s, groups, N_SPLITS, 30, RANKERS["MI"], None,
        name=f"MI30_{BOOST_TAG}_nasa_best",
    )
    print(f"  {f'MI Top-30 + {BOOST_TAG}_nasa_best':28s} MAE={res.pooled_mae:.3f}  R2={res.pooled_r2:+.3f}")
    reg_rows.append(pack_reg(f"MI Top-30 + {BOOST_TAG}_nasa_best", 30, res))

    for mode in ["minus_EEG", "minus_Log", "minus_AOI", "only_AOI", "only_Log"]:
        Xs, nfeat = subset(X, names, mode)
        res = pooled_cv(xgb_reg_shallow, Xs, y_s, groups, N_SPLITS, None, name=mode)
        print(f"  {mode:28s} MAE={res.pooled_mae:.3f}  R2={res.pooled_r2:+.3f}  n={nfeat}")
        reg_rows.append(pack_reg(f"{mode} + {BOOST_TAG}_shallow", nfeat, res))

    cls_rows: list[dict] = []
    print("\n=== classification ===")
    cls_models = [
        ("Dummy_stratified",
         lambda: DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), None, y_str),
        ("Dummy_most_frequent",
         lambda: DummyClassifier(strategy="most_frequent"), None, y_str),
        ("LR_L2_strong",
         lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
         median_impute_and_scale, y_str),
        ("RF_shallow",
         lambda: RandomForestClassifier(
             n_estimators=500, max_depth=4, min_samples_leaf=3,
             random_state=RANDOM_STATE, n_jobs=-1),
         median_impute_fold, y_str),
        (f"{BOOST_TAG}_shallow", xgb_cls_shallow, None, y_int),
    ]
    for name, factory, prep, y_this in cls_models:
        res = pooled_cv_cls(factory, X, y_this, groups, N_SPLITS, prep, name=name)
        print(f"  {name:28s} acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")
        cls_rows.append(pack_cls(f"Full + {name}", X.shape[1], res))

    res, _ = pooled_cv_cls_with_selection(
        xgb_cls_shallow, X, y_int, groups, N_SPLITS, 30, RANKERS_CLS["MI"], None,
        name=f"MI30_{BOOST_TAG}",
    )
    print(f"  {f'MI Top-30 + {BOOST_TAG}_shallow':28s} acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}")
    cls_rows.append(pack_cls(f"MI Top-30 + {BOOST_TAG}_shallow", 30, res))

    for mode in ["minus_EEG", "minus_Log", "minus_AOI", "only_AOI", "only_Log"]:
        Xs, nfeat = subset(X, names, mode)
        res = pooled_cv_cls(xgb_cls_shallow, Xs, y_int, groups, N_SPLITS, None, name=mode)
        print(f"  {mode:28s} acc={res.pooled_acc:.3f}  F1={res.pooled_macro_f1:.3f}  n={nfeat}")
        cls_rows.append(pack_cls(f"{mode} + {BOOST_TAG}_shallow", nfeat, res))

    payload = {"meta": meta, "regression": reg_rows, "classification": cls_rows}
    (REPORT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = write_report(meta, reg_rows, cls_rows)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\n[s_pred] wrote {REPORT_DIR}")


if __name__ == "__main__":
    main()
