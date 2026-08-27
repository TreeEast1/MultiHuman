#!/usr/bin/env python3
"""按流程图 + 圣袁划分，直接预测 S（不泄漏）。

流程
----
64 原始特征 → 被试内标准化 → ANOVA F-test 信息筛选
→ 保留眼动 / 行为 / 脑电 / 心率每个大模态
→ 多算法预测 S 并对比
→ 单 / 双 / 三 / 四模态分析

划分（跟圣袁 NASA 实验同一套）
------------------------------
GroupKFold(n_splits=5, groups=subject)，不打乱。
26 人分成 5 堆，每次 4 堆人训练、1 堆人考试，同一个人不跨折。
样本顺序与 regression_task_level/dataset 的 84 行对齐，因此每一折
的训练人 / 考试人与 compute_s_from_xgb_nasa.py 完全相同。

目标：直接预测 S（0.70 步骤 + 0.30 NASA 反向），不是先猜 NASA 再合成。
ANOVA 和填缺失都只在训练折上 fit。
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANDOM_STATE  # noqa: E402

from run_window64_shengyuan import (  # noqa: E402
    FEATURES_64,
    S_TABLE,
    STEP_W,
    XGB_CFG,
    _enable_xgboost_on_macos,
    load_windows,
    mix_s,
)

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
OUT_DIR = HERE / "reports_anova_s_flowchart"
N_SPLITS = 5
MODALITY_ORDER = ("眼动", "行为", "脑电", "心率")
QUOTA = {"眼动": 6, "行为": 12, "脑电": 5, "心率": 4}


def modality_of(name: str) -> str:
    if name.startswith("eeg_"):
        return "脑电"
    if name.startswith("hr_"):
        return "心率"
    if name.startswith("log_"):
        return "行为"
    if name.startswith("eye_") or name.startswith("blink_"):
        return "眼动"
    return "其他"


def zscore_within_subject(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].astype(np.float64)
    for _, idx in df.groupby("subject").groups.items():
        block = out.loc[idx, cols].to_numpy(dtype=np.float64)
        mu = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0, ddof=0)
        sd = np.where(sd < 1e-12, np.nan, sd)
        out.loc[idx, cols] = (block - mu) / sd
    return out


def build_task_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    win = load_windows()
    win = zscore_within_subject(win, FEATURES_64)
    agg = (
        win.groupby("sample_id", sort=False)
        .agg({c: "mean" for c in FEATURES_64} | {"subject": "first", "task": "first"})
        .reset_index()
    )
    samples_ref = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    groups_ref = np.load(NASA_DS / "groups_task.npy")
    y_nasa_ref = np.load(NASA_DS / "y_task.npy")
    agg["sample_id"] = agg["sample_id"].astype(str)
    agg = agg.set_index("sample_id").loc[samples_ref].reset_index()
    if not np.array_equal(agg["subject"].to_numpy(dtype=np.int64), groups_ref):
        raise RuntimeError("subject 与圣袁 84 行表对不齐，划分会对不上")

    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples_ref].reset_index()
    if not np.allclose(s_table["y_nasa"].to_numpy(), y_nasa_ref, atol=1e-8):
        raise RuntimeError("S 表 NASA 与回归标签对不齐")
    y_s = mix_s(
        s_table["weighted_step_score"].to_numpy(dtype=float),
        s_table["y_nasa"].to_numpy(dtype=float),
        STEP_W,
    )
    X = agg[FEATURES_64].to_numpy(dtype=np.float64)
    groups = agg["subject"].to_numpy(dtype=np.int64)
    return X, y_s, groups, samples_ref, FEATURES_64


def fold_subjects(groups: np.ndarray) -> list[dict]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    dummy = np.zeros(len(groups))
    rows = []
    for i, (tr, te) in enumerate(gkf.split(dummy, dummy, groups)):
        rows.append(
            {
                "fold": i + 1,
                "train_subjects": sorted(int(s) for s in np.unique(groups[tr])),
                "test_subjects": sorted(int(s) for s in np.unique(groups[te])),
                "n_train_samples": int(len(tr)),
                "n_test_samples": int(len(te)),
            }
        )
    return rows


def anova_quota_idx(X_tr: np.ndarray, y_tr: np.ndarray, names: list[str], mods: tuple[str, ...]) -> np.ndarray:
    F, _p = f_regression(X_tr, y_tr)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    picked: list[int] = []
    used: set[int] = set()
    for mod in mods:
        cand = [i for i, n in enumerate(names) if modality_of(n) == mod]
        cand.sort(key=lambda i: -F[i])
        k = min(QUOTA[mod], len(cand))
        for i in cand[:k]:
            picked.append(int(i))
            used.add(int(i))
    return np.array(picked, dtype=int)


def model_zoo():
    from xgboost import XGBRegressor

    return [
        ("Dummy_mean", lambda: DummyRegressor(strategy="mean"), "none"),
        ("Ridge", lambda: Ridge(alpha=10.0), "scale"),
        ("LinearSVR", lambda: LinearSVR(C=1.0, max_iter=20000, random_state=RANDOM_STATE), "scale"),
        ("SVR_RBF", lambda: SVR(kernel="rbf", C=1.0, gamma="scale"), "scale"),
        ("KNN_k5", lambda: KNeighborsRegressor(n_neighbors=5), "scale"),
        ("RF_shallow", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1), "impute"),
        ("XGB_nasa", lambda: XGBRegressor(**XGB_CFG), "impute"),
    ]


def prep_xy(X_tr, X_te, kind: str):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    if kind == "scale":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    return X_tr, X_te


def run_cv(X, y, groups, names, mods, factory, prep_kind, tag: str):
    gkf = GroupKFold(n_splits=N_SPLITS)
    y_hat = np.full(len(y), np.nan)
    fold_rows = []
    selected = []
    mask = np.array([modality_of(n) in mods for n in names])
    X_sub = X[:, mask]
    names_sub = [n for n, k in zip(names, mask) if k]
    dummy = np.zeros(len(y))
    for fold_idx, (tr, te) in enumerate(gkf.split(dummy, dummy, groups)):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_sub[tr])
        idx = anova_quota_idx(X_tr_imp, y[tr], names_sub, mods)
        selected.append([names_sub[i] for i in idx])
        X_tr_sel = X_sub[tr][:, idx]
        X_te_sel = X_sub[te][:, idx]
        X_tr_p, X_te_p = prep_xy(X_tr_sel, X_te_sel, prep_kind)
        model = factory()
        model.fit(X_tr_p, y[tr])
        pred = model.predict(X_te_p)
        y_hat[te] = pred
        fold_rows.append({
            "fold": fold_idx + 1,
            "n_selected": int(len(idx)),
            "mae": float(mean_absolute_error(y[te], pred)),
            "r2": float(r2_score(y[te], pred)) if len(te) > 1 else float("nan"),
        })
    if np.isnan(y_hat).any():
        raise RuntimeError(f"{tag} 折外预测不完整")
    return {
        "name": tag,
        "modalities": list(mods),
        "n_modalities": int(len(mods)),
        "n_features": int(len(selected[0])) if selected else 0,
        "pooled_mae": float(mean_absolute_error(y, y_hat)),
        "pooled_r2": float(r2_score(y, y_hat)),
        "fold_r2_mean": float(np.nanmean([f["r2"] for f in fold_rows])),
        "fold_r2_std": float(np.nanstd([f["r2"] for f in fold_rows])),
        "folds": fold_rows,
        "selected_per_fold": selected,
        "y_hat": y_hat,
    }


def pack_slim(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in {"y_hat", "selected_per_fold"}} | {
        "selected_per_fold": row["selected_per_fold"],
    }


def write_report(splits, algo_rows, combo_rows, best_name) -> str:
    lines = []
    lines.append("# 64 维 ANOVA 流程直接预测 S（划分与圣袁相同）\n\n")
    lines.append("图上的流程：64 原始特征 → 被试内标准化 → ANOVA F-test 信息筛选")
    lines.append("→ 保留眼动 / 行为 / 脑电 / 心率 → 多算法对比 → 单双三四模态。\n\n")
    lines.append("## 划分（圣袁怎么划的）\n\n")
    lines.append("圣袁 NASA 回归 / 合成 S 用的是 **按被试 GroupKFold 五折、不打乱**：\n\n")
    lines.append("- 26 人拆成 5 堆，每次 4 堆训练、1 堆考试\n")
    lines.append("- 同一个人的所有任务只出现在同一折（不泄漏）\n")
    lines.append("- `sklearn.model_selection.GroupKFold(n_splits=5)`，")
    lines.append("样本顺序与 `regression_task_level/dataset` 的 84 行一致，")
    lines.append("因此每一折的人与 `compute_s_from_xgb_nasa.py` 相同\n\n")
    lines.append("| 折 | 考试人数 | 考试被试 | 训练条数 | 考试条数 |\n|---|---:|---|---:|---:|\n")
    for s in splits:
        tes = ",".join(str(x) for x in s["test_subjects"])
        lines.append(
            f"| {s['fold']} | {len(s['test_subjects'])} | {tes} | "
            f"{s['n_train_samples']} | {s['n_test_samples']} |\n"
        )
    lines.append("\n## 输入\n\n")
    lines.append("- 64 个原始窗口特征（脑电原始功率 28 + 心率 5 + 眼动 13 + 眨眼 6 + 日志 win 12）\n")
    lines.append("- 每个特征按被试做 z-score，再对一次任务的窗口取平均，得到 84 × 64\n")
    lines.append("- 每一折在**训练人**上做 ANOVA F（`f_regression`），")
    lines.append("四个大模态定额：眼动 6 / 行为 12 / 脑电 5 / 心率 4\n")
    lines.append("- 直接预测 S = 0.70 × 步骤分 + 0.30 × (1 − NASA/10)\n\n")

    lines.append("## 四模态：多算法对比\n\n")
    lines.append("| 算法 | pooled R² | pooled MAE | 折间 R² (μ±σ) |\n|---|---:|---:|---:|\n")
    for r in sorted(algo_rows, key=lambda x: -x["pooled_r2"]):
        lines.append(
            f"| {r['name']} | {r['pooled_r2']:+.3f} | {r['pooled_mae']:.3f} | "
            f"{r['fold_r2_mean']:+.3f}±{r['fold_r2_std']:.3f} |\n"
        )
    lines.append(f"\n后面的模态组合用这一步最好的算法：**{best_name}**。\n\n")

    lines.append("## 单 / 双 / 三 / 四模态\n\n")
    lines.append("| 模态数 | 组合 | n_feat | pooled R² | pooled MAE |\n|---|---|---:|---:|---:|\n")
    for r in sorted(combo_rows, key=lambda x: (-x["n_modalities"], -x["pooled_r2"])):
        lines.append(
            f"| {r['n_modalities']} | {'+'.join(r['modalities'])} | {r['n_features']} | "
            f"{r['pooled_r2']:+.3f} | {r['pooled_mae']:.3f} |\n"
        )
    best4 = next(r for r in combo_rows if r["n_modalities"] == 4)
    best1 = max((r for r in combo_rows if r["n_modalities"] == 1), key=lambda x: x["pooled_r2"])
    lines.append(f"\n- 四模态直接预测 S：R² = {best4['pooled_r2']:+.3f}\n")
    lines.append(f"- 最强单模态：`{'+'.join(best1['modalities'])}`，R² = {best1['pooled_r2']:+.3f}\n")
    lines.append("\n这是**不泄漏、直接猜 S** 的数。0.98 那档仍是真步骤 + 预测 NASA，不是这条线。\n")
    return "".join(lines)


def main() -> None:
    _enable_xgboost_on_macos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, samples, names = build_task_table()
    splits = fold_subjects(groups)
    print(f"[anova-S] X={X.shape}  S=[{y.min():.3f},{y.max():.3f}]  被试 {len(np.unique(groups))}")
    print("[anova-S] 圣袁五折考试人：")
    for s in splits:
        print(f"  fold {s['fold']}: {s['test_subjects']}")

    print("\n===== 四模态 多算法 =====")
    algo_rows = []
    algo_hats = {}
    for name, factory, prep in model_zoo():
        row = run_cv(X, y, groups, names, MODALITY_ORDER, factory, prep, name)
        slim = pack_slim(row)
        algo_rows.append(slim)
        algo_hats[name] = row["y_hat"]
        print(f"  {name:12s}  R²={row['pooled_r2']:+.3f}  MAE={row['pooled_mae']:.3f}")

    best = max(algo_rows, key=lambda r: r["pooled_r2"])
    best_name = best["name"]
    best_factory, best_prep = next((f, p) for n, f, p in model_zoo() if n == best_name)
    print(f"\n最佳算法：{best_name}")

    print("\n===== 单双三四模态 =====")
    combo_rows = []
    for k in range(1, 5):
        for combo in combinations(MODALITY_ORDER, k):
            tag = "+".join(combo)
            row = run_cv(X, y, groups, names, combo, best_factory, best_prep, tag)
            combo_rows.append(pack_slim(row))
            print(f"  {k}模态 {tag:20s}  R²={row['pooled_r2']:+.3f}  n={row['n_features']}")

    payload = {
        "split": "GroupKFold n_splits=5 by subject, identical to Shengyuan NASA CV",
        "folds": splits,
        "n": int(len(y)),
        "n_features_raw": 64,
        "quota": QUOTA,
        "target": "S = 0.70 * step + 0.30 * (1 - NASA/10)",
        "best_algorithm": best_name,
        "algorithms": algo_rows,
        "modality_combos": combo_rows,
        "samples": samples.tolist(),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / "y_s_true.npy", y)
    np.save(OUT_DIR / "y_s_hat_best4.npy", algo_hats[best_name])
    (OUT_DIR / "report.md").write_text(write_report(splits, algo_rows, combo_rows, best_name), encoding="utf-8")
    print(f"\n[anova-S] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
