#!/usr/bin/env python3
"""子伟口径：窗口化 64 维直接进圣袁流程，中间不做窗平均/标准差。

和上一版的差别
--------------
上一版在被试内 z-score 之后，把一次任务的所有窗口收成 1 行（等于做了窗平均），
变成 84 × 64。子伟明确说不要这一步。

这一版：
    输入 = 12624 条窗口 × 64 维（30 s / 5 s 切好的表，不再聚合）
    划分 = 圣袁同一套按被试五折（同一批考试人）
    流程 = 被试内标准化 → 折内 ANOVA F、四模态定额 → 多算法直接预测 S
           → 单/双/三/四模态

S 仍是任务级标签，同一任务的窗口共用一个 S。主指标是把窗口预测按任务取中位数
再和 84 条真 S 比 R²（不泄漏：考试折没见过这个人）。
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANDOM_STATE  # noqa: E402

from exp_anova_s_flowchart import (  # noqa: E402
    MODALITY_ORDER,
    NASA_DS,
    QUOTA,
    anova_quota_idx,
    modality_of,
    pack_slim,
    prep_xy,
    zscore_within_subject,
)
from run_window64_shengyuan import (  # noqa: E402
    FEATURES_64,
    S_TABLE,
    STEP_W,
    XGB_CFG,
    _enable_xgboost_on_macos,
    load_windows,
    mix_s,
)

OUT_DIR = HERE / "reports_anova_s_window64"
N_SPLITS = 5


def shengyuan_subject_folds() -> list[set[int]]:
    """用 84 行任务表做 GroupKFold，得到与圣袁完全相同的考试人。"""
    groups = np.load(NASA_DS / "groups_task.npy")
    dummy = np.zeros(len(groups))
    folds = []
    for _tr, te in GroupKFold(n_splits=N_SPLITS).split(dummy, dummy, groups):
        folds.append(set(int(s) for s in np.unique(groups[te])))
    return folds


def load_window_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    win = load_windows()
    win = zscore_within_subject(win, FEATURES_64)
    samples_ref = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    y_nasa_ref = np.load(NASA_DS / "y_task.npy")
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples_ref].reset_index()
    if not np.allclose(s_table["y_nasa"].to_numpy(), y_nasa_ref, atol=1e-8):
        raise RuntimeError("S 表与 NASA 标签对不齐")
    s_map = {
        sid: float(mix_s(np.array([step]), np.array([nasa]), STEP_W)[0])
        for sid, step, nasa in zip(
            s_table["sample_id"], s_table["weighted_step_score"], s_table["y_nasa"]
        )
    }
    win["sample_id"] = win["sample_id"].astype(str)
    X = win[FEATURES_64].to_numpy(dtype=np.float64)
    y = np.array([s_map[s] for s in win["sample_id"]], dtype=np.float64)
    subjects = win["subject"].to_numpy(dtype=np.int64)
    sample_ids = win["sample_id"].to_numpy()
    return X, y, subjects, sample_ids, FEATURES_64


def make_splits(subjects: np.ndarray, fold_sets: list[set[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for test_subj in fold_sets:
        te = np.flatnonzero(np.isin(subjects, list(test_subj)))
        tr = np.flatnonzero(~np.isin(subjects, list(test_subj)))
        splits.append((tr, te))
    return splits


def task_metrics(sample_ids: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    rows_t, rows_p = [], []
    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        rows_t.append(float(y_true[mask][0]))
        rows_p.append(float(np.median(y_hat[mask])))
    yt = np.array(rows_t)
    yp = np.array(rows_p)
    return {
        "n_tasks": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def model_zoo():
    from xgboost import XGBRegressor

    # 窗口有 1 万多行，RBF-SVR 是 O(n²)，不跑。其余与圣袁多算法清单对齐。
    return [
        ("Dummy_mean", lambda: DummyRegressor(strategy="mean"), "none"),
        ("Ridge", lambda: Ridge(alpha=10.0), "scale"),
        ("LinearSVR", lambda: LinearSVR(C=1.0, max_iter=20000, random_state=RANDOM_STATE), "scale"),
        ("KNN_k5", lambda: KNeighborsRegressor(n_neighbors=5), "scale"),
        ("RF_shallow", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1), "impute"),
        ("XGB_nasa", lambda: XGBRegressor(**XGB_CFG), "impute"),
    ]


def run_cv(X, y, sample_ids, splits, names, mods, factory, prep_kind, tag: str):
    y_hat = np.full(len(y), np.nan)
    fold_rows = []
    selected = []
    mask = np.array([modality_of(n) in mods for n in names])
    X_sub = X[:, mask]
    names_sub = [n for n, k in zip(names, mask) if k]
    for fold_idx, (tr, te) in enumerate(splits):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_sub[tr])
        idx = anova_quota_idx(X_tr_imp, y[tr], names_sub, mods)
        selected.append([names_sub[i] for i in idx])
        X_tr_p, X_te_p = prep_xy(X_sub[tr][:, idx], X_sub[te][:, idx], prep_kind)
        model = factory()
        model.fit(X_tr_p, y[tr])
        pred = model.predict(X_te_p)
        y_hat[te] = pred
        fold_rows.append({
            "fold": fold_idx + 1,
            "n_train_windows": int(len(tr)),
            "n_test_windows": int(len(te)),
            "n_selected": int(len(idx)),
            "window_mae": float(mean_absolute_error(y[te], pred)),
            "window_r2": float(r2_score(y[te], pred)),
        })
        print(
            f"    [{tag}] fold {fold_idx + 1}  "
            f"win R²={fold_rows[-1]['window_r2']:+.3f}  n_sel={len(idx)}"
        )
    if np.isnan(y_hat).any():
        raise RuntimeError(f"{tag} 折外预测不完整")
    task = task_metrics(sample_ids, y, y_hat)
    return {
        "name": tag,
        "modalities": list(mods),
        "n_modalities": int(len(mods)),
        "n_features": int(len(selected[0])) if selected else 0,
        "window_mae": float(mean_absolute_error(y, y_hat)),
        "window_r2": float(r2_score(y, y_hat)),
        "task_mae": task["mae"],
        "task_r2": task["r2"],
        "folds": fold_rows,
        "selected_per_fold": selected,
        "y_hat": y_hat,
    }


def write_report(fold_sets, algo_rows, combo_rows, best_name) -> str:
    lines = []
    lines.append("# 窗口化 64 维 + 圣袁流程，直接预测 S\n\n")
    lines.append("按子伟：用做 NASA 的那 64 个输入，**不要**再对窗口求平均/标准差，")
    lines.append("把切好的窗口表直接送进圣袁流程（被试内标准化 → ANOVA → 四模态定额 → 多算法 → 模态组合）。\n\n")
    lines.append("划分仍是圣袁那套按人五折（考试人与 `compute_s_from_xgb_nasa.py` 相同）。\n\n")
    lines.append("| 折 | 考试被试 |\n|---|---|\n")
    for i, subj in enumerate(fold_sets, 1):
        lines.append(f"| {i} | {','.join(str(s) for s in sorted(subj))} |\n")
    lines.append("\n输入：12624 窗 × 64 列。主指标 = 窗口预测按任务取中位数，再和 84 条真 S 比 R²。\n\n")

    lines.append("## 四模态多算法（任务级 R²）\n\n")
    lines.append("| 算法 | 任务级 R² | 任务级 MAE | 窗口级 R² |\n|---|---:|---:|---:|\n")
    for r in sorted(algo_rows, key=lambda x: -x["task_r2"]):
        lines.append(
            f"| {r['name']} | {r['task_r2']:+.3f} | {r['task_mae']:.3f} | {r['window_r2']:+.3f} |\n"
        )
    lines.append(f"\n模态组合用：**{best_name}**。窗口级 RBF-SVR 因样本太多未跑。\n\n")

    lines.append("## 单 / 双 / 三 / 四模态（任务级）\n\n")
    lines.append("| 模态数 | 组合 | n_feat | 任务级 R² | 任务级 MAE |\n|---|---|---:|---:|---:|\n")
    for r in sorted(combo_rows, key=lambda x: (-x["n_modalities"], -x["task_r2"])):
        lines.append(
            f"| {r['n_modalities']} | {'+'.join(r['modalities'])} | {r['n_features']} | "
            f"{r['task_r2']:+.3f} | {r['task_mae']:.3f} |\n"
        )
    best4 = next(r for r in combo_rows if r["n_modalities"] == 4)
    best1 = max((r for r in combo_rows if r["n_modalities"] == 1), key=lambda x: x["task_r2"])
    lines.append(f"\n- 四模态直接预测 S（任务级）：R² = {best4['task_r2']:+.3f}\n")
    lines.append(f"- 最强单模态：`{'+'.join(best1['modalities'])}`，R² = {best1['task_r2']:+.3f}\n")
    lines.append("\n对照：上一版把窗口平均成 84 行之后，四模态最好是 SVR 0.16、XGB 0.07。")
    lines.append("本版不再平均，只看窗口表本身。\n")
    return "".join(lines)


def main() -> None:
    _enable_xgboost_on_macos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_sets = shengyuan_subject_folds()
    X, y, subjects, sample_ids, names = load_window_matrix()
    splits = make_splits(subjects, fold_sets)
    print(f"[win64] X={X.shape}  任务 {pd.unique(sample_ids).size}  被试 {len(np.unique(subjects))}")
    print("[win64] 圣袁考试人：")
    for i, s in enumerate(fold_sets, 1):
        print(f"  fold {i}: {sorted(s)}  考试窗={len(splits[i-1][1])}")

    print("\n===== 四模态 多算法 =====")
    algo_rows = []
    algo_hats = {}
    for name, factory, prep in model_zoo():
        print(f"  -- {name}")
        row = run_cv(X, y, sample_ids, splits, names, MODALITY_ORDER, factory, prep, name)
        algo_rows.append(pack_slim(row) | {
            "task_r2": row["task_r2"], "task_mae": row["task_mae"],
            "window_r2": row["window_r2"], "window_mae": row["window_mae"],
        })
        algo_hats[name] = row["y_hat"]
        print(f"     任务级 R²={row['task_r2']:+.3f}  窗口级 R²={row['window_r2']:+.3f}")

    best = max(algo_rows, key=lambda r: r["task_r2"])
    best_name = best["name"]
    best_factory, best_prep = next((f, p) for n, f, p in model_zoo() if n == best_name)
    print(f"\n最佳算法：{best_name}")

    print("\n===== 单双三四模态 =====")
    combo_rows = []
    for k in range(1, 5):
        for combo in combinations(MODALITY_ORDER, k):
            tag = "+".join(combo)
            print(f"  -- {tag}")
            row = run_cv(X, y, sample_ids, splits, names, combo, best_factory, best_prep, tag)
            combo_rows.append(pack_slim(row) | {
                "task_r2": row["task_r2"], "task_mae": row["task_mae"],
                "window_r2": row["window_r2"], "window_mae": row["window_mae"],
            })
            print(f"     任务级 R²={row['task_r2']:+.3f}")

    payload = {
        "input": "window-level 64-d, no task mean/std/median/slope",
        "n_windows": int(len(y)),
        "n_tasks": int(pd.unique(sample_ids).size),
        "split": "same subject GroupKFold as Shengyuan 84-row NASA CV",
        "folds": [{"fold": i + 1, "test_subjects": sorted(s)} for i, s in enumerate(fold_sets)],
        "quota": QUOTA,
        "best_algorithm": best_name,
        "algorithms": algo_rows,
        "modality_combos": combo_rows,
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / "y_s_hat_windows.npy", algo_hats[best_name])
    (OUT_DIR / "report.md").write_text(
        write_report(fold_sets, algo_rows, combo_rows, best_name), encoding="utf-8"
    )
    print(f"\n[win64] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
