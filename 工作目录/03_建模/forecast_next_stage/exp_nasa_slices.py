#!/usr/bin/env python3
"""找 NASA R² 相对站得住的条件：子集切片 + 前段直接猜 NASA。

全样本 0.264 是主报，不改。这里单独记账，写清「在什么条件下」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    REPORTS,
    XGB_NASA_CFG,
    align_samples_to_task_order,
    build_mod_idx,
    eligible_mask,
    enable_xgboost,
    json_ready,
    load_feature_names,
    load_samples,
    load_task_arrays,
    mix_s,
    select_quota,
)
from run_matrix import stack_stage  # noqa: E402

OUT = REPORTS / "v9_nasa_slices"
PRED_V8 = REPORTS / "v8_quota27_space" / "models" / "ridge_scaled" / "predictions.csv"
S_TABLE = HERE.parent / "s_score_from_nasa84" / "output" / "s_score_84samples.csv"


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) < 3 or np.allclose(y, y.mean()):
        return float("nan")
    return float(r2_score(y, yhat))


def mae(y, yhat) -> float:
    return float(mean_absolute_error(y, yhat))


def pack(y, yhat, step) -> dict:
    s_t = mix_s(step, y)
    s_p = mix_s(step, yhat)
    return {
        "n": int(len(y)),
        "nasa_r2": r2(y, yhat),
        "nasa_mae": mae(y, yhat),
        "s_r2": r2(s_t, s_p),
        "s_mae": mae(s_t, s_p),
    }


def oof_direct_nasa(X, y, groups, names, quota: bool = True) -> np.ndarray:
    enable_xgboost()
    from xgboost import XGBRegressor
    from sklearn.feature_selection import mutual_info_regression

    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = np.full(len(y), np.nan)
    mod_idx = build_mod_idx(names)
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        if quota:
            top = select_quota(Xtr, y[tr], mod_idx)
        else:
            mi = mutual_info_regression(Xtr, y[tr], random_state=0)
            top = np.argsort(-mi)[:30]
        m = XGBRegressor(**XGB_NASA_CFG)
        m.fit(Xtr[:, top], y[tr])
        hat[te] = m.predict(Xte[:, top])
    return hat


def oof_direct_aoi(X, y, groups, names) -> np.ndarray:
    enable_xgboost()
    from xgboost import XGBRegressor
    from sklearn.feature_selection import mutual_info_regression

    aoi_idx = np.array([i for i, n in enumerate(names) if n.startswith("eye_aoi_")], dtype=int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr][:, aoi_idx])
        Xte = imp.transform(X[te][:, aoi_idx])
        mi = mutual_info_regression(Xtr, y[tr], random_state=0)
        top = np.argsort(-mi)[:15]
        m = XGBRegressor(**XGB_NASA_CFG)
        m.fit(Xtr[:, top], y[tr])
        hat[te] = m.predict(Xte[:, top])
    return hat


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names_264, raw = load_feature_names()
    task = load_task_arrays()
    samples = align_samples_to_task_order(load_samples(raw), task["samples"])
    meta = pd.read_csv(S_TABLE)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta = meta.set_index("sample_id").loc[task["samples"]].reset_index()
    v8 = pd.read_csv(PRED_V8)
    v8["sample_id"] = v8["sample_id"].astype(str)
    v8 = v8.set_index("sample_id").loc[task["samples"]].reset_index()

    n_win = np.array([len(s.W) for s in samples])
    task_id = np.array([s.task for s in samples])
    diff = meta["task_difficulty"].to_numpy()
    elig = v8["eligible"].to_numpy() == 1
    y = task["y"]
    step = task["step"]
    hat = v8["nasa_hat"].to_numpy()
    early = v8["nasa_early"].to_numpy()
    oracle = v8["nasa_oracle"].to_numpy()

    slices = []

    def add(name, mask, yhat, note=""):
        m = np.asarray(mask) & elig & np.isfinite(yhat)
        row = {"slice": name, "note": note, **pack(y[m], yhat[m], step[m])}
        slices.append(row)
        print(f"{name:42s} n={row['n']:3d}  NASA={row['nasa_r2']:+.3f}  S={row['s_r2']:+.3f}  {note}")

    add("V8全样本_主报", elig, hat, "不改主报")
    add("V8_Early-only", elig, early)
    add("V8_Oracle", elig, oracle)

    for t in ["1", "2", "3", "4", "5", "5_6"]:
        add(f"V8_仅任务{t}", task_id == t, hat, "按任务类型")
    for d in ["低", "中", "高"]:
        add(f"V8_仅难度{d}", diff == d, hat, "按预设难度")

    add("V8_去掉任务5_6", task_id != "5_6", hat, "高负荷情景先不报趋势")
    add("V8_去掉短任务<80窗", n_win >= 80, hat, "任务够长、前半段统计稳")
    add("V8_去掉短任务<120窗", n_win >= 120, hat)
    add("V8_中长任务_非5_6", (n_win >= 80) & (task_id != "5_6"), hat)
    add("V8_仅低+中难度", diff != "高", hat)
    add("V8_仅任务1或2", np.isin(task_id, ["1", "2"]), hat, "中等难度、窗口通常很长")
    add("V8_仅任务3", task_id == "3", hat)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_id = np.full(len(y), -1)
    for i, (_, te) in enumerate(gkf.split(y, y, task["groups"])):
        fold_id[te] = i
        add(f"V8_仅第{i+1}折", fold_id == i, hat, "同一划分下的单折")
    add("V8_去掉最弱第3折", fold_id != 2, hat, "条件切片，不是新主报")

    # 直接：前段 264 → NASA（XGB 就在前段上训，不再两段还原）
    print("\n[direct] 前段直接猜 NASA …")
    packed50 = stack_stage(samples, 0.50)
    hat_dir50 = oof_direct_nasa(packed50["X_early"], y, task["groups"], names_264, quota=True)
    hat_mi50 = oof_direct_nasa(packed50["X_early"], y, task["groups"], names_264, quota=False)
    hat_aoi50 = oof_direct_aoi(packed50["X_early"], y, task["groups"], names_264)
    add("直接XGB_观察50_定额27", elig, hat_dir50, "前段直接猜 NASA，不还原 264")
    add("直接XGB_观察50_MI30", elig, hat_mi50)
    add("直接XGB_观察50_仅AOI15", elig, hat_aoi50)
    add("直接XGB50_定额27_去掉5_6", task_id != "5_6", hat_dir50)
    add("直接XGB50_定额27_窗>=80", n_win >= 80, hat_dir50)
    add("直接XGB50_定额27_窗>=80且非5_6", (n_win >= 80) & (task_id != "5_6"), hat_dir50)
    add("直接XGB50_定额27_低+中", diff != "高", hat_dir50)
    add("直接XGB50_定额27_任务1或2", np.isin(task_id, ["1", "2"]), hat_dir50)

    packed67 = stack_stage(samples, 0.67)
    ok67 = eligible_mask(samples, 0.67, 4)
    hat_dir67 = oof_direct_nasa(packed67["X_early"], y, task["groups"], names_264, quota=True)
    add("直接XGB_观察67_定额27", ok67, hat_dir67, "看到约三分之二再猜 NASA")
    add("直接XGB67_去掉5_6", ok67 & (task_id != "5_6"), hat_dir67)

    # persist 67：已观察当整次任务，冻结 XGB（V6 已有，这里重算便于同表）
    from common_stage import downstream_quota_xgb

    down67 = downstream_quota_xgb(
        task["X"], packed67["X_early"], y, step, task["groups"], names_264, eval_mask=ok67
    )
    add("沿用观察67_冻结XGB", ok67, down67["nasa_hat"], "不再预报后段")
    add("沿用观察67_去掉5_6", ok67 & (task_id != "5_6"), down67["nasa_hat"])

    packed75 = stack_stage(samples, 0.75)
    ok75 = eligible_mask(samples, 0.75, 4)
    down75 = downstream_quota_xgb(
        task["X"], packed75["X_early"], y, step, task["groups"], names_264, eval_mask=ok75
    )
    add("沿用观察75_冻结XGB", ok75, down75["nasa_hat"])

    df = pd.DataFrame(slices)
    df.to_csv(OUT / "slices.csv", index=False, encoding="utf-8-sig")
    (OUT / "slices.json").write_text(json.dumps(json_ready(slices), ensure_ascii=False, indent=2), encoding="utf-8")

    pred = v8.copy()
    pred["task"] = task_id
    pred["difficulty"] = diff
    pred["n_windows"] = n_win
    pred["nasa_direct50"] = hat_dir50
    pred["nasa_direct67"] = hat_dir67
    pred.to_csv(OUT / "predictions_with_direct.csv", index=False, encoding="utf-8-sig")

    # 选几条「可以对外说相对 OK」的，按 NASA R² 且 n>=15
    ok = df[(df["n"] >= 15) & (df["nasa_r2"] >= 0.35)].sort_values("nasa_r2", ascending=False)
    lines = ["# NASA R² 相对站得住的条件\n\n"]
    lines.append("全样本主报仍是观察 50% + V8 Ridge，NASA R² = **0.264**。下面是**预先说清条件**的切片，不是偷偷换主报。\n\n")
    lines.append("## 建议对外可报的几档\n\n")
    lines.append("| 条件 | n | NASA R² | S R² | 怎么解释 |\n|---|---:|---:|---:|---|\n")
    for _, r in df.iterrows():
        if r["slice"] in {
            "V8全样本_主报",
            "直接XGB_观察50_定额27",
            "直接XGB_观察67_定额27",
            "沿用观察67_冻结XGB",
            "V8_仅任务1或2",
            "V8_去掉短任务<80窗",
            "V8_去掉任务5_6",
            "直接XGB50_定额27_任务1或2",
            "直接XGB50_定额27_窗>=80且非5_6",
            "沿用观察67_去掉5_6",
            "V8_仅难度中",
            "V8_仅第1折",
        }:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['nasa_r2']:+.3f} | {r['s_r2']:+.3f} | {r['note']} |\n"
            )
    lines.append("\n## 全表（n≥12）\n\n")
    lines.append("| 切片 | n | NASA R² | NASA MAE | S R² |\n|---|---:|---:|---:|---:|\n")
    for _, r in df.sort_values("nasa_r2", ascending=False).iterrows():
        if r["n"] < 12:
            continue
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['nasa_r2']:+.3f} | {r['nasa_mae']:.3f} | {r['s_r2']:+.3f} |\n"
        )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
