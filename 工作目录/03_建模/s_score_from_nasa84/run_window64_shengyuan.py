#!/usr/bin/env python3
"""按子伟口径重跑圣袁流程：64 维原始窗口输入，不再做任务级 mean/std/median/slope。

背景
----
现行主线是 66 个窗口指标 × 4 个任务内统计量 = 264 维，再筛到 27 维去预测 NASA，
然后用真实步骤分合成 S。子伟的要求是：

1. 输入统一成 **64 个原始窗口特征**（脑电用原始功率，不是被试内 z 分数）
2. **只做窗口化**，不要再对每个任务求平均 / 方差 / 中位数 / 斜率
3. 30 s 窗、5 s 步（重叠 83%）本身会让相邻窗几乎一样、又共享同一个 NASA，
   这就是他说的「直接窗口化给他泄漏一点」
4. NASA 和 S **用同一套 64 维**，不再一边 264/27、一边另套特征
5. 模型配置跟圣袁那套浅树 XGB 一致，再按同一公式合成 S

64 维怎么来（对齐「历史 58 + 眨眼 6」）
------------------------------------
历史方案 58 = 脑电原始功率 28 + 心率 5 + 眼动 13 + 日志 win 12。
后来补了 6 个过滤后的眨眼列。眼动 13 = 瞳孔均值/波动 + 注视/扫视比例 + 9 个 AOI。
不含 eye_valid_ratio / eye_eyes_not_found_ratio（质量列），也不含 EEG z 分数、
日志累计列、眨眼 raw 审计列。

S 公式（与现行对外口径相同）
--------------------------
    S = 0.70 * 步骤分 + 0.30 * (1 - NASA / 10)
步骤分仍用真实操作记录，只有 NASA 是模型预测的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS  # noqa: E402

REPO = HERE.parents[2]
WIN_DIR = REPO / "工作目录" / "01_预处理" / "output_30s_step5s_final"
S_TABLE = HERE / "output" / "s_score_84samples.csv"
OUT_DIR = HERE / "reports_window64_shengyuan"

N_SPLITS = 5
TOP_K = 30
STEP_W = 0.70
WEIGHTS = (0.40, 0.50, 0.60, 0.70)

XGB_CFG = dict(
    max_depth=2,
    n_estimators=500,
    learning_rate=0.02,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=0,
)

REGIONS = ("frontal", "central", "parietal", "occipital")
BANDS = (
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "theta_alpha",
    "beta_alpha",
)
EEG_RAW_28 = [f"eeg_{region}_{band}" for region in REGIONS for band in BANDS]
HR_5 = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_slope_bpm_per_min"]
EYE_13 = [
    "eye_pupil_filtered_mean",
    "eye_pupil_filtered_std",
    "eye_fixation_ratio",
    "eye_saccade_ratio",
    "eye_aoi_interval_n",
    "eye_aoi_unique_hit_n",
    "eye_aoi_total_fix_ms",
    "eye_aoi_fixation_n",
    "eye_aoi_fixation_density_per_sec",
    "eye_aoi_coverage_ratio",
    "eye_aoi_max_share",
    "eye_aoi_entropy",
    "eye_aoi_pupil_weighted_mean",
]
BLINK_6 = [
    "blink_count",
    "blink_rate_per_min",
    "blink_duration_mean_ms",
    "blink_duration_std_ms",
    "blink_duration_median_ms",
    "blink_total_duration_ratio",
]
LOG_WIN_12 = [
    "log_action_count_win",
    "log_unique_device_count_win",
    "log_action_density_win",
    "log_correct_action_count_win",
    "log_error_action_count_win",
    "log_duplicate_action_count_win",
    "log_extra_action_count_win",
    "log_disallowed_action_count_win",
    "log_error_rate_win",
    "log_duplicate_rate_win",
    "log_extra_rate_win",
    "log_unique_step_count_win",
]
FEATURES_64 = EEG_RAW_28 + HR_5 + EYE_13 + BLINK_6 + LOG_WIN_12
assert len(FEATURES_64) == 64, len(FEATURES_64)
assert len(set(FEATURES_64)) == 64


def _enable_xgboost_on_macos() -> None:
    import ctypes
    import importlib.util
    import os
    import shutil
    from pathlib import Path as P

    import sklearn

    omp_lib = P(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if not omp_lib.exists():
        return
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(omp_lib.parent))
    os.environ.setdefault("DYLD_LIBRARY_PATH", str(omp_lib.parent))
    ctypes.CDLL(str(omp_lib), mode=ctypes.RTLD_GLOBAL)
    spec = importlib.util.find_spec("xgboost")
    if spec is None or not spec.origin:
        return
    dest = P(spec.origin).resolve().parent / "lib" / "libomp.dylib"
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(omp_lib)
    except OSError:
        shutil.copy2(omp_lib, dest)


def load_windows() -> pd.DataFrame:
    files = sorted(WIN_DIR.glob("subject_*_task_*.csv"))
    if not files:
        raise FileNotFoundError(f"没有窗口 CSV：{WIN_DIR}")
    frames = []
    for path in files:
        use = ["sample_id", "subject", "task", "nasa_tlx_weighted_task_label", *FEATURES_64]
        df = pd.read_csv(path, usecols=use)
        missing = [c for c in FEATURES_64 if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path.name} 缺列：{missing}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["sample_id"] = out["sample_id"].astype(str)
    out["subject"] = out["subject"].astype(int)
    out["task"] = out["task"].astype(str)
    return out


def aggregate_task(sample_ids: np.ndarray, y_true_win: np.ndarray, y_hat_win: np.ndarray) -> pd.DataFrame:
    rows = []
    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        rows.append(
            {
                "sample_id": str(sid),
                "n_windows": int(mask.sum()),
                "y_nasa_true": float(y_true_win[mask][0]),
                "y_nasa_hat": float(np.median(y_hat_win[mask])),
            }
        )
    return pd.DataFrame(rows)


def mix_s(step: np.ndarray, nasa: np.ndarray, a: float) -> np.ndarray:
    return a * step + (1.0 - a) * (1.0 - nasa / 10.0)


def metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "r2": float(r2_score(y_true, y_hat)),
    }


def run_cv(X: np.ndarray, y: np.ndarray, splits, names: list[str], top_k: int | None, tag: str):
    from xgboost import XGBRegressor

    y_hat = np.full(len(y), np.nan)
    fold_rows = []
    selected = []
    for fold_idx, (tr, te) in enumerate(splits):
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X[tr])
        X_te = imputer.transform(X[te])
        if top_k is None:
            idx = np.arange(X_tr.shape[1])
        else:
            idx = RANKERS["MI"](X_tr, y[tr])[:top_k]
        selected.append([names[i] for i in idx])
        model = XGBRegressor(**XGB_CFG)
        model.fit(X_tr[:, idx], y[tr])
        pred = model.predict(X_te[:, idx])
        y_hat[te] = pred
        fold_rows.append(
            {
                "fold": fold_idx + 1,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "window": metrics(y[te], pred),
                "n_selected": int(len(idx)),
            }
        )
        print(
            f"  [{tag}] fold {fold_idx + 1}/{N_SPLITS}  "
            f"win MAE={fold_rows[-1]['window']['mae']:.3f}  "
            f"R²={fold_rows[-1]['window']['r2']:+.3f}"
        )
    if np.isnan(y_hat).any():
        raise RuntimeError(f"{tag} 有窗口没拿到折外预测")
    return y_hat, fold_rows, selected


def pack_protocol(task_df: pd.DataFrame, s_table: pd.DataFrame, fold_rows, selected, note: str) -> dict:
    merged = s_table.merge(task_df, on="sample_id", how="inner")
    if len(merged) != len(s_table):
        missing = set(s_table["sample_id"]) - set(task_df["sample_id"])
        raise RuntimeError(f"对不齐 S 表：{sorted(missing)[:8]}")
    if not np.allclose(merged["y_nasa"].to_numpy(), merged["y_nasa_true"].to_numpy(), atol=1e-8):
        raise RuntimeError("任务级 NASA 真值与 S 表对不齐")

    step = merged["weighted_step_score"].to_numpy(dtype=float)
    y_true = merged["y_nasa_true"].to_numpy(dtype=float)
    y_hat = merged["y_nasa_hat"].to_numpy(dtype=float)
    s_true = mix_s(step, y_true, STEP_W)
    s_hat = mix_s(step, y_hat, STEP_W)
    weight_stats = {}
    for a in WEIGHTS:
        st = mix_s(step, y_true, a)
        sh = mix_s(step, y_hat, a)
        weight_stats[f"step{int(round(a * 10)):02d}"] = {
            "step": a,
            "nasa": 1.0 - a,
            **metrics(st, sh),
        }
    return {
        "note": note,
        "n_tasks": int(len(merged)),
        "nasa_task": metrics(y_true, y_hat),
        "S_step07": metrics(s_true, s_hat),
        "S_by_weight": weight_stats,
        "folds": fold_rows,
        "selected_per_fold": selected,
        "table": merged,
        "s_true": s_true,
        "s_hat": s_hat,
    }


def make_splits(kind: str, n: int, subjects: np.ndarray, sample_int: np.ndarray):
    if kind == "leak_kfold":
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
        dummy = np.zeros(n)
        return list(kf.split(dummy))
    if kind == "group_subject":
        gkf = GroupKFold(n_splits=N_SPLITS)
        dummy = np.zeros(n)
        return list(gkf.split(dummy, dummy, subjects))
    if kind == "group_sample":
        gkf = GroupKFold(n_splits=N_SPLITS)
        dummy = np.zeros(n)
        return list(gkf.split(dummy, dummy, sample_int))
    raise ValueError(kind)


def write_report(payload: dict) -> str:
    lines = []
    lines.append("# 64 维窗口输入 + 圣袁流程（按子伟口径）\n\n")
    lines.append("NASA 和 S 用**同一套 64 个原始窗口特征**。没有任务级 mean / std / median / slope。\n\n")
    lines.append("## 64 维构成\n\n")
    lines.append("| 块 | 个数 | 说明 |\n|---|---:|---|\n")
    lines.append("| 脑电原始功率 | 28 | 4 脑区 ×（δθ αβγ 功率 + θ/α、β/α），**不是** z 分数 |\n")
    lines.append("| 心率 | 5 | 窗内均值/标准差/最低/最高/斜率 |\n")
    lines.append("| 眼动 | 13 | 瞳孔均值/波动、注视/扫视比例、9 个 AOI |\n")
    lines.append("| 眨眼 | 6 | 历史 58 维之后补上的过滤眨眼 |\n")
    lines.append("| 操作日志（仅本窗） | 12 | 不含累计 `_cum` |\n")
    lines.append("| **合计** | **64** | 历史 58 + 眨眼 6 |\n\n")
    lines.append("切窗仍是 30 s / 5 s。相邻窗重叠 25 s，又共用同一个任务级 NASA，这就是窗口泄漏。\n\n")
    lines.append("模型：浅树 XGB（max_depth=2, n_estimators=500, learning_rate=0.02, "
                 "reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8），与圣袁 NASA 回归最优配置一致。\n\n")
    lines.append("```\n")
    lines.append("S = 0.70 × 真实步骤分 + 0.30 × (1 − 预测NASA / 10)\n")
    lines.append("```\n\n")
    lines.append("任务级 NASA 把一次任务里所有窗口的折外预测取**中位数**。\n\n")

    lines.append("## 结果\n\n")
    lines.append("| 划分 | 特征 | 任务级 NASA R² | NASA MAE | S R²（0.7/0.3） | S MAE |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    order = [
        "leak_kfold_all64",
        "leak_kfold_mi30",
        "group_sample_all64",
        "group_sample_mi30",
        "group_subject_all64",
        "group_subject_mi30",
    ]
    labels = {
        "leak_kfold_all64": "窗口乱序 5 折（泄漏） / 全部 64",
        "leak_kfold_mi30": "窗口乱序 5 折（泄漏） / MI Top-30",
        "group_sample_all64": "按任务分组（同任务不跨折） / 全部 64",
        "group_sample_mi30": "按任务分组 / MI Top-30",
        "group_subject_all64": "按被试分组（换人考试） / 全部 64",
        "group_subject_mi30": "按被试分组 / MI Top-30",
    }
    for key in order:
        r = payload["results"][key]
        lines.append(
            f"| {labels[key]} | {r['n_features']} | "
            f"{r['nasa_task']['r2']:+.3f} | {r['nasa_task']['mae']:.3f} | "
            f"{r['S_step07']['r2']:+.3f} | {r['S_step07']['mae']:.3f} |\n"
        )

    main = payload["results"]["leak_kfold_all64"]
    lines.append("\n子伟说的「走一遍」对应表里第一行：**64 维窗口输入 + 乱序折（让重叠窗漏一点）+ 圣袁 XGB + 同一公式合成 S**。\n\n")
    lines.append("## 主结果明细（泄漏 / 全部 64）\n\n")
    lines.append(f"- 任务数：{main['n_tasks']}\n")
    lines.append(f"- NASA 任务级：R² = {main['nasa_task']['r2']:+.3f}，MAE = {main['nasa_task']['mae']:.3f}\n")
    lines.append(f"- S（步骤 0.70）：R² = {main['S_step07']['r2']:+.3f}，MAE = {main['S_step07']['mae']:.3f}\n\n")
    lines.append("| 步骤 : NASA | S R² | S MAE |\n|---|---:|---:|\n")
    for tag, row in main["S_by_weight"].items():
        lines.append(
            f"| {row['step']:.2f} : {row['nasa']:.2f} | {row['r2']:+.3f} | {row['mae']:.3f} |\n"
        )
    lines.append("\n按被试分组那两行才是「没见过这个人」。乱序折会把同一次任务的重叠窗拆进训练和考试，"
                 "NASA 会明显更好看；这是子伟明确要的泄漏，不是诚实跨被试结论。\n")
    lines.append("\n明细：`s_from_window64.csv`（主结果 = 泄漏 + 全部 64）。\n")
    return "".join(lines)


def main() -> None:
    if len(FEATURES_64) != 64:
        raise RuntimeError("特征数不是 64")
    _enable_xgboost_on_macos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[window64] 读窗口 CSV …")
    win = load_windows()
    X = win[FEATURES_64].to_numpy(dtype=np.float64)
    y = win["nasa_tlx_weighted_task_label"].to_numpy(dtype=np.float64)
    subjects = win["subject"].to_numpy(dtype=np.int64)
    sample_ids = win["sample_id"].to_numpy()
    uniq = pd.unique(sample_ids)
    sid_to_int = {s: i for i, s in enumerate(uniq)}
    sample_int = np.array([sid_to_int[s] for s in sample_ids], dtype=np.int64)

    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    keep_s = ["sample_id", "subject", "task", "task_difficulty", "y_nasa", "weighted_step_score"]
    s_table = s_table[keep_s].copy()

    print(f"[window64] 窗口 {len(win)}  任务 {win['sample_id'].nunique()}  特征 {X.shape[1]}")
    print(f"[window64] 缺失率 {np.isnan(X).mean() * 100:.2f}%")

    protocols = [
        ("leak_kfold", "窗口乱序 5 折：同一次任务的重叠窗可以同时出现在训练和考试里（子伟说的泄漏）"),
        ("group_sample", "按 sample_id 分组：同一次任务的窗不跨折，但同一个人的不同任务可以"),
        ("group_subject", "按被试分组：考试折完全没见过这个人（现行诚实口径）"),
    ]
    selections = [
        ("all64", None, "全部 64 维，NASA 和 S 输入相同"),
        ("mi30", TOP_K, "圣袁流程里的折内 MI Top-30，候选仍是这 64 列"),
    ]

    results: dict[str, dict] = {}
    main_table = None
    for p_name, p_note in protocols:
        splits = make_splits(p_name, len(y), subjects, sample_int)
        for s_name, top_k, s_note in selections:
            key = f"{p_name}_{s_name}"
            print(f"\n===== {key} =====")
            y_hat_win, fold_rows, selected = run_cv(
                X, y, splits, FEATURES_64, top_k, key
            )
            task_df = aggregate_task(sample_ids, y, y_hat_win)
            packed = pack_protocol(task_df, s_table, fold_rows, selected, f"{p_note}；{s_note}")
            packed["n_features"] = 64 if top_k is None else TOP_K
            table = packed.pop("table")
            s_true = packed.pop("s_true")
            s_hat = packed.pop("s_hat")
            packed["nasa_task_r2"] = packed["nasa_task"]["r2"]
            results[key] = packed
            print(
                f"  任务级 NASA R²={packed['nasa_task']['r2']:+.3f}  "
                f"S R²={packed['S_step07']['r2']:+.3f}"
            )
            if key == "leak_kfold_all64":
                out = table.copy()
                out["S_true"] = s_true
                out["S_hat"] = s_hat
                out["S_hat_minus_S_true"] = s_hat - s_true
                main_table = out
                np.save(OUT_DIR / "y_nasa_task_hat.npy", table["y_nasa_hat"].to_numpy())
                np.save(OUT_DIR / "y_s_hat.npy", s_hat)

    if main_table is None:
        raise RuntimeError("没有主结果表")
    main_table.to_csv(OUT_DIR / "s_from_window64.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "feature_names_64.json").write_text(
        json.dumps(
            {
                "n": 64,
                "eeg_raw_28": EEG_RAW_28,
                "hr_5": HR_5,
                "eye_13": EYE_13,
                "blink_6": BLINK_6,
                "log_win_12": LOG_WIN_12,
                "all": FEATURES_64,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload_results = {}
    for k, v in results.items():
        payload_results[k] = {kk: vv for kk, vv in v.items() if kk != "selected_per_fold"}
        payload_results[k]["selected_per_fold_n"] = [len(s) for s in v["selected_per_fold"]]
        payload_results[k]["selected_per_fold"] = v["selected_per_fold"]
    payload = {
        "xgb": XGB_CFG,
        "formula": f"S = {STEP_W:.2f} * weighted_step + {1-STEP_W:.2f} * (1 - NASA/10)",
        "n_windows": int(len(win)),
        "n_tasks": int(win["sample_id"].nunique()),
        "n_features": 64,
        "results": payload_results,
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "report.md").write_text(write_report(payload), encoding="utf-8")
    print(f"\n[window64] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
