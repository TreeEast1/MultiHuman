#!/usr/bin/env python3
"""按正式口径实测各流水线墙钟时间，写出 reports_timing/timing.json。

口径：26 被试、84 条被试–任务；切窗 30 秒 / 步长 5 秒（12 624 窗）。
预处理 = 读入原始文件 + 与眼动锚点对齐 + 切窗；
特征提取 = 窗级指标 + 眨眼/EEG 被试内 z 分数 + 任务级 66×4 聚合；
绩效评估 = 折内定额 27 维浅树 XGB 预测 NASA，再按 0.70/0.30 合成 S；
趋势预测 = 已观察段 Ridge 预报 27 列 → 同一套冻结 XGB → 合成 S（主路径 ridge_scaled）。

EEG 的 EEGLAB 人工预处理（滤波/ICA 等）发生在 .set 产出之前，不计入本表。
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = HERE.parent / "01_预处理" / "scripts"
DATA = REPO / "data"
WIN_DIR = HERE.parent / "01_预处理" / "output_30s_step5s_final"
NASA_DS = HERE / "regression_task_level" / "dataset"
S_TABLE = HERE / "s_score_from_nasa84" / "output" / "s_score_84samples.csv"
OUT_DIR = HERE / "reports_timing"
WINDOW_SEC = 30.0
STEP_SEC = 5.0
N_WINDOWS_OFFICIAL = 12624
N_SAMPLES_OFFICIAL = 84

PROBE_IDS = [
    "subject_01_task_3",   # 53 窗
    "subject_02_task_5_6",  # 131 窗，对外案例
    "subject_04_task_2",    # 406 窗
]


def load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def now() -> float:
    return time.perf_counter()


def machine_info() -> dict:
    uname = platform.uname()
    info = {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "python": sys.version.split()[0],
    }
    try:
        import subprocess

        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["cpu"] = brand
        info["ram_gb"] = round(mem / 1024 ** 3, 1)
    except Exception:
        info["cpu"] = info["processor"]
        info["ram_gb"] = None
    return info


def used_eeg_files(s1) -> list[Path]:
    set_dir = DATA / "04_EEG" / "raw_set_256Hz"
    files = []
    for p in sorted(set_dir.glob("*_preprocessed_eeg.set")):
        subject, task, sample_id, repeat = s1.parse_set_name(p)
        if subject == "20" and task == "5_6" and not repeat:
            continue
        files.append(p)
    return files


def blink_segments(eye_g: pd.DataFrame) -> pd.DataFrame:
    if eye_g is None or eye_g.empty:
        return pd.DataFrame(columns=["start_sec", "end_sec", "duration_ms"])
    move = eye_g["Eye movement type"].astype(str).to_numpy()
    t = eye_g["time_sec"].to_numpy(dtype=float)
    is_miss = move == "EyesNotFound"
    rows = []
    i = 0
    n = len(is_miss)
    while i < n:
        if not is_miss[i]:
            i += 1
            continue
        j = i
        while j < n and is_miss[j]:
            j += 1
        t0, t1 = float(t[i]), float(t[j - 1])
        dur_ms = max(t1 - t0, 0.0) * 1000.0
        if 50.0 <= dur_ms <= 500.0:
            rows.append({"start_sec": t0, "end_sec": t1, "duration_ms": dur_ms})
        i = j
    return pd.DataFrame(rows)


def blink_window_feats(seg: pd.DataFrame, start: float, end: float) -> None:
    if seg.empty:
        return
    mid = 0.5 * (seg["start_sec"] + seg["end_sec"])
    hit = seg[(mid >= start) & (mid < end)]
    _ = int(len(hit)), float(hit["duration_ms"].mean()) if len(hit) else np.nan


def time_s_prediction() -> dict:
    sys.path.insert(0, str(HERE / "s_score_from_nasa84"))
    sys.path.insert(0, str(HERE / "common"))
    from exp_quota27_s import QUOTA, XGB_CFG, build_mod_idx, mix_s, select_quota
    from exp_s_fullmodal_mi30 import _enable_xgboost
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    _enable_xgboost()
    from xgboost import XGBRegressor

    t0 = now()
    X = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    step = s_table["weighted_step_score"].to_numpy(dtype=float)
    t_load = now() - t0

    t1 = now()
    mod_idx = build_mod_idx(names)
    gkf = GroupKFold(n_splits=5)
    hat = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        top = select_quota(Xtr, y[tr], mod_idx, classif=False)
        m = XGBRegressor(**XGB_CFG)
        m.fit(Xtr[:, top], y[tr])
        hat[te] = m.predict(Xte[:, top])
    s_true = mix_s(step, y, 0.70)
    s_hat = mix_s(step, hat, 0.70)
    t_fit = now() - t1
    return {
        "seconds_load": t_load,
        "seconds_fit_predict": t_fit,
        "seconds": t_load + t_fit,
        "nasa_r2": float(r2_score(y, hat)),
        "s_r2": float(r2_score(s_true, s_hat)),
        "s_mae": float(mean_absolute_error(s_true, s_hat)),
        "n": int(len(y)),
        "quota": QUOTA,
    }


def time_trend() -> dict:
    sys.path.insert(0, str(HERE / "forecast_next_stage"))
    from common_stage import (
        N_SPLITS,
        STEP_W,
        XGB_NASA_CFG,
        align_samples_to_task_order,
        build_mod_idx,
        eligible_mask,
        enable_xgboost,
        load_feature_names,
        load_samples,
        load_task_arrays,
        make_forecast_models,
        mix_s,
        select_quota,
    )
    from run_matrix import MIN_EACH, RATIO_MAIN, stack_stage
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold

    t0 = now()
    names_264, raw = load_feature_names()
    task = load_task_arrays()
    samples = align_samples_to_task_order(load_samples(raw), task["samples"])
    t_load = now() - t0

    t1 = now()
    enable_xgboost()
    from xgboost import XGBRegressor

    packed = stack_stage(samples, RATIO_MAIN)
    mask = eligible_mask(samples, RATIO_MAIN, MIN_EACH)
    md = make_forecast_models()
    nasa_hat = np.full(len(task["y"]), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    mod_idx = build_mod_idx(names_264)
    for tr, te in gkf.split(task["X"], task["y"], task["groups"]):
        imp_full = SimpleImputer(strategy="median")
        xtr_true = imp_full.fit_transform(task["X"][tr])
        xte_early = imp_full.transform(packed["X_early"][te])
        top = select_quota(xtr_true, task["y"][tr], mod_idx)
        imp_e = SimpleImputer(strategy="median")
        e_tr = imp_e.fit_transform(packed["X_early"][tr][:, top])
        e_te = imp_e.transform(packed["X_early"][te][:, top])
        model = md["ridge_scaled"]()
        model.fit(e_tr, xtr_true[:, top])
        pred27 = model.predict(e_te)
        xte_hat = xte_early.copy()
        xte_hat[:, top] = pred27
        m = XGBRegressor(**XGB_NASA_CFG)
        m.fit(xtr_true[:, top], task["y"][tr])
        nasa_hat[te] = m.predict(xte_hat[:, top])
    y, step = task["y"], task["step"]
    s_true = mix_s(step, y, STEP_W)
    msk = mask & np.isfinite(nasa_hat)
    from sklearn.metrics import mean_absolute_error, r2_score

    t_fit = now() - t1
    return {
        "seconds_load": t_load,
        "seconds_fit_predict": t_fit,
        "seconds": t_load + t_fit,
        "s_r2": float(r2_score(s_true[msk], mix_s(step, nasa_hat, STEP_W)[msk])),
        "s_mae": float(mean_absolute_error(s_true[msk], mix_s(step, nasa_hat, STEP_W)[msk])),
        "n_eval": int(msk.sum()),
        "n_samples": int(len(samples)),
        "n_windows": int(sum(len(s.W) for s in samples)),
    }


def time_task_aggregate() -> dict:
    sys.path.insert(0, str(HERE / "regression_task_level"))
    from make_dataset_task import aggregate_one_sample, build_raw_feature_columns

    t0 = now()
    csv_files = sorted(WIN_DIR.glob("subject_*_task_*.csv"))
    first = pd.read_csv(csv_files[0])
    raw = build_raw_feature_columns(first)
    n_by = defaultdict(int)
    for c in raw:
        if c.startswith("eeg_"):
            n_by["脑电"] += 1
        elif c.startswith("hr_"):
            n_by["心率"] += 1
        elif c.startswith("log_"):
            n_by["行为"] += 1
        else:
            n_by["眼动"] += 1
    for f in csv_files:
        df = pd.read_csv(f)
        aggregate_one_sample(df, raw)
    elapsed = now() - t0
    total_n = sum(n_by.values()) or 1
    return {
        "seconds": elapsed,
        "n_files": len(csv_files),
        "n_raw": len(raw),
        "by_mod": dict(n_by),
        "share": {k: elapsed * v / total_n for k, v in n_by.items()},
    }


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1 = load_mod(SCRIPTS / "step1_build_window_samples.py", "step1")
    s2 = load_mod(SCRIPTS / "step2_add_log_features.py", "step2")
    index = pd.read_csv(WIN_DIR / "index.csv")
    n_windows = int(index["window_n"].sum())
    assert n_windows == N_WINDOWS_OFFICIAL, (n_windows, N_WINDOWS_OFFICIAL)
    assert len(index) == N_SAMPLES_OFFICIAL

    eeg_files = used_eeg_files(s1)
    eeg_by_sid = {}
    for p in eeg_files:
        _s, _t, sid, _r = s1.parse_set_name(p)
        eeg_by_sid[sid] = p

    print("[1] 心率 / NASA 读入")
    t0 = now()
    hr = s1.load_hr(DATA / "03_心率" / "心率_逐点时序_26被试.xlsx")
    t_hr_load = now() - t0
    t0 = now()
    nasa = s1.load_nasa(DATA / "01_NASA_TLX" / "NASA_TLX_原始评分_26被试82条.xlsx")
    t_nasa_load = now() - t0
    print(f"    HR {t_hr_load:.2f}s  NASA {t_nasa_load:.2f}s  rows={len(hr)}")

    print("[2] 眼动 TSV 读入与解析")
    eye_dir = DATA / "05_眼动" / "raw_tsv"
    eye_data_paths = sorted(eye_dir.glob("task* Data export.tsv"))
    eye_metric_paths = sorted(eye_dir.glob("task* Metrics.tsv"))
    t0 = now()
    eye_starts = s1.load_eye_starts(eye_data_paths)
    t_eye_starts = now() - t0
    t0 = now()
    eye_ts = s1.load_eye_timeseries(eye_data_paths)
    t_eye_ts = now() - t0
    t0 = now()
    eye_aoi, top_aoi = s1.load_eye_aoi_metrics(eye_metric_paths)
    t_eye_aoi = now() - t0
    t_eye_load = t_eye_starts + t_eye_ts + t_eye_aoi
    print(f"    starts {t_eye_starts:.2f}s  ts {t_eye_ts:.2f}s  aoi {t_eye_aoi:.2f}s")

    print("[3] 脑电 .set 读入（84 条全量）")
    probe_sids = [sid for sid in PROBE_IDS if sid in eeg_by_sid]
    t0 = now()
    bytes_all = 0
    for p in eeg_files:
        s1.read_eeg(p)
        bytes_all += p.stat().st_size
    t_eeg_load = now() - t0
    print(f"    {len(eeg_files)} files {bytes_all/1e6:.1f} MB in {t_eeg_load:.2f}s")

    print("[4] 窗级特征（探针样本外推 12 624 窗）")
    feat = defaultdict(float)
    pre = defaultdict(float)
    windows_timed = 0
    old_dir = DATA / "06_任务表现与操作日志" / "被试01-12_原始日志与分析"
    new_dir = DATA / "06_任务表现与操作日志" / "被试13-26_原始日志与分析"

    for sid in probe_sids:
        row = index[index["sample_id"] == sid].iloc[0]
        eeg_path = eeg_by_sid[sid]
        subject, task, sample_id, repeat = s1.parse_set_name(eeg_path)
        data_task = f"{task}_{repeat}" if repeat and task == "5_6" else task

        t0 = now()
        eeg = s1.read_eeg(eeg_path)
        t_read = now() - t0
        duration_sec = float(eeg["duration_sec"])
        windows = s1.make_windows(duration_sec, WINDOW_SEC, STEP_SEC)
        hr_group = hr[(hr["subject"] == subject) & (hr["task"] == data_task)].copy()
        eye_row = eye_starts[(eye_starts["subject"] == subject) & (eye_starts["task"] == task)]
        if not eye_row.empty:
            task_start = eye_row.iloc[0]["eye_start_datetime"]
        elif not hr_group.empty:
            task_start = hr_group["datetime"].min()
        else:
            task_start = pd.NaT
        eye_g = eye_ts.get((subject, task))
        aoi_g = eye_aoi.get((subject, task))
        t0 = now()
        segs = blink_segments(eye_g)
        t_blink_seg = now() - t0

        subj_pad = f"{int(subject):02d}"
        t0 = now()
        log_path, _src = s2.find_log_path(subj_pad, task, old_dir, new_dir)
        actions = s2.parse_log_file(log_path) if log_path else None
        t_log_parse = now() - t0

        t_eeg = t_hr = t_eye = t_aoi = t_blink = t_log = t_align = 0.0
        for win_start, win_end in windows:
            t0 = now()
            start_time = task_start + pd.to_timedelta(win_start, unit="s") if pd.notna(task_start) else pd.NaT
            end_time = task_start + pd.to_timedelta(win_end, unit="s") if pd.notna(task_start) else pd.NaT
            t_align += now() - t0

            t0 = now()
            s1.eeg_window_features(eeg, win_start, win_end)
            t_eeg += now() - t0

            t0 = now()
            if pd.notna(start_time):
                s1.summarize_hr_window(hr_group, start_time, end_time)
            t_hr += now() - t0

            t0 = now()
            s1.summarize_eye_window(eye_g, win_start, win_end)
            t_eye += now() - t0

            t0 = now()
            s1.summarize_eye_aoi_window(aoi_g, win_start, win_end, top_aoi)
            t_aoi += now() - t0

            t0 = now()
            blink_window_feats(segs, win_start, win_end)
            t_blink += now() - t0

            t0 = now()
            if actions is not None and len(actions):
                s2.compute_log_features(actions, win_start, win_end, win_end)
            t_log += now() - t0

        n_w = len(windows)
        windows_timed += n_w
        pre["脑电"] += t_read
        pre["心率"] += 0.0
        pre["眼动"] += t_align
        feat["脑电"] += t_eeg
        feat["心率"] += t_hr
        feat["眼动"] += t_eye + t_aoi + t_blink_seg + t_blink
        feat["行为"] += t_log_parse + t_log
        print(f"    {sid}: {n_w} 窗  EEG {t_eeg:.2f}s  eye {t_eye+t_aoi:.2f}s  hr {t_hr:.2f}s  log {t_log:.2f}s")

    scale = n_windows / max(windows_timed, 1)
    # 共享读入只发生一次，不随窗数外推；窗级计算按 12 624 窗外推。
    t_pre_eye = t_eye_load + pre["眼动"] * scale
    t_pre_eeg = t_eeg_load  # 读入已按 84 条外推；切窗本身计入对齐
    t_pre_hr = t_hr_load + t_nasa_load
    t_feat_eye = feat["眼动"] * scale
    t_feat_eeg = feat["脑电"] * scale
    t_feat_hr = feat["心率"] * scale
    t_feat_log = feat["行为"] * scale

    print("[5] 任务级 264 维聚合（84 个 CSV 全量）")
    agg = time_task_aggregate()
    t_feat_eeg += agg["share"].get("脑电", 0.0)
    t_feat_hr += agg["share"].get("心率", 0.0)
    t_feat_eye += agg["share"].get("眼动", 0.0)
    t_feat_log += agg["share"].get("行为", 0.0)
    print(f"    aggregate {agg['seconds']:.2f}s")

    print("[6] 绩效 S 回归（五折定额 27 维 XGB）")
    s_pred = time_s_prediction()
    print(f"    {s_pred['seconds']:.2f}s  S R²={s_pred['s_r2']:.3f}")

    print("[7] 趋势预测（v8 ridge_scaled）")
    trend = time_trend()
    print(f"    {trend['seconds']:.2f}s  S R²={trend['s_r2']:.3f}  n={trend['n_eval']}")

    payload = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": machine_info(),
        "protocol": {
            "n_subjects": 26,
            "n_samples": N_SAMPLES_OFFICIAL,
            "n_windows": n_windows,
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_eeg_set": len(eeg_files),
            "probe_sample_ids": probe_sids,
            "windows_timed": windows_timed,
            "window_scale": scale,
        },
        "rows": {
            "眼动预处理": t_pre_eye,
            "脑电预处理": t_pre_eeg,
            "心率预处理": t_pre_hr,
            "行为特征提取": t_feat_log,
            "眼动特征提取": t_feat_eye,
            "脑电特征提取": t_feat_eeg,
            "心率特征提取": t_feat_hr,
            "绩效 S 回归预测": s_pred["seconds"],
            "绩效 S 趋势预测": trend["seconds"],
        },
        "detail": {
            "eye_load": {"starts": t_eye_starts, "timeseries": t_eye_ts, "aoi": t_eye_aoi},
            "eeg_load": {
                "n_files": len(eeg_files),
                "all_bytes": bytes_all,
                "seconds": t_eeg_load,
            },
            "hr_load": t_hr_load,
            "nasa_load": t_nasa_load,
            "window_features_probe": {k: v for k, v in feat.items()},
            "align_probe": dict(pre),
            "aggregate": agg,
            "s_prediction": {k: v for k, v in s_pred.items() if k != "quota"},
            "trend": trend,
        },
    }
    out = OUT_DIR / "timing.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    for k, v in payload["rows"].items():
        print(f"  {k:12s} {v:8.2f}s")
    print("  TOTAL", sum(payload["rows"].values()))
    return out


if __name__ == "__main__":
    main()
