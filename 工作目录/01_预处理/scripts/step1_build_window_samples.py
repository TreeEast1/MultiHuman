#!/usr/bin/env python3
"""Build aligned EEG-HR-eye window samples, one subject-task per file.

This is a weak-supervision dataset builder: each row is a process window, while
NASA/behavior labels remain task-level outcomes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch


TASK_DIFFICULTY = {
    "1": "中",
    "2": "中",
    "4": "中",
    "3": "低",
    "5": "低",
    "5_6": "高",
}

TASK_REQUIRED_TIME_MIN = {
    "1": 30.0,
    "2": 30.0,
    "3": 10.0,
    "4": 22.0,
    "5": 12.0,
    "6": 12.0,
    "5_6": 18.0,
}

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

REGIONS = {
    "frontal": ["FP1", "FP2", "F11", "F7", "F3", "FZ", "F4", "F8", "F12"],
    "central": ["FC3", "FCZ", "FC4", "C3", "CZ", "C4"],
    "parietal": ["CP3", "CPZ", "CP4", "P7", "P3", "PZ", "P4", "P8"],
    "occipital": ["O1", "OZ", "O2"],
}

# Task interfaces use different AOI names. Specific AOI fix-ratio columns can
# encode task identity, so keep only task-comparable aggregate AOI features.
TOP_AOI_N = 0


def normalize_task(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\.0+$", "", text)
    text = text.replace("-", "_")
    if text in {"05_06", "5_06", "05_6"}:
        return "5_6"
    parts = []
    for part in text.split("_"):
        parts.append(str(int(part)) if part.isdigit() else part)
    return "_".join(parts)


def safe_col_token(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_set_name(path: Path) -> tuple[str, str, str, str]:
    match = re.match(r"(?P<subject>\d{2})(?P<body>.+)_preprocessed_eeg\.set$", path.name)
    if not match:
        raise ValueError(f"Cannot parse EEG file name: {path.name}")
    subject = str(int(match.group("subject")))
    body = match.group("body")
    repeat = ""
    if body.startswith("05_06"):
        task = "5_6"
        rest = body.removeprefix("05_06")
        if rest.startswith("_") and rest[1:].isdigit():
            repeat = rest[1:]
    elif body == "0506":
        task = "5_6"
    else:
        task = normalize_task(body[:2])
        rest = body[2:]
        if rest.startswith("_") and rest[1:].isdigit():
            repeat = rest[1:]
    sample_id = f"subject_{int(subject):02d}_task_{task}" + (f"_repeat_{repeat}" if repeat else "")
    return subject, task, sample_id, repeat


def parse_eye_participant(value: object) -> tuple[str, str]:
    text = str(value).strip()
    match = re.match(r"^(?P<subject>\d{2})(?P<body>.+)$", text)
    if not match:
        return "", ""
    subject = str(int(match.group("subject")))
    subject_num = int(subject)
    body = match.group("body").replace("-", "_")
    if "_06" in body or body in {"0506", "05_06"} or (subject_num <= 12 and body.startswith("05")):
        task = "5_6"
    else:
        task = normalize_task(body[:2])
    return subject, task


def read_eeg(path: Path) -> dict[str, object]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    eeg = mat["EEG"]
    labels = [str(ch.labels).upper() for ch in np.atleast_1d(eeg.chanlocs)]
    return {
        "data": np.asarray(eeg.data, dtype=float),
        "srate": float(eeg.srate),
        "pnts": int(eeg.pnts),
        "nbchan": int(eeg.nbchan),
        "labels": labels,
        "duration_sec": float(int(eeg.pnts) / float(eeg.srate)),
    }


def band_power_from_psd(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.nan
    return float(np.nanmean(np.trapezoid(psd[..., mask], freqs[mask], axis=-1)))


def eeg_window_features(eeg: dict[str, object], start_sec: float, end_sec: float) -> dict[str, float]:
    data = eeg["data"]
    srate = float(eeg["srate"])
    labels = list(eeg["labels"])
    start = max(0, int(round(start_sec * srate)))
    end = min(data.shape[1], int(round(end_sec * srate)))
    features: dict[str, float] = {}
    if end <= start:
        return features
    for region, channels in REGIONS.items():
        idx = [labels.index(ch) for ch in channels if ch in labels]
        if not idx:
            continue
        segment = data[idx, start:end]
        if segment.shape[-1] < max(8, int(srate * 2)):
            continue
        nperseg = min(int(srate * 4), segment.shape[-1])
        freqs, psd = welch(segment, fs=srate, nperseg=nperseg, axis=-1)
        powers = {}
        for band, (lo, hi) in BANDS.items():
            powers[band] = band_power_from_psd(freqs, psd, lo, hi)
            features[f"eeg_{region}_{band}_power"] = powers[band]
        features[f"eeg_{region}_theta_alpha"] = powers["theta"] / powers["alpha"] if powers.get("alpha") else np.nan
        features[f"eeg_{region}_beta_alpha"] = powers["beta"] / powers["alpha"] if powers.get("alpha") else np.nan
    return features


def load_hr(path: Path) -> pd.DataFrame:
    hr = pd.read_excel(path, sheet_name="合并数据")
    hr = hr.rename(columns={"被试": "subject", "任务": "task", "日期时间": "datetime", "心率": "hr", "任务难度": "task_difficulty"})
    hr["subject"] = hr["subject"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    hr["task"] = hr["task"].map(normalize_task)
    # Known entry error: HR's subject 11 task 2 is actually 1101.
    hr.loc[(hr["subject"] == "11") & (hr["task"] == "2"), "task"] = "1"
    hr["datetime"] = pd.to_datetime(hr["datetime"], errors="coerce")
    hr["hr"] = pd.to_numeric(hr["hr"], errors="coerce")
    return hr.dropna(subset=["subject", "task", "datetime", "hr"]).sort_values(["subject", "task", "datetime"])


def load_nasa(path: Path) -> pd.DataFrame:
    nasa = pd.read_excel(path, sheet_name=0)
    nasa = nasa.rename(columns={"被试": "subject", "任务": "task", "加权总分": "nasa_tlx_weighted"})
    nasa["subject"] = nasa["subject"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    nasa["task"] = nasa["task"].map(normalize_task)
    keep = ["subject", "task", "任务难度", "nasa_tlx_weighted", "脑力需求", "体力需求", "时间压力", "操作表现", "努力程度", "挫败感"]
    return nasa[[c for c in keep if c in nasa.columns]]


def load_eye_starts(paths: list[Path]) -> pd.DataFrame:
    needed = {
        "Participant name",
        "Recording date",
        "Recording start time",
        "Recording date UTC",
        "Recording start time UTC",
        "Recording duration",
    }
    frames = []
    for path in paths:
        available = pd.read_csv(path, sep="\t", nrows=0).columns
        usecols = [col for col in available if col in needed]
        frames.append(pd.read_csv(path, sep="\t", usecols=usecols, dtype=str))
    eye = pd.concat(frames, ignore_index=True)
    date_col = "Recording date" if "Recording date" in eye.columns else "Recording date UTC"
    start_col = "Recording start time" if "Recording start time" in eye.columns else "Recording start time UTC"
    eye = eye.drop_duplicates(subset=["Participant name", date_col, start_col])
    eye["participant"] = eye["Participant name"].astype(str).str.strip()
    parsed = eye["participant"].map(parse_eye_participant)
    eye["subject"] = parsed.map(lambda x: x[0])
    eye["task"] = parsed.map(lambda x: x[1])
    eye["eye_start_datetime"] = pd.to_datetime(
        eye[date_col].str.strip() + " " + eye[start_col].str.strip(),
        errors="coerce",
    )
    if date_col.endswith("UTC") or start_col.endswith("UTC"):
        eye["eye_start_datetime"] = eye["eye_start_datetime"] + pd.to_timedelta(8, unit="h")
    eye["eye_duration_sec"] = pd.to_numeric(eye["Recording duration"], errors="coerce") / 1000.0
    return eye[["subject", "task", "eye_start_datetime", "eye_duration_sec"]].dropna(subset=["eye_start_datetime"])


def load_eye_timeseries(paths: list[Path]) -> dict[tuple[str, str], pd.DataFrame]:
    cols = [
        "Recording timestamp",
        "Participant name",
        "Pupil diameter left",
        "Pupil diameter right",
        "Pupil diameter filtered",
        "Validity left",
        "Validity right",
        "Eye movement type",
    ]
    frames = [pd.read_csv(path, sep="\t", usecols=cols, dtype=str) for path in paths]
    eye = pd.concat(frames, ignore_index=True)
    eye["participant"] = eye["Participant name"].astype(str).str.strip()
    parsed = eye["participant"].map(parse_eye_participant)
    eye["subject"] = parsed.map(lambda x: x[0])
    eye["task"] = parsed.map(lambda x: x[1])
    eye["time_sec"] = pd.to_numeric(eye["Recording timestamp"], errors="coerce") / 1000.0
    for col in ["Pupil diameter left", "Pupil diameter right", "Pupil diameter filtered"]:
        eye[col] = pd.to_numeric(eye[col], errors="coerce")
    eye["valid_both"] = ((eye["Validity left"] == "Valid") & (eye["Validity right"] == "Valid")).astype(float)
    result = {}
    for key, group in eye.dropna(subset=["time_sec"]).groupby(["subject", "task"], sort=False):
        result[key] = group.sort_values("time_sec").reset_index(drop=True)
    return result


def load_eye_aoi_metrics(paths: list[Path]) -> tuple[dict[tuple[str, str], pd.DataFrame], list[str]]:
    frames = [pd.read_csv(path, sep="\t", dtype=str) for path in paths]
    eye = pd.concat(frames, ignore_index=True)
    eye = eye[eye["TOI"].astype(str) != "Entire Recording"].copy()
    eye["participant"] = eye["Participant"].astype(str).str.strip()
    parsed = eye["participant"].map(parse_eye_participant)
    eye["subject"] = parsed.map(lambda x: x[0])
    eye["task"] = parsed.map(lambda x: x[1])
    for col in [
        "Duration_of_interval",
        "Start_of_interval",
        "Total_duration_of_fixations",
        "Number_of_fixations",
        "Average_pupil_diameter",
    ]:
        eye[col] = pd.to_numeric(eye[col], errors="coerce")
    eye["start_sec"] = eye["Start_of_interval"] / 1000.0
    eye["end_sec"] = (eye["Start_of_interval"] + eye["Duration_of_interval"]) / 1000.0
    eye["fix_ms"] = eye["Total_duration_of_fixations"].fillna(0.0)
    eye["fix_n"] = eye["Number_of_fixations"].fillna(0.0)
    valid = eye.dropna(subset=["subject", "task", "AOI", "start_sec", "end_sec"]).copy()
    top_aoi = (
        valid.groupby("AOI")["fix_ms"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_AOI_N)
        .index.astype(str)
        .tolist()
    )
    result = {}
    keep_cols = [
        "subject",
        "task",
        "TOI",
        "Interval",
        "AOI",
        "start_sec",
        "end_sec",
        "Duration_of_interval",
        "fix_ms",
        "fix_n",
        "Average_pupil_diameter",
    ]
    for key, group in valid[keep_cols].groupby(["subject", "task"], sort=False):
        result[key] = group.sort_values(["start_sec", "AOI"]).reset_index(drop=True)
    return result, top_aoi


def summarize_hr_window(group: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> dict[str, float]:
    g = group[(group["datetime"] >= start_time) & (group["datetime"] < end_time)]
    hr = g["hr"].astype(float)
    out = {
        "hr_n": int(hr.notna().sum()),
        "hr_mean": hr.mean(),
        "hr_std": hr.std(ddof=1),
        "hr_min": hr.min(),
        "hr_max": hr.max(),
    }
    if len(g) >= 2:
        t_min = (g["datetime"] - g["datetime"].min()).dt.total_seconds().to_numpy() / 60.0
        if np.nanmax(t_min) > np.nanmin(t_min):
            out["hr_slope_bpm_per_min"] = float(np.polyfit(t_min, hr.to_numpy(dtype=float), 1)[0])
        else:
            out["hr_slope_bpm_per_min"] = np.nan
    else:
        out["hr_slope_bpm_per_min"] = np.nan
    return out


def summarize_eye_window(group: pd.DataFrame | None, start_sec: float, end_sec: float) -> dict[str, float]:
    if group is None:
        return {"eye_n": 0}
    g = group[(group["time_sec"] >= start_sec) & (group["time_sec"] < end_sec)]
    if g.empty:
        return {"eye_n": 0}
    move = g["Eye movement type"].astype(str)
    return {
        "eye_n": int(len(g)),
        "eye_pupil_filtered_mean": g["Pupil diameter filtered"].mean(),
        "eye_pupil_filtered_std": g["Pupil diameter filtered"].std(ddof=1),
        "eye_valid_ratio": g["valid_both"].mean(),
        "eye_fixation_ratio": float((move == "Fixation").mean()),
        "eye_saccade_ratio": float((move == "Saccade").mean()),
        "eye_eyes_not_found_ratio": float((move == "EyesNotFound").mean()),
    }


def summarize_eye_aoi_window(
    group: pd.DataFrame | None,
    start_sec: float,
    end_sec: float,
    top_aoi: list[str],
) -> dict[str, float]:
    features: dict[str, float] = {
        "eye_aoi_interval_n": 0,
        "eye_aoi_unique_hit_n": 0,
        "eye_aoi_total_fix_ms": 0.0,
        "eye_aoi_fixation_n": 0.0,
        "eye_aoi_fixation_density_per_sec": np.nan,
        "eye_aoi_coverage_ratio": np.nan,
        "eye_aoi_max_share": np.nan,
        "eye_aoi_entropy": np.nan,
        "eye_aoi_pupil_weighted_mean": np.nan,
    }
    for aoi in top_aoi:
        features[f"eye_aoi_{safe_col_token(aoi)}_fix_ratio"] = np.nan

    if group is None or group.empty:
        return features

    g = group[(group["end_sec"] > start_sec) & (group["start_sec"] < end_sec)].copy()
    if g.empty:
        return features

    overlap = np.minimum(g["end_sec"].to_numpy(dtype=float), end_sec) - np.maximum(g["start_sec"].to_numpy(dtype=float), start_sec)
    overlap = np.clip(overlap, 0.0, None)
    duration = g["Duration_of_interval"].replace(0, np.nan).to_numpy(dtype=float) / 1000.0
    weight = np.divide(overlap, duration, out=np.zeros_like(overlap, dtype=float), where=np.isfinite(duration) & (duration > 0))
    g["weighted_fix_ms"] = g["fix_ms"].to_numpy(dtype=float) * weight
    g["weighted_fix_n"] = g["fix_n"].to_numpy(dtype=float) * weight

    by_aoi = g.groupby("AOI", as_index=True)["weighted_fix_ms"].sum()
    hit = by_aoi[by_aoi > 0]
    total_fix_ms = float(by_aoi.sum())
    fix_n = float(g["weighted_fix_n"].sum())
    window_ms = max((end_sec - start_sec) * 1000.0, 1.0)

    features.update(
        {
            "eye_aoi_interval_n": int(len(g)),
            "eye_aoi_unique_hit_n": int(len(hit)),
            "eye_aoi_total_fix_ms": total_fix_ms,
            "eye_aoi_fixation_n": fix_n,
            "eye_aoi_fixation_density_per_sec": fix_n / max(end_sec - start_sec, 1e-9),
            "eye_aoi_coverage_ratio": total_fix_ms / window_ms,
            "eye_aoi_max_share": float(hit.max() / total_fix_ms) if total_fix_ms > 0 and len(hit) else np.nan,
        }
    )
    if total_fix_ms > 0 and len(hit):
        shares = hit.to_numpy(dtype=float) / total_fix_ms
        features["eye_aoi_entropy"] = float(-np.sum(shares * np.log(shares + 1e-12)))
        pupil = pd.to_numeric(g["Average_pupil_diameter"], errors="coerce")
        weights = g["weighted_fix_ms"].to_numpy(dtype=float)
        valid = pupil.notna().to_numpy() & np.isfinite(weights) & (weights > 0)
        if valid.any():
            features["eye_aoi_pupil_weighted_mean"] = float(np.average(pupil.to_numpy(dtype=float)[valid], weights=weights[valid]))
    for aoi in top_aoi:
        features[f"eye_aoi_{safe_col_token(aoi)}_fix_ratio"] = float(by_aoi.get(aoi, 0.0) / total_fix_ms) if total_fix_ms > 0 else np.nan
    return features


def make_windows(duration_sec: float, window_sec: float, step_sec: float) -> list[tuple[float, float]]:
    if duration_sec <= 0:
        return []
    windows = []
    start = 0.0
    while start + window_sec <= duration_sec + 1e-6:
        windows.append((start, start + window_sec))
        start += step_sec
    if not windows or windows[-1][1] < duration_sec:
        tail_start = max(0.0, duration_sec - window_sec)
        tail = (tail_start, duration_sec)
        if not windows or abs(tail[0] - windows[-1][0]) > 1e-6:
            windows.append(tail)
    return windows


def quality_label(hr_inside: int, hr_extra_after_sec: float, anchor_source: str) -> str:
    if hr_inside == 0:
        return "bad_no_hr_inside"
    if anchor_source == "eye" and abs(hr_extra_after_sec) <= 120:
        return "good_eye_anchor"
    if abs(hr_extra_after_sec) <= 120:
        return "usable_hr_anchor"
    return "check_large_duration_gap"


def build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    set_dir = data_dir / "04_EEG" / "raw_set_256Hz"
    out_dir = Path(args.output_dir)
    sample_dir = out_dir / f"window_features_{int(args.window_sec)}s_step{int(args.step_sec)}s"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in sample_dir.glob("*.csv"):
        stale_csv.unlink()

    hr = load_hr(data_dir / "03_心率" / "心率_逐点时序_26被试.xlsx")
    nasa = load_nasa(data_dir / "01_NASA_TLX" / "NASA_TLX_原始评分_26被试82条.xlsx")
    eye_dir = data_dir / "05_眼动" / "raw_tsv"
    eye_data_paths = sorted(eye_dir.glob("task* Data export.tsv"))
    eye_metric_paths = sorted(eye_dir.glob("task* Metrics.tsv"))
    if not eye_data_paths:
        raise FileNotFoundError(f"No eye data exports found under {eye_dir}")
    if not eye_metric_paths:
        raise FileNotFoundError(f"No eye metrics exports found under {eye_dir}")
    eye_starts = load_eye_starts(eye_data_paths)
    eye_ts = load_eye_timeseries(eye_data_paths)
    eye_aoi, top_aoi = load_eye_aoi_metrics(eye_metric_paths)

    audit_rows = []
    index_rows = []
    for eeg_path in sorted(set_dir.glob("*_preprocessed_eeg.set")):
        subject, task, sample_id, repeat = parse_set_name(eeg_path)
        if subject == "20" and task == "5_6" and not repeat:
            continue
        eeg = read_eeg(eeg_path)
        duration_sec = float(eeg["duration_sec"])

        data_task = f"{task}_{repeat}" if repeat and task == "5_6" else task
        hr_group = hr[(hr["subject"] == subject) & (hr["task"] == data_task)].copy()
        eye_row = eye_starts[(eye_starts["subject"] == subject) & (eye_starts["task"] == task)]
        if not eye_row.empty:
            anchor_source = "eye"
            task_start = eye_row.iloc[0]["eye_start_datetime"]
            eye_duration_sec = float(eye_row.iloc[0]["eye_duration_sec"])
        elif not hr_group.empty:
            anchor_source = "hr_first"
            task_start = hr_group["datetime"].min()
            eye_duration_sec = np.nan
        else:
            anchor_source = "missing"
            task_start = pd.NaT
            eye_duration_sec = np.nan

        task_end = task_start + pd.to_timedelta(duration_sec, unit="s") if pd.notna(task_start) else pd.NaT
        hr_inside = hr_group[(hr_group["datetime"] >= task_start) & (hr_group["datetime"] <= task_end)] if pd.notna(task_start) else hr_group.iloc[0:0]
        if hr_inside.empty and not hr_group.empty:
            anchor_source = "hr_first"
            task_start = hr_group["datetime"].min()
            task_end = task_start + pd.to_timedelta(duration_sec, unit="s")
            hr_inside = hr_group[(hr_group["datetime"] >= task_start) & (hr_group["datetime"] <= task_end)]
        hr_start = hr_group["datetime"].min() if not hr_group.empty else pd.NaT
        hr_end = hr_group["datetime"].max() if not hr_group.empty else pd.NaT
        hr_extra_after_sec = (hr_end - task_end).total_seconds() if pd.notna(hr_end) and pd.notna(task_end) else np.nan
        hr_offset_from_anchor_sec = (hr_start - task_start).total_seconds() if pd.notna(hr_start) and pd.notna(task_start) else np.nan

        label_task = data_task
        label_row = nasa[(nasa["subject"] == subject) & (nasa["task"] == label_task)]
        if label_row.empty and label_task != task:
            label_row = nasa[(nasa["subject"] == subject) & (nasa["task"] == task)]
        label = label_row.iloc[0].to_dict() if not label_row.empty else {}
        difficulty = label.get("任务难度") or TASK_DIFFICULTY.get(task, "")

        rows = []
        for i, (win_start, win_end) in enumerate(make_windows(duration_sec, args.window_sec, args.step_sec), start=1):
            start_time = task_start + pd.to_timedelta(win_start, unit="s") if pd.notna(task_start) else pd.NaT
            end_time = task_start + pd.to_timedelta(win_end, unit="s") if pd.notna(task_start) else pd.NaT
            row = {
                "sample_id": sample_id,
                "subject": subject,
                "task": task,
                "repeat_id": repeat,
                "label_task": label_task,
                "window_id": i,
                "window_start_sec": win_start,
                "window_end_sec": win_end,
                "progress_end_ratio": win_end / duration_sec if duration_sec else np.nan,
                "task_required_time_min": TASK_REQUIRED_TIME_MIN.get(task, np.nan),
                "elapsed_required_ratio": win_end / (TASK_REQUIRED_TIME_MIN.get(task, np.nan) * 60.0),
                "anchor_source": anchor_source,
                "task_difficulty": difficulty,
                "nasa_tlx_weighted_task_label": label.get("nasa_tlx_weighted", np.nan),
            }
            row.update(summarize_hr_window(hr_group, start_time, end_time) if pd.notna(start_time) else {"hr_n": 0})
            row.update(eeg_window_features(eeg, win_start, win_end))
            row.update(summarize_eye_window(eye_ts.get((subject, task)), win_start, win_end))
            row.update(summarize_eye_aoi_window(eye_aoi.get((subject, task)), win_start, win_end, top_aoi))
            rows.append(row)

        sample_path = sample_dir / f"{sample_id}.csv"
        pd.DataFrame(rows).to_csv(sample_path, index=False, encoding="utf-8-sig")

        audit_rows.append(
            {
                "sample_id": sample_id,
                "eeg_file": eeg_path.name,
                "subject": subject,
                "task": task,
                "repeat_id": repeat,
                "label_task": label_task,
                "anchor_source": anchor_source,
                "task_start_datetime": task_start,
                "task_end_datetime": task_end,
                "eeg_duration_sec": duration_sec,
                "eeg_srate": eeg["srate"],
                "eeg_pnts": eeg["pnts"],
                "hr_start_datetime": hr_start,
                "hr_end_datetime": hr_end,
                "hr_total_points": int(len(hr_group)),
                "hr_points_inside_eeg_window": int(len(hr_inside)),
                "hr_offset_from_anchor_sec": hr_offset_from_anchor_sec,
                "hr_extra_duration_after_eeg_sec": hr_extra_after_sec,
                "eye_duration_sec": eye_duration_sec,
                "window_file": str(sample_path.relative_to(out_dir)),
                "window_n": len(rows),
                "sync_quality": quality_label(int(len(hr_inside)), float(hr_extra_after_sec) if pd.notna(hr_extra_after_sec) else np.nan, anchor_source),
            }
        )
        index_rows.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "task": task,
                "repeat_id": repeat,
                "label_task": label_task,
                "task_difficulty": difficulty,
                "anchor_source": anchor_source,
                "window_file": str(sample_path.relative_to(out_dir)),
                "window_n": len(rows),
                "has_hr_inside": int(len(hr_inside) > 0),
                "has_eye_timeseries": int((subject, task) in eye_ts),
                "has_eye_aoi_metrics": int((subject, task) in eye_aoi),
                "nasa_tlx_weighted_task_label": label.get("nasa_tlx_weighted", np.nan),
            }
        )

    audit = pd.DataFrame(audit_rows)
    index = pd.DataFrame(index_rows)
    audit.to_csv(out_dir / "alignment_audit.csv", index=False, encoding="utf-8-sig")
    index.to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "alignment_summary.xlsx") as writer:
        index.to_excel(writer, sheet_name="index", index=False)
        audit.to_excel(writer, sheet_name="alignment_audit", index=False)
        audit["sync_quality"].value_counts(dropna=False).rename_axis("sync_quality").reset_index(name="n").to_excel(
            writer, sheet_name="quality_counts", index=False
        )
        pd.DataFrame({"top_aoi": top_aoi}).to_excel(writer, sheet_name="eye_top_aoi", index=False)

    print(f"Saved index: {out_dir / 'index.csv'}")
    print(f"Saved audit: {out_dir / 'alignment_audit.csv'}")
    print(f"Saved sample files: {sample_dir}")
    print(f"Samples: {len(index)}, windows: {int(index['window_n'].sum())}")
    print(audit["sync_quality"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data", help="Path to the data/ directory")
    parser.add_argument("--output-dir", default="./output", help="Where to write window sample CSVs")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--step-sec", type=float, default=10.0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
