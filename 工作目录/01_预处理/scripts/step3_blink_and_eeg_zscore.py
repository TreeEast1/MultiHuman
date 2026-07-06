#!/usr/bin/env python3
"""Step 3: add blink features (from eye-tracker EyesNotFound segments) and
per-subject z-score standardization for all EEG features.

Inputs:
- 工作目录/01_预处理/output_30s_step5s_with_log/*.csv (from step2)
- data/05_眼动/raw_tsv/task*.tsv (raw eye stream)

Outputs (工作目录/01_预处理/output_30s_step5s_final/):
- subject_XX_task_Y.csv    (original columns + blink_* + eeg_*_z_within_subject)
- index.csv, blink_audit.csv, README.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers reused from step1 (kept local so this script stays self-contained)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Blink extraction
# --------------------------------------------------------------------------- #

BLINK_MIN_MS_DEFAULT = 50   # 过滤追踪抖动导致的短时丢失（<50ms 不作为疑似眨眼）
BLINK_MAX_MS_DEFAULT = 500  # 超出500ms更可能是长时间遮挡/离屏而非单次眨眼


def load_eye_blink_segments(paths: list[Path]) -> dict[tuple[str, str], pd.DataFrame]:
    """Return per (subject, task) DataFrame of EyesNotFound segments with columns:
    start_sec, end_sec, duration_ms.
    A "segment" is a run of consecutive rows whose Eye movement type == EyesNotFound.
    """
    cols = ["Recording timestamp", "Participant name", "Eye movement type"]
    frames = [pd.read_csv(p, sep="\t", usecols=cols, dtype=str) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df["participant"] = df["Participant name"].astype(str).str.strip()
    parsed = df["participant"].map(parse_eye_participant)
    df["subject"] = parsed.map(lambda x: x[0])
    df["task"] = parsed.map(lambda x: x[1])
    df["t_ms"] = pd.to_numeric(df["Recording timestamp"], errors="coerce")
    df = df.dropna(subset=["t_ms", "subject", "task"]).reset_index(drop=True)

    result: dict[tuple[str, str], pd.DataFrame] = {}
    for (subject, task), g in df.groupby(["subject", "task"], sort=False):
        g = g.sort_values("t_ms").reset_index(drop=True)
        etype = g["Eye movement type"].astype(str)
        # 段变化点：type变化或 subject/task 隐式已经按分组分开，因此仅需按 type 变化划段
        seg_id = (etype != etype.shift()).cumsum()
        # 采样间隔（末端补一次典型间隔以覆盖最后一帧）
        dt = g["t_ms"].diff().median()
        if not np.isfinite(dt) or dt <= 0:
            dt = 10.0
        seg = pd.DataFrame({"t_ms": g["t_ms"].values, "etype": etype.values, "seg": seg_id.values})
        enf = seg[seg["etype"] == "EyesNotFound"]
        if enf.empty:
            result[(subject, task)] = pd.DataFrame(columns=["start_sec", "end_sec", "duration_ms"])
            continue
        agg = enf.groupby("seg")["t_ms"].agg(["min", "max", "size"]).reset_index()
        agg["duration_ms"] = (agg["max"] - agg["min"]) + dt  # 帧跨度 + 一个采样步 = 段总时长
        agg["start_sec"] = agg["min"] / 1000.0
        agg["end_sec"] = (agg["max"] + dt) / 1000.0
        result[(subject, task)] = agg[["start_sec", "end_sec", "duration_ms"]].reset_index(drop=True)
    return result


def summarize_blink_window(
    segs: pd.DataFrame | None,
    win_start: float,
    win_end: float,
    min_ms: float,
    max_ms: float,
) -> dict[str, float]:
    """Aggregate blink metrics inside [win_start, win_end).

    We use a "segment-anchored" convention: a segment is counted for the window
    that contains its midpoint. This avoids double-counting a blink that
    straddles two windows. Fractional overlap for total duration is still
    reported separately if you want that view later.
    """
    win_dur = max(win_end - win_start, 1e-9)
    out = {
        "blink_count_raw": 0,          # 未过滤的 EyesNotFound 段数
        "blink_total_ms_raw": 0.0,     # 未过滤总时长（含短时抖动）
        "blink_count": 0,              # 过滤后疑似眨眼次数（min_ms<=dur<=max_ms）
        "blink_rate_per_min": 0.0,     # 眨眼频率
        "blink_duration_mean_ms": np.nan,
        "blink_duration_std_ms": np.nan,
        "blink_duration_median_ms": np.nan,
        "blink_total_duration_ratio": 0.0,  # 眨眼总时长 / 窗口时长
    }
    if segs is None or segs.empty:
        return out

    mids = (segs["start_sec"] + segs["end_sec"]) / 2.0
    in_win = segs[(mids >= win_start) & (mids < win_end)]
    if in_win.empty:
        return out

    out["blink_count_raw"] = int(len(in_win))
    out["blink_total_ms_raw"] = float(in_win["duration_ms"].sum())

    dur = in_win["duration_ms"].astype(float)
    blink_mask = (dur >= min_ms) & (dur <= max_ms)
    blinks = dur[blink_mask]
    n_blink = int(blink_mask.sum())
    out["blink_count"] = n_blink
    out["blink_rate_per_min"] = n_blink / (win_dur / 60.0)
    if n_blink > 0:
        out["blink_duration_mean_ms"] = float(blinks.mean())
        out["blink_duration_std_ms"] = float(blinks.std(ddof=1)) if n_blink >= 2 else np.nan
        out["blink_duration_median_ms"] = float(blinks.median())
        out["blink_total_duration_ratio"] = float(blinks.sum() / (win_dur * 1000.0))
    return out


# --------------------------------------------------------------------------- #
# EEG per-subject z-score
# --------------------------------------------------------------------------- #

def zscore_eeg_within_subject(df_all: pd.DataFrame, eeg_cols: list[str]) -> pd.DataFrame:
    """For every subject compute (x - mean) / std over that subject's rows.
    Add columns "{col}_z_within_subject".
    """
    grouped = df_all.groupby("subject", sort=False)
    z_frames = []
    for col in eeg_cols:
        z = grouped[col].transform(lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0.0)
        z.name = f"{col}_z_within_subject"
        z_frames.append(z)
    z_df = pd.concat(z_frames, axis=1)
    return pd.concat([df_all, z_df], axis=1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../output_30s_step5s_with_log",
                        help="Directory of step2 CSVs (one per subject_task)")
    parser.add_argument("--eye-dir", default="../../../data/05_眼动/raw_tsv",
                        help="Path to raw eye-tracker tsv exports")
    parser.add_argument("--output-dir", default="../output_30s_step5s_final",
                        help="Output directory")
    parser.add_argument("--blink-min-ms", type=float, default=BLINK_MIN_MS_DEFAULT)
    parser.add_argument("--blink-max-ms", type=float, default=BLINK_MAX_MS_DEFAULT)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    in_dir = (script_dir / args.input_dir).resolve()
    eye_dir = (script_dir / args.eye_dir).resolve()
    out_dir = (script_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load per-(subject,task) blink segments
    eye_paths = sorted(eye_dir.glob("task* Data export.tsv"))
    if not eye_paths:
        raise FileNotFoundError(f"No eye exports under {eye_dir}")
    print(f"[step3] loading eye segments from {len(eye_paths)} tsv...")
    blink_segments = load_eye_blink_segments(eye_paths)
    print(f"[step3] blink segments extracted for {len(blink_segments)} (subject,task) pairs")

    # Blink audit rows
    audit_rows = []
    for (subject, task), segs in blink_segments.items():
        dur = segs["duration_ms"].astype(float)
        audit_rows.append({
            "subject": subject,
            "task": task,
            "n_segments_all": int(len(segs)),
            "n_segments_50_500ms": int(((dur >= args.blink_min_ms) & (dur <= args.blink_max_ms)).sum()),
            "seg_duration_median_ms": float(dur.median()) if len(dur) else np.nan,
            "seg_duration_p95_ms": float(dur.quantile(0.95)) if len(dur) else np.nan,
        })
    pd.DataFrame(audit_rows).sort_values(["subject", "task"]).to_csv(
        out_dir / "blink_audit.csv", index=False, encoding="utf-8-sig"
    )

    # Step 2: process each subject_task CSV
    per_file_rows: list[tuple[str, pd.DataFrame]] = []
    for csv_path in sorted(in_dir.glob("subject_*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # normalize types
        df["subject"] = df["subject"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        # per row (window) add blink features
        subj = df["subject"].iloc[0]
        task = str(df["task"].iloc[0])
        segs = blink_segments.get((str(int(subj)), task))
        blink_records = []
        for _, row in df.iterrows():
            blink_records.append(
                summarize_blink_window(
                    segs,
                    float(row["window_start_sec"]),
                    float(row["window_end_sec"]),
                    args.blink_min_ms,
                    args.blink_max_ms,
                )
            )
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(blink_records)], axis=1)
        per_file_rows.append((csv_path.name, df))

    if not per_file_rows:
        raise RuntimeError(f"No input csv found under {in_dir}")

    # Step 3: concatenate to compute per-subject z-score, then split back
    combined = pd.concat([d for _, d in per_file_rows], ignore_index=True)
    eeg_cols = [c for c in combined.columns if c.startswith("eeg_")]
    print(f"[step3] EEG columns to z-score: {len(eeg_cols)}")
    combined = zscore_eeg_within_subject(combined, eeg_cols)

    # Write back per-file, preserving original ordering
    index_rows = []
    start = 0
    for name, orig_df in per_file_rows:
        end = start + len(orig_df)
        chunk = combined.iloc[start:end].reset_index(drop=True)
        out_path = out_dir / name
        chunk.to_csv(out_path, index=False, encoding="utf-8-sig")
        sample_id = str(chunk["sample_id"].iloc[0])
        index_rows.append({
            "sample_id": sample_id,
            "subject": str(chunk["subject"].iloc[0]),
            "task": str(chunk["task"].iloc[0]),
            "window_n": len(chunk),
            "window_file": name,
            "task_difficulty": str(chunk["task_difficulty"].iloc[0]) if "task_difficulty" in chunk.columns else "",
            "nasa_tlx_weighted_task_label": chunk["nasa_tlx_weighted_task_label"].iloc[0]
            if "nasa_tlx_weighted_task_label" in chunk.columns else np.nan,
            "has_blink_segments": int((str(int(chunk["subject"].iloc[0])), str(chunk["task"].iloc[0])) in blink_segments),
        })
        start = end

    pd.DataFrame(index_rows).to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")

    # QC summary
    total_windows = len(combined)
    total_samples = len(index_rows)
    print(f"[step3] wrote {total_samples} sample files, {total_windows} windows")
    print(f"[step3] blink_count coverage: {(combined['blink_count'] > 0).mean():.1%} of windows have >=1 blink")
    print(f"[step3] output dir: {out_dir}")


if __name__ == "__main__":
    main()
