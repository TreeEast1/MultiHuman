#!/usr/bin/env python3
"""
Embed log behavioral features into existing 30s/10s window samples.

Strategy:
  - Parse annotated logs from 第二次测试（415） and 第三次测试（517）.
  - Each action line has a relative timestamp (seconds or MM:SS).
  - Classify actions via # 标注: tags when available; otherwise count only.
  - For each existing window, compute:
      A. Window-level (within [start, end]) log features.
      B. Cumulative (task start → window end) log features.
  - Output: enhanced CSVs + audit report + feature summary.

No task-level completion rate, abnormal ratio, or time ratio are used as
features — only window-level and cumulative process features derived from
actions observed up to and including the current window.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_REQUIRED_TIME_MIN = {
    "1": 30.0, "2": 30.0, "3": 10.0, "4": 22.0,
    "5": 12.0, "6": 12.0, "5_6": 18.0,
}

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_timestamp_sec(token: str) -> float:
    """Parse a timestamp token into seconds from task start.

    Formats: '15', '01:11', '03:04:52' (H:MM:SS), '2:48:53'
    """
    token = token.strip()
    parts = token.replace("：", ":").split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return np.nan


# Regex for an action line with optional timestamp
ACTION_RE = re.compile(
    r"^\s*(?P<ts>[\d]{1,2}(?::[\d]{2}){0,2})\s+"
    r"(?P<action>\w+)\s+"
    r"(?P<detail>.+?)\s*"
    r"(?:#\s*标注:\s*(?P<label>.+))?$"
)

# Regex for "resume/restart/--" style lines that reset context
SKIP_PREFIXES = {"reset", "Load", "exit", "unload"}

# Annotation categories
CORRECT_TAGS   = {"正确", "正确-可选"}
ERROR_TAGS     = {"错误"}
DUPLICATE_TAGS = {"重复步骤"}
EXTRA_TAGS     = {"多做"}
ALLOWED_REPEAT = {"允许重复"}  # not counted as error or duplicate
OPTIONAL_TAGS  = {"正确-可选"}


def parse_log_file(path: Path) -> pd.DataFrame | None:
    """Parse one annotated or raw log file into a DataFrame of actions.

    Returns DataFrame with columns:
      timestamp_sec, action, device, detail, label_category, step_info
    or None if the file cannot be parsed meaningfully.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines = text.split("\n")
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip meta-commands
        first_word = line.split()[0] if line.split() else ""
        if first_word in SKIP_PREFIXES:
            continue
        if first_word in {"Run", "Freeze", "unload", "ICM"}:
            continue

        m = ACTION_RE.match(line)
        if not m:
            # Try to match without annotation
            simple = re.match(
                r"^\s*(?P<ts>[\d]{1,2}(?::[\d]{2}){0,2})\s+(?P<action>\w+)\s+(?P<detail>.+)",
                line,
            )
            if simple:
                ts = parse_timestamp_sec(simple.group("ts"))
                action = simple.group("action")
                detail = simple.group("detail").strip()
                label = None
            else:
                continue
        else:
            ts = parse_timestamp_sec(m.group("ts"))
            action = m.group("action")
            detail = m.group("detail").strip()
            label = m.group("label")

        # Extract device from detail
        # detail looks like: "CLOG_3KCO015KM_XJ38.VALUE=1089"
        device_match = re.match(r"([A-Za-z0-9_]+)\.", detail)
        device = device_match.group(1) if device_match else ""

        # Classify label
        category = "unknown"
        if label:
            label_clean = label.strip()
            if any(t in label_clean for t in CORRECT_TAGS):
                category = "correct"
            elif any(t in label_clean for t in DUPLICATE_TAGS):
                category = "duplicate"
            elif any(t in label_clean for t in ERROR_TAGS):
                category = "error"
            elif any(t in label_clean for t in EXTRA_TAGS):
                category = "extra"
            elif any(t in label_clean for t in ALLOWED_REPEAT):
                category = "allowed_repeat"
            else:
                category = "correct"  # default benign

        # Extract step info
        step_info = ""
        if label:
            step_match = re.search(r"步骤(\d+)[：:]", label)
            if step_match:
                step_info = step_match.group(1)

        records.append({
            "timestamp_sec": ts,
            "action": action,
            "device": device,
            "detail": detail,
            "label_category": category,
            "step_info": step_info,
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp_sec").reset_index(drop=True)
    # Filter out rows with NaN timestamps
    df = df.dropna(subset=["timestamp_sec"])
    if df.empty:
        return None
    return df


def is_annotated(df: pd.DataFrame) -> bool:
    """Return True if the parsed log has meaningful annotations."""
    return (df["label_category"] != "unknown").sum() > 0


# ---------------------------------------------------------------------------
# Window feature computation
# ---------------------------------------------------------------------------

def compute_log_features(
    actions: pd.DataFrame,
    win_start: float,
    win_end: float,
    cum_end: float,
) -> dict[str, float]:
    """Compute window-level and cumulative log features.

    Parameters
    ----------
    actions : DataFrame with timestamp_sec, label_category, device
    win_start, win_end : boundaries of the current window (seconds).
    cum_end : end of the cumulative period (typically same as win_end for
              cumulative features that grow monotonically through the task).

    Returns
    -------
    dict of feature_name -> value. NaN if no data.
    """
    feats: dict[str, float] = {}
    annotated = is_annotated(actions)

    if len(actions) == 0:
        return feats

    # ---- Window-level features ----
    win = actions[(actions["timestamp_sec"] >= win_start) & (actions["timestamp_sec"] < win_end)]
    win_n = len(win)
    win_dur = max(win_end - win_start, 1e-6)

    feats["log_action_count_win"] = float(win_n)
    feats["log_unique_device_count_win"] = float(win["device"].replace("", np.nan).dropna().nunique())
    feats["log_action_density_win"] = float(win_n) / win_dur

    if annotated:
        feats["log_correct_action_count_win"]   = float((win["label_category"] == "correct").sum())
        feats["log_error_action_count_win"]     = float((win["label_category"] == "error").sum())
        feats["log_duplicate_action_count_win"]  = float((win["label_category"] == "duplicate").sum())
        feats["log_extra_action_count_win"]      = float((win["label_category"] == "extra").sum())
        feats["log_disallowed_action_count_win"] = float(
            win["label_category"].isin({"error", "duplicate", "extra"}).sum()
        )
        feats["log_error_rate_win"] = (
            feats["log_disallowed_action_count_win"] / max(float(win_n), 1.0)
        )
        feats["log_duplicate_rate_win"] = (
            feats["log_duplicate_action_count_win"] / max(float(win_n), 1.0)
        )
        feats["log_extra_rate_win"] = (
            feats["log_extra_action_count_win"] / max(float(win_n), 1.0)
        )
    else:
        for k in ["correct", "error", "duplicate", "extra", "disallowed"]:
            feats[f"log_{k}_action_count_win"] = np.nan
        for k in ["error_rate", "duplicate_rate", "extra_rate"]:
            feats[f"log_{k}_win"] = np.nan

    # Unique steps touched in window
    step_col = actions["step_info"].replace("", np.nan)
    if step_col.notna().any():
        win_steps = win["step_info"].replace("", np.nan).dropna()
        feats["log_unique_step_count_win"] = float(win_steps.nunique())
    else:
        feats["log_unique_step_count_win"] = np.nan

    # ---- Cumulative features (task start to cum_end) ----
    cum = actions[actions["timestamp_sec"] < cum_end]
    cum_n = len(cum)
    cum_dur = max(cum_end, 1e-6)

    feats["log_action_count_cum"] = float(cum_n)
    feats["log_unique_device_count_cum"] = float(cum["device"].replace("", np.nan).dropna().nunique())
    feats["log_action_density_cum"] = float(cum_n) / cum_dur

    if annotated:
        feats["log_correct_action_count_cum"]   = float((cum["label_category"] == "correct").sum())
        feats["log_error_action_count_cum"]     = float((cum["label_category"] == "error").sum())
        feats["log_duplicate_action_count_cum"]  = float((cum["label_category"] == "duplicate").sum())
        feats["log_extra_action_count_cum"]      = float((cum["label_category"] == "extra").sum())
        feats["log_disallowed_action_count_cum"] = float(
            cum["label_category"].isin({"error", "duplicate", "extra"}).sum()
        )
        feats["log_error_rate_cum"] = (
            feats["log_disallowed_action_count_cum"] / max(float(cum_n), 1.0)
        )
        feats["log_duplicate_rate_cum"] = (
            feats["log_duplicate_action_count_cum"] / max(float(cum_n), 1.0)
        )
        feats["log_extra_rate_cum"] = (
            feats["log_extra_action_count_cum"] / max(float(cum_n), 1.0)
        )
        # Cumulative unique completed steps
        cum_steps = cum["step_info"].replace("", np.nan).dropna()
        feats["log_unique_step_count_cum"] = float(cum_steps.nunique())
    else:
        for k in ["correct", "error", "duplicate", "extra", "disallowed"]:
            feats[f"log_{k}_action_count_cum"] = np.nan
        for k in ["error_rate", "duplicate_rate", "extra_rate"]:
            feats[f"log_{k}_cum"] = np.nan
        feats["log_unique_step_count_cum"] = np.nan

    # Time since last action (relative to cum_end)
    if cum_n > 0:
        last_ts = cum["timestamp_sec"].max()
        feats["log_time_since_last_action_sec"] = max(0.0, cum_end - last_ts)
    else:
        feats["log_time_since_last_action_sec"] = np.nan

    # Cumulative idle time: sum of gaps > 5 seconds between consecutive actions
    if cum_n >= 2:
        gaps = cum["timestamp_sec"].diff().iloc[1:]
        feats["log_cum_gap_count"] = float((gaps > 5.0).sum())
        feats["log_cum_idle_gap_total_sec"] = float(gaps[gaps > 5.0].sum())
        feats["log_cum_idle_ratio"] = feats["log_cum_idle_gap_total_sec"] / cum_dur
        feats["log_cum_mean_inter_action_sec"] = float(gaps.mean()) if len(gaps) else np.nan
    else:
        feats["log_cum_gap_count"] = 0.0
        feats["log_cum_idle_gap_total_sec"] = 0.0
        feats["log_cum_idle_ratio"] = 0.0
        feats["log_cum_mean_inter_action_sec"] = np.nan

    # Action type diversity: count of distinct action verbs
    feats["log_action_type_diversity_cum"] = float(cum["action"].nunique())

    # Recent action rate (last 60 seconds before cum_end)
    recent = cum[cum["timestamp_sec"] >= max(0, cum_end - 60)]
    feats["log_recent_60s_action_count"] = float(len(recent))
    feats["log_recent_60s_action_density"] = float(len(recent)) / min(cum_end, 60.0)

    return feats


# ---------------------------------------------------------------------------
# Log → sample_id mapping
# ---------------------------------------------------------------------------

def log_task_to_sample_task(log_task: str) -> str:
    """Convert log's 2-digit task code to sample's task string.

    log_task: '01','02','03','04','05'
    → sample task: '1','2','3','4','5_6'
    """
    mapping = {"01": "1", "02": "2", "03": "3", "04": "4", "05": "5_6"}
    return mapping.get(log_task, log_task)


def find_log_path(
    subject: str,
    sample_task: str,
    old_test_dir: Path,
    new_test_dir: Path,
) -> tuple[Path | None, str]:
    """Find log file for a subject-task pair from primary sources.

    Mapping:
      - Subjects 01-12 use 第二次测试（415） logs, the old experiment.
      - Subjects 13-26 use 第三次测试（517） logs, the new experiment.

    Returns (path, source) where source is 'annotated' or 'no_log'.
    """
    subject_num = int(subject)
    lookup_subject = subject
    if subject_num <= 12:
        search_dirs = [old_test_dir]
    else:
        search_dirs = [new_test_dir]

    if sample_task == "5_6":
        log_names = [f"{lookup_subject}05_06.txt", f"{lookup_subject}05.txt", "05_06.txt"]
    else:
        log_names = [f"{lookup_subject}{int(sample_task):02d}.txt"]

    for log_dir in search_dirs:
        # First try subject-named subdirectory (like 第二次测试/01/0102.txt)
        subj_dir = log_dir / lookup_subject
        if subj_dir.is_dir():
            for log_name in log_names:
                p = subj_dir / log_name
                if p.exists():
                    return p, "annotated"
        # Then search across all subdirectories (like 第一次测试/第一组/0102.txt)
        for sub_dir in sorted(log_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for log_name in log_names:
                p = sub_dir / log_name
                if p.exists():
                    return p, "annotated"

    return None, "no_log"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_test_dir = data_dir / "06_任务表现与操作日志" / "被试01-12_原始日志与分析"
    new_test_dir = data_dir / "06_任务表现与操作日志" / "被试13-26_原始日志与分析"

    # Load index
    index = pd.read_csv(Path(args.index_csv))
    audit_rows = []
    all_feat_names: set[str] = set()

    for _, row in index.iterrows():
        sample_id = row["sample_id"]
        subject = f"{int(row['subject']):02d}"
        task = str(row["task"])
        label_task = str(row["label_task"])

        # Find log
        log_path, log_source = find_log_path(subject, task, old_test_dir, new_test_dir)

        # Parse log
        actions = parse_log_file(log_path) if log_path else None

        # Read window sample
        sample_file = sample_dir / f"{sample_id}.csv"
        if not sample_file.exists():
            print(f"  SKIP (no window file): {sample_id}")
            continue
        window_df = pd.read_csv(sample_file)

        # Log audit info
        n_actions = len(actions) if actions is not None else 0
        has_ts = actions is not None and "timestamp_sec" in actions.columns and actions["timestamp_sec"].notna().any()
        annotated = is_annotated(actions) if actions is not None else False
        alignment = "relative_time" if has_ts else ("sequence_progress" if actions is not None else "no_log")

        n_feat = 0
        if actions is not None and len(actions) > 0:
            # Add log features to each window
            new_cols = {}
            for idx, wrow in window_df.iterrows():
                ws = float(wrow["window_start_sec"])
                we = float(wrow["window_end_sec"])
                feats = compute_log_features(actions, ws, we, we)
                for k, v in feats.items():
                    if k not in new_cols:
                        new_cols[k] = [np.nan] * len(window_df)
                    new_cols[k][idx] = v
                all_feat_names.update(feats.keys())
            n_feat = len(new_cols)

            for k, vals in new_cols.items():
                window_df[k] = vals
        else:
            n_feat = 0

        # Save enhanced sample
        out_path = out_dir / f"{sample_id}.csv"
        window_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Actions outside EEG duration
        extra_actions = 0
        if actions is not None and len(actions) > 0:
            eeg_dur = row.get("eeg_duration_sec", float(window_df["window_end_sec"].max()))
            # Actually use window max
            max_win = float(window_df["window_end_sec"].max())
            extra_actions = int((actions["timestamp_sec"] > max_win + 10).sum())

        audit_rows.append({
            "sample_id": sample_id,
            "subject": subject,
            "task": task,
            "label_task": label_task,
            "eeg_window_exists": True,
            "log_exists": log_path is not None,
            "log_source": log_source,
            "log_action_count": n_actions,
            "has_timestamps": has_ts,
            "is_annotated": annotated,
            "alignment_method": alignment,
            "n_log_features_added": n_feat,
            "actions_beyond_eeg_duration": extra_actions,
            "notes": "",
        })
        print(f"  {sample_id}: {log_source}, {n_actions} actions, +{n_feat} features")

    # Save audit
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "log_alignment_audit.csv", index=False, encoding="utf-8-sig")

    # Save new index
    new_index = index.copy()
    new_index.to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")

    # Summary
    print(f"\n=== Audit Summary ===")
    print(f"Total samples: {len(audit)}")
    print(f"Log coverage: {audit['log_exists'].sum()} / {len(audit)}")
    print(f"  Annotated: {(audit['log_source'] == 'annotated').sum()}")
    print(f"  Raw only:  {(audit['log_source'] == 'raw').sum()}")
    print(f"  No log:    {(audit['log_source'] == 'no_log').sum()}")
    print(f"Avg log actions: {audit.loc[audit['log_exists'], 'log_action_count'].mean():.1f}")
    print(f"Avg features added: {audit.loc[audit['log_exists'], 'n_log_features_added'].mean():.1f}")
    print(f"Output dir: {out_dir}")
    print(f"Total log feature names: {len(all_feat_names)}")
    for fn in sorted(all_feat_names):
        print(f"  {fn}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../../data", help="Path to the data/ directory")
    parser.add_argument("--index-csv", required=True, help="index.csv produced by step1")
    parser.add_argument("--sample-dir", required=True, help="Window sample CSV dir produced by step1")
    parser.add_argument("--output-dir", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
