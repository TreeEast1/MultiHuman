#!/usr/bin/env python3
"""抽出指定 15 条样本的评估 S、趋势预警 S，并画出对照图与心率趋势曲线。

评估 S：四模态 27 维浅树 XGB 预测 NASA，再按 0.70／0.30 合成（五折折外）。
趋势 S：已观察 27 维 → Ridge 补整场 27 维 → 同一套冻结 XGB → 公式 S。
S 一场任务只有一个数；图上的曲线是窗级心率，不是 S 随时间起伏。

运行：
    uv run --with pandas --with numpy --with scikit-learn --with matplotlib \\
        python export_selected_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
WIN_DIR = MODEL.parent / "01_预处理" / "output_30s_step5s" / "window_features_30s_step5s"
S_TABLE = MODEL / "s_score_from_nasa84" / "output" / "s_score_84samples.csv"
PRED_DIR = MODEL / "s_score_from_nasa84" / "reports_s_fullmodal"
NASA_DS = MODEL / "regression_task_level" / "dataset"
TREND_CSV = MODEL / "forecast_next_stage" / "reports" / "v8_quota27_space" / "models" / "ridge_scaled" / "predictions.csv"
FIG = HERE / "figures"
REP = HERE / "reports"

# 表里 subject_05_task_1 出现两次，去重后按原顺序保留 15 条
SAMPLE_IDS = [
    "subject_05_task_1",
    "subject_20_task_4",
    "subject_17_task_4",
    "subject_15_task_5",
    "subject_12_task_5_6",
    "subject_02_task_5_6",
    "subject_05_task_4",
    "subject_06_task_2",
    "subject_12_task_2",
    "subject_26_task_5_6",
    "subject_16_task_3",
    "subject_15_task_5_6",
    "subject_07_task_3",
    "subject_04_task_5_6",
    "subject_10_task_2",
]
S_THR = 0.51
RATIO = 0.50
OBS = "#1A1D23"
TF = "#E86B2A"
RD = "#1F4E79"
NOW = "#8A8F99"
OK = "#1F8A4C"
WARN = "#C0392B"
C_TRUE = "#2B2B2B"
C_EVAL = "#3A7CA5"
C_TREND = "#E07B39"


def setup_font() -> None:
    for name in ("Songti SC", "STSong", "PingFang SC", "Heiti SC"):
        matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
        if matches:
            plt.rcParams["font.family"] = matches[0].name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["savefig.dpi"] = 220


def mix_s(step, nasa, a=0.70):
    return a * np.asarray(step, dtype=float) + (1.0 - a) * (1.0 - np.asarray(nasa, dtype=float) / 10.0)


def fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return np.zeros_like(y)
    if ok.sum() < len(y):
        idx = np.arange(len(y))
        y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def split_index(n: int, ratio: float = RATIO) -> int:
    cut = int(np.floor(n * ratio))
    return min(max(cut, 1), n - 1)


def minutes(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * 5.0 / 60.0


def mean_revert(early: np.ndarray, n_late: int, rho=0.90) -> np.ndarray:
    y = fill_nan(early)
    mu, last = float(y.mean()), float(y[-1])
    k = np.arange(1, n_late + 1, dtype=np.float64)
    return mu + (last - mu) * (rho**k)


def ridge_overall(early: np.ndarray, n_late: int) -> np.ndarray:
    y = fill_nan(early)
    if len(y) < 2:
        return np.full(n_late, float(y[-1]) if len(y) else 0.0)
    x = np.arange(len(y), dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    last = float(y[-1])
    k = np.arange(1, n_late + 1, dtype=np.float64)
    return last + slope * k


def zh_name(sid: str) -> str:
    subj, task = sid.replace("subject_", "").split("_task_")
    return f"被试 {int(subj)} · 任务 {task}"


def load_table() -> pd.DataFrame:
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    y_nasa = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    st = pd.read_csv(S_TABLE)
    st["sample_id"] = st["sample_id"].astype(str)
    st["task"] = st["task"].astype(str)
    st = st.set_index("sample_id").loc[samples].reset_index()
    step = st["weighted_step_score"].to_numpy(dtype=float)
    y_s07 = np.load(PRED_DIR / "y_s07.npy")
    yhat_s07 = np.load(PRED_DIR / "yhat_s07_xgb.npy")
    yhat_nasa = np.load(PRED_DIR / "yhat_nasa_quota27_mi_reg.npy")
    if not np.allclose(y_s07, mix_s(step, y_nasa), atol=1e-8):
        raise RuntimeError("y_s07 与 0.70／0.30 公式对不齐")
    if not np.allclose(yhat_s07, mix_s(step, yhat_nasa), atol=1e-5):
        raise RuntimeError("yhat_s07 与预测 NASA 合成对不齐")

    tr = pd.read_csv(TREND_CSV)
    tr["sample_id"] = tr["sample_id"].astype(str)
    tr = tr.set_index("sample_id")

    rows = []
    idx = {sid: i for i, sid in enumerate(samples)}
    for sid in SAMPLE_IDS:
        i = idx[sid]
        trow = tr.loc[sid]
        s_true = float(y_s07[i])
        s_eval = float(yhat_s07[i])
        s_trend = float(trow["S_hat"])
        rows.append(
            {
                "sample_id": sid,
                "label": zh_name(sid),
                "subject": int(st.loc[i, "subject"]),
                "task": str(st.loc[i, "task"]),
                "difficulty": str(st.loc[i, "task_difficulty"]),
                "n_windows": int(st.loc[i, "n_windows"]),
                "step": float(step[i]),
                "nasa_true": float(y_nasa[i]),
                "nasa_eval": float(yhat_nasa[i]),
                "nasa_trend": float(trow["nasa_hat"]),
                "S_true": s_true,
                "S_eval": s_eval,
                "S_trend": s_trend,
                "dS_eval": s_eval - s_true,
                "dS_trend": s_trend - s_true,
                "status": "正常" if s_trend >= S_THR else "预警",
            }
        )
    df = pd.DataFrame(rows)
    stats = {
        "n": int(len(df)),
        "n_warn": int((df["status"] == "预警").sum()),
        "n_ok": int((df["status"] == "正常").sum()),
        "eval_r2": float(r2_score(df["S_true"], df["S_eval"])),
        "eval_mae": float(mean_absolute_error(df["S_true"], df["S_eval"])),
        "trend_r2": float(r2_score(df["S_true"], df["S_trend"])),
        "trend_mae": float(mean_absolute_error(df["S_true"], df["S_trend"])),
        "S_true_min": float(df["S_true"].min()),
        "S_true_max": float(df["S_true"].max()),
        "S_true_mean": float(df["S_true"].mean()),
        "threshold": S_THR,
        "note": "评估 S 为完整观测 27 维公式法五折折外；趋势 S 为半场 Ridge→冻结 XGB。",
    }
    return df, stats


def savefig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig", p)
    return p


def fig_eval_scatter(df: pd.DataFrame, stats: dict) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(
        df["S_true"], df["S_eval"], s=42, c="white",
        edgecolors=C_TRUE, linewidths=0.95, zorder=3,
    )
    for _, r in df.iterrows():
        ax.annotate(
            f'{int(r.subject)}-{r.task}',
            (r.S_true, r.S_eval),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7, color="#4B5563",
        )
    lim = (0.10, 0.95)
    ax.plot(lim, lim, color=C_TRUE, lw=0.9, ls="--")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("真值 S")
    ax.set_ylabel("评估预测 S（完整观测 27 维公式法）")
    ax.set_title(
        f"本批 15 条：R²＝{stats['eval_r2']:.3f}，MAE＝{stats['eval_mae']:.3f}",
        loc="left", fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return savefig(fig, "fig1_eval_scatter.png")


def fig_trend_scatter(df: pd.DataFrame, stats: dict) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ok = df["status"] == "正常"
    ax.scatter(
        df.loc[ok, "S_true"], df.loc[ok, "S_trend"], s=46, c="white",
        edgecolors=OK, linewidths=1.05, zorder=3, label="正常",
    )
    ax.scatter(
        df.loc[~ok, "S_true"], df.loc[~ok, "S_trend"], s=46, c="white",
        edgecolors=WARN, linewidths=1.05, zorder=3, label="预警",
    )
    for _, r in df.iterrows():
        ax.annotate(
            f'{int(r.subject)}-{r.task}',
            (r.S_true, r.S_trend),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7, color="#4B5563",
        )
    lim = (0.10, 0.95)
    ax.plot(lim, lim, color=C_TRUE, lw=0.9, ls="--")
    ax.axhline(S_THR, color=WARN, lw=0.8, ls=":", label=f"阈值 {S_THR:.2f}")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("真值 S")
    ax.set_ylabel("趋势预警预测 S（半场 Ridge）")
    ax.set_title(
        f"本批 15 条：R²＝{stats['trend_r2']:.3f}，MAE＝{stats['trend_mae']:.3f}",
        loc="left", fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return savefig(fig, "fig2_trend_scatter.png")


def fig_s_compare(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    x = np.arange(len(df))
    w = 0.26
    ax.bar(x - w, df["S_true"], w, label="真值 S", color=C_TRUE, alpha=0.85)
    ax.bar(x, df["S_eval"], w, label="评估预测 S", color=C_EVAL, alpha=0.90)
    ax.bar(x + w, df["S_trend"], w, label="趋势预测 S", color=C_TREND, alpha=0.90)
    ax.axhline(S_THR, color=WARN, lw=0.9, ls="--")
    ax.text(len(df) - 0.4, S_THR + 0.02, f"预警阈值 {S_THR:.2f}", color=WARN, fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{int(r.subject)}-{r.task}' for _, r in df.iterrows()],
        rotation=40, ha="right", fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("S")
    ax.set_title("本批样本的真值 S、评估预测 S 与趋势预测 S", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return savefig(fig, "fig3_s_compare.png")


def fig_s_alpha(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    alphas = np.linspace(0.0, 1.0, 101)
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    for i, r in df.iterrows():
        nasa_rev = 1.0 - r.nasa_true / 10.0
        curve = alphas * r.step + (1.0 - alphas) * nasa_rev
        ax.plot(alphas, curve, color=colors[i], lw=1.15, label=f'{int(r.subject)}-{r.task}')
        ax.plot([0.70], [r.S_true], marker="o", color=colors[i], markersize=4.5, markeredgecolor="white")
    ax.axvline(0.70, color=C_TRUE, ls="--", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("α（客观步骤分的权重）")
    ax.set_ylabel("S")
    ax.set_title("本批 15 条的 S 随 α 变化（圆点为正式口径 α＝0.70）", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return savefig(fig, "fig4_s_alpha.png")


def fig_warning(df: pd.DataFrame) -> Path:
    order = df.sort_values("S_trend", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(order))
    colors = [WARN if s == "预警" else OK for s in order["status"]]
    ax.hlines(y, S_THR, order["S_trend"], color="#D0D4DA", lw=1.2)
    ax.scatter(order["S_trend"], y, s=48, c=colors, zorder=3)
    ax.axvline(S_THR, color=WARN, lw=0.95, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f'{int(r.subject)}-{r.task}' for _, r in order.iterrows()], fontsize=9)
    ax.set_xlabel("趋势预测 S")
    ax.set_xlim(0.10, 0.95)
    ax.set_title(f"趋势预警判定（阈值 {S_THR:.2f}；红＝预警，绿＝正常）", loc="left", fontsize=11)
    for yi, (_, r) in enumerate(order.iterrows()):
        ax.text(r.S_trend + 0.012, yi, f'{r.S_trend:.3f}  {r.status}', va="center", fontsize=8, color=colors[yi])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return savefig(fig, "fig5_warning.png")


def load_hr(sid: str) -> np.ndarray:
    path = WIN_DIR / f"{sid}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    w = pd.read_csv(path)
    return fill_nan(w["hr_mean"].to_numpy(dtype=float))


def fig_hr_grid(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(5, 3, figsize=(12.4, 13.2), sharex=False)
    axes = axes.ravel()
    for i, r in df.iterrows():
        ax = axes[i]
        y = load_hr(r.sample_id)
        cut = split_index(len(y))
        t = minutes(len(y))
        y_tf = mean_revert(y[:cut], len(y) - cut)
        y_rd = ridge_overall(y[:cut], len(y) - cut)
        last = float(y[cut - 1])
        t_f = np.concatenate([[t[cut - 1]], t[cut:]])
        ax.plot(t[:cut], y[:cut], color=OBS, lw=1.15)
        ax.axvspan(t[cut - 1], t[-1], color="#FFF4EC", alpha=0.45, zorder=0)
        ax.axvline(t[cut - 1], color=NOW, lw=0.7, ls="--")
        ax.plot(t_f, np.concatenate([[last], y_tf]), color=TF, lw=1.15, ls=(0, (4.2, 2.2)))
        ax.plot(t_f, np.concatenate([[last], y_rd]), color=RD, lw=1.35, ls=(0, (1.1, 1.3)))
        tag = r.status
        ax.set_title(
            f'{r.label}    Ŝ＝{r.S_trend:.3f}  {tag}',
            loc="left", fontsize=9,
            color=WARN if tag == "预警" else OBS,
        )
        ax.tick_params(labelsize=7.5, colors="#4B5563")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i >= 12:
            ax.set_xlabel("任务时间（分钟）", fontsize=8)
        if i % 3 == 0:
            ax.set_ylabel("bpm", fontsize=8)
    fig.suptitle(
        "本批 15 条的趋势预警主图（心率均值）\n黑实线＝已观察；橙虚线＝瞬时走势；蓝点线＝整体走势；右侧浅底＝预测段",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, "fig6_trend_hr_grid.png")


def main() -> None:
    setup_font()
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    df, stats = load_table()
    df.to_csv(REP / "selected_samples.csv", index=False, encoding="utf-8-sig")
    (REP / "selected_samples.json").write_text(
        json.dumps({"stats": stats, "rows": df.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(df[["sample_id", "S_true", "S_eval", "S_trend", "status"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("stats", json.dumps(stats, ensure_ascii=False))
    fig_eval_scatter(df, stats)
    fig_trend_scatter(df, stats)
    fig_s_compare(df)
    fig_s_alpha(df)
    fig_warning(df)
    fig_hr_grid(df)
    print("wrote", REP / "selected_samples.csv")


if __name__ == "__main__":
    main()
