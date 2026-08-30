#!/usr/bin/env python3
"""窗级人因时序预报：已观察段 → 未来段每个窗的几条过程曲线。

目标不是 S（S 一场一个数）。这里只看老师界面那种「实线已发生、虚线未来」
能不能用人因过程量画出来，以及几种算法跨被试是否站得住。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    REPORTS,
    eligible_mask,
    load_feature_names,
    load_samples,
    load_task_arrays,
    split_index,
)

OUT = REPORTS / "v10_timeseries_hf"
FIG = HERE / "figures"
RATIO = 0.50
MIN_EACH = 4

SERIES = [
    ("hr_mean", "心率均值"),
    ("hr_std", "心率波动"),
    ("eye_pupil_filtered_mean", "瞳孔直径"),
    ("eye_aoi_coverage_ratio", "AOI覆盖"),
    ("log_action_density_win", "操作密度"),
    ("log_action_count_win", "操作次数"),
    ("eeg_frontal_theta_alpha_z_within_subject", "额区θ/α"),
    ("blink_rate_per_min", "眨眼频率"),
]


def setup_font() -> None:
    for name in ("Songti SC", "STSong", "Heiti SC", "PingFang SC"):
        matches = [f for f in font_manager.fontManager.ttflist if name in f.name]
        if matches:
            plt.rcParams["font.family"] = matches[0].name
            break
    plt.rcParams["axes.unicode_minus"] = False


def safe_r2(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 4:
        return float("nan")
    yt, yh = y[m], yhat[m]
    if np.allclose(yt, yt.mean()):
        return float("nan")
    return float(r2_score(yt, yh))


def safe_mae(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y[m], yhat[m]))


def safe_pearson(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 4:
        return float("nan")
    a, b = y[m], yhat[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_extend(early: np.ndarray, n_late: int) -> np.ndarray:
    n = len(early)
    x = np.arange(n, dtype=np.float64)
    out = np.empty((n_late, early.shape[1]), dtype=np.float64)
    xt = np.arange(n, n + n_late, dtype=np.float64)
    for j in range(early.shape[1]):
        col = early[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            fill = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
            out[:, j] = fill
            continue
        coef = np.polyfit(x[ok], col[ok], 1)
        out[:, j] = coef[0] * xt + coef[1]
    return out


def metrics_block(y_true, y_hat, names, zh) -> list[dict]:
    rows = []
    for j, (name, label) in enumerate(zip(names, zh)):
        yt, yh = y_true[:, j], y_hat[:, j]
        rows.append(
            {
                "feature": name,
                "label": label,
                "r2": safe_r2(yt, yh),
                "mae": safe_mae(yt, yh),
                "pearson": safe_pearson(yt, yh),
                "n": int(np.isfinite(yt).sum()),
            }
        )
    return rows


def fit_predict_oof(X, Y, groups, factory) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = np.full_like(Y, np.nan, dtype=np.float64)
    for tr, te in gkf.split(X, Y[:, 0], groups):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        yimp = SimpleImputer(strategy="median")
        Ytr = yimp.fit_transform(Y[tr])
        m = factory()
        m.fit(Xtr, Ytr)
        pred = m.predict(Xte)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        hat[te] = pred
    return hat


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    idx = [col[n] for n, _ in SERIES]
    labels = [zh for _, zh in SERIES]
    feat_names = [n for n, _ in SERIES]
    del names_264

    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: k for k, s in enumerate(samples)}
    order = [by_id[sid] for sid in task["samples"]]
    samples = [samples[i] for i in order]
    mask = eligible_mask(samples, RATIO, MIN_EACH)
    groups_task = task["groups"]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_of = np.full(len(samples), -1, dtype=int)
    for f, (_, te) in enumerate(gkf.split(task["X"], task["y"], groups_task)):
        fold_of[te] = f

    persist_last, persist_mean, linear, truth = [], [], [], []
    meta_rows = []
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        n = len(s.W)
        cut = split_index(n, RATIO)
        early, late = s.W[:cut][:, idx], s.W[cut:][:, idx]
        n1, n2 = len(early), len(late)
        last = early[-1]
        mu = np.nanmean(early, axis=0)
        persist_last.append(np.repeat(last[None, :], n2, axis=0))
        persist_mean.append(np.repeat(mu[None, :], n2, axis=0))
        linear.append(linear_extend(early, n2))
        truth.append(late)
        meta_rows.append(
            {
                "i": i,
                "sample_id": s.sample_id,
                "subject": s.subject,
                "task": s.task,
                "fold": int(fold_of[i]),
                "n_early": n1,
                "n_late": n2,
            }
        )

    Y = np.vstack(truth)
    hats = {
        "persist_last": np.vstack(persist_last),
        "persist_mean": np.vstack(persist_mean),
        "linear_trend": np.vstack(linear),
    }

    # 条件预报：前段 8 维统计 + 末窗 + 时间位置 → 该未来窗
    xs, ys, grp = [], [], []
    for rec, s in ((r, samples[r["i"]]) for r in meta_rows):
        cut = split_index(len(s.W), RATIO)
        early, late = s.W[:cut][:, idx], s.W[cut:][:, idx]
        n1, n2 = len(early), len(late)
        n = len(s.W)
        last = early[-1]
        mu = np.nanmean(early, axis=0)
        sd = np.nanstd(early, axis=0, ddof=0)
        sl = linear_extend(early, 1)[0] - mu
        ctx = np.concatenate([mu, sd, last, sl])
        for k, row in enumerate(late):
            t_frac = (n1 + k) / max(n - 1, 1)
            remain = (n2 - k) / max(n, 1)
            xs.append(np.concatenate([ctx, [t_frac, remain, n1 / 100.0]]))
            ys.append(row)
            grp.append(s.subject)
    Xh = np.asarray(xs, dtype=np.float64)
    Yh = np.asarray(ys, dtype=np.float64)
    gh = np.asarray(grp)

    factories = {
        "ridge_horizon": lambda: Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
        ),
        "knn5_horizon": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    print(f"[v10] horizon rows {len(Xh)}  series {len(SERIES)}")
    for name, fac in factories.items():
        print(f"  fit {name} …")
        hats[name] = fit_predict_oof(Xh, Yh, gh, fac)

    # 自回归 hop=6：当前 8 维 → 约 30s 后的 8 维，从末窗滚到后段
    xar, yar, gar = [], [], []
    hop = 6
    for s in samples:
        w = s.W[:, idx]
        if len(w) <= hop:
            continue
        for t in range(0, len(w) - hop):
            xar.append(w[t])
            yar.append(w[t + hop])
            gar.append(s.subject)
    Xar = np.asarray(xar, dtype=np.float64)
    Yar = np.asarray(yar, dtype=np.float64)
    print(f"[v10] AR pairs {len(Xar)}")
    ar_models = {}
    gkf = GroupKFold(n_splits=N_SPLITS)
    subj_to_fold = {}
    for f, (tr, te) in enumerate(gkf.split(task["X"], task["y"], groups_task)):
        for i in te:
            subj_to_fold[int(groups_task[i])] = f
        tr_subj = set(int(groups_task[i]) for i in tr)
        sel = np.array([g in tr_subj for g in gar])
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(Xar[sel])
        yimp = SimpleImputer(strategy="median")
        Ytr = yimp.fit_transform(Yar[sel])
        m = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        m.fit(Xtr, Ytr)
        ar_models[f] = (imp, m)

    ar_chunks = []
    for rec in meta_rows:
        s = samples[rec["i"]]
        cut = split_index(len(s.W), RATIO)
        early = s.W[:cut][:, idx]
        n2 = rec["n_late"]
        f = rec["fold"]
        imp, m = ar_models[f]
        cur = early[-1].copy()
        steps = int(np.ceil(n2 / hop))
        pred = []
        for _ in range(max(steps, 1)):
            nxt = m.predict(imp.transform(cur.reshape(1, -1)))[0]
            pred.append(nxt)
            cur = nxt
        seq = np.repeat(np.vstack(pred), hop, axis=0)[:n2]
        ar_chunks.append(seq)
    hats["ridge_ar_hop6"] = np.vstack(ar_chunks)

    row_fold = np.concatenate(
        [np.full(r["n_late"], r["fold"], dtype=int) for r in meta_rows]
    )
    near_mask = []
    for r in meta_rows:
        n2 = r["n_late"]
        k = min(12, n2)
        near_mask.append(np.arange(n2) < k)
    near_mask = np.concatenate(near_mask)

    summary = {"n_windows": int(len(Y)), "n_tasks": int(len(meta_rows)), "series": feat_names}
    table = []
    for model, Yh_hat in hats.items():
        rows = metrics_block(Y, Yh_hat, feat_names, labels)
        m1 = row_fold == 0
        rows1 = metrics_block(Y[m1], Yh_hat[m1], feat_names, labels) if m1.any() else []
        rows_near = metrics_block(Y[near_mask], Yh_hat[near_mask], feat_names, labels)
        mean_r2 = float(np.nanmean([r["r2"] for r in rows]))
        mean_r = float(np.nanmean([r["pearson"] for r in rows]))
        mean_r2_f1 = float(np.nanmean([r["r2"] for r in rows1])) if rows1 else float("nan")
        mean_r2_near = float(np.nanmean([r["r2"] for r in rows_near]))
        table.append(
            {
                "model": model,
                "mean_r2": mean_r2,
                "mean_pearson": mean_r,
                "fold1_mean_r2": mean_r2_f1,
                "near12_mean_r2": mean_r2_near,
                "per_series": rows,
                "fold1_per_series": rows1,
                "near12_per_series": rows_near,
            }
        )
        print(
            f"  {model:16s}  mean R²={mean_r2:+.3f}  Pearson={mean_r:+.3f}  "
            f"fold1 R²={mean_r2_f1:+.3f}  near12 R²={mean_r2_near:+.3f}"
        )

    (OUT / "metrics.json").write_text(
        json.dumps({"summary": summary, "models": table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 画第 1 折三条任务：心率 + 操作密度
    preferred = ["subject_02_task_1", "subject_07_task_2", "subject_12_task_2"]
    rec_by = {r["sample_id"]: r for r in meta_rows}
    fold1_ids = [r["sample_id"] for r in meta_rows if r["fold"] == 0]
    plot_ids = [sid for sid in preferred if sid in rec_by]
    for sid in fold1_ids:
        if sid not in plot_ids:
            plot_ids.append(sid)
        if len(plot_ids) >= 3:
            break
    if not plot_ids:
        plot_ids = [r["sample_id"] for r in meta_rows[:3]]
    offsets = np.cumsum([0] + [r["n_late"] for r in meta_rows])
    pos = {r["sample_id"]: (offsets[k], offsets[k + 1]) for k, r in enumerate(meta_rows)}

    fig, axes = plt.subplots(len(plot_ids), 2, figsize=(8.4, 7.2))
    show_models = ["persist_last", "linear_trend", "ridge_horizon"]
    colors = {"persist_last": "0.45", "linear_trend": "#C45C26", "ridge_horizon": "#1F4E79"}
    series_plot = [0, 4]  # 心率均值、操作密度
    for ri, sid in enumerate(plot_ids):
        rec = rec_by[sid]
        s = samples[rec["i"]]
        cut = split_index(len(s.W), RATIO)
        a, b = pos[sid]
        for ci, sj in enumerate(series_plot):
            ax = axes[ri, ci]
            t_all = np.arange(len(s.W))
            ax.plot(t_all[:cut], s.W[:cut, idx[sj]], color="black", lw=1.1, label="已观察（真值）")
            ax.plot(t_all[cut:], s.W[cut:, idx[sj]], color="0.55", lw=1.0, ls=":", label="未来（真值）")
            for mn in show_models:
                ax.plot(
                    t_all[cut:],
                    hats[mn][a:b, sj],
                    color=colors[mn],
                    lw=1.15,
                    ls="--",
                    label=mn,
                )
            ax.axvline(cut - 0.5, color="0.3", lw=0.7, ls="--")
            ax.set_title(f"{sid}  {labels[sj]}", loc="left", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_timeseries_hf.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # markdown
    lines = [
        "# 窗级人因时序预报（不是 S）\n",
        "S 仍是一场一个数。这里预报的是未来段每个窗口的过程指标，用来回答：",
        "老师界面那种实线/虚线，用人因曲线能不能画、跨被试准不准。\n",
        f"评测：{int(mask.sum())} 条可切段任务的后半段窗口，共 {len(Y)} 个窗；按被试五折。\n",
        "## 八条曲线平均（跨被试）\n",
        "| 算法 | 后半段平均 R² | 平均 Pearson | 第 1 折 R² | 近 12 窗 R² |\n|---|---:|---:|---:|---:|\n",
    ]
    for r in table:
        lines.append(
            f"| {r['model']} | {r['mean_r2']:+.3f} | {r['mean_pearson']:+.3f} | "
            f"{r['fold1_mean_r2']:+.3f} | {r['near12_mean_r2']:+.3f} |\n"
        )
    flat = []
    for r in table:
        for row in r["per_series"]:
            flat.append({"model": r["model"], "horizon": "late_half", **row})
        for row in r["near12_per_series"]:
            flat.append({"model": r["model"], "horizon": "near12", **row})
    pd.DataFrame(flat).to_csv(OUT / "metrics.csv", index=False)
    lines.append("\n## 分指标（全样本）\n")
    best = max(table, key=lambda r: r["mean_r2"] if np.isfinite(r["mean_r2"]) else -99)
    lines.append(f"下表为 `{best['model']}`。\n\n")
    lines.append("| 指标 | R² | MAE | Pearson |\n|---|---:|---:|---:|\n")
    for row in best["per_series"]:
        lines.append(
            f"| {row['label']} | {row['r2']:+.3f} | {row['mae']:.3f} | {row['pearson']:+.3f} |\n"
        )
    lines.append("\n图：`figures/fig_timeseries_hf.png`（验证折三条任务的心率、操作密度）。\n")
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v10] wrote", OUT / "report.md")
    print("[v10] fig", FIG / "fig_timeseries_hf.png")


if __name__ == "__main__":
    main()
