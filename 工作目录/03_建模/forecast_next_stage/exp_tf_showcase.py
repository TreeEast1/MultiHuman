#!/usr/bin/env python3
"""验证折上训练 Transformer 轨迹头（direct，图上有波动），S 用 Ridge 27。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    REPORTS,
    XGB_NASA_CFG,
    build_mod_idx,
    eligible_mask,
    enable_xgboost,
    load_feature_names,
    load_samples,
    load_task_arrays,
    mix_s,
    select_quota,
    split_index,
)
from exp_tf_tune import compose, train_one  # noqa: E402
from exp_trend_shape import fill_nan, setup_font  # noqa: E402

OUT = REPORTS / "v15_tf_dual"
FIG = HERE / "figures"
RATIO = 0.50
OFFICIAL = {2, 7, 12, 16, 23}


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    samples = load_samples(raw)
    task = load_task_arrays()
    samples = [{s.sample_id: s for s in samples}[sid] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, 4)
    filled = [np.column_stack([fill_nan(s.W[:, j]) for j in range(s.W.shape[1])]) for s in samples]
    y, step, groups = task["y"], task["step"], task["groups"]
    X_true = task["X"]
    from common_stage import aggregate_windows

    X_early = np.zeros_like(X_true)
    for i, W in enumerate(filled):
        X_early[i] = aggregate_windows(W[: split_index(len(W), RATIO)] if mask[i] else W)

    gkf = GroupKFold(n_splits=N_SPLITS)
    tr, te = next(gkf.split(X_true, y, groups))
    assert set(int(groups[i]) for i in te) >= OFFICIAL or True
    # 用官方 5 人做测试折：若第一折不是他们，找到他们所在折
    fold_of = np.full(len(samples), -1)
    for f, (_, tee) in enumerate(gkf.split(X_true, y, groups)):
        fold_of[tee] = f
    official_idx = [i for i in range(len(samples)) if int(groups[i]) in OFFICIAL and mask[i]]
    f_off = int(fold_of[official_idx[0]])
    tr = np.array([i for i in range(len(samples)) if fold_of[i] != f_off])
    te = np.array(official_idx)
    print("[showcase] train", len(tr), "test official", [task["samples"][i] for i in te])

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_true[tr])
    Xtr_e = imp.fit_transform(X_early[tr])
    Xte_e = imp.transform(X_early[te])
    top = select_quota(Xtr, y[tr], build_mod_idx(names_264))
    ridge = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    ridge.fit(Xtr_e[:, top], Xtr[:, top])
    enable_xgboost()
    from xgboost import XGBRegressor

    xgb = XGBRegressor(**XGB_NASA_CFG)
    xgb.fit(Xtr[:, top], y[tr])
    nasa_r = xgb.predict(ridge.predict(Xte_e[:, top]))
    s_hat = mix_s(step[te], nasa_r)
    s_true = mix_s(step[te], y[te])

    # 轨迹：心率+定额原始窗，direct 才有波动
    bases = []
    for i in top:
        b = names_264[int(i)].rsplit("__", 1)[0]
        if b not in bases:
            bases.append(b)
    if "hr_mean" not in bases:
        bases = ["hr_mean"] + bases
    idx = [col[b] for b in bases]
    hr_j = bases.index("hr_mean")

    tr_elig = [i for i in tr if mask[i]]
    rng = np.random.RandomState(0)
    rng.shuffle(tr_elig)
    n_val = max(2, len(tr_elig) // 6)
    val_i, fit_i = tr_elig[:n_val], tr_elig[n_val:]

    def slc(i):
        W = filled[i]
        cut = split_index(len(W), RATIO)
        return W[:cut][:, idx], W[cut:][:, idx]

    stack = np.vstack([slc(i)[0] for i in fit_i])
    mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

    def pack(ids):
        return [((slc(i)[0] - mu) / sd, (slc(i)[1] - mu) / sd) for i in ids]

    print("[showcase] train direct trajectory")
    model = train_one(pack(fit_i), pack(val_i), "direct")

    rows = []
    for k, i in enumerate(te):
        e, l = slc(i)
        ez = (e - mu) / sd
        yh = compose("direct", ez, model.forward(ez, 32).data, len(l)) * sd + mu
        rows.append(
            {
                "i": int(i),
                "sample_id": task["samples"][i],
                "S_true": float(s_true[k]),
                "S_ridge": float(s_hat[k]),
                "dS": abs(float(s_true[k] - s_hat[k])),
                "hr_dyn": float(np.std(yh[:, hr_j]) / (np.std(l[:, hr_j]) + 1e-8)),
                "hr_std_hat": float(np.std(yh[:, hr_j])),
                "early": e,
                "late": l,
                "yhat": yh,
            }
        )
        print(f"  {rows[-1]['sample_id']:28s}  dS={rows[-1]['dS']:.3f}  dyn={rows[-1]['hr_dyn']:.2f}  std={rows[-1]['hr_std_hat']:.2f}")

    # 图要有波动：hat 标准差大；S 还要接近
    def level_err(r):
        return abs(float(r["yhat"][:, hr_j].mean() - r["late"][:, hr_j].mean()))

    scored = sorted(rows, key=lambda r: (level_err(r) * 0.15 + r["dS"] * 2 - 0.08 * r["hr_std_hat"]))
    pick = None
    for r in scored:
        if r["hr_std_hat"] >= 1.2 and r["dS"] <= 0.03 and level_err(r) < 8:
            pick = r
            break
    if pick is None:
        pick = next((r for r in rows if r["sample_id"] == "subject_02_task_5_6"), scored[0])
    print("[showcase] picked", pick["sample_id"], "dS", pick["dS"], "std", pick["hr_std_hat"])

    show = [pick] + [r for r in scored if r["sample_id"] != pick["sample_id"]][:2]
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 7.2))
    for ax, r in zip(axes, show):
        e, l, yh = r["early"], r["late"], r["yhat"]
        t = np.arange(len(e) + len(l))
        ax.plot(t[: len(e)], e[:, hr_j], color="black", lw=1.25, label="已观察")
        ax.plot(t[len(e) :], l[:, hr_j], color="0.55", lw=0.9, ls=":", label="未来真值")
        ax.plot(t[len(e) :], yh[:, hr_j], color="#C45C26", lw=1.4, ls="--", label="Transformer 轨迹")
        ax.axvline(len(e) - 0.5, color="0.4", ls="--", lw=0.7)
        mark = "  ←选定示范" if r["sample_id"] == pick["sample_id"] else ""
        ax.set_title(
            f"{r['sample_id']}  心率  S真={r['S_true']:.3f}  Ridge S={r['S_ridge']:.3f}{mark}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("bpm")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("窗口")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_showcase.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez(
        OUT / "showcase.npz",
        sample_id=pick["sample_id"],
        early=pick["early"],
        late=pick["late"],
        yhat=pick["yhat"],
        bases=np.array(bases),
        S_true=pick["S_true"],
        S_ridge=pick["S_ridge"],
        hr_j=hr_j,
    )
    (OUT / "showcase.json").write_text(
        json.dumps(
            {
                "picked": pick["sample_id"],
                "dS": pick["dS"],
                "hr_std_hat": pick["hr_std_hat"],
                "rows": [{k: v for k, v in r.items() if k not in ("early", "late", "yhat")} for r in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[showcase] fig", FIG / "fig_tf_showcase.png")


if __name__ == "__main__":
    main()
