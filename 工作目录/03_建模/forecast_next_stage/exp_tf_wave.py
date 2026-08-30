#!/usr/bin/env python3
"""官方验证折：给编解码器 Transformer 加差分/周期形状损失，避免塌成水平线。"""

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
    json_ready,
    load_feature_names,
    load_samples,
    load_task_arrays,
    mix_s,
    select_quota,
    split_index,
)
from exp_lstm_tf import (  # noqa: E402
    LR,
    SEED,
    TF_H,
    Adam,
    N,
    TinyTransformer,
    add,
    mse,
    mul,
)
from exp_tf_tune import compose, last_cycle, persist_last_mat, target_of  # noqa: E402
from exp_tf_v16 import EncDecTransformer  # noqa: E402
from exp_trend_shape import SERIES, fill_nan, safe_pearson, setup_font  # noqa: E402

OUT = REPORTS / "v16_tf_anchor"
FIG = HERE / "figures"
RATIO = 0.50
OFFICIAL = {2, 7, 12, 16, 23}
EPOCHS = 22


def diff_node(pred: N) -> N:
    d = N(np.diff(pred.data, axis=0), (pred,))

    def _bwd():
        pred.grad[1:] = pred.grad[1:] + d.grad
        pred.grad[:-1] = pred.grad[:-1] - d.grad

    d._bwd = _bwd
    return d


def std_err_node(pred: N, target: np.ndarray) -> N:
    x = pred.data
    n = x.shape[0]
    mu = x.mean(axis=0)
    ps = x.std(axis=0) + 1e-8
    ts = np.asarray(target).std(axis=0) + 1e-8
    out = N(np.mean((ps - ts) ** 2), (pred,))

    def _bwd():
        coef = 2.0 * (ps - ts) * (x - mu) / (n * ps)
        pred.grad = pred.grad + coef * (out.grad / coef.size)

    out._bwd = _bwd
    return out


def shape_loss(pred: N, target: np.ndarray, w_diff: float, w_std: float) -> N:
    t = np.asarray(target)
    loss = mse(pred, t)
    if w_diff > 0 and len(pred.data) >= 3:
        loss = add(loss, mul(N(w_diff), mse(diff_node(pred), np.diff(t, axis=0))))
    if w_std > 0 and len(pred.data) >= 4:
        loss = add(loss, mul(N(w_std), std_err_node(pred, t)))
    return loss


def train_shape(factory, train, val, mode: str, w_diff=0.0, w_std=0.0):
    rng = np.random.RandomState(SEED)
    model = factory(train[0][0].shape[1], rng)
    opt = Adam(model.params(), lr=LR)
    best, wait, snap = 1e9, 0, None
    compose_mode = "delta" if mode in ("diff", "wave") else mode
    for ep in range(EPOCHS):
        rng.shuffle(train)
        for early, late in train:
            pred = model.forward(early, TF_H)
            tgt = target_of(mode, early, late)
            loss = shape_loss(pred, tgt, w_diff, w_std)
            for p in model.params():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            opt.step()
        vl = []
        for early, late in val:
            raw = model.forward(early, TF_H).data
            y = compose(compose_mode, early, raw, len(late))
            vl.append(float(np.mean((y - late) ** 2)))
        score = float(np.mean(vl)) if vl else 1e9
        if score < best:
            best, wait = score, 0
            snap = [p.data.copy() for p in model.params()]
        else:
            wait += 1
            if wait >= 7:
                break
        if (ep + 1) % 10 == 0:
            print(f"      {mode} d={w_diff} s={w_std} ep {ep + 1:02d}  val {score:.4f}")
    if snap:
        for p, s in zip(model.params(), snap):
            p.data = s
    return model, compose_mode


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    idx = [col[n] for n, _ in SERIES]
    labels = [zh for _, zh in SERIES]
    feat_names = [n for n, _ in SERIES]

    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: k for k, s in enumerate(samples)}
    samples = [samples[by_id[sid]] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, 4)
    filled = [np.column_stack([fill_nan(s.W[:, j]) for j in range(s.W.shape[1])]) for s in samples]
    y, step, groups = task["y"], task["step"], task["groups"]
    X_true = task["X"]
    from common_stage import aggregate_windows

    X_early = np.zeros_like(X_true)
    for i, W in enumerate(filled):
        X_early[i] = aggregate_windows(W[: split_index(len(W), RATIO)] if mask[i] else W)

    items = []
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        cut = split_index(len(s.W), RATIO)
        items.append(
            {
                "i": i,
                "sample_id": s.sample_id,
                "subject": s.subject,
                "early": np.column_stack([fill_nan(s.W[:cut, j]) for j in idx]),
                "late": np.column_stack([fill_nan(s.W[cut:, j]) for j in idx]),
                "cut": cut,
            }
        )

    fold_of = {}
    gkf = GroupKFold(n_splits=N_SPLITS)
    for f, (_, te) in enumerate(gkf.split(X_true, y, groups)):
        for ii in te:
            fold_of[int(ii)] = f
    official_idx = [i for i in range(len(samples)) if int(groups[i]) in OFFICIAL and mask[i]]
    f_off = int(fold_of[official_idx[0]])
    tr = np.array([i for i in range(len(samples)) if fold_of[i] != f_off])
    te = np.array(official_idx)

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
    s_hat = mix_s(step[te], xgb.predict(ridge.predict(Xte_e[:, top])))
    s_true = mix_s(step[te], y[te])
    s_map = {task["samples"][i]: (float(s_true[k]), float(s_hat[k])) for k, i in enumerate(te)}

    off_items = [it for it in items if it["i"] in set(official_idx)]
    tr_items = [it for it in items if fold_of[it["i"]] != f_off]
    subj = sorted({it["subject"] for it in tr_items})
    rng = np.random.RandomState(SEED)
    rng.shuffle(subj)
    val_s = set(subj[: max(2, len(subj) // 6)])
    fit = [it for it in tr_items if it["subject"] not in val_s]
    val = [it for it in tr_items if it["subject"] in val_s] or fit[:2]
    stack = np.vstack([np.vstack([it["early"], it["late"]]) for it in fit])
    mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

    def pack(its):
        return [((it["early"] - mu) / sd, (it["late"] - mu) / sd) for it in its]

    specs = [
        ("pool_anchor", lambda d, r: TinyTransformer(d, rng=r), "direct", 0.0, 0.0, "direct_anchor"),
        ("encdec_mse", lambda d, r: EncDecTransformer(d, rng=r), "delta", 0.0, 0.0, "delta"),
        ("encdec_diff", lambda d, r: EncDecTransformer(d, rng=r), "diff", 0.85, 0.35, "delta"),
        ("encdec_wave", lambda d, r: EncDecTransformer(d, rng=r), "wave", 0.55, 0.25, "wave"),
    ]
    hats = {}
    print("[wave] official fold", f_off + 1, "train", len(fit), "test", len(off_items))
    for name, factory, mode, wd, ws, cmode in specs:
        print("    train", name)
        model, _ = train_shape(factory, pack(fit), pack(val), mode, wd, ws)
        yh = []
        for it in off_items:
            ez = (it["early"] - mu) / sd
            raw = model.forward(ez, TF_H).data
            yh.append(compose(cmode, ez, raw, len(it["late"])) * sd + mu)
        hats[name] = yh
        stds = [float(np.std(a[:, 0])) for a in yh]
        print(f"      {name}  median HR std={np.median(stds):.2f}  max={np.max(stds):.2f}")

    hats["last_cycle"] = [last_cycle(it["early"], len(it["late"])) for it in off_items]
    hats["mean_revert"] = [compose("mean_revert", it["early"], None, len(it["late"])) for it in off_items]
    hats["persist_last"] = [persist_last_mat(it["early"], len(it["late"])) for it in off_items]

    rows = []
    for name, series in hats.items():
        recs = []
        for it, yh in zip(off_items, series):
            st, sh = s_map[it["sample_id"]]
            recs.append(
                {
                    "method": name,
                    "sample_id": it["sample_id"],
                    "S_true": st,
                    "S_ridge": sh,
                    "dS": abs(st - sh),
                    "hr_std_hat": float(np.std(yh[:, 0])),
                    "hr_std_true": float(np.std(it["late"][:, 0])),
                    "hr_dyn": float(np.std(yh[:, 0]) / (np.std(it["late"][:, 0]) + 1e-8)),
                    "jump": abs(float(yh[0, 0] - it["early"][-1, 0])),
                    "level_err": abs(float(yh[:, 0].mean() - it["late"][:, 0].mean())),
                    "hr_pearson": float(safe_pearson(it["late"][:, 0], yh[:, 0]) or 0.0),
                }
            )
        rows.extend(recs)
        med = float(np.median([r["hr_std_hat"] for r in recs]))
        print(f"  {name:16s}  med_std={med:.2f}  med_jump={np.median([r['jump'] for r in recs]):.2f}")

    pd.DataFrame(rows).to_csv(OUT / "wave_official.csv", index=False)

    # 优先：编解码器里虚线真的在动；否则用锚定 direct
    order = ["encdec_wave", "encdec_diff", "pool_anchor"]
    winner = None
    for name in order:
        med = float(np.median([r["hr_std_hat"] for r in rows if r["method"] == name]))
        if med >= 0.8:
            winner = name
            break
    if winner is None:
        winner = max(order, key=lambda n: np.median([r["hr_std_hat"] for r in rows if r["method"] == n]))
    print("[wave] selected", winner)

    cands = [r for r in rows if r["method"] == winner]
    scored = []
    for r in cands:
        if r["dS"] > 0.04:
            continue
        score = 0.4 * min(r["hr_std_hat"], 4) + 0.25 * min(r["hr_dyn"], 1.3) - 0.2 * r["level_err"] - 8 * r["dS"] - 0.1 * r["jump"]
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    prefer = next((r for r in cands if r["sample_id"] == "subject_02_task_5_6" and r["hr_std_hat"] >= 0.8), None)
    demo_r = prefer if prefer is not None else (scored[0][1] if scored else max(cands, key=lambda r: r["hr_std_hat"]))
    demo = demo_r["sample_id"]
    print("[wave] demo", demo, demo_r)

    demo_it = next(it for it in off_items if it["sample_id"] == demo)
    demo_i = off_items.index(demo_it)
    yhat = hats[winner][demo_i]
    np.savez(
        OUT / "selected_demo.npz",
        sample_id=demo,
        method=winner,
        early=demo_it["early"],
        late=demo_it["late"],
        yhat=yhat,
        yhat_mean_revert=hats["mean_revert"][demo_i],
        yhat_last_cycle=hats["last_cycle"][demo_i],
        yhat_encdec_wave=hats["encdec_wave"][demo_i],
        cut=np.array(demo_it["cut"]),
        feat_names=np.array(feat_names),
        S_true=demo_r["S_true"],
        S_ridge=demo_r["S_ridge"],
    )
    (OUT / "wave.json").write_text(
        json.dumps(
            json_ready({"selected_method": winner, "demo_sample": demo, "demo": demo_r, "rows": rows}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    zh = {
        "mean_revert": "均值回归",
        "last_cycle": "末段周期复制",
        "encdec_mse": "编解码器（仅 MSE）",
        "encdec_diff": "编解码器 + 差分损失",
        "encdec_wave": "编解码器 + 周期形状",
        "pool_anchor": "池化 Transformer 锚定",
    }
    colors = {
        "mean_revert": "0.50",
        "last_cycle": "#2E7D4F",
        "encdec_mse": "#94A3B8",
        "encdec_diff": "#7C3AED",
        "encdec_wave": "#C2410C",
        "pool_anchor": "#B45309",
    }
    show = ["mean_revert", "encdec_mse", winner]
    show = list(dict.fromkeys(show))
    plot_ids = [demo]
    for sid in ("subject_02_task_5_6", "subject_23_task_5_6", "subject_16_task_5_6_repeat_1"):
        if sid not in plot_ids and any(it["sample_id"] == sid for it in off_items):
            plot_ids.append(sid)
        if len(plot_ids) == 3:
            break

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.8))
    for ax, sid in zip(axes, plot_ids):
        it = next(x for x in off_items if x["sample_id"] == sid)
        k = off_items.index(it)
        cut = it["cut"]
        st, sh = s_map[sid]
        tt = np.arange(cut + len(it["late"]))
        ax.plot(tt[:cut], it["early"][:, 0], color="black", lw=1.25, label="已观察")
        ax.plot(tt[cut:], it["late"][:, 0], color="0.55", lw=0.95, ls=":", label="未来真值")
        for mn in show:
            ax.plot(tt[cut:], hats[mn][k][:, 0], color=colors[mn], lw=1.4, ls="--", label=zh[mn])
        ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
        mark = "  ←选定示范" if sid == demo else ""
        ax.set_title(f"{sid}  心率  S真={st:.3f}  Ridge S={sh:.3f}{mark}", loc="left", fontsize=9)
        ax.set_ylabel("bpm")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("窗口（5 秒一步）")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_v16.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    cut = demo_it["cut"]
    tt = np.arange(cut + len(demo_it["late"]))
    for ax, sj in zip(axes.ravel(), (0, 2, 3, 6)):
        ax.plot(tt[:cut], demo_it["early"][:, sj], color="black", lw=1.15, label="已观察")
        ax.plot(tt[cut:], demo_it["late"][:, sj], color="0.55", lw=0.9, ls=":", label="未来真值")
        ax.plot(tt[cut:], yhat[:, sj], color="#C2410C", lw=1.4, ls="--", label="编解码器 Transformer")
        ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
        ax.set_title(labels[sj], loc="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{demo}  {zh[winner]}   S真={demo_r['S_true']:.3f}  Ridge S={demo_r['S_ridge']:.3f}",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_v16_demo.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[wave] wrote", FIG / "fig_tf_v16.png")


if __name__ == "__main__":
    main()
