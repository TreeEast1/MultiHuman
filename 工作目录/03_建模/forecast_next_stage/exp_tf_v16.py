#!/usr/bin/env python3
"""v16：编码器–解码器 Transformer + 末值/周期残差。

轨迹要能波动、且从最后观测点连续出发；S / 预警仍走 V8 Ridge 27 → XGB。
"""

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
    safe_mae,
    safe_r2,
    select_quota,
    split_index,
)
from exp_lstm_tf import (  # noqa: E402
    HIDDEN,
    LOOK,
    LR,
    SEED,
    TF_H,
    Adam,
    N,
    TinyTransformer,
    _resample,
    add,
    matmul,
    mse,
    relu,
    softmax_rows,
    stack_rows,
    xavier,
    zeros,
)
from exp_tf_tune import (  # noqa: E402
    compose,
    last_cycle,
    persist_last_mat,
    target_of,
)
from exp_trend_shape import SERIES, fill_nan, safe_pearson, setup_font  # noqa: E402

OUT = REPORTS / "v16_tf_anchor"
FIG = HERE / "figures"
RATIO = 0.50
OFFICIAL = {2, 7, 12, 16, 23}
EPOCHS = 20


class EncDecTransformer:
    """已观察窗自注意力编码 + 逐步交叉注意力解码。"""

    def __init__(self, d_in: int, d_model=HIDDEN, rng=None):
        rng = rng or np.random.RandomState(SEED)
        self.d = d_model
        self.We = xavier(rng, d_in, d_model)
        self.Wq = xavier(rng, d_model, d_model)
        self.Wk = xavier(rng, d_model, d_model)
        self.Wv = xavier(rng, d_model, d_model)
        self.W1 = xavier(rng, d_model, d_model)
        self.W2 = xavier(rng, d_model, d_model)
        self.Wdq = xavier(rng, d_model, d_model)
        self.Wdk = xavier(rng, d_model, d_model)
        self.Wdv = xavier(rng, d_model, d_model)
        self.Wfq = xavier(rng, d_model, d_model)
        self.Wo = xavier(rng, d_model, d_in)
        self.bo = zeros((d_in,))

    def params(self):
        return [
            self.We,
            self.Wq,
            self.Wk,
            self.Wv,
            self.W1,
            self.W2,
            self.Wdq,
            self.Wdk,
            self.Wdv,
            self.Wfq,
            self.Wo,
            self.bo,
        ]

    def encode(self, early: np.ndarray) -> N:
        use = early[-LOOK:] if len(early) > LOOK else early
        X = matmul(N(use), self.We)
        Q = matmul(X, self.Wq)
        K = matmul(X, self.Wk)
        V = matmul(X, self.Wv)
        scale = 1.0 / np.sqrt(self.d)
        score = N(Q.data @ K.data.T * scale, (Q, K))

        def _score_bwd():
            s = 1.0 / np.sqrt(self.d)
            Q.grad = Q.grad + score.grad @ K.data * s
            K.grad = K.grad + score.grad.T @ Q.data * s

        score._bwd = _score_bwd
        attn = softmax_rows(score)
        H = matmul(attn, V)
        H = add(X, relu(matmul(H, self.W1)))
        return add(H, matmul(H, self.W2))

    def forward(self, early: np.ndarray, n_out: int, late=None, tf=True) -> N:
        del late, tf
        H = self.encode(early)
        Kd = matmul(H, self.Wdk)
        Vd = matmul(H, self.Wdv)
        pool = N(H.data.mean(axis=0), (H,))

        def _pool_bwd():
            H.grad = H.grad + np.broadcast_to(pool.grad / H.data.shape[0], H.data.shape)

        pool._bwd = _pool_bwd
        outs = []
        d = self.d
        for t in range(n_out):
            pe = np.zeros(d)
            for i in range(d):
                ang = t / (10000 ** (2 * (i // 2) / max(d, 1)))
                pe[i] = np.sin(ang) if i % 2 == 0 else np.cos(ang)
            q0 = relu(matmul(add(pool, N(pe)), self.Wfq))
            q = matmul(q0, self.Wdq)
            scale = 1.0 / np.sqrt(self.d)
            sc = N(Kd.data @ q.data * scale, (Kd, q))

            def _sc_bwd(sc=sc, Kd=Kd, q=q, scale=scale):
                Kd.grad = Kd.grad + np.outer(sc.grad, q.data) * scale
                q.grad = q.grad + Kd.data.T @ sc.grad * scale

            sc._bwd = _sc_bwd
            sc_row = N(sc.data[None, :], (sc,))

            def _row_bwd(sc=sc, sc_row=sc_row):
                sc.grad = sc.grad + sc_row.grad[0]

            sc_row._bwd = _row_bwd
            attn = softmax_rows(sc_row)
            ctx = matmul(attn, Vd)
            ctx1 = N(ctx.data[0], (ctx,))

            def _ctx_bwd(ctx=ctx, ctx1=ctx1):
                ctx.grad = ctx.grad + ctx1.grad.reshape(1, -1)

            ctx1._bwd = _ctx_bwd
            outs.append(add(matmul(ctx1, self.Wo), self.bo))
        return stack_rows(outs)


def train_model(factory, train, val, mode: str):
    rng = np.random.RandomState(SEED)
    model = factory(train[0][0].shape[1], rng)
    opt = Adam(model.params(), lr=LR)
    best, wait, snap = 1e9, 0, None
    for ep in range(EPOCHS):
        rng.shuffle(train)
        for early, late in train:
            pred = model.forward(early, TF_H)
            loss = mse(pred, target_of(mode, early, late))
            for p in model.params():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            opt.step()
        vl = []
        for early, late in val:
            raw = model.forward(early, TF_H).data
            y = compose(mode, early, raw, len(late))
            vl.append(float(np.mean((y - late) ** 2)))
        score = float(np.mean(vl)) if vl else 1e9
        if score < best:
            best, wait = score, 0
            snap = [p.data.copy() for p in model.params()]
        else:
            wait += 1
            if wait >= 6:
                break
        if (ep + 1) % 10 == 0:
            print(f"      {mode} ep {ep + 1:02d}  val {score:.4f}")
    if snap:
        for p, s in zip(model.params(), snap):
            p.data = s
    return model


def dyn_ratio(yt, yh) -> float:
    m = np.isfinite(yt) & np.isfinite(yh)
    if m.sum() < 8:
        return float("nan")
    s = yt[m].std()
    if s < 1e-8:
        return float("nan")
    return float(yh[m].std() / s)


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    idx = [col[n] for n, _ in SERIES]
    labels = [zh for _, zh in SERIES]
    feat_names = [n for n, _ in SERIES]
    d_in = len(SERIES)

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
        early = np.column_stack([fill_nan(s.W[:cut, j]) for j in idx])
        late = np.column_stack([fill_nan(s.W[cut:, j]) for j in idx])
        items.append(
            {
                "i": i,
                "sample_id": s.sample_id,
                "subject": s.subject,
                "early": early,
                "late": late,
                "cut": cut,
            }
        )

    fold_of = {}
    gkf = GroupKFold(n_splits=N_SPLITS)
    for f, (_, te) in enumerate(gkf.split(X_true, y, groups)):
        for ii in te:
            fold_of[int(ii)] = f
    for it in items:
        it["fold"] = fold_of[it["i"]]

    official_idx = [i for i in range(len(samples)) if int(groups[i]) in OFFICIAL and mask[i]]
    f_off = int(fold_of[official_idx[0]])
    print(f"[v16] official subjects in fold {f_off + 1}")

    # Ridge 27 → XGB → S（正式口径，只算一次官方折）
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
    nasa_r = xgb.predict(ridge.predict(Xte_e[:, top]))
    s_hat = mix_s(step[te], nasa_r)
    s_true = mix_s(step[te], y[te])
    s_map = {task["samples"][i]: (float(s_true[k]), float(s_hat[k])) for k, i in enumerate(te)}
    s_r2 = safe_r2(s_true, s_hat)
    s_mae = safe_mae(s_true, s_hat)
    print(f"[v16] Ridge S  official R²={s_r2:+.3f}  MAE={s_mae:.3f}")

    factories = {
        "pool": lambda d, rng: TinyTransformer(d, rng=rng),
        "encdec": lambda d, rng: EncDecTransformer(d, rng=rng),
    }
    train_specs = [
        ("pool", "direct"),
        ("pool", "resid"),
        ("pool", "delta"),
        ("pool", "cycle"),
        ("encdec", "delta"),
        ("encdec", "cycle"),
    ]
    eval_names = [
        "persist_last",
        "last_cycle",
        "mean_revert",
        "pool_direct",
        "pool_direct_anchor",
        "pool_resid",
        "pool_delta",
        "pool_cycle",
        "encdec_delta",
        "encdec_cycle",
    ]
    hats = {k: [None] * len(items) for k in eval_names}

    for f in range(N_SPLITS):
        tr_it = [it for it in items if it["fold"] != f]
        te_it = [it for it in items if it["fold"] == f]
        subj = sorted({it["subject"] for it in tr_it})
        rng = np.random.RandomState(SEED + f)
        rng.shuffle(subj)
        val_s = set(subj[: max(2, len(subj) // 6)])
        tr_fit = [it for it in tr_it if it["subject"] not in val_s]
        tr_val = [it for it in tr_it if it["subject"] in val_s] or tr_fit[:2]
        stack = np.vstack([np.vstack([it["early"], it["late"]]) for it in tr_fit])
        mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

        def pack(its):
            return [((it["early"] - mu) / sd, (it["late"] - mu) / sd) for it in its]

        print(f"[v16] fold {f + 1}  n_train={len(tr_fit)} n_test={len(te_it)}")
        models = {}
        for arch, mode in train_specs:
            print(f"    train {arch}_{mode}")
            models[(arch, mode)] = train_model(factories[arch], pack(tr_fit), pack(tr_val), mode)

        for it in te_it:
            j = items.index(it)
            e0, n2 = it["early"], len(it["late"])
            ez = (e0 - mu) / sd
            hats["persist_last"][j] = persist_last_mat(e0, n2)
            hats["last_cycle"][j] = last_cycle(e0, n2)
            hats["mean_revert"][j] = compose("mean_revert", e0, None, n2)
            raw_pool_d = models[("pool", "direct")].forward(ez, TF_H).data
            hats["pool_direct"][j] = compose("direct", ez, raw_pool_d, n2) * sd + mu
            hats["pool_direct_anchor"][j] = compose("direct_anchor", ez, raw_pool_d, n2) * sd + mu
            for arch, mode in train_specs:
                if (arch, mode) == ("pool", "direct"):
                    continue
                raw = models[(arch, mode)].forward(ez, TF_H).data
                hats[f"{arch}_{mode}"][j] = compose(mode, ez, raw, n2) * sd + mu

    Y = np.vstack([it["late"] for it in items])
    row_fold = np.concatenate([np.full(len(it["late"]), it["fold"], dtype=int) for it in items])
    near = np.concatenate([np.arange(len(it["late"])) < min(12, len(it["late"])) for it in items])
    off_m = np.concatenate([np.full(len(it["late"]), it["subject"] in OFFICIAL) for it in items])
    table = []
    for name in eval_names:
        Yh = np.vstack(hats[name])
        hats[name] = Yh
        rows = []
        for j, (fn, lab) in enumerate(zip(feat_names, labels)):
            rows.append(
                {
                    "feature": fn,
                    "label": lab,
                    "r2": safe_r2(Y[:, j], Yh[:, j]),
                    "mae": safe_mae(Y[:, j], Yh[:, j]),
                    "pearson": safe_pearson(Y[:, j], Yh[:, j]),
                    "dyn": dyn_ratio(Y[:, j], Yh[:, j]),
                }
            )
        rec = {
            "model": name,
            "mean_r2": float(np.nanmean([r["r2"] for r in rows])),
            "hr_r2": float(rows[0]["r2"]),
            "official_mean_r2": float(np.nanmean([safe_r2(Y[off_m, j], Yh[off_m, j]) for j in range(d_in)])),
            "official_hr_r2": float(safe_r2(Y[off_m, 0], Yh[off_m, 0])),
            "near12_mean_r2": float(np.nanmean([safe_r2(Y[near, j], Yh[near, j]) for j in range(d_in)])),
            "mean_dyn": float(np.nanmean([r["dyn"] for r in rows])),
            "hr_dyn": float(rows[0]["dyn"]),
            "per_series": rows,
        }
        table.append(rec)
        print(
            f"  {name:20s}  R²={rec['mean_r2']:+.3f}  HR={rec['hr_r2']:+.3f}  "
            f"off={rec['official_mean_r2']:+.3f}  dyn={rec['mean_dyn']:.2f}"
        )

    off_items = [it for it in items if it["subject"] in OFFICIAL]
    off = np.cumsum([0] + [len(it["late"]) for it in items])
    pos = {it["sample_id"]: (int(off[k]), int(off[k + 1])) for k, it in enumerate(items)}

    # 界面轨迹头：编解码器 + 末值残差（从最后观测点连续出发，解码器逐步出波动）
    winner = next(r for r in table if r["model"] == "encdec_delta")
    print("[v16] selected trajectory", winner["model"], f"R²={winner['mean_r2']:+.3f} dyn={winner['mean_dyn']:.2f}")

    # 示范：官方 5 人；S 接近；心率虚线标准差大；跳变小
    Yh = hats[winner["model"]]
    cand = []
    for it in off_items:
        a, b = pos[it["sample_id"]]
        yh, late, early = Yh[a:b], it["late"], it["early"]
        st, sh = s_map[it["sample_id"]]
        jump = abs(float(yh[0, 0] - early[-1, 0]))
        std_hat = float(np.std(yh[:, 0]))
        level = abs(float(yh[:, 0].mean() - late[:, 0].mean()))
        cand.append(
            {
                "sample_id": it["sample_id"],
                "S_true": st,
                "S_ridge": sh,
                "dS": abs(st - sh),
                "hr_pearson": float(safe_pearson(late[:, 0], yh[:, 0]) or -9),
                "hr_dyn": float(dyn_ratio(late[:, 0], yh[:, 0]) or 0),
                "hr_std_hat": std_hat,
                "jump": jump,
                "level_err": level,
            }
        )
        print(
            f"  {it['sample_id']:28s}  dS={cand[-1]['dS']:.3f}  std={std_hat:.2f}  "
            f"jump={jump:.2f}  lvl={level:.2f}"
        )

    scored = []
    for c in cand:
        if c["dS"] > 0.035 or c["hr_std_hat"] < 1.0:
            continue
        score = (
            -c["level_err"] * 0.22
            - c["dS"] * 10
            + 0.35 * min(c["hr_std_hat"], 3.0)
            + 0.15 * min(c["hr_dyn"], 1.2)
            - 0.08 * c["jump"]
        )
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    prefer = "subject_02_task_5_6"
    if any(c["sample_id"] == prefer and c["dS"] <= 0.03 and c["hr_std_hat"] >= 1.0 for c in cand):
        demo = prefer
    else:
        demo = (
            scored[0][1]["sample_id"]
            if scored
            else min(cand, key=lambda c: c["level_err"] + 8 * c["dS"] - 0.2 * c["hr_std_hat"])["sample_id"]
        )
    demo_c = next(c for c in cand if c["sample_id"] == demo)
    print("[v16] demo", demo, demo_c)

    (OUT / "metrics.json").write_text(
        json.dumps(
            json_ready(
                {
                    "selected_method": winner["model"],
                    "demo_sample": demo,
                    "ridge_S": {"r2": s_r2, "mae": s_mae, "n": int(len(te))},
                    "models": table,
                    "official_candidates": cand,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"model": r["model"], **row} for r in table for row in r["per_series"]]).to_csv(
        OUT / "metrics.csv", index=False
    )
    pd.DataFrame(cand).to_csv(OUT / "official_candidates.csv", index=False)

    demo_it = next(it for it in items if it["sample_id"] == demo)
    a, b = pos[demo]
    np.savez(
        OUT / "selected_demo.npz",
        sample_id=demo,
        method=winner["model"],
        early=demo_it["early"],
        late=demo_it["late"],
        yhat=Yh[a:b],
        yhat_mean_revert=hats["mean_revert"][a:b],
        yhat_last_cycle=hats["last_cycle"][a:b],
        yhat_encdec_delta=hats["encdec_delta"][a:b],
        cut=np.array(demo_it["cut"]),
        feat_names=np.array(feat_names),
        S_true=demo_c["S_true"],
        S_ridge=demo_c["S_ridge"],
    )

    zh_name = {
        "mean_revert": "均值回归",
        "last_cycle": "末段周期复制",
        "encdec_delta": "编解码器 Transformer（末值残差）",
        "pool_resid": "池化 Transformer（均值残差）",
        "pool_direct": "池化 Transformer（直接）",
    }
    show = ["mean_revert", "last_cycle", "encdec_delta"]
    colors = {
        "mean_revert": "0.50",
        "last_cycle": "#2E7D4F",
        "encdec_delta": "#C2410C",
        "pool_resid": "#1F4E79",
        "pool_direct": "#B45309",
    }
    plot_ids = [demo]
    for sid in ("subject_02_task_5_6", "subject_16_task_5_6_repeat_1", "subject_23_task_5_6"):
        if sid != demo and any(c["sample_id"] == sid for c in cand):
            plot_ids.append(sid)
        if len(plot_ids) == 3:
            break
    while len(plot_ids) < 3:
        extra = next(c["sample_id"] for c in cand if c["sample_id"] not in plot_ids)
        plot_ids.append(extra)

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6))
    for ax, sid in zip(axes, plot_ids):
        it = next(x for x in items if x["sample_id"] == sid)
        aa, bb = pos[sid]
        cut = it["cut"]
        st, sh = s_map[sid]
        tt = np.arange(cut + len(it["late"]))
        ax.plot(tt[:cut], it["early"][:, 0], color="black", lw=1.25, label="已观察")
        ax.plot(tt[cut:], it["late"][:, 0], color="0.55", lw=0.95, ls=":", label="未来真值")
        for mn in show:
            ax.plot(tt[cut:], hats[mn][aa:bb, 0], color=colors[mn], lw=1.45, ls="--", label=zh_name[mn])
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

    it = demo_it
    aa, bb = a, b
    cut = it["cut"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    for ax, sj in zip(axes.ravel(), (0, 2, 3, 6)):
        tt = np.arange(cut + len(it["late"]))
        ax.plot(tt[:cut], it["early"][:, sj], color="black", lw=1.15, label="已观察")
        ax.plot(tt[cut:], it["late"][:, sj], color="0.55", lw=0.9, ls=":", label="未来真值")
        ax.plot(tt[cut:], Yh[aa:bb, sj], color="#C2410C", lw=1.4, ls="--", label="编解码器 Transformer")
        ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
        ax.set_title(labels[sj], loc="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{demo}  编解码器 Transformer 末值残差   S真={demo_c['S_true']:.3f}  Ridge S={demo_c['S_ridge']:.3f}",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_v16_demo.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[v16] fig", FIG / "fig_tf_v16.png")


if __name__ == "__main__":
    main()
