#!/usr/bin/env python3
"""Transformer 双头：序列编码读出整场 27 维（走正式 XGB→S）+ 轨迹头画波动虚线。"""

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
    STEP_W,
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
    SEED,
    Adam,
    N,
    TinyTransformer,
    add,
    matmul,
    mse,
    relu,
    xavier,
    zeros,
)
from exp_tf_tune import compose, train_one  # noqa: E402
from exp_trend_shape import fill_nan, setup_font  # noqa: E402

OUT = REPORTS / "v15_tf_dual"
FIG = HERE / "figures"
RATIO = 0.50
EPOCHS = 25
LR = 8e-4


def concat1d(a, b):
    out = N(np.concatenate([np.ravel(a.data), np.ravel(b.data)]), (a, b))

    def _bwd():
        na = a.data.size
        a.grad = a.grad + out.grad[:na].reshape(a.data.shape)
        b.grad = b.grad + out.grad[na:].reshape(b.data.shape)

    out._bwd = _bwd
    return out


class FuseDelta27:
    """pool ⊕ 已观察 27 → 残差补全整场 27。"""

    def __init__(self, d_in_seq: int, d27: int, rng):
        self.enc = TinyTransformer(d_in_seq, rng=rng)
        self.W = xavier(rng, self.enc.d + d27, 32)
        self.W2 = xavier(rng, 32, d27)
        self.b = zeros((d27,))

    def params(self):
        return self.enc.params() + [self.W, self.W2, self.b]

    def pred27(self, early_z: np.ndarray, e27_z: np.ndarray) -> N:
        p = self.enc.encode_pool(early_z)
        h = relu(matmul(concat1d(p, N(e27_z)), self.W))
        return add(N(e27_z), add(matmul(h, self.W2), self.b))


class Pool27:
    def __init__(self, d_in_seq: int, d27: int, rng):
        self.enc = TinyTransformer(d_in_seq, rng=rng)
        self.W = xavier(rng, self.enc.d, 32)
        self.W2 = xavier(rng, 32, d27)
        self.b = zeros((d27,))

    def params(self):
        return self.enc.params() + [self.W, self.W2, self.b]

    def pred27(self, early_z: np.ndarray, e27_z: np.ndarray) -> N:
        del e27_z
        h = relu(matmul(self.enc.encode_pool(early_z), self.W))
        return add(matmul(h, self.W2), self.b)


def raw_bases(names_264, idx27):
    seen = []
    for i in idx27:
        b = names_264[int(i)].rsplit("__", 1)[0]
        if b not in seen:
            seen.append(b)
    return seen


def train_readout(cls, train, val, d_seq, d27):
    rng = np.random.RandomState(SEED)
    model = cls(d_seq, d27, rng)
    opt = Adam(model.params(), lr=LR)
    best, wait, snap = 1e9, 0, None
    for ep in range(EPOCHS):
        rng.shuffle(train)
        for ez, e27, y27 in train:
            loss = mse(model.pred27(ez, e27), y27)
            for p in model.params():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            opt.step()
        vl = []
        for ez, e27, y27 in val:
            yh = model.pred27(ez, e27).data
            vl.append(float(np.mean((yh - y27) ** 2)))
        score = float(np.mean(vl)) if vl else 1e9
        if score < best:
            best, wait = score, 0
            snap = [p.data.copy() for p in model.params()]
        else:
            wait += 1
            if wait >= 7:
                break
        if (ep + 1) % 10 == 0:
            print(f"      {cls.__name__} ep {ep+1:02d}  val {score:.4f}")
    if snap:
        for p, s in zip(model.params(), snap):
            p.data = s
    return model


def fold_metrics(y, step, nasa, groups, mask):
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    dummy = np.zeros(len(y))
    for f, (_, te) in enumerate(gkf.split(dummy, y, groups)):
        m = mask[te] & np.isfinite(nasa[te])
        st, yt, nh = step[te][m], y[te][m], nasa[te][m]
        rows.append(
            {
                "fold": f,
                "n": int(m.sum()),
                "nasa_r2": safe_r2(yt, nh),
                "s_r2": safe_r2(mix_s(st, yt), mix_s(st, nh)),
                "s_mae": safe_mae(mix_s(st, yt), mix_s(st, nh)),
            }
        )
    return rows


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    enable_xgboost()
    from xgboost import XGBRegressor

    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    samples = load_samples(raw)
    task = load_task_arrays()
    samples = [{s.sample_id: s for s in samples}[sid] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, 4)
    y, step, groups = task["y"], task["step"], task["groups"]
    X_true = task["X"]

    from common_stage import aggregate_windows

    filled = [np.column_stack([fill_nan(s.W[:, j]) for j in range(s.W.shape[1])]) for s in samples]
    X_early = np.zeros_like(X_true)
    for i, W in enumerate(filled):
        X_early[i] = aggregate_windows(W[: split_index(len(W), RATIO)] if mask[i] else W)

    names = ["persist_early", "ridge_27", "tf_fuse_delta", "tf_pool"]
    nasa = {k: np.full(len(y), np.nan) for k in names}
    plot_hat = {}
    gkf = GroupKFold(n_splits=N_SPLITS)
    mod_idx = build_mod_idx(names_264)

    for f, (tr, te) in enumerate(gkf.split(X_true, y, groups)):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_true[tr])
        Xte_true = imp.transform(X_true[te])
        Xte_early = imp.transform(X_early[te])
        Xtr_early = imp.transform(X_early[tr])
        top = select_quota(Xtr, y[tr], mod_idx)
        bases = raw_bases(names_264, top)
        idx = [col[b] for b in bases]
        print(f"[v15] fold {f+1}  raw={len(idx)}")

        # Ridge 对照（与正式 V8 相同）
        ridge = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        ridge.fit(Xtr_early[:, top], Xtr[:, top])
        pred_ridge = ridge.predict(Xte_early[:, top])

        tr_elig = [i for i in tr if mask[i]]
        subj = sorted({int(groups[i]) for i in tr_elig})
        rng = np.random.RandomState(SEED + f)
        rng.shuffle(subj)
        val_s = set(subj[: max(2, len(subj) // 6)])
        tr_fit = [i for i in tr_elig if int(groups[i]) not in val_s]
        tr_val = [i for i in tr_elig if int(groups[i]) in val_s] or tr_fit[:2]

        def seq_of(i):
            W = filled[i]
            cut = split_index(len(W), RATIO)
            return W[:cut][:, idx], W[cut:][:, idx]

        seq_stack = np.vstack([seq_of(i)[0] for i in tr_fit])
        mu_s, sd_s = seq_stack.mean(0), np.where(seq_stack.std(0) < 1e-6, 1.0, seq_stack.std(0))
        mu27, sd27 = Xtr[:, top].mean(0), np.where(Xtr[:, top].std(0) < 1e-6, 1.0, Xtr[:, top].std(0))

        def pack(ids):
            rows = []
            for i in ids:
                e, _ = seq_of(i)
                ez = (e - mu_s) / sd_s
                e27 = (X_early[i, top] - mu27) / sd27
                y27 = (X_true[i, top] - mu27) / sd27
                rows.append((ez, e27, y27))
            return rows

        print("    train fuse_delta / pool / plot-resid")
        fuse = train_readout(FuseDelta27, pack(tr_fit), pack(tr_val), len(idx), len(top))
        poolm = train_readout(Pool27, pack(tr_fit), pack(tr_val), len(idx), len(top))
        plot_pairs = []
        for i in tr_fit:
            e, l = seq_of(i)
            plot_pairs.append(((e - mu_s) / sd_s, (l - mu_s) / sd_s))
        val_pairs = []
        for i in tr_val:
            e, l = seq_of(i)
            val_pairs.append(((e - mu_s) / sd_s, (l - mu_s) / sd_s))
        plot_m = train_one(plot_pairs, val_pairs, "resid")

        xgb = XGBRegressor(**XGB_NASA_CFG)
        xgb.fit(Xtr[:, top], y[tr])
        nasa["persist_early"][te] = xgb.predict(Xte_early[:, top])
        nasa["ridge_27"][te] = xgb.predict(pred_ridge)

        def apply_tf(model, i):
            if not mask[i]:
                return X_early[i, top]
            e, _ = seq_of(i)
            ez = (e - mu_s) / sd_s
            e27 = (X_early[i, top] - mu27) / sd27
            return model.pred27(ez, e27).data * sd27 + mu27

        pred_fuse = np.vstack([apply_tf(fuse, i) for i in te])
        pred_pool = np.vstack([apply_tf(poolm, i) for i in te])
        nasa["tf_fuse_delta"][te] = xgb.predict(pred_fuse)
        nasa["tf_pool"][te] = xgb.predict(pred_pool)

        for i in te:
            if not mask[i]:
                continue
            e, l = seq_of(i)
            ez = (e - mu_s) / sd_s
            raw32 = plot_m.forward(ez, 32).data
            yhat = compose("resid", ez, raw32, len(l)) * sd_s + mu_s
            plot_hat[int(i)] = (e, l, yhat, bases, idx)

    table = []
    s_true = mix_s(step, y)
    for name in names:
        msk = mask & np.isfinite(nasa[name])
        folds = fold_metrics(y, step, nasa[name], groups, mask)
        rec = {
            "model": name,
            "nasa_r2": safe_r2(y[msk], nasa[name][msk]),
            "s_r2": safe_r2(s_true[msk], mix_s(step, nasa[name])[msk]),
            "s_mae": safe_mae(s_true[msk], mix_s(step, nasa[name])[msk]),
            "fold1_nasa_r2": folds[0]["nasa_r2"],
            "fold1_s_r2": folds[0]["s_r2"],
            "fold1_s_mae": folds[0]["s_mae"],
            "folds": folds,
        }
        table.append(rec)
        print(
            f"  {name:14s}  S={rec['s_r2']:+.3f}  fold1 S={rec['fold1_s_r2']:+.3f}  "
            f"NASA={rec['nasa_r2']:+.3f}  fold1 NASA={rec['fold1_nasa_r2']:+.3f}"
        )

    pred = pd.DataFrame(
        {
            "sample_id": task["samples"],
            "subject": groups,
            "eligible": mask.astype(int),
            "S_true": s_true,
            "y_nasa": y,
            **{f"S_{k}": mix_s(step, nasa[k]) for k in names},
            **{f"nasa_{k}": nasa[k] for k in names},
        }
    )
    pred.to_csv(OUT / "predictions.csv", index=False)

    # 验证折里：S 接近 且 轨迹有波动
    fold0 = [i for i, s in enumerate(samples) if mask[i] and int(groups[i]) in (2, 7, 12, 16, 23)]
    # fold 0 是 GroupKFold 第一折，被试不一定是 2,7,12... 应用 fold_of
    fold_of = np.full(len(samples), -1)
    for ff, (_, tee) in enumerate(gkf.split(X_true, y, groups)):
        fold_of[tee] = ff
    cand = []
    for i in np.where((fold_of == 0) & mask)[0]:
        if i not in plot_hat:
            continue
        e, l, yh, bases, idx = plot_hat[i]
        # 心率若在 bases 里
        hr_j = bases.index("hr_mean") if "hr_mean" in bases else 0
        dyn = float(np.std(yh[:, hr_j]) / (np.std(l[:, hr_j]) + 1e-8))
        ds = abs(float(pred.loc[i, "S_tf_fuse_delta"] - pred.loc[i, "S_true"]))
        cand.append((task["samples"][i], ds, dyn, i, hr_j))
    cand.sort(key=lambda x: (x[1], -x[2]))
    demo_sid, _, _, demo_i, hr_j = cand[0] if cand else (task["samples"][0], 0, 0, 0, 0)
    print("[v15] demo", demo_sid, "dS", cand[0][1] if cand else None, "dyn", cand[0][2] if cand else None)

    # 画示范 + 另外两场波动大的
    more = [c for c in cand if c[0] != demo_sid][:2]
    show = [(demo_sid, demo_i, hr_j)] + [(c[0], c[3], c[4]) for c in more]
    fig, axes = plt.subplots(len(show), 1, figsize=(7.6, 2.4 * max(len(show), 1)), squeeze=False)
    for ri, (sid, ii, hj) in enumerate(show):
        e, l, yh, bases, idx = plot_hat[ii]
        ax = axes[ri, 0]
        tt = np.arange(len(e) + len(l))
        ax.plot(tt[: len(e)], e[:, hj], color="black", lw=1.2, label="已观察")
        ax.plot(tt[len(e) :], l[:, hj], color="0.55", lw=0.9, ls=":", label="未来真值")
        ax.plot(tt[len(e) :], yh[:, hj], color="#C45C26", lw=1.3, ls="--", label="Transformer 轨迹头")
        ax.axvline(len(e) - 0.5, color="0.4", ls="--", lw=0.7)
        ax.set_title(
            f"{sid}  {bases[hj]}  S真={pred.loc[ii,'S_true']:.3f}  "
            f"S_TF={pred.loc[ii,'S_tf_fuse_delta']:.3f}  S_Ridge={pred.loc[ii,'S_ridge_27']:.3f}",
            loc="left",
            fontsize=9,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ri == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_dual.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    e, l, yh, bases, idx = plot_hat[demo_i]
    np.savez(
        OUT / "selected_demo.npz",
        sample_id=demo_sid,
        early=e,
        late=l,
        yhat=yh,
        bases=np.array(bases),
        S_true=float(pred.loc[demo_i, "S_true"]),
        S_tf=float(pred.loc[demo_i, "S_tf_fuse_delta"]),
        S_ridge=float(pred.loc[demo_i, "S_ridge_27"]),
    )

    (OUT / "metrics.json").write_text(
        json.dumps({"demo": demo_sid, "models": json_ready(table), "candidates": cand[:8]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Transformer 双头：轨迹波动 + 正式 S\n\n",
        "不再把预报窗拼回 264（那条路 NASA R² 为负）。",
        "改为共享序列编码、两个读出头：\n\n",
        "1. **任务头**：编码已观察窗 + 已观察 27 维，残差补全整场 27 维，再进冻结 XGB 得 NASA、公式 S。\n",
        "2. **轨迹头**：同一套定额原始窗序列上做残差解码，只用于实线/虚线图。\n\n",
        f"**示范样本**：`{demo_sid}`\n\n",
        "| 方法 | 全样本 S R² | 验证折 S R² | 验证折 NASA R² | 验证折 S MAE |\n|---|---:|---:|---:|---:|\n",
    ]
    for r in table:
        lines.append(
            f"| {r['model']} | {r['s_r2']:+.3f} | {r['fold1_s_r2']:+.3f} | "
            f"{r['fold1_nasa_r2']:+.3f} | {r['fold1_s_mae']:.3f} |\n"
        )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v15] wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
