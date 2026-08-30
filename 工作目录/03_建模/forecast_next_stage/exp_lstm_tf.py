#!/usr/bin/env python3
"""窗级人因：小型 LSTM / Transformer（numpy），对照保留的均值回归。

不依赖 PyTorch。按被试五折，与 v11 同一批 8 条曲线。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    REPORTS,
    eligible_mask,
    load_feature_names,
    load_samples,
    load_task_arrays,
    split_index,
)
from exp_trend_shape import (  # noqa: E402
    SERIES,
    fill_nan,
    mean_revert,
    safe_mae,
    safe_pearson,
    safe_r2,
    setup_font,
)

OUT = REPORTS / "v12_lstm_tf"
FIG = HERE / "figures"
RATIO = 0.50
MIN_EACH = 4
EPOCHS = 20
LR = 8e-4
SEED = 0
HIDDEN = 16
LOOK = 24
CHUNK = 16


class N:
    """极小 ndarray 自动微分。"""

    def __init__(self, data, parents=(), op=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self.op = op

    def _bwd(self):
        return

    def backward(self):
        order = []
        seen = set()

        def walk(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v.parents:
                walk(p)
            order.append(v)

        walk(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(order):
            v._bwd()


def leaf(arr, rng, scale=None):
    a = np.asarray(arr, dtype=np.float64)
    if scale is None:
        scale = np.sqrt(2.0 / max(a.shape[-1], 1))
    x = N(rng.normal(0.0, scale, size=a.shape) if a.size else a)
    return x


def _bin(a, b, data, db_a, db_b):
    out = N(data, (a, b))

    def _bwd():
        ga, gb = db_a(out.grad), db_b(out.grad)
        a.grad = a.grad + ga
        b.grad = b.grad + gb

    out._bwd = _bwd
    return out


def add(a, b):
    if not isinstance(b, N):
        b = N(b)
    return _bin(a, b, a.data + b.data, lambda g: g, lambda g: g)


def sub(a, b):
    if not isinstance(b, N):
        b = N(b)
    return _bin(a, b, a.data - b.data, lambda g: g, lambda g: -g)


def mul(a, b):
    if not isinstance(b, N):
        b = N(b)
    return _bin(a, b, a.data * b.data, lambda g: g * b.data, lambda g: g * a.data)


def matmul(a, b):
    out = N(a.data @ b.data, (a, b))

    def _bwd():
        ad, bd, g = a.data, b.data, out.grad
        if ad.ndim == 1 and bd.ndim == 2:
            a.grad = a.grad + g @ bd.T
            b.grad = b.grad + np.outer(ad, g)
        else:
            a.grad = a.grad + g @ bd.T
            b.grad = b.grad + ad.T @ g

    out._bwd = _bwd
    return out


def tanh(a):
    y = np.tanh(a.data)
    out = N(y, (a,))

    def _bwd():
        a.grad = a.grad + out.grad * (1.0 - y**2)

    out._bwd = _bwd
    return out


def sigmoid(a):
    y = 1.0 / (1.0 + np.exp(-np.clip(a.data, -20, 20)))
    out = N(y, (a,))

    def _bwd():
        a.grad = a.grad + out.grad * y * (1.0 - y)

    out._bwd = _bwd
    return out


def relu(a):
    y = np.maximum(a.data, 0.0)
    out = N(y, (a,))

    def _bwd():
        a.grad = a.grad + out.grad * (a.data > 0)

    out._bwd = _bwd
    return out


def softmax_rows(a):
    z = a.data - a.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=-1, keepdims=True)
    out = N(y, (a,))

    def _bwd():
        gy = out.grad
        a.grad = a.grad + y * (gy - (gy * y).sum(axis=-1, keepdims=True))

    out._bwd = _bwd
    return out


def mean2(a):
    out = N(a.data.mean(), (a,))

    def _bwd():
        a.grad = a.grad + out.grad * (np.ones_like(a.data) / a.data.size)

    out._bwd = _bwd
    return out


def mse(pred, target):
    t = N(target) if not isinstance(target, N) else target
    d = sub(pred, t)
    return mean2(mul(d, d))


def stack_rows(nodes: list[N]) -> N:
    data = np.stack([n.data for n in nodes], axis=0)
    out = N(data, tuple(nodes))

    def _bwd():
        for i, n in enumerate(nodes):
            n.grad = n.grad + out.grad[i]

    out._bwd = _bwd
    return out


class Adam:
    def __init__(self, params: list[N], lr=LR, wd=1e-3):
        self.params = params
        self.lr = lr
        self.wd = wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p.data
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)
            mhat = self.m[i] / (1 - b1**self.t)
            vhat = self.v[i] / (1 - b2**self.t)
            p.data = p.data - self.lr * mhat / (np.sqrt(vhat) + 1e-8)
            p.grad = np.zeros_like(p.data)


def xavier(rng, rows, cols):
    return N(rng.normal(0.0, np.sqrt(2.0 / (rows + cols)), size=(rows, cols)))


def zeros(shape):
    return N(np.zeros(shape, dtype=np.float64))


class LSTMForecast:
    def __init__(self, d_in: int, hidden=HIDDEN, rng=None):
        rng = rng or np.random.RandomState(SEED)
        self.h = hidden
        self.Wih = xavier(rng, d_in, 4 * hidden)
        self.Whh = xavier(rng, hidden, 4 * hidden)
        self.b = zeros((4 * hidden,))
        self.Who = xavier(rng, hidden, d_in)
        self.bo = zeros((d_in,))

    def params(self):
        return [self.Wih, self.Whh, self.b, self.Who, self.bo]

    def _step_split(self, z, h, c):
        hs = self.h

        def sl(start, fn):
            chunk = N(z.data[start : start + hs], (z,))

            def _bwd():
                z.grad[start : start + hs] = z.grad[start : start + hs] + chunk.grad

            chunk._bwd = _bwd
            return fn(chunk)

        i = sl(0, sigmoid)
        f = sl(hs, sigmoid)
        g = sl(2 * hs, tanh)
        o = sl(3 * hs, sigmoid)
        c2 = add(mul(f, c), mul(i, g))
        h2 = mul(o, tanh(c2))
        return h2, c2

    def _run_step(self, x, h, c):
        z = add(add(matmul(x, self.Wih), matmul(h, self.Whh)), self.b)
        return self._step_split(z, h, c)

    def teacher_chunk(self, seq: np.ndarray) -> N:
        """seq: (L+1, D)，逐步预报下一窗。"""
        h = zeros((self.h,))
        c = zeros((self.h,))
        outs = []
        for t in range(len(seq) - 1):
            h, c = self._run_step(N(seq[t]), h, c)
            outs.append(add(matmul(h, self.Who), self.bo))
        return stack_rows(outs)

    def rollout(self, early: np.ndarray, n_out: int) -> np.ndarray:
        use = early[-LOOK:] if len(early) > LOOK else early
        h = zeros((self.h,))
        c = zeros((self.h,))
        for t in range(len(use)):
            h, c = self._run_step(N(use[t]), h, c)
        x = N(use[-1])
        ys = []
        for _ in range(n_out):
            h, c = self._run_step(x, h, c)
            y = add(matmul(h, self.Who), self.bo)
            ys.append(y.data.copy())
            x = N(y.data)
        return np.stack(ys, axis=0)


class TinyTransformer:
    def __init__(self, d_in: int, d_model=HIDDEN, rng=None):
        rng = rng or np.random.RandomState(SEED)
        self.d = d_model
        self.We = xavier(rng, d_in, d_model)
        self.Wq = xavier(rng, d_model, d_model)
        self.Wk = xavier(rng, d_model, d_model)
        self.Wv = xavier(rng, d_model, d_model)
        self.W1 = xavier(rng, d_model, d_model)
        self.W2 = xavier(rng, d_model, d_model)
        self.Wfq = xavier(rng, d_model, d_model)
        self.Wo = xavier(rng, d_model, d_in)
        self.bo = zeros((d_in,))

    def params(self):
        return [self.We, self.Wq, self.Wk, self.Wv, self.W1, self.W2, self.Wfq, self.Wo, self.bo]

    def encode_pool(self, early: np.ndarray) -> N:
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
        H = add(H, matmul(H, self.W2))
        pool = N(H.data.mean(axis=0), (H,))

        def _pool_bwd():
            H.grad = H.grad + np.broadcast_to(pool.grad / H.data.shape[0], H.data.shape)

        pool._bwd = _pool_bwd
        return pool

    def forward(self, early: np.ndarray, n_out: int, late=None, tf=True) -> N:
        del late, tf
        pool = self.encode_pool(early)
        d = self.d
        outs = []
        for t in range(n_out):
            pe = np.zeros(d)
            for i in range(d):
                ang = t / (10000 ** (2 * (i // 2) / d))
                pe[i] = np.sin(ang) if i % 2 == 0 else np.cos(ang)
            q = add(pool, N(pe))
            q = relu(matmul(q, self.Wfq))
            outs.append(add(matmul(q, self.Wo), self.bo))
        return stack_rows(outs)


def _chunks(pairs, rng):
    out = []
    for early, late in pairs:
        seq = np.vstack([early, late])
        if len(seq) < CHUNK + 1:
            continue
        for start in range(0, len(seq) - CHUNK, 4):
            out.append(seq[start : start + CHUNK + 1])
    rng.shuffle(out)
    return out[:240]


def train_lstm(train, val, test, mu, sd):
    rng = np.random.RandomState(SEED)
    model = LSTMForecast(train[0][0].shape[1], rng=rng)
    opt = Adam(model.params(), lr=LR)
    chunks = _chunks(train, rng)
    best, wait, snap = 1e9, 0, None
    for ep in range(EPOCHS):
        rng.shuffle(chunks)
        for seq in chunks:
            pred = model.teacher_chunk(seq)
            loss = mse(pred, seq[1:])
            for p in model.params():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            opt.step()
        vl = []
        for early, late in val:
            y = model.rollout(early, len(late))
            vl.append(float(np.mean((y - late) ** 2)))
        score = float(np.mean(vl)) if vl else 1e9
        if score < best:
            best, wait = score, 0
            snap = [p.data.copy() for p in model.params()]
        else:
            wait += 1
            if wait >= 6:
                break
        if (ep + 1) % 5 == 0:
            print(f"    lstm ep {ep+1:02d}  val {score:.4f}")
    if snap:
        for p, s in zip(model.params(), snap):
            p.data = s
    return [model.rollout(e, len(y)) * sd + mu for e, y in test]


def _resample(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == n:
        return y
    x = np.linspace(0.0, 1.0, len(y))
    xt = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(xt, x, y[:, j]) for j in range(y.shape[1])])


TF_H = 32


def train_tf(train, val, test, mu, sd):
    rng = np.random.RandomState(SEED)
    model = TinyTransformer(train[0][0].shape[1], rng=rng)
    opt = Adam(model.params(), lr=LR)
    best, wait, snap = 1e9, 0, None
    for ep in range(EPOCHS):
        rng.shuffle(train)
        for early, late in train:
            tgt = _resample(late, TF_H)
            pred = model.forward(early, TF_H)
            loss = mse(pred, tgt)
            for p in model.params():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            opt.step()
        vl = []
        for early, late in val:
            y = _resample(model.forward(early, TF_H).data, len(late))
            vl.append(float(np.mean((y - late) ** 2)))
        score = float(np.mean(vl)) if vl else 1e9
        if score < best:
            best, wait = score, 0
            snap = [p.data.copy() for p in model.params()]
        else:
            wait += 1
            if wait >= 6:
                break
        if (ep + 1) % 5 == 0:
            print(f"    transformer ep {ep+1:02d}  val {score:.4f}")
    if snap:
        for p, s in zip(model.params(), snap):
            p.data = s
    out = []
    for e, y in test:
        pred = _resample(model.forward(e, TF_H).data, len(y))
        out.append(pred * sd + mu)
    return out


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    _, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    idx = [col[n] for n, _ in SERIES]
    labels = [zh for _, zh in SERIES]
    feat_names = [n for n, _ in SERIES]

    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: k for k, s in enumerate(samples)}
    samples = [samples[by_id[sid]] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, MIN_EACH)

    items = []
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        cut = split_index(len(s.W), RATIO)
        early = np.column_stack([fill_nan(s.W[:cut, j]) for j in idx])
        late = np.column_stack([fill_nan(s.W[cut:, j]) for j in idx])
        items.append(
            {"i": i, "sample_id": s.sample_id, "subject": s.subject, "early": early, "late": late, "cut": cut}
        )

    fold_of = {}
    gkf = GroupKFold(n_splits=N_SPLITS)
    for f, (_, te) in enumerate(gkf.split(task["X"], task["y"], task["groups"])):
        for ii in te:
            fold_of[int(ii)] = f
    for it in items:
        it["fold"] = fold_of[it["i"]]

    hats = {k: [None] * len(items) for k in ("mean_revert", "lstm", "transformer")}
    d_in = len(SERIES)

    for f in range(N_SPLITS):
        tr = [it for it in items if it["fold"] != f]
        te = [it for it in items if it["fold"] == f]
        subj = sorted({it["subject"] for it in tr})
        rng = np.random.RandomState(SEED + f)
        rng.shuffle(subj)
        val_s = set(subj[: max(2, len(subj) // 6)])
        tr_fit = [it for it in tr if it["subject"] not in val_s]
        tr_val = [it for it in tr if it["subject"] in val_s] or tr_fit[:2]
        stack = np.vstack([np.vstack([it["early"], it["late"]]) for it in tr_fit])
        mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

        def pack(its):
            return [((it["early"] - mu) / sd, (it["late"] - mu) / sd) for it in its]

        print(f"[v12] fold {f+1}  train {len(tr_fit)}  val {len(tr_val)}  test {len(te)}")
        lstm_pred = train_lstm(pack(tr_fit), pack(tr_val), pack(te), mu, sd)
        tf_pred = train_tf(pack(tr_fit), pack(tr_val), pack(te), mu, sd)
        for k, it in enumerate(te):
            j = items.index(it)
            n2 = len(it["late"])
            hats["mean_revert"][j] = np.column_stack([mean_revert(it["early"][:, c], n2) for c in range(d_in)])
            hats["lstm"][j] = lstm_pred[k]
            hats["transformer"][j] = tf_pred[k]

    Y = np.vstack([it["late"] for it in items])
    row_fold = np.concatenate([np.full(len(it["late"]), it["fold"], dtype=int) for it in items])
    near = np.concatenate([np.arange(len(it["late"])) < min(12, len(it["late"])) for it in items])
    table = []
    for name in hats:
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
                }
            )
        m1 = row_fold == 0
        rec = {
            "model": name,
            "mean_r2": float(np.nanmean([r["r2"] for r in rows])),
            "mean_pearson": float(np.nanmean([r["pearson"] for r in rows])),
            "fold1_mean_r2": float(np.nanmean([safe_r2(Y[m1, j], Yh[m1, j]) for j in range(d_in)])),
            "near12_mean_r2": float(np.nanmean([safe_r2(Y[near, j], Yh[near, j]) for j in range(d_in)])),
            "per_series": rows,
        }
        table.append(rec)
        print(
            f"  {name:12s}  R²={rec['mean_r2']:+.3f}  near12={rec['near12_mean_r2']:+.3f}  "
            f"fold1={rec['fold1_mean_r2']:+.3f}"
        )

    (OUT / "metrics.json").write_text(
        json.dumps({"n": int(len(Y)), "n_tasks": len(items), "models": table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{"model": r["model"], **row} for r in table for row in r["per_series"]]).to_csv(
        OUT / "metrics.csv", index=False
    )

    plot_ids = ["subject_02_task_1", "subject_07_task_2", "subject_12_task_2"]
    rec_by = {it["sample_id"]: it for it in items}
    off = np.cumsum([0] + [len(it["late"]) for it in items])
    pos = {it["sample_id"]: (off[k], off[k + 1]) for k, it in enumerate(items)}
    colors = {"mean_revert": "0.4", "lstm": "#1F4E79", "transformer": "#C45C26"}
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 7.2))
    for ri, sid in enumerate(plot_ids):
        it = rec_by[sid]
        cut, a, b = it["cut"], *pos[sid]
        for ci, sj in enumerate((0, 4)):
            ax = axes[ri, ci]
            tt = np.arange(it["cut"] + len(it["late"]))
            ax.plot(tt[:cut], it["early"][:, sj], color="black", lw=1.1, label="已观察")
            ax.plot(tt[cut:], it["late"][:, sj], color="0.6", lw=0.9, ls=":", label="未来真值")
            for mn in ("mean_revert", "lstm", "transformer"):
                ax.plot(tt[cut:], hats[mn][a:b, sj], color=colors[mn], lw=1.15, ls="--", label=mn)
            ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
            ax.set_title(f"{sid}  {labels[sj]}", loc="left", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lstm_tf.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "# LSTM / Transformer 对照均值回归\n\n",
        "均值回归保留。小型 LSTM 与单层注意力 Transformer，按被试五折，8 条窗级曲线。\n\n",
        "| 算法 | 后半段 R² | 近 12 窗 R² | 第 1 折 R² |\n|---|---:|---:|---:|\n",
    ]
    for r in table:
        lines.append(
            f"| {r['model']} | {r['mean_r2']:+.3f} | {r['near12_mean_r2']:+.3f} | {r['fold1_mean_r2']:+.3f} |\n"
        )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v12] wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
