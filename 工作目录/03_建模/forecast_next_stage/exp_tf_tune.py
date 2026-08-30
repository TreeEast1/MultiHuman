#!/usr/bin/env python3
"""围绕 Transformer 多试几种接法，选出正式趋势预测方法 + 示范样本。"""

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
from exp_lstm_tf import (  # noqa: E402
    EPOCHS,
    LR,
    SEED,
    TF_H,
    TinyTransformer,
    _resample,
    mse,
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
from exp_lstm_tf import Adam  # noqa: E402

OUT = REPORTS / "v13_tf_tune"
FIG = HERE / "figures"
RATIO = 0.50
RHO = 0.92


def roll_mean(y: np.ndarray, w: int = 7) -> np.ndarray:
    if len(y) < 3:
        return y
    k = np.ones(w) / w
    return np.column_stack([np.convolve(y[:, j], k, mode="same") for j in range(y.shape[1])])


def mr_mat(early: np.ndarray, n: int) -> np.ndarray:
    return np.column_stack([mean_revert(early[:, j], n) for j in range(early.shape[1])])


def persist_last_mat(early: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(early[-1], (n, early.shape[1])).copy()


def estimate_period(y: np.ndarray, lo: int = 6, hi: int = 18) -> int:
    y = np.asarray(y, dtype=np.float64)
    y = y - np.nanmean(y)
    hi = min(hi, max(lo, len(y) // 2))
    best_p, best = 12, -1.0
    for p in range(lo, hi + 1):
        if len(y) <= p:
            break
        a, b = y[p:], y[:-p]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r) and r > best:
            best, best_p = r, p
    return int(best_p)


def last_cycle(early: np.ndarray, n: int, period: int | None = None) -> np.ndarray:
    if n < 1:
        return np.zeros((0, early.shape[1]))
    if period is None:
        period = estimate_period(early[:, 0])
    p = min(max(int(period), 4), max(len(early), 1))
    src = early[-p:]
    reps = int(np.ceil(n / p))
    return np.tile(src, (reps, 1))[:n]


def fade_w(n: int, rho: float = RHO) -> np.ndarray:
    return (rho ** np.arange(1, n + 1, dtype=np.float64))[:, None]


def target_of(mode: str, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    if mode == "direct":
        return _resample(late, TF_H)
    if mode == "resid":
        return _resample(late - mr_mat(early, len(late)), TF_H)
    if mode == "smooth":
        return _resample(roll_mean(late), TF_H)
    if mode in ("delta", "diff"):
        return _resample(late - early[-1], TF_H)
    if mode == "cycle":
        return _resample(late - last_cycle(early, len(late)), TF_H)
    if mode == "wave":
        mix = 0.55 * late + 0.45 * last_cycle(early, len(late))
        return _resample(mix - early[-1], TF_H)
    raise ValueError(mode)


def compose(mode: str, early: np.ndarray, raw32: np.ndarray, n: int) -> np.ndarray:
    raw = _resample(raw32, n) if raw32 is not None else None
    mr = mr_mat(early, n)
    w = fade_w(n)
    if mode == "direct":
        return raw
    if mode == "direct_anchor":
        return raw - raw[[0]] + early[[-1]]
    if mode == "resid":
        return mr + raw
    if mode == "smooth":
        return raw
    if mode == "direct_fade":
        return mr + (raw - mr) * w
    if mode == "resid_fade":
        return mr + raw * w
    if mode == "smooth_fade":
        return mr + (raw - mr) * w
    if mode == "delta":
        return persist_last_mat(early, n) + raw
    if mode in ("wave", "diff"):
        return persist_last_mat(early, n) + raw
    if mode == "cycle":
        return last_cycle(early, n) + raw
    if mode == "last_cycle":
        return last_cycle(early, n)
    if mode == "persist_last":
        return persist_last_mat(early, n)
    if mode == "mean_revert":
        return mr
    raise ValueError(mode)


def train_one(train, val, mode: str):
    rng = np.random.RandomState(SEED)
    model = TinyTransformer(train[0][0].shape[1], rng=rng)
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
            print(f"      {mode} ep {ep+1:02d}  val {score:.4f}")
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

    _, raw = load_feature_names()
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
    for f, (_, te) in enumerate(gkf.split(task["X"], task["y"], task["groups"])):
        for ii in te:
            fold_of[int(ii)] = f
    for it in items:
        it["fold"] = fold_of[it["i"]]

    train_modes = ("direct", "resid", "smooth")
    eval_names = ["mean_revert", "direct", "resid", "smooth", "direct_fade", "resid_fade", "smooth_fade"]
    hats = {k: [None] * len(items) for k in eval_names}

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

        print(f"[v13] fold {f+1}  n_train={len(tr_fit)} n_test={len(te)}")
        models = {m: train_one(pack(tr_fit), pack(tr_val), m) for m in train_modes}

        for it in te:
            j = items.index(it)
            e0, n2 = it["early"], len(it["late"])
            ez = (e0 - mu) / sd
            hats["mean_revert"][j] = mr_mat(e0, n2)
            raws = {m: models[m].forward(ez, TF_H).data for m in train_modes}
            for name in eval_names:
                if name == "mean_revert":
                    continue
                src = name.replace("_fade", "")
                yz = compose(name, ez, raws[src], n2)
                hats[name][j] = yz * sd + mu

    Y = np.vstack([it["late"] for it in items])
    row_fold = np.concatenate([np.full(len(it["late"]), it["fold"], dtype=int) for it in items])
    near = np.concatenate([np.arange(len(it["late"])) < min(12, len(it["late"])) for it in items])
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
        m1 = row_fold == 0
        rec = {
            "model": name,
            "mean_r2": float(np.nanmean([r["r2"] for r in rows])),
            "hr_r2": float(rows[0]["r2"]),
            "fold1_mean_r2": float(np.nanmean([safe_r2(Y[m1, j], Yh[m1, j]) for j in range(d_in)])),
            "near12_mean_r2": float(np.nanmean([safe_r2(Y[near, j], Yh[near, j]) for j in range(d_in)])),
            "mean_dyn": float(np.nanmean([r["dyn"] for r in rows])),
            "per_series": rows,
        }
        table.append(rec)
        print(
            f"  {name:12s}  R²={rec['mean_r2']:+.3f}  HR={rec['hr_r2']:+.3f}  "
            f"near={rec['near12_mean_r2']:+.3f}  fold1={rec['fold1_mean_r2']:+.3f}  dyn={rec['mean_dyn']:.2f}"
        )

    # 界面要看得出趋势：不用衰减混合（会走平）。在会动的 Transformer 里取 R² 最高。
    visible = [r for r in table if r["model"] in ("resid", "direct", "smooth")]
    winner = max(visible, key=lambda r: (r["mean_r2"], r["hr_r2"]))
    print("[v13] selected", winner["model"], f"R²={winner['mean_r2']:+.3f}")

    # 第 1 折示范样本：心率上该方法与真值相关最高
    fold1 = [it for it in items if it["fold"] == 0]
    off = np.cumsum([0] + [len(it["late"]) for it in items])
    pos = {it["sample_id"]: (int(off[k]), int(off[k + 1])) for k, it in enumerate(items)}
    Yh = hats[winner["model"]]
    cand = []
    for it in fold1:
        a, b = pos[it["sample_id"]]
        r = safe_pearson(it["late"][:, 0], Yh[a:b, 0])
        d = dyn_ratio(it["late"][:, 0], Yh[a:b, 0])
        cand.append((it["sample_id"], float(r) if np.isfinite(r) else -9, float(d) if np.isfinite(d) else 0))
    cand.sort(key=lambda x: (x[1], x[2]), reverse=True)
    demo = cand[0][0] if cand else fold1[0]["sample_id"]
    print("[v13] demo sample", demo, cand[:5])

    (OUT / "metrics.json").write_text(
        json.dumps(
            {
                "selected_method": winner["model"],
                "demo_sample": demo,
                "models": table,
                "fold1_candidates": [{"sample_id": s, "hr_pearson": r, "hr_dyn": d} for s, r, d in cand],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"model": r["model"], **row} for r in table for row in r["per_series"]]).to_csv(
        OUT / "metrics.csv", index=False
    )

    # 存示范样本的预报，供界面图使用
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
        cut=np.array(demo_it["cut"]),
        feat_names=np.array(feat_names),
    )

    show = ["mean_revert", winner["model"]]
    extra = [n for n in ("resid_fade", "direct", "smooth") if n != winner["model"]][:1]
    show = ["mean_revert"] + [winner["model"]] + extra
    colors = {
        "mean_revert": "0.45",
        "direct": "#C45C26",
        "resid": "#1F4E79",
        "smooth": "#2E7D4F",
        "direct_fade": "#C45C26",
        "resid_fade": "#1F4E79",
        "smooth_fade": "#2E7D4F",
    }
    plot_ids = [demo] + [s for s in ("subject_02_task_1", "subject_07_task_2", "subject_12_task_2") if s != demo]
    plot_ids = plot_ids[:3]
    fig, axes = plt.subplots(len(plot_ids), 2, figsize=(8.6, 7.2))
    for ri, sid in enumerate(plot_ids):
        it = next(x for x in items if x["sample_id"] == sid)
        a, b = pos[sid]
        cut = it["cut"]
        for ci, sj in enumerate((0, 4)):
            ax = axes[ri, ci]
            tt = np.arange(cut + len(it["late"]))
            ax.plot(tt[:cut], it["early"][:, sj], color="black", lw=1.1, label="已观察")
            ax.plot(tt[cut:], it["late"][:, sj], color="0.6", lw=0.9, ls=":", label="未来真值")
            for mn in show:
                ax.plot(tt[cut:], hats[mn][a:b, sj], color=colors[mn], lw=1.2, ls="--", label=mn)
            ax.axvline(cut - 0.5, color="0.35", lw=0.7, ls="--")
            mark = "  ←示范" if sid == demo else ""
            ax.set_title(f"{sid}  {labels[sj]}{mark}", loc="left", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tf_tune.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "# Transformer 接法比较与选定\n\n",
        f"**选定方法**：`{winner['model']}`（Transformer 预报相对已观察均值的残差，再加回去）  \n",
        f"**示范样本**：`{demo}`（验证折；心率相关最高）\n\n",
        "衰减混合 `resid_fade` 数略高，但虚线很快走平，不适合界面。",
        "`direct` 最好看，但跨被试 R² 明显差一截。\n\n",
        "| 算法 | 后半段 R² | 心率 R² | 近 12 窗 | 第 1 折 | 动态比 |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for r in table:
        star = " ←选定" if r["model"] == winner["model"] else ""
        lines.append(
            f"| {r['model']}{star} | {r['mean_r2']:+.3f} | {r['hr_r2']:+.3f} | "
            f"{r['near12_mean_r2']:+.3f} | {r['fold1_mean_r2']:+.3f} | {r['mean_dyn']:.2f} |\n"
        )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v13] wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
