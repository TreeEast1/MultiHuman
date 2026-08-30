#!/usr/bin/env python3
"""Transformer 窗级预报 → 聚合成 264 → 同一套定额 XGB → S R²。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    REPORTS,
    STEP_W,
    aggregate_windows,
    build_mod_idx,
    downstream_quota_xgb,
    eligible_mask,
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
from exp_tf_tune import compose, train_one  # noqa: E402
from exp_trend_shape import fill_nan, mean_revert  # noqa: E402

OUT = REPORTS / "v14_tf_to_s"
RATIO = 0.50
V8 = REPORTS / "v8_quota27_space" / "models" / "ridge_scaled" / "predictions.csv"
SEED = 0


def raw_bases(names_264: list[str], idx27: np.ndarray) -> list[str]:
    seen: list[str] = []
    for i in idx27:
        base = names_264[int(i)].rsplit("__", 1)[0]
        if base not in seen:
            seen.append(base)
    return seen


def recon_264(early_66: np.ndarray, late_66: np.ndarray) -> np.ndarray:
    return aggregate_windows(np.vstack([early_66, late_66]))


def mr_late(early: np.ndarray, n: int) -> np.ndarray:
    return np.column_stack([mean_revert(early[:, j], n) for j in range(early.shape[1])])


def fold_s(y, step, nasa, groups, mask) -> list[dict]:
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
                "s_r2": safe_r2(mix_s(st, yt, STEP_W), mix_s(st, nh, STEP_W)),
                "s_mae": safe_mae(mix_s(st, yt, STEP_W), mix_s(st, nh, STEP_W)),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: s for s in samples}
    samples = [by_id[sid] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, 4)
    y, step, groups = task["y"], task["step"], task["groups"]
    X_true = task["X"]

    X_early = np.zeros_like(X_true)
    filled = []
    for i, s in enumerate(samples):
        W = np.column_stack([fill_nan(s.W[:, j]) for j in range(s.W.shape[1])])
        filled.append(W)
        if not mask[i]:
            X_early[i] = aggregate_windows(W)
            continue
        cut = split_index(len(W), RATIO)
        X_early[i] = aggregate_windows(W[:cut])

    n, d66 = len(samples), len(raw)
    hats = {
        "early_only": X_early.copy(),
        "win_mean_revert": np.zeros_like(X_true),
        "tf_resid": np.zeros_like(X_true),
    }
    for i, W in enumerate(filled):
        if not mask[i]:
            hats["win_mean_revert"][i] = X_early[i]
            hats["tf_resid"][i] = X_early[i]
            continue
        cut = split_index(len(W), RATIO)
        early, n2 = W[:cut], len(W) - cut
        hats["win_mean_revert"][i] = recon_264(early, mr_late(early, n2))

    gkf = GroupKFold(n_splits=N_SPLITS)
    mod_idx = build_mod_idx(names_264)
    fold_bases: list[list[str]] = []

    for f, (tr, te) in enumerate(gkf.split(X_true, y, groups)):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_true[tr])
        top = select_quota(Xtr, y[tr], mod_idx)
        bases = raw_bases(names_264, top)
        fold_bases.append(bases)
        idx = [col[b] for b in bases if b in col]
        print(f"[v14] fold {f+1}  quota raw {len(idx)}  {bases[:6]}...")

        tr_elig = [i for i in tr if mask[i]]
        te_elig = [i for i in te if mask[i]]
        subj = sorted({int(groups[i]) for i in tr_elig})
        rng = np.random.RandomState(SEED + f)
        rng.shuffle(subj)
        val_s = set(subj[: max(2, len(subj) // 6)])
        tr_fit = [i for i in tr_elig if int(groups[i]) not in val_s]
        tr_val = [i for i in tr_elig if int(groups[i]) in val_s] or tr_fit[:2]

        def pair(i: int):
            W = filled[i]
            cut = split_index(len(W), RATIO)
            e, l = W[:cut][:, idx], W[cut:][:, idx]
            return e, l

        stack = np.vstack([np.vstack(pair(i)) for i in tr_fit])
        mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

        def pack(ids):
            out = []
            for i in ids:
                e, l = pair(i)
                out.append(((e - mu) / sd, (l - mu) / sd))
            return out

        model = train_one(pack(tr_fit), pack(tr_val), "resid")
        for i in te_elig:
            W = filled[i]
            cut = split_index(len(W), RATIO)
            early, n2 = W[:cut], len(W) - cut
            late = mr_late(early, n2)
            ez = (early[:, idx] - mu) / sd
            raw32 = model.forward(ez, 32).data
            late[:, idx] = compose("resid", ez, raw32, n2) * sd + mu
            hats["tf_resid"][i] = recon_264(early, late)

    downs = {}
    for name, Xh in hats.items():
        print(f"[v14] XGB {name} …")
        downs[name] = downstream_quota_xgb(
            X_true, Xh, y, step, groups, names_264, eval_mask=mask, X_early=X_early
        )

    v8 = pd.read_csv(V8)
    v8["sample_id"] = v8["sample_id"].astype(str)
    v8 = v8.set_index("sample_id").loc[task["samples"]].reset_index()
    nasa_v8 = v8["nasa_hat"].to_numpy()

    methods_nasa = {k: downs[k]["nasa_hat"] for k in downs}
    methods_nasa["v8_ridge27"] = nasa_v8
    methods_nasa["oracle"] = downs["tf_resid"]["nasa_oracle"]

    table = []
    for name, nasa in methods_nasa.items():
        msk = mask & np.isfinite(nasa)
        s_true = mix_s(step, y, STEP_W)
        s_hat = mix_s(step, nasa, STEP_W)
        folds = fold_s(y, step, nasa, groups, mask)
        rec = {
            "model": name,
            "nasa_r2": safe_r2(y[msk], nasa[msk]),
            "s_r2": safe_r2(s_true[msk], s_hat[msk]),
            "s_mae": safe_mae(s_true[msk], s_hat[msk]),
            "fold1_nasa_r2": folds[0]["nasa_r2"],
            "fold1_s_r2": folds[0]["s_r2"],
            "fold1_s_mae": folds[0]["s_mae"],
            "folds": folds,
        }
        table.append(rec)
        print(
            f"  {name:16s}  S R²={rec['s_r2']:+.3f}  fold1 S R²={rec['fold1_s_r2']:+.3f}  "
            f"NASA={rec['nasa_r2']:+.3f}"
        )

    pred = pd.DataFrame(
        {
            "sample_id": task["samples"],
            "subject": groups,
            "eligible": mask.astype(int),
            "S_true": mix_s(step, y, STEP_W),
            "y_nasa": y,
            "S_tf_resid": mix_s(step, methods_nasa["tf_resid"], STEP_W),
            "S_win_mean_revert": mix_s(step, methods_nasa["win_mean_revert"], STEP_W),
            "S_v8_ridge": mix_s(step, nasa_v8, STEP_W),
            "S_early": mix_s(step, methods_nasa["early_only"], STEP_W),
            "nasa_tf_resid": methods_nasa["tf_resid"],
            "nasa_v8": nasa_v8,
        }
    )
    pred.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "metrics.json").write_text(
        json.dumps(
            {"fold_quota_raw": fold_bases, "models": json_ready(table)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    demo = pred[pred["sample_id"] == "subject_12_task_3"].iloc[0]
    lines = [
        "# Transformer 接到正式 S\n\n",
        "窗级预报定额 27 用到的原始指标 → 与已观察段拼回全任务 264 → 折内定额 27 → 冻结 XGB → 公式 S。\n",
        "验证折为被试 2、7、12、16、23。\n\n",
        "| 方法 | 全样本 S R² | 验证折 S R² | 验证折 S MAE | NASA R² |\n|---|---:|---:|---:|---:|\n",
    ]
    for r in table:
        lines.append(
            f"| {r['model']} | {r['s_r2']:+.3f} | {r['fold1_s_r2']:+.3f} | "
            f"{r['fold1_s_mae']:.3f} | {r['nasa_r2']:+.3f} |\n"
        )
    lines.append(
        f"\n示范 `subject_12_task_3`：真值 S={demo['S_true']:.3f}，"
        f"Transformer={demo['S_tf_resid']:.3f}，V8 Ridge={demo['S_v8_ridge']:.3f}。\n"
    )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("[v14] wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
