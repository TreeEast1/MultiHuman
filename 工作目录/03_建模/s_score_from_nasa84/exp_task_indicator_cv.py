#!/usr/bin/env python3
"""五折：在全模态 MI Top-30 上加入任务指示，看 NASA R² 是否提升。

推理时任务编号已知（做哪一关事先定了）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS  # noqa: E402

from exp_s_fullmodal_mi30 import (  # noqa: E402
    NASA_DS,
    N_SPLITS,
    S_TABLE,
    STEP_W,
    TOP_K,
    XGB_CFG,
    _enable_xgboost,
    make_fullmodal_ranker,
)

OUT = HERE / "reports_s_fullmodal"


def onehot(vals: np.ndarray, vocab: list[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(vals), len(vocab)), dtype=float)
    for i, v in enumerate(vals):
        if v in idx:
            m[i, idx[v]] = 1.0
    return m


def task_prior(task_tr, y_tr, task_te):
    means = pd.Series(y_tr).groupby(pd.Series(task_tr)).mean()
    g = float(np.mean(y_tr))
    prior_tr = pd.Series(task_tr).map(means).fillna(g).to_numpy(dtype=float)
    prior_te = pd.Series(task_te).map(means).fillna(g).to_numpy(dtype=float)
    return prior_tr, prior_te


def eval_cv(name, predict_fold, y, groups, step) -> dict:
    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = np.full(len(y), np.nan)
    fold_r2 = []
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        yhat_te = predict_fold(tr, te)
        hat[te] = yhat_te
        fold_r2.append(float(r2_score(y[te], yhat_te)) if len(te) > 1 else float("nan"))
    y_s = STEP_W * step + (1.0 - STEP_W) * (1.0 - y / 10.0)
    s_hat = STEP_W * step + (1.0 - STEP_W) * (1.0 - hat / 10.0)
    row = {
        "name": name,
        "nasa_pooled_r2": float(r2_score(y, hat)),
        "nasa_pooled_mae": float(mean_absolute_error(y, hat)),
        "nasa_fold_r2_mean": float(np.nanmean(fold_r2)),
        "nasa_fold_r2_std": float(np.nanstd(fold_r2)),
        "s_composed_r2": float(r2_score(y_s, s_hat)),
        "s_composed_mae": float(mean_absolute_error(y_s, s_hat)),
    }
    print(
        f"{name:48s} NASA R²={row['nasa_pooled_r2']:+.3f} MAE={row['nasa_pooled_mae']:.3f}  "
        f"合成S R²={row['s_composed_r2']:+.3f}"
    )
    return row, hat


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor

    X = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    tbl = pd.read_csv(NASA_DS / "task_level_table.csv")
    tbl["sample_id"] = tbl["sample_id"].astype(str)
    tbl = tbl.set_index("sample_id").loc[pd.Index(samples)].reset_index()
    tasks = tbl["task"].astype(str).to_numpy()
    diffs = tbl["task_difficulty"].astype(str).to_numpy()
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    step = s_table["weighted_step_score"].to_numpy(dtype=float)

    task_vocab = sorted(set(tasks.tolist()), key=lambda t: (len(t), t))
    diff_vocab = ["低", "中", "高"]
    ranker = make_fullmodal_ranker(names, 1)

    def select30(Xtr_imp, ytr):
        return ranker(Xtr_imp, ytr)[:TOP_K]

    rows = []
    hats = {}

    def xgb():
        return XGBRegressor(**XGB_CFG)

    # A. 全模态 30，无任务
    def pred_base(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = select30(Xtr, y[tr])
        m = xgb()
        m.fit(Xtr[:, top], y[tr])
        return m.predict(Xte[:, top])

    row, hat = eval_cv("fullmodal30_no_task", pred_base, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # B. 全模态 30 + 任务 one-hot（6 维）
    def pred_task_oh(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = select30(Xtr, y[tr])
        extra_tr = onehot(tasks[tr], task_vocab)
        extra_te = onehot(tasks[te], task_vocab)
        m = xgb()
        m.fit(np.column_stack([Xtr[:, top], extra_tr]), y[tr])
        return m.predict(np.column_stack([Xte[:, top], extra_te]))

    row, hat = eval_cv("fullmodal30_plus_task_onehot", pred_task_oh, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # C. 全模态 30 + 难度 one-hot
    def pred_diff_oh(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = select30(Xtr, y[tr])
        extra_tr = onehot(diffs[tr], diff_vocab)
        extra_te = onehot(diffs[te], diff_vocab)
        m = xgb()
        m.fit(np.column_stack([Xtr[:, top], extra_tr]), y[tr])
        return m.predict(np.column_stack([Xte[:, top], extra_te]))

    row, hat = eval_cv("fullmodal30_plus_difficulty_onehot", pred_diff_oh, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # D. 全模态 30 + 任务 one-hot + 难度
    def pred_both(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = select30(Xtr, y[tr])
        extra_tr = np.column_stack([onehot(tasks[tr], task_vocab), onehot(diffs[tr], diff_vocab)])
        extra_te = np.column_stack([onehot(tasks[te], task_vocab), onehot(diffs[te], diff_vocab)])
        m = xgb()
        m.fit(np.column_stack([Xtr[:, top], extra_tr]), y[tr])
        return m.predict(np.column_stack([Xte[:, top], extra_te]))

    row, hat = eval_cv("fullmodal30_plus_task_and_diff", pred_both, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # E. 只猜任务平均（无生理）
    def pred_task_mean(tr, te):
        _, prior_te = task_prior(tasks[tr], y[tr], tasks[te])
        return prior_te

    row, hat = eval_cv("task_mean_only", pred_task_mean, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # F. 任务平均 + 全模态30 残差（任务与生理耦合）
    def pred_resid(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = select30(Xtr, y[tr])
        prior_tr, prior_te = task_prior(tasks[tr], y[tr], tasks[te])
        m = xgb()
        m.fit(Xtr[:, top], y[tr] - prior_tr)
        return prior_te + m.predict(Xte[:, top])

    row, hat = eval_cv("task_mean_plus_fullmodal30_residual", pred_resid, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    # G. 无约束 MI30 + 任务 one-hot（对照）
    def pred_free_task(tr, te):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        top = RANKERS["MI"](Xtr, y[tr])[:TOP_K]
        extra_tr = onehot(tasks[tr], task_vocab)
        extra_te = onehot(tasks[te], task_vocab)
        m = xgb()
        m.fit(np.column_stack([Xtr[:, top], extra_tr]), y[tr])
        return m.predict(np.column_stack([Xte[:, top], extra_te]))

    row, hat = eval_cv("freeMI30_plus_task_onehot", pred_free_task, y, groups, step)
    rows.append(row)
    hats[row["name"]] = hat

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_vocab": task_vocab,
        "note": "任务指示只在训练折统计/编码；考试折只用同名任务列，不见考试人的 NASA。",
        "rows": rows,
    }
    (OUT / "results_task_indicator.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez(OUT / "yhat_task_indicator.npz", **{k: v for k, v in hats.items()})
    print("写完", OUT / "results_task_indicator.json")


if __name__ == "__main__":
    main()
