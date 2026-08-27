#!/usr/bin/env python3
"""在同一 20/6 划分上再试训练方式，把 NASA 往上推。

配置只看训练 20 人内部 5 堆交叉验证；6 个考试人只验收。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
OUT_DIR = HERE / "output_one_model"
N_SPLITS = 5
RANDOM_STATE = 0


def _enable_xgboost() -> None:
    import ctypes
    import os
    from pathlib import Path as P

    import sklearn

    omp = P(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp.exists():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(omp.parent))
        ctypes.CDLL(str(omp), mode=ctypes.RTLD_GLOBAL)


def _split_random(groups: np.ndarray, n_test: int, seed: int):
    subjects = np.array(sorted(set(int(g) for g in groups)))
    rng = np.random.RandomState(seed)
    test_subjects = np.sort(rng.choice(subjects, size=n_test, replace=False))
    train_subjects = np.array([s for s in subjects if s not in set(test_subjects)])
    return train_subjects, test_subjects


def _task_prior(tasks_tr, y_tr, tasks_te):
    means = pd.Series(y_tr).groupby(pd.Series(tasks_tr)).mean()
    g = float(np.mean(y_tr))
    prior_tr = pd.Series(tasks_tr).map(means).fillna(g).to_numpy(dtype=float)
    prior_te = pd.Series(tasks_te).map(means).fillna(g).to_numpy(dtype=float)
    return prior_tr, prior_te


def _diff_prior(diff_tr, y_tr, diff_te):
    return _task_prior(diff_tr, y_tr, diff_te)


def _oof_predict(fit_predict, X, y, groups, tasks, diffs, n_splits=N_SPLITS):
    """fit_predict(Xtr,ytr,Xte,task_tr,task_te,diff_tr,diff_te) -> yhat_te"""
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        pred[te] = fit_predict(
            X[tr], y[tr], X[te],
            tasks[tr], tasks[te], diffs[tr], diffs[te],
        )
    return pred


def _xgb(cfg=None):
    from xgboost import XGBRegressor
    base = dict(
        max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, random_state=0,
    )
    if cfg:
        base.update(cfg)
    return XGBRegressor(**base)


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor  # noqa: F401

    X_raw = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy").astype(int)
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    tbl = pd.read_csv(NASA_DS / "task_level_table.csv")
    tbl["sample_id"] = tbl["sample_id"].astype(str)
    order = pd.Index(samples)
    tbl = tbl.set_index("sample_id").loc[order].reset_index()
    tasks = tbl["task"].astype(str).to_numpy()
    diffs = tbl["task_difficulty"].astype(str).to_numpy()
    n_windows = tbl["n_windows"].to_numpy(dtype=float)

    keep = np.array([not n.startswith("hr_") for n in names])
    X_raw = X_raw[:, keep]
    names = [n for n, k in zip(json.loads((NASA_DS / "feature_names_task.json").read_text()), keep) if k]

    train_subj, test_subj = _split_random(groups, 6, RANDOM_STATE)
    tr = np.isin(groups, train_subj)
    te = np.isin(groups, test_subj)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_raw[tr])
    X_te = imp.transform(X_raw[te])
    y_tr, y_te = y[tr], y[te]
    g_tr = groups[tr]
    task_tr, task_te = tasks[tr], tasks[te]
    diff_tr, diff_te = diffs[tr], diffs[te]
    win_tr, win_te = n_windows[tr], n_windows[te]

    mi_rank = RANKERS["MI"](X_tr, y_tr)
    rf_ranker = RANKERS["RF_importance"](X_tr, y_tr)

    def cols(rank, k):
        idx = rank[:k]
        return X_tr[:, idx], X_te[:, idx], idx

    results = []

    def record(name, inner_hat, test_hat):
        row = {
            "name": name,
            "inner_r2": float(r2_score(y_tr, inner_hat)),
            "inner_mae": float(mean_absolute_error(y_tr, inner_hat)),
            "test_r2": float(r2_score(y_te, test_hat)),
            "test_mae": float(mean_absolute_error(y_te, test_hat)),
        }
        results.append(row)
        print(f"  {name:40s} inner R²={row['inner_r2']:+.3f}  考试 R²={row['test_r2']:+.3f} MAE={row['test_mae']:.3f}")
        return row

    # ----- 1. 当前生理模型 -----
    def make_physio(rank, k, xgb_cfg=None, weights=None):
        Xa, Xb, _ = cols(rank, k)

        def fp(Xtr, ytr, Xte, *_):
            m = _xgb(xgb_cfg)
            sw = None
            if weights == "tail":
                z = np.abs(ytr - ytr.mean()) / (ytr.std() + 1e-6)
                sw = 1.0 + 1.5 * z
            elif weights == "tail2":
                z = np.abs(ytr - ytr.mean()) / (ytr.std() + 1e-6)
                sw = 1.0 + z ** 2
            m.fit(Xtr, ytr, sample_weight=sw)
            return m.predict(Xte)

        inner = _oof_predict(fp, Xa, y_tr, g_tr, task_tr, diff_tr)
        m = _xgb(xgb_cfg)
        sw = None
        if weights == "tail":
            z = np.abs(y_tr - y_tr.mean()) / (y_tr.std() + 1e-6)
            sw = 1.0 + 1.5 * z
        elif weights == "tail2":
            z = np.abs(y_tr - y_tr.mean()) / (y_tr.std() + 1e-6)
            sw = 1.0 + z ** 2
        m.fit(Xa, y_tr, sample_weight=sw)
        hat = m.predict(Xb)
        return inner, hat

    print("\n===== 生理-only =====")
    record("physio_MI15_xgbmid", *make_physio(mi_rank, 15))
    record("physio_MI10_xgbmid", *make_physio(mi_rank, 10))
    record("physio_MI20_xgbmid", *make_physio(mi_rank, 20))
    record("physio_RF15_xgbmid", *make_physio(rf_ranker, 15))
    record("physio_MI15_xgbslow", *make_physio(mi_rank, 15, dict(learning_rate=0.02, n_estimators=500, reg_lambda=2.0)))
    record("physio_MI15_tailw", *make_physio(mi_rank, 15, weights="tail"))
    record("physio_MI15_tail2", *make_physio(mi_rank, 15, weights="tail2"))

    def make_sk(rank, k, factory, scale=False):
        Xa, Xb, _ = cols(rank, k)

        def fp(Xtr, ytr, Xte, *_):
            if scale:
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
                Xte = sc.transform(Xte)
            m = factory()
            m.fit(Xtr, ytr)
            return m.predict(Xte)

        inner = _oof_predict(fp, Xa, y_tr, g_tr, task_tr, diff_tr)
        if scale:
            sc = StandardScaler()
            A = sc.fit_transform(Xa)
            B = sc.transform(Xb)
        else:
            A, B = Xa, Xb
        m = factory()
        m.fit(A, y_tr)
        return inner, m.predict(B)

    record("physio_MI15_rf", *make_sk(mi_rank, 15, lambda: RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)))
    record("physio_MI15_et", *make_sk(mi_rank, 15, lambda: ExtraTreesRegressor(
        n_estimators=400, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)))
    record("physio_MI15_huber", *make_sk(mi_rank, 15, lambda: HuberRegressor(max_iter=500), scale=True))
    record("physio_MI15_ridge", *make_sk(mi_rank, 15, lambda: Ridge(alpha=10), scale=True))
    record("physio_MI15_svr", *make_sk(mi_rank, 15, lambda: SVR(C=3.0, epsilon=0.2, gamma="scale"), scale=True))

    # bagging 5 seeds
    Xa, Xb, _ = cols(mi_rank, 15)

    def fp_bag(Xtr, ytr, Xte, *_):
        preds = []
        for seed in range(5):
            m = _xgb(dict(random_state=seed, subsample=0.7, colsample_bytree=0.7))
            m.fit(Xtr, ytr)
            preds.append(m.predict(Xte))
        return np.mean(preds, axis=0)

    inner = _oof_predict(fp_bag, Xa, y_tr, g_tr, task_tr, diff_tr)
    bag_te = []
    for seed in range(5):
        m = _xgb(dict(random_state=seed, subsample=0.7, colsample_bytree=0.7))
        m.fit(Xa, y_tr)
        bag_te.append(m.predict(Xb))
    record("physio_MI15_bag5", inner, np.mean(bag_te, axis=0))

    # 生理 + 任务时长
    Xa_w = np.column_stack([Xa, win_tr.reshape(-1, 1)])
    Xb_w = np.column_stack([Xb, win_te.reshape(-1, 1)])

    def fp_win(Xtr, ytr, Xte, *_):
        m = _xgb()
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    inner = _oof_predict(fp_win, Xa_w, y_tr, g_tr, task_tr, diff_tr)
    m = _xgb(); m.fit(Xa_w, y_tr)
    record("physio_MI15_plus_nwindows", inner, m.predict(Xb_w))

    # ----- 2. 任务先验（考试时任务已知）-----
    print("\n===== 任务先验 / 残差 =====")

    def fp_task(Xtr, ytr, Xte, task_tr, task_te, *_):
        _, prior_te = _task_prior(task_tr, ytr, task_te)
        return prior_te

    inner = _oof_predict(fp_task, Xa, y_tr, g_tr, task_tr, diff_tr)
    _, prior_te_full = _task_prior(task_tr, y_tr, task_te)
    record("task_mean_only", inner, prior_te_full)

    def fp_diff(Xtr, ytr, Xte, task_tr, task_te, diff_tr, diff_te):
        _, prior_te = _diff_prior(diff_tr, ytr, diff_te)
        return prior_te

    inner = _oof_predict(fp_diff, Xa, y_tr, g_tr, task_tr, diff_tr)
    _, diff_te_full = _diff_prior(diff_tr, y_tr, diff_te)
    record("difficulty_mean_only", inner, diff_te_full)

    def make_residual(rank, k, prior="task", model="xgb"):
        Xa, Xb, _ = cols(rank, k)

        def fp(Xtr, ytr, Xte, task_tr, task_te, diff_tr, diff_te):
            if prior == "task":
                prior_tr, prior_te = _task_prior(task_tr, ytr, task_te)
            else:
                prior_tr, prior_te = _diff_prior(diff_tr, ytr, diff_te)
            if model == "xgb":
                m = _xgb()
            elif model == "rf":
                m = RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=0, n_jobs=-1)
            else:
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr)
                Xte_s = sc.transform(Xte)
                m = Ridge(alpha=10)
                m.fit(Xtr_s, ytr - prior_tr)
                return prior_te + m.predict(Xte_s)
            m.fit(Xtr, ytr - prior_tr)
            return prior_te + m.predict(Xte)

        inner = _oof_predict(fp, Xa, y_tr, g_tr, task_tr, diff_tr)
        if prior == "task":
            prior_tr, prior_te = _task_prior(task_tr, y_tr, task_te)
        else:
            prior_tr, prior_te = _diff_prior(diff_tr, y_tr, diff_te)
        if model == "ridge":
            sc = StandardScaler()
            A = sc.fit_transform(Xa)
            B = sc.transform(Xb)
            m = Ridge(alpha=10)
            m.fit(A, y_tr - prior_tr)
            hat = prior_te + m.predict(B)
        else:
            m = _xgb() if model == "xgb" else RandomForestRegressor(
                n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=0, n_jobs=-1)
            m.fit(Xa, y_tr - prior_tr)
            hat = prior_te + m.predict(Xb)
        return inner, hat

    for k in (10, 15, 20):
        record(f"resid_task_MI{k}_xgb", *make_residual(mi_rank, k, "task", "xgb"))
    record("resid_diff_MI15_xgb", *make_residual(mi_rank, 15, "diff", "xgb"))
    record("resid_task_MI15_rf", *make_residual(mi_rank, 15, "task", "rf"))
    record("resid_task_MI15_ridge", *make_residual(mi_rank, 15, "task", "ridge"))

    # 凸组合：a * task_prior + (1-a) * physio，a 用内部 OOF 网格选
    def fp_phys(Xtr, ytr, Xte, *_):
        m = _xgb(); m.fit(Xtr, ytr); return m.predict(Xte)

    oof_phys = _oof_predict(fp_phys, Xa, y_tr, g_tr, task_tr, diff_tr)
    oof_task = _oof_predict(fp_task, Xa, y_tr, g_tr, task_tr, diff_tr)
    oof_diff = _oof_predict(fp_diff, Xa, y_tr, g_tr, task_tr, diff_tr)

    m = _xgb(); m.fit(Xa, y_tr)
    te_phys = m.predict(Xb)
    _, te_task = _task_prior(task_tr, y_tr, task_te)
    _, te_diff = _diff_prior(diff_tr, y_tr, diff_te)

    best_a, best_inner = None, -1e9
    for a in np.round(np.linspace(0, 1, 11), 2):
        hat_in = a * oof_task + (1 - a) * oof_phys
        r2 = r2_score(y_tr, hat_in)
        if r2 > best_inner:
            best_inner, best_a = r2, float(a)
        record(f"mix_task{a:.1f}_phys{1-a:.1f}", hat_in, a * te_task + (1 - a) * te_phys)

    best_b, best_inner_d = None, -1e9
    for b in np.round(np.linspace(0, 1, 11), 2):
        hat_in = b * oof_diff + (1 - b) * oof_phys
        r2 = r2_score(y_tr, hat_in)
        if r2 > best_inner_d:
            best_inner_d, best_b = r2, float(b)
        record(f"mix_diff{b:.1f}_phys{1-b:.1f}", hat_in, b * te_diff + (1 - b) * te_phys)

    # 任务 one-hot 拼进特征
    all_tasks = sorted(set(task_tr.tolist()))
    all_diffs = sorted(set(diff_tr.tolist()))

    def onehot(vals, vocab):
        idx = {v: i for i, v in enumerate(vocab)}
        M = np.zeros((len(vals), len(vocab)))
        for i, v in enumerate(vals):
            if v in idx:
                M[i, idx[v]] = 1.0
        return M

    Xa_t = np.column_stack([Xa, onehot(task_tr, all_tasks)])
    Xb_t = np.column_stack([Xb, onehot(task_te, all_tasks)])
    inner = _oof_predict(fp_phys, Xa_t, y_tr, g_tr, task_tr, diff_tr)
    m = _xgb(); m.fit(Xa_t, y_tr)
    record("physio_MI15_plus_taskOH", inner, m.predict(Xb_t))

    Xa_d = np.column_stack([Xa, onehot(diff_tr, all_diffs)])
    Xb_d = np.column_stack([Xb, onehot(diff_te, all_diffs)])
    inner = _oof_predict(fp_phys, Xa_d, y_tr, g_tr, task_tr, diff_tr)
    m = _xgb(); m.fit(Xa_d, y_tr)
    record("physio_MI15_plus_diffOH", inner, m.predict(Xb_d))

    # 残差再和先验按网格混（收缩残差，防过拟合）
    def fp_resid(Xtr, ytr, Xte, task_tr, task_te, *_):
        prior_tr, prior_te = _task_prior(task_tr, ytr, task_te)
        m = _xgb(); m.fit(Xtr, ytr - prior_tr)
        return prior_te + m.predict(Xte)

    oof_resid = _oof_predict(fp_resid, Xa, y_tr, g_tr, task_tr, diff_tr)
    prior_tr_full, te_task = _task_prior(task_tr, y_tr, task_te)
    m = _xgb(); m.fit(Xa, y_tr - prior_tr_full)
    te_resid = te_task + m.predict(Xb)
    for shrink in (0.0, 0.25, 0.5, 0.75, 1.0):
        # shrink=0 纯先验；1 全残差模型
        hat_in = (1 - shrink) * oof_task + shrink * oof_resid
        hat_te = (1 - shrink) * te_task + shrink * te_resid
        record(f"resid_shrink{shrink:.2f}", hat_in, hat_te)

    results.sort(key=lambda r: -r["inner_r2"])
    best = results[0]
    print("\n===== 按训练内部选出 =====")
    print(best)
    print("（考试分仅验收，不参与选择）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": {"train_subjects": train_subj.tolist(), "test_subjects": test_subj.tolist()},
        "best_by_inner": best,
        "top8_inner": results[:8],
        "all": results,
        "mix_best_task_a": best_a,
        "mix_best_diff_b": best_b,
        "note": "inner_r2 选配置；test_r2 只验收。task/difficulty 表示考试时任务已知。",
    }
    (OUT_DIR / "try_more_train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("写完", OUT_DIR / "try_more_train.json")


if __name__ == "__main__":
    main()
