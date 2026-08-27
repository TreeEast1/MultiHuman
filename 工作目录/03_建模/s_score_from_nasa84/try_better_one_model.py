#!/usr/bin/env python3
"""在「一个模型、20 人训练 / 6 人考试」下试更好的特征和模型。

规则：
- 配置只看训练 20 人内部的 4 堆交叉验证来挑
- 6 个考试人只用于最后验收，不参与选配置
- 同时对照：原来的随机 20/6、按人均 NASA 分层的 20/6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
OUT_DIR = HERE / "output_one_model"
TOP_K_DEFAULT = 30
XGB_CFG = dict(
    max_depth=2,
    learning_rate=0.02,
    reg_lambda=2.0,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=0,
)


def _enable_xgboost() -> None:
    import ctypes
    import os
    from pathlib import Path as P

    import sklearn

    omp = P(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp.exists():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(omp.parent))
        ctypes.CDLL(str(omp), mode=ctypes.RTLD_GLOBAL)


def _mask_names(names: list[str], kind: str) -> np.ndarray:
    arr = np.array(names)
    if kind == "all":
        return np.ones(len(arr), dtype=bool)
    if kind == "only_std":
        return np.array([n.endswith("__std") for n in names])
    if kind == "aoi_log":
        return np.array([n.startswith("eye_aoi_") or n.startswith("log_") for n in names])
    if kind == "aoi_log_std":
        return np.array(
            [(n.startswith("eye_aoi_") or n.startswith("log_")) and n.endswith("__std") for n in names]
        )
    if kind == "minus_eeg":
        return np.array([not n.startswith("eeg_") for n in names])
    if kind == "minus_hr":
        return np.array([not n.startswith("hr_") for n in names])
    if kind == "aoi_log_eye":
        return np.array([n.startswith(("eye_", "log_", "blink_")) for n in names])
    raise KeyError(kind)


def _split_random(groups: np.ndarray, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.array(sorted(set(int(g) for g in groups)))
    rng = np.random.RandomState(seed)
    test_subjects = np.sort(rng.choice(subjects, size=n_test, replace=False))
    train_subjects = np.array([s for s in subjects if s not in set(test_subjects)])
    return train_subjects, test_subjects


def _split_stratified(y: np.ndarray, groups: np.ndarray, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按每人平均 NASA 分成低/中/高，每档抽 2 人考试。"""
    subjects = np.array(sorted(set(int(g) for g in groups)))
    means = np.array([y[groups == s].mean() for s in subjects])
    bins = np.array_split(subjects[np.argsort(means)], 3)
    rng = np.random.RandomState(seed)
    test = []
    for b in bins:
        test.extend(rng.choice(b, size=min(2, len(b)), replace=False).tolist())
    test_subjects = np.sort(np.array(test, dtype=int))
    train_subjects = np.array([s for s in subjects if s not in set(test_subjects)])
    return train_subjects, test_subjects


def _inner_cv_r2(model_factory, X, y, groups, n_splits=4) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return float(r2_score(y, preds))


def _fit_eval(model_factory, X_tr, y_tr, X_te, y_te):
    m = model_factory()
    m.fit(X_tr, y_tr)
    hat = m.predict(X_te)
    return m, hat, float(mean_absolute_error(y_te, hat)), float(r2_score(y_te, hat))


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor

    X_raw = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy").astype(int)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())

    splits = {
        "random_seed0": _split_random(groups, 6, 0),
        "strat_seed0": _split_stratified(y, groups, 6, 0),
        "strat_seed1": _split_stratified(y, groups, 6, 1),
    }

    def xgb():
        return XGBRegressor(**XGB_CFG)

    def xgb_mid():
        return XGBRegressor(
            max_depth=2, learning_rate=0.05, reg_lambda=5.0, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0,
        )

    def rf():
        return RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)

    def et():
        return ExtraTreesRegressor(n_estimators=400, max_depth=4, min_samples_leaf=2, random_state=0, n_jobs=-1)

    def hgb():
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)

    models = {
        "XGB_best": xgb,
        "XGB_mid": xgb_mid,
        "RF": rf,
        "ET": et,
        "HGB": hgb,
    }

    pools = ["all", "only_std", "aoi_log", "aoi_log_std", "minus_eeg", "minus_hr", "aoi_log_eye"]
    ks = [15, 20, 30, 50, 80]

    all_rows = []
    best_by_split = {}

    for split_name, (train_subj, test_subj) in splits.items():
        tr = np.isin(groups, train_subj)
        te = np.isin(groups, test_subj)
        y_tr, y_te = y[tr], y[te]
        g_tr = groups[tr]
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_raw[tr])
        X_te_imp = imp.transform(X_raw[te])

        print(f"\n===== {split_name}  训练{train_subj.tolist()}  考试{test_subj.tolist()}  "
              f"n={int(tr.sum())}/{int(te.sum())}  "
              f"NASA训练{y_tr.mean():.2f}±{y_tr.std():.2f} 考试{y_te.mean():.2f}±{y_te.std():.2f}")

        split_rows = []
        for pool in pools:
            col_mask = _mask_names(names, pool)
            pool_idx = np.where(col_mask)[0]
            if len(pool_idx) < 5:
                continue
            Xtr_p = X_tr_imp[:, pool_idx]
            Xte_p = X_te_imp[:, pool_idx]
            # MI 只算一次
            mi_rank = RANKERS["MI"](Xtr_p, y_tr)
            for k in ks:
                if k > len(pool_idx):
                    continue
                sel = mi_rank[:k]
                Xtr_k, Xte_k = Xtr_p[:, sel], Xte_p[:, sel]
                for mname, factory in models.items():
                    inner = _inner_cv_r2(factory, Xtr_k, y_tr, g_tr, n_splits=4)
                    _, hat, mae, r2 = _fit_eval(factory, Xtr_k, y_tr, Xte_k, y_te)
                    row = {
                        "split": split_name,
                        "pool": pool,
                        "k": int(k),
                        "n_pool": int(len(pool_idx)),
                        "model": mname,
                        "inner_r2": inner,
                        "test_mae": mae,
                        "test_r2": r2,
                    }
                    split_rows.append(row)
                    all_rows.append(row)

            # Ridge：标准化后全池 / top30
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_p)
            Xte_s = scaler.transform(Xte_p)
            for k in [30, len(pool_idx)]:
                if k > len(pool_idx):
                    continue
                if k < len(pool_idx):
                    sel = mi_rank[:k]
                    Xa, Xb = Xtr_s[:, sel], Xte_s[:, sel]
                    tag = f"Ridge_k{k}"
                else:
                    Xa, Xb = Xtr_s, Xte_s
                    tag = "Ridge_fullpool"
                factory = lambda: Ridge(alpha=10.0)
                inner = _inner_cv_r2(factory, Xa, y_tr, g_tr, 4)
                _, _, mae, r2 = _fit_eval(factory, Xa, y_tr, Xb, y_te)
                row = {
                    "split": split_name,
                    "pool": pool,
                    "k": int(k),
                    "n_pool": int(len(pool_idx)),
                    "model": tag,
                    "inner_r2": inner,
                    "test_mae": mae,
                    "test_r2": r2,
                }
                split_rows.append(row)
                all_rows.append(row)

        # 集成：XGB + RF，MI30 all
        mi_all = RANKERS["MI"](X_tr_imp, y_tr)[:30]
        xgb_m = xgb(); rf_m = rf()
        xgb_m.fit(X_tr_imp[:, mi_all], y_tr)
        rf_m.fit(X_tr_imp[:, mi_all], y_tr)
        hat = 0.5 * xgb_m.predict(X_te_imp[:, mi_all]) + 0.5 * rf_m.predict(X_te_imp[:, mi_all])
        # inner for ensemble: average two inner preds
        gkf = GroupKFold(n_splits=4)
        ip = np.full(len(y_tr), np.nan)
        for itr, ite in gkf.split(X_tr_imp[:, mi_all], y_tr, g_tr):
            a = xgb(); b = rf()
            a.fit(X_tr_imp[:, mi_all][itr], y_tr[itr])
            b.fit(X_tr_imp[:, mi_all][itr], y_tr[itr])
            ip[ite] = 0.5 * a.predict(X_tr_imp[:, mi_all][ite]) + 0.5 * b.predict(X_tr_imp[:, mi_all][ite])
        row = {
            "split": split_name,
            "pool": "all",
            "k": 30,
            "n_pool": 264,
            "model": "XGB+RF",
            "inner_r2": float(r2_score(y_tr, ip)),
            "test_mae": float(mean_absolute_error(y_te, hat)),
            "test_r2": float(r2_score(y_te, hat)),
        }
        split_rows.append(row)
        all_rows.append(row)

        split_rows.sort(key=lambda r: r["inner_r2"], reverse=True)
        best = split_rows[0]
        best_by_split[split_name] = {
            "train_subjects": train_subj.tolist(),
            "test_subjects": test_subj.tolist(),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "best_by_inner": best,
            "top5_inner": split_rows[:5],
            "best_test_cheat": max(split_rows, key=lambda r: r["test_r2"]),
        }
        print(f"  按训练内部选：{best['model']} {best['pool']} k={best['k']}  "
              f"inner R²={best['inner_r2']:+.3f}  考试 R²={best['test_r2']:+.3f} MAE={best['test_mae']:.3f}")
        cheat = best_by_split[split_name]["best_test_cheat"]
        print(f"  （若偷看考试，最好能到 R²={cheat['test_r2']:+.3f}，不采用）")

    # 原配置在多个随机划分上的波动
    seed_rows = []
    for seed in range(12):
        tr_s, te_s = _split_random(groups, 6, seed)
        tr = np.isin(groups, tr_s)
        te = np.isin(groups, te_s)
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_raw[tr])
        Xte = imp.transform(X_raw[te])
        sel = RANKERS["MI"](Xtr, y[tr])[:30]
        m = xgb()
        m.fit(Xtr[:, sel], y[tr])
        hat = m.predict(Xte[:, sel])
        seed_rows.append({
            "seed": seed,
            "test_subjects": te_s.tolist(),
            "test_mae": float(mean_absolute_error(y[te], hat)),
            "test_r2": float(r2_score(y[te], hat)),
            "test_n": int(te.sum()),
        })
    r2s = np.array([r["test_r2"] for r in seed_rows])
    print("\n===== 原配置 MI30+XGB 在 12 个随机 20/6 上 =====")
    print(f"  R² 中位 {np.median(r2s):+.3f}  均值 {r2s.mean():+.3f}  范围 {r2s.min():+.3f} ~ {r2s.max():+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "note": "inner_r2 才是选配置依据；test_r2 是验收。best_test_cheat 仅对照，不采用。",
        "best_by_split": best_by_split,
        "random_split_variance": {
            "r2_mean": float(r2s.mean()),
            "r2_median": float(np.median(r2s)),
            "r2_min": float(r2s.min()),
            "r2_max": float(r2s.max()),
            "rows": seed_rows,
        },
        "all_rows": all_rows,
    }
    (OUT_DIR / "try_better_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n写完", OUT_DIR / "try_better_results.json")


if __name__ == "__main__":
    main()
