#!/usr/bin/env python3
"""用 NASA 分类那套 27 维配额，五折预测 NASA 再合成 S。

27 = 眼动6 + 脑电5 + 心率4 + 行为12
每一折在各模态内部按互信息取 Top-K，不是全实验锁死 27 个名字。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANDOM_STATE  # noqa: E402

from exp_s_fullmodal_mi30 import (  # noqa: E402
    NASA_DS,
    S_TABLE,
    STEP_W,
    XGB_CFG,
    _enable_xgboost,
    modality_of,
)

CLS_DS = HERE.parent / "classification_task_level_nasa" / "dataset"
OUT = HERE / "reports_s_fullmodal"
N_SPLITS = 5
QUOTA = {"眼动": 6, "脑电": 5, "心率": 4, "行为": 12}  # 27


def build_mod_idx(names: list[str]) -> dict[str, np.ndarray]:
    mods = {"眼动": [], "脑电": [], "心率": [], "行为": []}
    for i, n in enumerate(names):
        mods[modality_of(n)].append(i)
    return {k: np.array(v, dtype=int) for k, v in mods.items()}


def select_quota(X_tr: np.ndarray, y_rank: np.ndarray, mod_idx: dict, classif: bool) -> np.ndarray:
    picked = []
    for mod, idx in mod_idx.items():
        k = QUOTA[mod]
        Xmod = X_tr[:, idx]
        if classif:
            mi = mutual_info_classif(Xmod, y_rank, random_state=RANDOM_STATE)
        else:
            mi = mutual_info_regression(Xmod, y_rank, random_state=RANDOM_STATE)
        order = np.argsort(-mi)[:k]
        picked.extend(idx[order].tolist())
    return np.array(picked, dtype=int)


def mix_s(step, nasa, a):
    return a * step + (1.0 - a) * (1.0 - nasa / 10.0)


def eval_s(step, y_true, y_hat, alphas=(0.50, 0.70)):
    out = {
        "nasa_r2": float(r2_score(y_true, y_hat)),
        "nasa_mae": float(mean_absolute_error(y_true, y_hat)),
    }
    for a in alphas:
        st = mix_s(step, y_true, a)
        sp = mix_s(step, y_hat, a)
        out[f"s_r2_step{int(a*10):02d}"] = float(r2_score(st, sp))
        out[f"s_mae_step{int(a*10):02d}"] = float(mean_absolute_error(st, sp))
    return out


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor

    X = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    step = s_table["weighted_step_score"].to_numpy(dtype=float)
    y_s07 = mix_s(step, y, 0.70)
    y_s05 = mix_s(step, y, 0.50)

    cls_samples = np.load(CLS_DS / "sample_cls.npy", allow_pickle=True).astype(str)
    y_cls = np.load(CLS_DS / "y_cls_int.npy")
    cls_map = {sid: int(c) for sid, c in zip(cls_samples, y_cls)}
    y_cls_aligned = np.array([cls_map[s] for s in samples])

    mod_idx = build_mod_idx(names)
    gkf = GroupKFold(n_splits=N_SPLITS)

    def run(name, y_rank, classif, y_target):
        hat = np.full(len(y_target), np.nan)
        fold_counts = []
        for tr, te in gkf.split(X, y_target, groups):
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(X[tr])
            Xte = imp.transform(X[te])
            top = select_quota(Xtr, y_rank[tr], mod_idx, classif=classif)
            c = Counter(modality_of(names[i]) for i in top)
            fold_counts.append({m: int(c[m]) for m in QUOTA})
            m = XGBRegressor(**XGB_CFG)
            m.fit(Xtr[:, top], y_target[tr])
            hat[te] = m.predict(Xte[:, top])
        stats = eval_s(step, y, hat) if np.allclose(y_target, y) else {
            "target_r2": float(r2_score(y_target, hat)),
            "target_mae": float(mean_absolute_error(y_target, hat)),
        }
        # 若直接猜 S，再报对 0.5/0.7 真 S 的 R²
        if not np.allclose(y_target, y):
            stats["s05_as_target_r2"] = float(r2_score(y_s05, hat)) if abs(y_target.mean() - y_s05.mean()) < 0.05 else None
        print(name, json.dumps(stats, ensure_ascii=False))
        return {"name": name, "fold_counts": fold_counts, **stats, "hat": hat}

    print("配额", QUOTA, "合计", sum(QUOTA.values()))
    results = []

    r = run("27_MI对NASA连续_预测NASA", y, classif=False, y_target=y)
    results.append({k: v for k, v in r.items() if k != "hat"})
    np.save(OUT / "yhat_nasa_quota27_mi_reg.npy", r["hat"])

    r = run("27_MI对NASA三档_预测NASA", y_cls_aligned, classif=True, y_target=y)
    results.append({k: v for k, v in r.items() if k != "hat"})
    np.save(OUT / "yhat_nasa_quota27_mi_cls.npy", r["hat"])

    # 直接猜 S（0.50 和 0.70 两套标签）
    for a, y_s in ((0.50, y_s05), (0.70, y_s07)):
        hat = np.full(len(y_s), np.nan)
        for tr, te in gkf.split(X, y_s, groups):
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(X[tr])
            Xte = imp.transform(X[te])
            top = select_quota(Xtr, y_s[tr], mod_idx, classif=False)
            m = XGBRegressor(**XGB_CFG)
            m.fit(Xtr[:, top], y_s[tr])
            hat[te] = m.predict(Xte[:, top])
        row = {
            "name": f"27_MI对S_直接预测S_step{int(a*10):02d}",
            "s_direct_r2": float(r2_score(y_s, hat)),
            "s_direct_mae": float(mean_absolute_error(y_s, hat)),
            "alpha_step": a,
        }
        print(row["name"], f"R²={row['s_direct_r2']:+.3f} MAE={row['s_direct_mae']:.3f}")
        results.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_quota27.json").write_text(
        json.dumps({"quota": QUOTA, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("写完", OUT / "results_quota27.json")


if __name__ == "__main__":
    main()
