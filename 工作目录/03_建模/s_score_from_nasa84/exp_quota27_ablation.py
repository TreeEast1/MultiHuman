#!/usr/bin/env python3
"""27 维定额、NASA 公式法：模态消融。

与 exp_quota27_s.py 同一协议：按被试五折，各模态内部 MI 定额，
XGB 预测 NASA，再按步骤 0.70 / NASA 反向 0.30 合成 S。
去掉某个模态时，只用该组合对应定额，不再补到 27。
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANDOM_STATE  # noqa: E402

from exp_s_fullmodal_mi30 import NASA_DS, S_TABLE, XGB_CFG, _enable_xgboost, modality_of
from exp_quota27_s import QUOTA, mix_s

OUT = HERE / "reports_s_fullmodal"
N_SPLITS = 5
MODS = ("眼动", "脑电", "心率", "行为")


def build_mod_idx(names: list[str]) -> dict[str, np.ndarray]:
    mods = {m: [] for m in MODS}
    for i, n in enumerate(names):
        mods[modality_of(n)].append(i)
    return {k: np.array(v, dtype=int) for k, v in mods.items()}


def select_quota_combo(
    X_tr: np.ndarray, y_tr: np.ndarray, mod_idx: dict, combo: tuple[str, ...]
) -> np.ndarray:
    picked = []
    for mod in combo:
        k = QUOTA[mod]
        idx = mod_idx[mod]
        mi = mutual_info_regression(X_tr[:, idx], y_tr, random_state=RANDOM_STATE)
        order = np.argsort(-mi)[:k]
        picked.extend(idx[order].tolist())
    return np.array(picked, dtype=int)


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
    y_s = mix_s(step, y, 0.70)

    mod_idx = build_mod_idx(names)
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []

    combos = []
    for k in range(1, 5):
        combos.extend(itertools.combinations(MODS, k))

    for combo in combos:
        hat = np.full(len(y), np.nan)
        n_feat = sum(QUOTA[m] for m in combo)
        for tr, te in gkf.split(X, y, groups):
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(X[tr])
            Xte = imp.transform(X[te])
            top = select_quota_combo(Xtr, y[tr], mod_idx, combo)
            m = XGBRegressor(**XGB_CFG)
            m.fit(Xtr[:, top], y[tr])
            hat[te] = m.predict(Xte[:, top])
        s_hat = mix_s(step, hat, 0.70)
        row = {
            "n_mod": len(combo),
            "combo": "+".join(combo),
            "n_feat": n_feat,
            "nasa_r2": float(r2_score(y, hat)),
            "nasa_mae": float(mean_absolute_error(y, hat)),
            "s_r2": float(r2_score(y_s, s_hat)),
            "s_mae": float(mean_absolute_error(y_s, s_hat)),
        }
        rows.append(row)
        print(
            f"{row['n_mod']}  {row['combo']:<16}  n={n_feat:2d}  "
            f"NASA R²={row['nasa_r2']:+.3f}  S R²={row['s_r2']:+.3f}"
        )

    rows.sort(key=lambda r: (-r["s_r2"], -r["nasa_r2"]))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "results_quota27_ablation.json"
    out_path.write_text(
        json.dumps({"quota": QUOTA, "alpha_step": 0.70, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("写完", out_path)


if __name__ == "__main__":
    main()
