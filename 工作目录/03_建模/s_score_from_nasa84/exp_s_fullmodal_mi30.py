#!/usr/bin/env python3
"""五折交叉验证：按 S 做互信息 Top-30，并保证全模态。

需求：
- 仍按被试 GroupKFold 5 折，只留一套 pooled R²
- Top-30 的互信息对的是 S（0.70 步骤 + 0.30 NASA反向），不再对 NASA
- 30 列里脑电 / 心率 / 眼动 / 行为每类至少 1 个（另做至少 2 个的对照）
- 保底之后，其余名额仍按互信息从高到低填，重要模态会自然占满

对照：
- 不对 S 做约束的普通 MI Top-30
- 旧路径：MI 对 NASA → 预测 NASA → 再合成 S（已有结果，这里重算一遍对齐）
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS, RANDOM_STATE, pooled_cv_with_selection  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
S_TABLE = HERE / "output" / "s_score_84samples.csv"
OUT_DIR = HERE / "reports_s_fullmodal"
N_SPLITS = 5
TOP_K = 30
STEP_W = 0.70
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
MODALITY_ORDER = ("脑电", "心率", "眼动", "行为")


def _enable_xgboost() -> None:
    import ctypes
    import os
    from pathlib import Path as P

    import sklearn

    omp = P(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp.exists():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(omp.parent))
        ctypes.CDLL(str(omp), mode=ctypes.RTLD_GLOBAL)


def modality_of(name: str) -> str:
    if name.startswith("eeg_"):
        return "脑电"
    if name.startswith("hr_"):
        return "心率"
    if name.startswith("log_"):
        return "行为"
    if name.startswith("blink_") or name.startswith("eye_"):
        return "眼动"
    return "其他"


def make_fullmodal_ranker(names: list[str], min_per: int):
    """先给每个模态留 min_per 个最高 MI，再按全局 MI 填满。"""

    def ranker(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
        mi = mutual_info_regression(X_tr, y_tr, random_state=RANDOM_STATE)
        order = np.argsort(-mi)
        selected: list[int] = []
        used: set[int] = set()
        for mod in MODALITY_ORDER:
            cand = [i for i in order if modality_of(names[i]) == mod and i not in used]
            for i in cand[:min_per]:
                selected.append(int(i))
                used.add(int(i))
        for i in order:
            if len(selected) >= TOP_K:
                break
            ii = int(i)
            if ii not in used:
                selected.append(ii)
                used.add(ii)
        rest = [int(i) for i in order if int(i) not in used]
        return np.array(selected + rest, dtype=int)

    return ranker


def counts_of(names: list[str], idx: np.ndarray) -> dict[str, int]:
    c = Counter(modality_of(names[i]) for i in idx)
    return {m: int(c.get(m, 0)) for m in MODALITY_ORDER}


def fold_summary(names: list[str], selected: list[np.ndarray]) -> dict:
    per_fold = []
    sets = []
    for k, idx in enumerate(selected):
        sel_names = [names[i] for i in idx]
        per_fold.append({
            "fold": k + 1,
            "counts": counts_of(names, idx),
            "names": sel_names,
        })
        sets.append(set(sel_names))
    stable = sorted(set.intersection(*sets)) if sets else []
    avg = {m: float(np.mean([f["counts"][m] for f in per_fold])) for m in MODALITY_ORDER}
    return {"per_fold": per_fold, "avg_counts": avg, "stable_all_folds": stable}


def pack_cv(res, names, selected) -> dict:
    return {
        "pooled_mae": float(res.pooled_mae),
        "pooled_r2": float(res.pooled_r2),
        "fold_r2_mean": float(res.fold_r2_mean),
        "fold_r2_std": float(res.fold_r2_std),
        "fold_mae_mean": float(res.fold_mae_mean),
        "fold_mae_std": float(res.fold_mae_std),
        "selection": fold_summary(names, selected),
    }


def main() -> None:
    _enable_xgboost()
    from xgboost import XGBRegressor

    X = np.load(NASA_DS / "X_task.npy")
    y_nasa = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text())
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    if not np.allclose(s_table["y_nasa"].to_numpy(), y_nasa, atol=1e-8):
        raise RuntimeError("S 表与 NASA 标签对不齐")
    step = s_table["weighted_step_score"].to_numpy(dtype=float)
    y_s = STEP_W * step + (1.0 - STEP_W) * (1.0 - y_nasa / 10.0)

    catalog = Counter(modality_of(n) for n in names)
    print("264 列构成:", dict(catalog))
    print(f"S(0.70/0.30) 范围 [{y_s.min():.3f}, {y_s.max():.3f}]  均 {y_s.mean():.3f}")

    def xgb():
        return XGBRegressor(**XGB_CFG)

    results: dict[str, dict] = {}

    print("\n===== 旧路径：MI 对 NASA → 预测 NASA → 合成 S =====")
    res_nasa, sel_nasa = pooled_cv_with_selection(
        xgb, X, y_nasa, groups, N_SPLITS, TOP_K, RANKERS["MI"], None, "NASA_MI30",
    )
    y_nasa_hat = res_nasa.y_pred_pooled
    s_from_nasa = STEP_W * step + (1.0 - STEP_W) * (1.0 - y_nasa_hat / 10.0)
    results["old_MI_vs_NASA_then_compose_S"] = {
        **pack_cv(res_nasa, names, sel_nasa),
        "target": "NASA",
        "composed_S_r2": float(r2_score(y_s, s_from_nasa)),
        "composed_S_mae": float(mean_absolute_error(y_s, s_from_nasa)),
        "note": "互信息对 NASA；模型预测 NASA；S 用真步骤 + 预测 NASA",
    }
    print(f"  NASA pooled R²={res_nasa.pooled_r2:+.3f}  MAE={res_nasa.pooled_mae:.3f}")
    print(f"  合成 S     R²={results['old_MI_vs_NASA_then_compose_S']['composed_S_r2']:+.3f}  "
          f"MAE={results['old_MI_vs_NASA_then_compose_S']['composed_S_mae']:.3f}")
    print("  各折 Top-30 模态:", results["old_MI_vs_NASA_then_compose_S"]["selection"]["avg_counts"])

    configs = [
        ("S_MI30_free", RANKERS["MI"], "互信息对 S，不强制模态"),
        ("S_MI30_fullmodal_min1", make_fullmodal_ranker(names, 1), "互信息对 S，四模态各至少 1 个"),
        ("S_MI30_fullmodal_min2", make_fullmodal_ranker(names, 2), "互信息对 S，四模态各至少 2 个"),
    ]
    print("\n===== 直接预测 S =====")
    for name, ranker, note in configs:
        res, sel = pooled_cv_with_selection(
            xgb, X, y_s, groups, N_SPLITS, TOP_K, ranker, None, name,
        )
        results[name] = {
            **pack_cv(res, names, sel),
            "target": "S",
            "note": note,
        }
        avg = results[name]["selection"]["avg_counts"]
        print(f"  {name:28s}  S pooled R²={res.pooled_r2:+.3f}  MAE={res.pooled_mae:.3f}  "
              f"折间 R² {res.fold_r2_mean:+.3f}±{res.fold_r2_std:.3f}  平均构成 {avg}")
        np.save(OUT_DIR / f"yhat_{name}.npy", res.y_pred_pooled)

    np.save(OUT_DIR / "y_s_true.npy", y_s)
    np.save(OUT_DIR / "y_nasa_hat_old.npy", y_nasa_hat)
    np.save(OUT_DIR / "s_from_nasa_hat.npy", s_from_nasa)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": int(len(y_s)),
        "n_splits": N_SPLITS,
        "top_k": TOP_K,
        "formula_S": "S = 0.70 * weighted_step + 0.30 * (1 - NASA/10)",
        "xgb": XGB_CFG,
        "modality_catalog_264": dict(catalog),
        "fullmodal_rule": "每折训练堆上算 MI(S, 特征)；四模态各先取 min_per 个最高 MI，其余按全局 MI 补到 30",
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# 五折：按 S 筛全模态 Top-30\n\n"]
    lines.append("**划分**：26 人 5 折按被试分组，训练折上筛 30 列，考试折只推理。主指标是 84 条拼在一起的 pooled R²。\n\n")
    lines.append("**S 公式**：`S = 0.70 × 步骤分 + 0.30 × (1 − NASA/10)`。步骤分用真实操作记录。\n\n")
    lines.append("**全模态规则**：脑电 / 心率 / 眼动 / 行为每类至少 1 个（对照至少 2 个）。先在该类里取互信息最高的，再按全局互信息把重要特征填进剩下名额。\n\n")
    lines.append("## 结果\n\n")
    lines.append("| 方案 | 预测目标 | pooled R² | pooled MAE | 平均 脑电/心率/眼动/行为 |\n")
    lines.append("|---|---|---:|---:|---|\n")
    old = results["old_MI_vs_NASA_then_compose_S"]
    c = old["selection"]["avg_counts"]
    lines.append(
        f"| 旧：MI→NASA 再合成 S | NASA（合成 S） | {old['composed_S_r2']:+.3f} | {old['composed_S_mae']:.3f} | "
        f"{c['脑电']:.1f} / {c['心率']:.1f} / {c['眼动']:.1f} / {c['行为']:.1f} |\n"
    )
    lines.append(
        f"| 旧路径的 NASA 本身 | NASA | {old['pooled_r2']:+.3f} | {old['pooled_mae']:.3f} | 同上 |\n"
    )
    for key in ("S_MI30_free", "S_MI30_fullmodal_min1", "S_MI30_fullmodal_min2"):
        r = results[key]
        c = r["selection"]["avg_counts"]
        lines.append(
            f"| {r['note']} | S | {r['pooled_r2']:+.3f} | {r['pooled_mae']:.3f} | "
            f"{c['脑电']:.1f} / {c['心率']:.1f} / {c['眼动']:.1f} / {c['行为']:.1f} |\n"
        )
    lines.append("\n无约束时心率经常进不了 30；全模态约束后每折都有心率。主推荐：**互信息对 S + 四模态各至少 1 个 + XGB**。\n")
    (OUT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")
    print("\n写完", OUT_DIR / "results.json")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
