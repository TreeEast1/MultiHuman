#!/usr/bin/env python3
"""只训一个 NASA 模型：20 人训练，6 人测试。

现行配置（训练 20 人内部交叉验证选出）：
    预测 = 该任务在训练人中的平均 NASA
         + XGB 残差（去心率，MI Top-10）

考试时任务是已知的（做哪一关事先知道）。生理特征只负责解释「比该任务平均更高/更低」。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from exp_utils import RANKERS  # noqa: E402

NASA_DS = HERE.parent / "regression_task_level" / "dataset"
OUT_DIR = HERE / "output_one_model"
TOP_K = 10
N_TEST_SUBJECTS = 6
RANDOM_STATE = 0
DROP_HR = True
XGB_CFG = dict(
    max_depth=2,
    learning_rate=0.05,
    reg_lambda=5.0,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=0,
)


def _enable_xgboost_on_macos() -> None:
    import ctypes
    import os
    from pathlib import Path as P

    import sklearn

    omp_lib = P(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp_lib.exists():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(omp_lib.parent))
        ctypes.CDLL(str(omp_lib), mode=ctypes.RTLD_GLOBAL)


def _modality(name: str) -> str:
    if name.startswith("eeg_"):
        return "脑电"
    if name.startswith("hr_"):
        return "心率"
    if name.startswith("log_"):
        return "行为"
    if name.startswith("blink_"):
        return "眨眼"
    if name.startswith("eye_"):
        return "眼动"
    return "其他"


def main() -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score

    _enable_xgboost_on_macos()
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

    keep = np.array([not n.startswith("hr_") for n in names]) if DROP_HR else np.ones(len(names), dtype=bool)
    keep_idx = np.where(keep)[0]
    names_kept = [names[i] for i in keep_idx]
    X = X[:, keep_idx]

    subjects = np.array(sorted(set(int(g) for g in groups)))
    rng = np.random.RandomState(RANDOM_STATE)
    test_subjects = np.sort(rng.choice(subjects, size=N_TEST_SUBJECTS, replace=False))
    train_subjects = np.array([s for s in subjects if s not in set(test_subjects)])

    tr = np.isin(groups, train_subjects)
    te = np.isin(groups, test_subjects)

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X[tr])
    X_te = imputer.transform(X[te])

    rank = RANKERS["MI"](X_tr, y[tr])
    top_idx = rank[:TOP_K]
    top_names = [names_kept[i] for i in top_idx]
    counts = Counter(_modality(n) for n in top_names)

    y_tr = y[tr]
    task_tr = tasks[tr]
    task_te = tasks[te]
    task_means = {t: float(y_tr[task_tr == t].mean()) for t in sorted(set(task_tr))}
    global_mean = float(y_tr.mean())
    prior_tr = np.array([task_means.get(t, global_mean) for t in task_tr])
    prior_te = np.array([task_means.get(t, global_mean) for t in task_te])

    model = XGBRegressor(**XGB_CFG)
    model.fit(X_tr[:, top_idx], y_tr - prior_tr)
    y_hat = prior_te + model.predict(X_te[:, top_idx])

    mae = float(mean_absolute_error(y[te], y_hat))
    r2 = float(r2_score(y[te], y_hat))
    mae_prior = float(mean_absolute_error(y[te], prior_te))
    r2_prior = float(r2_score(y[te], prior_te))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(OUT_DIR / "nasa_xgb_one_model.json")
    pred_rows = [
        {
            "sample_id": samples[i],
            "subject": int(groups[i]),
            "task": str(tasks[i]),
            "y_nasa_true": float(y[i]),
            "y_nasa_prior": float(prior_te[k]),
            "y_nasa_pred": float(y_hat[k]),
        }
        for k, i in enumerate(np.where(te)[0])
    ]
    (OUT_DIR / "test_predictions.json").write_text(
        json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "method": "task_mean_prior + XGB residual",
        "n_train_subjects": int(len(train_subjects)),
        "n_test_subjects": int(len(test_subjects)),
        "train_subjects": train_subjects.tolist(),
        "test_subjects": test_subjects.tolist(),
        "n_train_samples": int(tr.sum()),
        "n_test_samples": int(te.sum()),
        "drop_hr": DROP_HR,
        "top_k": TOP_K,
        "top_names": top_names,
        "top_counts": dict(counts),
        "task_means_from_train": task_means,
        "global_mean": global_mean,
        "imputer_medians": imputer.statistics_.tolist(),
        "test_mae": mae,
        "test_r2": r2,
        "task_prior_only_mae": mae_prior,
        "task_prior_only_r2": r2_prior,
        "xgb": XGB_CFG,
        "split_random_state": RANDOM_STATE,
        "model_path": "nasa_xgb_one_model.json",
        "selection": "inner GroupKFold on 20 train subjects; resid_task_MI10_xgb won",
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("训练人:", train_subjects.tolist(), f"共 {int(tr.sum())} 条")
    print("考试人:", test_subjects.tolist(), f"共 {int(te.sum())} 条")
    print("任务先验（训练人各任务平均 NASA）:", {k: round(v, 2) for k, v in task_means.items()})
    print("残差特征:", dict(counts), top_names)
    print(f"只猜任务平均  MAE={mae_prior:.3f}  R²={r2_prior:+.3f}")
    print(f"任务平均+残差  MAE={mae:.3f}  R²={r2:+.3f}")
    print("模型已保存:", OUT_DIR / "nasa_xgb_one_model.json")


if __name__ == "__main__":
    main()
