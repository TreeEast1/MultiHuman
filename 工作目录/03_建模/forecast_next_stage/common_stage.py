#!/usr/bin/env python3
"""下一阶段人因指标预测 → 冻结 27 维定额 XGB → NASA → S 的共用工具。

约定
----
- 窗口级 66 维与任务级 264 维（mean/std/median/slope）与
  ``regression_task_level/make_dataset_task.py`` 完全一致。
- 下游 NASA 模型与 ``exp_quota27_s.py`` 相同：按被试 GroupKFold，
  折内互信息定额 27 维（眼动 6 + 脑电 5 + 心率 4 + 行为 12），浅树 XGB。
- S = 0.70 × 真实步骤分 + 0.30 × (1 − NASA/10)。步骤分不走模型。
- 所有填充 / 特征选择 / 模型只在训练被试上 fit。
"""

from __future__ import annotations

import ctypes
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NASA_DS = HERE.parent / "regression_task_level" / "dataset"
WIN_DIR = HERE.parents[1] / "01_预处理" / "output_30s_step5s_final"
S_TABLE = HERE.parent / "s_score_from_nasa84" / "output" / "s_score_84samples.csv"
CACHE_DIR = HERE / "cache"
REPORTS = HERE / "reports"

RANDOM_STATE = 0
N_SPLITS = 5
STEP_W = 0.70
QUOTA = {"眼动": 6, "脑电": 5, "心率": 4, "行为": 12}
STATS = ("mean", "std", "median", "slope")
TASK_ORDER = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "5_6": 5}

XGB_NASA_CFG = dict(
    max_depth=2,
    learning_rate=0.02,
    reg_lambda=2.0,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

XGB_FORECAST_CFG = dict(
    max_depth=2,
    learning_rate=0.05,
    reg_lambda=2.0,
    n_estimators=80,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


def enable_xgboost() -> None:
    import sklearn

    omp = Path(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
    if omp.exists():
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


def mix_s(step: np.ndarray, nasa: np.ndarray, alpha: float = STEP_W) -> np.ndarray:
    return alpha * step + (1.0 - alpha) * (1.0 - nasa / 10.0)


def load_feature_names() -> tuple[list[str], list[str]]:
    names_264 = json.loads((NASA_DS / "feature_names_task.json").read_text(encoding="utf-8"))
    raw: list[str] = []
    for n in names_264:
        base = n.rsplit("__", 1)[0]
        if base not in raw:
            raw.append(base)
    if len(raw) * 4 != len(names_264):
        raise RuntimeError(f"264 列与 66×4 对不上：raw={len(raw)} names={len(names_264)}")
    return names_264, raw


def slope_1d(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(v)
    if mask.sum() < 2:
        return np.nan
    y = v[mask]
    x = np.arange(len(v), dtype=np.float64)[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def aggregate_windows(W: np.ndarray) -> np.ndarray:
    """(n_win, 66) → (264,) 顺序与 feature_names_task.json 一致。"""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W 必须是二维")
    n_feat = W.shape[1]
    out = np.empty(n_feat * 4, dtype=np.float64)
    with np.errstate(all="ignore"):
        means = np.nanmean(W, axis=0)
        stds = np.nanstd(W, axis=0, ddof=0)
        meds = np.nanmedian(W, axis=0)
    finite_any = np.isfinite(W).any(axis=0)
    means = np.where(finite_any, means, np.nan)
    stds = np.where(finite_any, stds, np.nan)
    meds = np.where(finite_any, meds, np.nan)
    slopes = np.array([slope_1d(W[:, j]) for j in range(n_feat)], dtype=np.float64)
    out[0::4] = means
    out[1::4] = stds
    out[2::4] = meds
    out[3::4] = slopes
    return out


def pool_two_stage(early_264: np.ndarray, late_264: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """用两段聚合近似还原全任务 264。median / slope 是近似，报告里会标明。"""
    n1 = max(int(n1), 1)
    n2 = max(int(n2), 1)
    n = float(n1 + n2)
    out = np.empty_like(early_264, dtype=np.float64)
    for j in range(early_264.size // 4):
        i = 4 * j
        m1, s1, med1, sl1 = early_264[i : i + 4]
        m2, s2, med2, sl2 = late_264[i : i + 4]
        mean = (n1 * m1 + n2 * m2) / n
        e2 = (n1 * (s1**2 + m1**2) + n2 * (s2**2 + m2**2)) / n
        var = e2 - mean**2
        std = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else 0.0
        median = med1 if n1 >= n2 else med2
        # 用两段均值差估计全任务漂移（每窗）
        slope = (m2 - m1) / max(n2, 1)
        if not np.isfinite(mean):
            mean = m1 if np.isfinite(m1) else m2
        if not np.isfinite(std):
            std = s1 if np.isfinite(s1) else s2
        if not np.isfinite(median):
            median = med1 if np.isfinite(med1) else med2
        if not np.isfinite(slope):
            slope = sl1 if np.isfinite(sl1) else sl2
        out[i : i + 4] = (mean, std, median, slope)
    return out


def split_index(n: int, ratio: float) -> int:
    """前 ratio 为已观察段，至少 1 窗，后段至少 1 窗。"""
    if n < 2:
        raise ValueError("窗口数不足 2，无法切段")
    cut = int(np.floor(n * ratio))
    cut = min(max(cut, 1), n - 1)
    return cut


@dataclass
class SampleWindows:
    sample_id: str
    subject: int
    task: str
    difficulty: str
    y_nasa: float
    window_id: np.ndarray
    progress: np.ndarray
    W: np.ndarray  # (n, 66)


def build_window_cache(raw_names: list[str], force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet = CACHE_DIR / "windows_66.parquet"
    pkl = CACHE_DIR / "windows_66.pkl"
    meta_path = CACHE_DIR / "windows_meta.json"
    for candidate in (parquet, pkl):
        if candidate.exists() and meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("raw_names") == raw_names:
                return candidate
    out_path = parquet
    rows = []
    csv_files = sorted(WIN_DIR.glob("subject_*_task_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"没有窗口 CSV：{WIN_DIR}")
    for f in csv_files:
        df = pd.read_csv(f)
        missing = [c for c in raw_names if c not in df.columns]
        if missing:
            raise RuntimeError(f"{f.name} 缺列 {missing[:5]}")
        keep = {
            "sample_id": df["sample_id"].astype(str),
            "subject": df["subject"].astype(int),
            "task": df["task"].astype(str),
            "task_difficulty": df["task_difficulty"].astype(str),
            "y_nasa": df["nasa_tlx_weighted_task_label"].astype(float),
            "window_id": df["window_id"].astype(int),
            "progress_end_ratio": df["progress_end_ratio"].astype(float),
        }
        block = pd.DataFrame(keep)
        for c in raw_names:
            block[c] = df[c].to_numpy(dtype=np.float64)
        rows.append(block)
    all_df = pd.concat(rows, ignore_index=True)
    try:
        all_df.to_parquet(out_path, index=False)
    except Exception:
        out_path = CACHE_DIR / "windows_66.pkl"
        all_df.to_pickle(out_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_rows": int(len(all_df)),
                "n_samples": int(all_df["sample_id"].nunique()),
                "raw_names": raw_names,
                "source": str(WIN_DIR),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def load_samples(raw_names: list[str]) -> list[SampleWindows]:
    path = build_window_cache(raw_names)
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        df = pd.read_parquet(path)
    samples: list[SampleWindows] = []
    for sid, g in df.groupby("sample_id", sort=True):
        g = g.sort_values("window_id")
        samples.append(
            SampleWindows(
                sample_id=str(sid),
                subject=int(g["subject"].iloc[0]),
                task=str(g["task"].iloc[0]),
                difficulty=str(g["task_difficulty"].iloc[0]),
                y_nasa=float(g["y_nasa"].iloc[0]),
                window_id=g["window_id"].to_numpy(dtype=np.int64),
                progress=g["progress_end_ratio"].to_numpy(dtype=np.float64),
                W=g[raw_names].to_numpy(dtype=np.float64),
            )
        )
    return samples


def load_task_arrays() -> dict:
    X = np.load(NASA_DS / "X_task.npy")
    y = np.load(NASA_DS / "y_task.npy")
    groups = np.load(NASA_DS / "groups_task.npy")
    samples = np.load(NASA_DS / "sample_task.npy", allow_pickle=True).astype(str)
    names = json.loads((NASA_DS / "feature_names_task.json").read_text(encoding="utf-8"))
    s_table = pd.read_csv(S_TABLE)
    s_table["sample_id"] = s_table["sample_id"].astype(str)
    s_table = s_table.set_index("sample_id").loc[samples].reset_index()
    if not np.allclose(s_table["y_nasa"].to_numpy(), y, atol=1e-6):
        raise RuntimeError("S 表与 y_task 的 NASA 对不齐")
    step = s_table["weighted_step_score"].to_numpy(dtype=np.float64)
    return {
        "X": X,
        "y": y,
        "groups": groups,
        "samples": samples,
        "names": names,
        "step": step,
        "s_true": mix_s(step, y, STEP_W),
        "s_table": s_table,
    }


def align_samples_to_task_order(
    samples: list[SampleWindows], task_sids: np.ndarray
) -> list[SampleWindows]:
    by_id = {s.sample_id: s for s in samples}
    missing = [sid for sid in task_sids if sid not in by_id]
    if missing:
        raise RuntimeError(f"窗口缓存缺样本：{missing[:8]}")
    return [by_id[sid] for sid in task_sids]


def eligible_mask(samples: list[SampleWindows], ratio: float, min_each: int = 4) -> np.ndarray:
    ok = []
    for s in samples:
        n = len(s.W)
        if n < min_each * 2:
            ok.append(False)
            continue
        cut = split_index(n, ratio)
        ok.append(cut >= min_each and (n - cut) >= min_each)
    return np.array(ok, dtype=bool)


def stage_split(sample: SampleWindows, ratio: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    cut = split_index(len(sample.W), ratio)
    early, late = sample.W[:cut], sample.W[cut:]
    return early, late, int(len(early)), int(len(late))


def build_mod_idx(names: list[str]) -> dict[str, np.ndarray]:
    mods = {"眼动": [], "脑电": [], "心率": [], "行为": []}
    for i, n in enumerate(names):
        mods[modality_of(n)].append(i)
    return {k: np.array(v, dtype=int) for k, v in mods.items()}


def select_quota(X_tr: np.ndarray, y_tr: np.ndarray, mod_idx: dict[str, np.ndarray]) -> np.ndarray:
    picked = []
    for mod, idx in mod_idx.items():
        k = QUOTA[mod]
        mi = mutual_info_regression(X_tr[:, idx], y_tr, random_state=RANDOM_STATE)
        order = np.argsort(-mi)[:k]
        picked.extend(idx[order].tolist())
    return np.array(picked, dtype=int)


def safe_r2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 2:
        return float("nan")
    yt, yh = y_true[mask], y_hat[mask]
    if np.allclose(yt, yt.mean()):
        return float("nan")
    return float(r2_score(yt, yh))


def safe_mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y_true[mask], y_hat[mask]))


def col_r2_table(Y_true: np.ndarray, Y_hat: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "feature": name,
                "modality": modality_of(name),
                "r2": safe_r2(Y_true[:, j], Y_hat[:, j]),
                "mae": safe_mae(Y_true[:, j], Y_hat[:, j]),
            }
        )
    return rows


def modality_mean_r2(col_rows: list[dict]) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in col_rows:
        if np.isfinite(r["r2"]):
            out.setdefault(r["modality"], []).append(r["r2"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def make_forecast_models():
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=10.0),
        "ridge_scaled": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        ),
        "pls6": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=6)),
            ]
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "knn5": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
    }


def oof_multioutput(model_factory, X: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """按被试五折，折内中位数填充后拟合多输出回归。"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = np.full_like(Y, np.nan, dtype=np.float64)
    for tr, te in gkf.split(X, Y[:, 0], groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        Ytr = Y[tr].copy()
        col_imp = SimpleImputer(strategy="median")
        Ytr = col_imp.fit_transform(Ytr)
        m = model_factory()
        m.fit(Xtr, Ytr)
        pred = m.predict(Xte)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        hat[te] = pred
    return hat


def oof_persist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """把输入原样当作输出（维度须一致）。"""
    if X.shape != Y.shape:
        raise ValueError(f"persist 要求同形，got {X.shape} vs {Y.shape}")
    return X.copy()


def oof_xgb_means(
    X: np.ndarray,
    Y_264: np.ndarray,
    groups: np.ndarray,
    persist_rest: np.ndarray,
) -> np.ndarray:
    """只对 66 个 __mean 列训 XGB，其余统计量用 persist_rest。"""
    enable_xgboost()
    from xgboost import XGBRegressor

    gkf = GroupKFold(n_splits=N_SPLITS)
    hat = persist_rest.copy()
    mean_idx = np.arange(0, Y_264.shape[1], 4)
    for tr, te in gkf.split(X, Y_264[:, 0], groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        Ytr = Y_264[tr]
        for j in mean_idx:
            ycol = Ytr[:, j]
            fill = np.nanmedian(ycol)
            ycol = np.where(np.isfinite(ycol), ycol, fill)
            m = XGBRegressor(**XGB_FORECAST_CFG)
            m.fit(Xtr, ycol)
            hat[te, j] = m.predict(Xte)
    return hat


def downstream_quota_xgb(
    X_true: np.ndarray,
    X_hat: np.ndarray,
    y: np.ndarray,
    step: np.ndarray,
    groups: np.ndarray,
    names: list[str],
    eval_mask: np.ndarray | None = None,
    X_early: np.ndarray | None = None,
) -> dict:
    """冻结协议：训练折用真 264 选 27 维并训 XGB；考试折把预测 264 送进同一套列。"""
    enable_xgboost()
    from xgboost import XGBRegressor

    if eval_mask is None:
        eval_mask = np.ones(len(y), dtype=bool)
    mod_idx = build_mod_idx(names)
    gkf = GroupKFold(n_splits=N_SPLITS)
    nasa_hat = np.full(len(y), np.nan)
    nasa_oracle = np.full(len(y), np.nan)
    nasa_early = np.full(len(y), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(gkf.split(X_true, y, groups)):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_true[tr])
        Xte_true = imp.transform(X_true[te])
        Xte_hat = imp.transform(X_hat[te])
        top = select_quota(Xtr, y[tr], mod_idx)
        counts = Counter(modality_of(names[i]) for i in top)
        m = XGBRegressor(**XGB_NASA_CFG)
        m.fit(Xtr[:, top], y[tr])
        nasa_hat[te] = m.predict(Xte_hat[:, top])
        nasa_oracle[te] = m.predict(Xte_true[:, top])
        if X_early is not None:
            nasa_early[te] = m.predict(imp.transform(X_early[te])[:, top])
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(te)),
                "n_test_eval": int(eval_mask[te].sum()),
                "quota": {k: int(counts[k]) for k in QUOTA},
                "nasa_r2_hat": safe_r2(y[te][eval_mask[te]], nasa_hat[te][eval_mask[te]]),
                "nasa_r2_oracle": safe_r2(y[te][eval_mask[te]], nasa_oracle[te][eval_mask[te]]),
            }
        )

    def pack(nasa_pred: np.ndarray, tag: str) -> dict:
        msk = eval_mask & np.isfinite(nasa_pred)
        s_true = mix_s(step, y, STEP_W)
        s_pred = mix_s(step, nasa_pred, STEP_W)
        return {
            f"{tag}_nasa_r2": safe_r2(y[msk], nasa_pred[msk]),
            f"{tag}_nasa_mae": safe_mae(y[msk], nasa_pred[msk]),
            f"{tag}_s_r2": safe_r2(s_true[msk], s_pred[msk]),
            f"{tag}_s_mae": safe_mae(s_true[msk], s_pred[msk]),
            f"{tag}_n": int(msk.sum()),
        }

    out = {
        "n_eval": int(eval_mask.sum()),
        "folds": fold_rows,
        **pack(nasa_hat, "hat"),
        **pack(nasa_oracle, "oracle"),
    }
    if X_early is not None:
        out.update(pack(nasa_early, "early"))
    out["nasa_hat"] = nasa_hat
    out["nasa_oracle"] = nasa_oracle
    out["nasa_early"] = nasa_early
    return out


def json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj
