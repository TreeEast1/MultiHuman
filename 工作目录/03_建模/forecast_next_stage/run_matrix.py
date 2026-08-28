#!/usr/bin/env python3
"""下一阶段人因预报对照矩阵。

每个版本写到 reports/<version_id>/ ，互不覆盖。
结束时汇总 COMPARISON.md 与 figures/。
"""

import json
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    REPORTS,
    STEP_W,
    TASK_ORDER,
    aggregate_windows,
    align_samples_to_task_order,
    col_r2_table,
    downstream_quota_xgb,
    eligible_mask,
    json_ready,
    load_feature_names,
    load_samples,
    load_task_arrays,
    make_forecast_models,
    mix_s,
    modality_mean_r2,
    oof_multioutput,
    oof_xgb_means,
    pool_two_stage,
    split_index,
    stage_split,
)

FIG_DIR = HERE / "figures"
RATIO_MAIN = 0.50
MIN_EACH = 4


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stack_stage(samples, ratio: float):
    n = len(samples)
    x_early = np.zeros((n, 264), dtype=np.float64)
    y_late = np.zeros((n, 264), dtype=np.float64)
    y_full = np.zeros((n, 264), dtype=np.float64)
    last66 = np.zeros((n, 66), dtype=np.float64)
    late_mean66 = np.zeros((n, 66), dtype=np.float64)
    n1 = np.zeros(n, dtype=int)
    n2 = np.zeros(n, dtype=int)
    for i, s in enumerate(samples):
        y_full[i] = aggregate_windows(s.W)
        if len(s.W) < 2:
            x_early[i] = y_full[i]
            y_late[i] = y_full[i]
            last66[i] = s.W[-1]
            late_mean66[i] = np.nanmean(s.W, axis=0)
            n1[i], n2[i] = len(s.W), 0
            continue
        early, late, a, b = stage_split(s, ratio)
        x_early[i] = aggregate_windows(early)
        y_late[i] = aggregate_windows(late)
        last66[i] = early[-1]
        with np.errstate(all="ignore"):
            late_mean66[i] = np.nanmean(late, axis=0)
        n1[i], n2[i] = a, b
    return {
        "X_early": x_early,
        "Y_late": y_late,
        "Y_full": y_full,
        "last66": last66,
        "late_mean66": late_mean66,
        "n1": n1,
        "n2": n2,
    }


def persist_lastwin_full(samples, ratio, y_full_true, eval_mask):
    hat = y_full_true.copy()
    for i, s in enumerate(samples):
        if not eval_mask[i] or len(s.W) < 2:
            continue
        early, late, _, n_late = stage_split(s, ratio)
        tiled = np.repeat(early[-1][None, :], n_late, axis=0)
        hat[i] = aggregate_windows(np.vstack([early, tiled]))
    return hat


def tile_late_mean_full(samples, ratio, late_mean_hat, y_full_true, eval_mask):
    hat = y_full_true.copy()
    for i, s in enumerate(samples):
        if not eval_mask[i] or len(s.W) < 2:
            continue
        early, _, _, n_late = stage_split(s, ratio)
        tiled = np.repeat(late_mean_hat[i][None, :], max(n_late, 1), axis=0)
        hat[i] = aggregate_windows(np.vstack([early, tiled]))
    return hat


def pool_late_full(x_early, y_late_hat, n1, n2, y_full_true, eval_mask):
    hat = y_full_true.copy()
    for i in range(len(x_early)):
        if not eval_mask[i]:
            continue
        hat[i] = pool_two_stage(x_early[i], y_late_hat[i], int(n1[i]), int(n2[i]))
    return hat


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_pred_csv(path: Path, task: dict, nasa_hat, nasa_oracle, nasa_early, mask):
    s_true = task["s_true"]
    step = task["step"]
    df = pd.DataFrame(
        {
            "sample_id": task["samples"],
            "subject": task["groups"],
            "y_nasa": task["y"],
            "step": step,
            "S_true": s_true,
            "nasa_hat": nasa_hat,
            "nasa_oracle": nasa_oracle,
            "nasa_early": nasa_early,
            "S_hat": mix_s(step, nasa_hat, STEP_W),
            "S_oracle": mix_s(step, nasa_oracle, STEP_W),
            "eligible": mask.astype(int),
        }
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")


def score_forecast(y_true, y_hat, names, mask) -> dict:
    yt, yh = y_true[mask], y_hat[mask]
    cols = col_r2_table(yt, yh, names)
    return {
        "pooled_mae": float(mean_absolute_error(np.nan_to_num(yt), np.nan_to_num(yh))),
        "n": int(mask.sum()),
        "modality_r2": modality_mean_r2(cols),
        "col_r2": cols,
    }


def strip_down(down: dict) -> dict:
    keep = {k: v for k, v in down.items() if k not in ("nasa_hat", "nasa_oracle", "nasa_early", "folds")}
    keep["folds"] = down.get("folds", [])
    return keep


def run_downstream_and_save(
    out_dir: Path,
    model_name: str,
    x_full_hat: np.ndarray,
    feat_names: list[str],
    y_feat_true: np.ndarray,
    task: dict,
    x_early: np.ndarray,
    mask: np.ndarray,
    extra: dict,
):
    feat_stats = score_forecast(y_feat_true, x_full_hat, feat_names, mask)
    down = downstream_quota_xgb(
        task["X"],
        x_full_hat,
        task["y"],
        task["step"],
        task["groups"],
        task["names"],
        eval_mask=mask,
        X_early=x_early,
    )
    row = {
        "model": model_name,
        "forecast": {k: v for k, v in feat_stats.items() if k != "col_r2"},
        "downstream": strip_down(down),
        **extra,
    }
    sub = out_dir / "models" / model_name
    sub.mkdir(parents=True, exist_ok=True)
    save_json(sub / "metrics.json", row)
    pd.DataFrame(feat_stats["col_r2"]).to_csv(sub / "feature_r2.csv", index=False, encoding="utf-8-sig")
    write_pred_csv(
        sub / "predictions.csv",
        task,
        down["nasa_hat"],
        down["nasa_oracle"],
        down["nasa_early"],
        mask,
    )
    print(
        f"    {model_name:22s}  NASA R²={row['downstream'].get('hat_nasa_r2', np.nan):+.3f}  "
        f"S R²={row['downstream'].get('hat_s_r2', np.nan):+.3f}  "
        f"oracle NASA={row['downstream'].get('oracle_nasa_r2', np.nan):+.3f}  "
        f"early NASA={row['downstream'].get('early_nasa_r2', np.nan):+.3f}",
        flush=True,
    )
    return row


def models_for_264():
    md = make_forecast_models()
    return md


def run_v1_v2_v3(samples, task, names_264, ratio=RATIO_MAIN) -> list[dict]:
    packed = stack_stage(samples, ratio)
    mask = eligible_mask(samples, ratio, MIN_EACH)
    x_early, y_late, y_full = packed["X_early"], packed["Y_late"], packed["Y_full"]
    groups = task["groups"]
    results = []

    specs = [
        (
            "v1_stage_late_pool",
            "前段264→后段264，矩合并还原全任务264",
            y_late,
            lambda yhat: pool_late_full(x_early, yhat, packed["n1"], packed["n2"], y_full, mask),
            y_late,
        ),
        (
            "v2_direct_full",
            "前段264→直接预报全任务264",
            y_full,
            lambda yhat: yhat,
            y_full,
        ),
        (
            "v3_tile_late_mean",
            "前段264→后段66维均值，铺窗后再聚合",
            packed["late_mean66"],
            lambda yhat: tile_late_mean_full(samples, ratio, yhat, y_full, mask),
            packed["late_mean66"],
        ),
    ]

    for ver_id, title, y_target, recon, y_for_score in specs:
        print(f"\n== {ver_id}  n_eval={int(mask.sum())}  {title}", flush=True)
        out_dir = REPORTS / ver_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            out_dir / "meta.json",
            {
                "version": ver_id,
                "title": title,
                "ratio": ratio,
                "n_eval": int(mask.sum()),
                "dropped": [s.sample_id for s, ok in zip(samples, mask) if not ok],
            },
        )
        rows = []

        # persist / last-window tile
        if ver_id != "v3_tile_late_mean":
            y_persist = x_early if y_target.shape[1] == 264 else packed["late_mean66"]
            if y_target.shape[1] == 264:
                rows.append(
                    run_downstream_and_save(
                        out_dir, "persist_early", recon(y_persist), names_264, y_full,
                        task, x_early, mask, {"note": "前段264原样当作预报目标"},
                    )
                )
        rows.append(
            run_downstream_and_save(
                out_dir,
                "persist_lastwin_tile",
                persist_lastwin_full(samples, ratio, y_full, mask),
                names_264,
                y_full,
                task,
                x_early,
                mask,
                {"note": "最后一窗铺满后段再聚合"},
            )
        )

        for name, factory in models_for_264().items():
            try:
                yhat = oof_multioutput(factory, x_early, y_target, groups)
                xhat = recon(yhat)
                rows.append(
                    run_downstream_and_save(
                        out_dir, name, xhat, names_264, y_full, task, x_early, mask, {}
                    )
                )
            except Exception as exc:
                print(f"    {name} FAILED: {exc}")
                save_json(out_dir / "models" / name / "error.json", {"error": str(exc), "trace": traceback.format_exc()})

        if ver_id == "v2_direct_full":
            resid = y_full - x_early
            try:
                rhat = oof_multioutput(models_for_264()["ridge_scaled"], x_early, resid, groups)
                rows.append(
                    run_downstream_and_save(
                        out_dir, "ridge_residual", x_early + rhat, names_264, y_full,
                        task, x_early, mask, {"note": "预报(全任务-前段)再加回"},
                    )
                )
            except Exception as exc:
                print("    ridge_residual FAILED", exc)
            try:
                xhat = oof_xgb_means(x_early, y_full, groups, persist_rest=x_early)
                rows.append(
                    run_downstream_and_save(
                        out_dir, "xgb_means", xhat, names_264, y_full, task, x_early, mask,
                        {"note": "只预报66个mean，其余沿用前段"},
                    )
                )
            except Exception as exc:
                print("    xgb_means FAILED", exc)

        if ver_id == "v3_tile_late_mean":
            try:
                dummy_mean = oof_multioutput(models_for_264()["dummy_mean"], x_early, packed["late_mean66"], groups)
                rows.append(
                    run_downstream_and_save(
                        out_dir, "dummy_mean_tile", recon(dummy_mean), names_264, y_full,
                        task, x_early, mask, {},
                    )
                )
            except Exception as exc:
                print("    dummy_mean_tile extra skip", exc)

        save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
        results.extend([{"version": ver_id, **r} for r in rows])
    return results


def collect_horizon_rows(samples, ratio, mask):
    xs, ys, grp, sidx = [], [], [], []
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        early, late, n1, n2 = stage_split(s, ratio)
        e264 = aggregate_windows(early)
        last = early[-1]
        n = len(s.W)
        for k, row in enumerate(late):
            t_frac = (n1 + k) / max(n - 1, 1)
            remain = (n2 - k) / max(n, 1)
            xs.append(np.concatenate([e264, last, [t_frac, remain, n1 / 100.0, n2 / 100.0]]))
            ys.append(row)
            grp.append(s.subject)
            sidx.append(i)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(grp), np.asarray(sidx, dtype=int)


def stitch_horizon(samples, ratio, sidx, win_hat, y_full, mask):
    buckets: dict[int, list] = defaultdict(list)
    for row, si in zip(win_hat, sidx):
        buckets[int(si)].append(row)
    hat = y_full.copy()
    for i, s in enumerate(samples):
        if not mask[i]:
            continue
        early, late, _, _ = stage_split(s, ratio)
        pred = np.vstack(buckets[i]) if buckets[i] else late
        if pred.shape[0] != late.shape[0]:
            pred = pred[: late.shape[0]]
            if pred.shape[0] < late.shape[0]:
                pad = np.repeat(pred[-1][None, :], late.shape[0] - pred.shape[0], axis=0)
                pred = np.vstack([pred, pad])
        hat[i] = aggregate_windows(np.vstack([early, pred]))
    return hat


def run_v4(samples, task, names_264) -> list[dict]:
    ver_id = "v4_horizon_windows"
    title = "前段264+末窗66+时间→后段每个窗66维，再聚合"
    print(f"\n== {ver_id}")
    mask = eligible_mask(samples, RATIO_MAIN, MIN_EACH)
    packed = stack_stage(samples, RATIO_MAIN)
    xs, ys, grp, sidx = collect_horizon_rows(samples, RATIO_MAIN, mask)
    out_dir = REPORTS / ver_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "meta.json", {"version": ver_id, "title": title, "n_window_rows": int(len(xs)), "n_eval": int(mask.sum())})
    rows = []
    md = make_forecast_models()
    use = ["dummy_mean", "ridge_scaled", "pls6", "extra_trees", "knn5"]
    for name in use:
        try:
            yhat = oof_multioutput(md[name], xs, ys, grp)
            xhat = stitch_horizon(samples, RATIO_MAIN, sidx, yhat, packed["Y_full"], mask)
            rows.append(
                run_downstream_and_save(
                    out_dir, name, xhat, names_264, packed["Y_full"], task, packed["X_early"], mask,
                    {"n_window_rows": int(len(xs))},
                )
            )
        except Exception as exc:
            print(f"    {name} FAILED: {exc}")
            save_json(out_dir / "models" / name / "error.json", {"error": str(exc), "trace": traceback.format_exc()})
    save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
    return [{"version": ver_id, **r} for r in rows]


def collect_ar_pairs(samples, hop: int):
    xs, ys, grp = [], [], []
    for s in samples:
        w = s.W
        if len(w) <= hop:
            continue
        for t in range(0, len(w) - hop):
            xs.append(w[t])
            ys.append(w[t + hop])
            grp.append(s.subject)
    return np.asarray(xs, np.float64), np.asarray(ys, np.float64), np.asarray(grp)


def rollout_late(model, imputer, start, n_late, hop):
    cur = np.asarray(start, dtype=np.float64)
    chunks = []
    n_steps = int(np.ceil(n_late / hop))
    for _ in range(max(n_steps, 1)):
        nxt = model.predict(imputer.transform(cur.reshape(1, -1)))[0]
        chunks.append(nxt)
        cur = nxt
    seq = np.repeat(np.vstack(chunks), hop, axis=0)[:n_late]
    return seq


def run_v5(samples, task, names_264) -> list[dict]:
    results = []
    packed = stack_stage(samples, RATIO_MAIN)
    mask = eligible_mask(samples, RATIO_MAIN, MIN_EACH)
    md = make_forecast_models()
    for hop, tag in ((1, "hop1_overlap"), (6, "hop6_30s")):
        ver_id = f"v5_ar_rollout_{tag}"
        title = f"窗级自回归滚动，hop={hop}（{'重叠5s' if hop == 1 else '约30s不重叠'}）"
        print(f"\n== {ver_id}")
        out_dir = REPORTS / ver_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "meta.json", {"version": ver_id, "title": title, "hop": hop, "n_eval": int(mask.sum())})
        rows = []
        for name in ("ridge_scaled", "extra_trees"):
            try:
                x_full_hat = packed["Y_full"].copy()
                gkf = GroupKFold(n_splits=N_SPLITS)
                for tr, te in gkf.split(task["X"], task["y"], task["groups"]):
                    tr_samples = [samples[i] for i in tr]
                    xar, yar, _ = collect_ar_pairs(tr_samples, hop)
                    imp = SimpleImputer(strategy="median")
                    xar_i = imp.fit_transform(xar)
                    yimp = SimpleImputer(strategy="median")
                    yar_i = yimp.fit_transform(yar)
                    model = md[name]()
                    model.fit(xar_i, yar_i)
                    for i in te:
                        if not mask[i]:
                            continue
                        early, late, _, n_late = stage_split(samples[i], RATIO_MAIN)
                        late_hat = rollout_late(model, imp, early[-1], n_late, hop)
                        x_full_hat[i] = aggregate_windows(np.vstack([early, late_hat]))
                rows.append(
                    run_downstream_and_save(
                        out_dir, name, x_full_hat, names_264, packed["Y_full"], task,
                        packed["X_early"], mask, {"hop": hop},
                    )
                )
            except Exception as exc:
                print(f"    {name} FAILED: {exc}")
                save_json(out_dir / "models" / name / "error.json", {"error": str(exc), "trace": traceback.format_exc()})
        save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
        results.extend([{"version": ver_id, **r} for r in rows])
    return results


def run_v6(samples, task, names_264) -> list[dict]:
    ver_id = "v6_observe_ratio"
    title = "观察比例扫描（数据处理：直接预报全任务264）"
    print(f"\n== {ver_id}")
    out_dir = REPORTS / ver_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    use_models = ["persist_early", "ridge_scaled", "extra_trees"]
    md = make_forecast_models()
    for ratio in (0.25, 0.33, 0.50, 0.67, 0.75):
        packed = stack_stage(samples, ratio)
        mask = eligible_mask(samples, ratio, MIN_EACH)
        print(f"  ratio={ratio:.2f}  n_eval={int(mask.sum())}")
        for name in use_models:
            try:
                if name == "persist_early":
                    xhat = packed["X_early"]
                else:
                    xhat = oof_multioutput(md[name], packed["X_early"], packed["Y_full"], task["groups"])
                tag = f"r{int(ratio * 100):02d}_{name}"
                row = run_downstream_and_save(
                    out_dir, tag, xhat, names_264, packed["Y_full"], task,
                    packed["X_early"], mask, {"ratio": ratio, "base_model": name},
                )
                row["ratio"] = ratio
                row["base_model"] = name
                rows.append(row)
            except Exception as exc:
                print(f"    {name} @ {ratio} FAILED: {exc}")
    save_json(out_dir / "meta.json", {"version": ver_id, "title": title})
    save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
    return [{"version": ver_id, **r} for r in rows]


def next_task_pairs(samples) -> list[tuple[int, int, int]]:
    by = defaultdict(list)
    for i, s in enumerate(samples):
        by[s.subject].append(i)
    pairs = []
    for subj, idxs in by.items():
        idxs = sorted(idxs, key=lambda i: (TASK_ORDER.get(samples[i].task, 99), samples[i].sample_id))
        for a, b in zip(idxs, idxs[1:]):
            pairs.append((a, b, int(subj)))
    return pairs


def run_v7(samples, task, names_264) -> list[dict]:
    ver_id = "v7_next_task"
    title = "跨任务：本任务264 → 同一被试下一任务264"
    print(f"\n== {ver_id}")
    pairs = next_task_pairs(samples)
    out_dir = REPORTS / ver_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src = np.array([a for a, b, g in pairs], dtype=int)
    dst = np.array([b for a, b, g in pairs], dtype=int)
    grp = np.array([g for a, b, g in pairs], dtype=int)
    x_in = task["X"][src]
    y_out = task["X"][dst]
    mask = np.zeros(len(samples), dtype=bool)
    mask[dst] = True
    save_json(
        out_dir / "meta.json",
        {
            "version": ver_id,
            "title": title,
            "n_pairs": len(pairs),
            "pairs": [
                {"from": samples[a].sample_id, "to": samples[b].sample_id, "subject": g}
                for a, b, g in pairs
            ],
        },
    )
    rows = []
    dummy_xhat = task["X"].copy()
    dummy_xhat[dst] = x_in  # persist: 用上一任务当下一任务
    rows.append(
        run_downstream_and_save(
            out_dir, "persist_prev_task", dummy_xhat, names_264, task["X"], task,
            dummy_xhat, mask, {"n_pairs": len(pairs)},
        )
    )
    md = make_forecast_models()
    for name in ("dummy_mean", "ridge_scaled", "pls6", "extra_trees", "knn5"):
        try:
            yhat_pairs = oof_multioutput(md[name], x_in, y_out, grp)
            xhat = task["X"].copy()
            xhat[dst] = yhat_pairs
            rows.append(
                run_downstream_and_save(
                    out_dir, name, xhat, names_264, task["X"], task, dummy_xhat, mask,
                    {"n_pairs": len(pairs)},
                )
            )
        except Exception as exc:
            print(f"    {name} FAILED: {exc}")
            save_json(out_dir / "models" / name / "error.json", {"error": str(exc)})
    save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
    return [{"version": ver_id, **r} for r in rows]


def run_v8(samples, task, names_264) -> list[dict]:
    ver_id = "v8_quota27_space"
    title = "只预报下游 XGB 实际使用的 27 列（折内定额）"
    print(f"\n== {ver_id}")
    packed = stack_stage(samples, RATIO_MAIN)
    mask = eligible_mask(samples, RATIO_MAIN, MIN_EACH)
    out_dir = REPORTS / ver_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "meta.json", {"version": ver_id, "title": title, "n_eval": int(mask.sum())})
    from common_stage import build_mod_idx, select_quota, enable_xgboost, XGB_NASA_CFG

    enable_xgboost()
    from xgboost import XGBRegressor

    md = make_forecast_models()
    rows = []
    for name in ("persist_early", "ridge_scaled", "pls6", "extra_trees"):
        nasa_hat = np.full(len(task["y"]), np.nan)
        nasa_oracle = np.full(len(task["y"]), np.nan)
        nasa_early = np.full(len(task["y"]), np.nan)
        gkf = GroupKFold(n_splits=N_SPLITS)
        mod_idx = build_mod_idx(names_264)
        for tr, te in gkf.split(task["X"], task["y"], task["groups"]):
            imp_full = SimpleImputer(strategy="median")
            xtr_true = imp_full.fit_transform(task["X"][tr])
            xte_true = imp_full.transform(task["X"][te])
            xte_early = imp_full.transform(packed["X_early"][te])
            top = select_quota(xtr_true, task["y"][tr], mod_idx)
            # 预报这 27 列
            if name == "persist_early":
                xte_hat = xte_early.copy()
            else:
                imp_e = SimpleImputer(strategy="median")
                e_tr = imp_e.fit_transform(packed["X_early"][tr][:, top])
                e_te = imp_e.transform(packed["X_early"][te][:, top])
                y_tr = xtr_true[:, top]
                model = md[name]()
                model.fit(e_tr, y_tr)
                pred27 = model.predict(e_te)
                xte_hat = xte_early.copy()
                xte_hat[:, top] = pred27
            m = XGBRegressor(**XGB_NASA_CFG)
            m.fit(xtr_true[:, top], task["y"][tr])
            nasa_hat[te] = m.predict(xte_hat[:, top])
            nasa_oracle[te] = m.predict(xte_true[:, top])
            nasa_early[te] = m.predict(xte_early[:, top])
        # wrap as downstream-like dict
        from common_stage import safe_r2, safe_mae

        y, step = task["y"], task["step"]
        s_true = mix_s(step, y)
        msk = mask & np.isfinite(nasa_hat)

        def pack(pred, tag):
            return {
                f"{tag}_nasa_r2": safe_r2(y[msk], pred[msk]),
                f"{tag}_nasa_mae": safe_mae(y[msk], pred[msk]),
                f"{tag}_s_r2": safe_r2(s_true[msk], mix_s(step, pred)[msk]),
                f"{tag}_s_mae": safe_mae(s_true[msk], mix_s(step, pred)[msk]),
                f"{tag}_n": int(msk.sum()),
            }

        down = {**pack(nasa_hat, "hat"), **pack(nasa_oracle, "oracle"), **pack(nasa_early, "early")}
        row = {"model": name, "forecast": {}, "downstream": down}
        sub = out_dir / "models" / name
        sub.mkdir(parents=True, exist_ok=True)
        save_json(sub / "metrics.json", row)
        write_pred_csv(sub / "predictions.csv", task, nasa_hat, nasa_oracle, nasa_early, mask)
        print(f"    {name:22s}  NASA R²={down['hat_nasa_r2']:+.3f}  S R²={down['hat_s_r2']:+.3f}")
        rows.append(row)
    save_json(out_dir / "summary.json", {"version": ver_id, "title": title, "models": rows})
    return [{"version": ver_id, **r} for r in rows]


def flatten_leaderboard(all_rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in all_rows:
        d = r.get("downstream") or {}
        recs.append(
            {
                "version": r.get("version"),
                "model": r.get("model"),
                "ratio": r.get("ratio", r.get("extra", {}).get("ratio") if isinstance(r.get("extra"), dict) else None),
                "nasa_r2": d.get("hat_nasa_r2"),
                "nasa_mae": d.get("hat_nasa_mae"),
                "s_r2": d.get("hat_s_r2"),
                "s_mae": d.get("hat_s_mae"),
                "oracle_nasa_r2": d.get("oracle_nasa_r2"),
                "early_nasa_r2": d.get("early_nasa_r2"),
                "n": d.get("hat_n"),
                "mod_r2_眼动": (r.get("forecast") or {}).get("modality_r2", {}).get("眼动"),
                "mod_r2_脑电": (r.get("forecast") or {}).get("modality_r2", {}).get("脑电"),
                "mod_r2_心率": (r.get("forecast") or {}).get("modality_r2", {}).get("心率"),
                "mod_r2_行为": (r.get("forecast") or {}).get("modality_r2", {}).get("行为"),
            }
        )
    return pd.DataFrame(recs)


def write_comparison(df: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.to_csv(REPORTS / "leaderboard.csv", index=False, encoding="utf-8-sig")
    # 主方案：任务内、观察 50%、不含 next_task
    main = df[df["version"].isin(["v1_stage_late_pool", "v2_direct_full", "v3_tile_late_mean", "v4_horizon_windows", "v8_quota27_space"])]
    main = main.dropna(subset=["nasa_r2"])
    best = None
    if len(main):
        best = main.sort_values("nasa_r2", ascending=False).iloc[0]

    lines = []
    lines.append("# 下一阶段人因预报对照总表\n\n")
    lines.append(f"生成时间：{_now()}\n\n")
    lines.append("主指标是 **NASA pooled R²**（人因预报是否有用）。S 的 R² 含 70% 真实步骤，只作并列。\n\n")
    if best is not None:
        lines.append("## 当前有效路径（按 NASA R²）\n\n")
        lines.append(
            f"**{best['version']} / {best['model']}**：NASA R² = {best['nasa_r2']:+.3f}，"
            f"MAE = {best['nasa_mae']:.3f}；S R² = {best['s_r2']:+.3f}。"
            f"同折 Oracle NASA R² = {best['oracle_nasa_r2']:+.3f}，"
            f"Early-only = {best['early_nasa_r2']:+.3f}。\n\n"
        )
        beat_early = best["nasa_r2"] > (best["early_nasa_r2"] if pd.notna(best["early_nasa_r2"]) else -9)
        beat_zero = best["nasa_r2"] > 0
        if beat_early and beat_zero:
            lines.append("该路径超过 Early-only 且 NASA R²>0，可作为验收主结果。\n\n")
        elif beat_zero:
            lines.append("NASA R²>0，但未稳定超过 Early-only：预报有信号，拼段方式仍需看表。\n\n")
        else:
            lines.append("尚未超过均值基线。表中仍保留全部对照，便于改切段或特征后再训。\n\n")

    lines.append("## 全量对照\n\n")
    lines.append("| 版本 | 模型 | n | NASA R² | NASA MAE | S R² | Oracle NASA | Early NASA |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    show = df.sort_values(["version", "nasa_r2"], ascending=[True, False])
    for _, r in show.iterrows():
        def fmt(x):
            return "" if pd.isna(x) else f"{x:+.3f}" if isinstance(x, float) else str(x)

        def fmt_mae(x):
            return "" if pd.isna(x) else f"{x:.3f}"

        lines.append(
            f"| {r['version']} | {r['model']} | {'' if pd.isna(r['n']) else int(r['n'])} | "
            f"{fmt(r['nasa_r2'])} | {fmt_mae(r['nasa_mae'])} | {fmt(r['s_r2'])} | "
            f"{fmt(r['oracle_nasa_r2'])} | {fmt(r['early_nasa_r2'])} |\n"
        )
    (REPORTS / "COMPARISON.md").write_text("".join(lines), encoding="utf-8")

    selected = []
    selected.append("# 验收选用\n\n")
    selected.append("选择规则：主方案（任务内切段）里 NASA R² 最高，且须看是否超过 Early-only 与 Oracle 缺口。\n\n")
    if best is not None:
        selected.append(f"- **主报**：`{best['version']}` + `{best['model']}`\n")
        selected.append(f"- NASA R² = {best['nasa_r2']:+.3f}（Oracle {best['oracle_nasa_r2']:+.3f}，Early {best['early_nasa_r2']:+.3f}）\n")
        selected.append(f"- S R² = {best['s_r2']:+.3f}（步骤 0.70 为真值，不解释成人因预报有这么准）\n")
        selected.append(f"- 明细：`reports/{best['version']}/models/{best['model']}/`\n")
    selected.append("\n其余路径全部保留在 `reports/`，不删除。\n")
    (REPORTS / "SELECTED.md").write_text("".join(selected), encoding="utf-8")


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    main_vers = [
        "v1_stage_late_pool",
        "v2_direct_full",
        "v3_tile_late_mean",
        "v4_horizon_windows",
        "v8_quota27_space",
    ]
    sub = df[df["version"].isin(main_vers)].dropna(subset=["nasa_r2"])
    if len(sub):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels, vals, colors = [], [], []
        for _, r in sub.sort_values("nasa_r2", ascending=False).head(18).iterrows():
            labels.append(f"{r['version'].replace('v1_stage_late_pool','V1').replace('v2_direct_full','V2').replace('v3_tile_late_mean','V3').replace('v4_horizon_windows','V4').replace('v8_quota27_space','V8')}/{r['model']}")
            vals.append(r["nasa_r2"])
            colors.append("#3A7CA5")
        ax.barh(range(len(vals))[::-1], vals[::-1], color="#3A7CA5")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("NASA pooled R²")
        ax.set_title("任务内下一阶段：各路径 NASA R²（越高越好）")
        ax.axvline(0.0, color="#666", lw=0.8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_nasa_r2_main.png", dpi=160)
        plt.close(fig)

    v6 = df[df["version"] == "v6_observe_ratio"].copy()
    if len(v6) and "ratio" in v6.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for model, g in v6.groupby("model"):
            base = str(model)
            if "_" in base and base[0] == "r":
                # r25_ridge_scaled
                parts = base.split("_", 1)
                ratio = g["ratio"]
                ax.plot(g["ratio"], g["nasa_r2"], marker="o", label=parts[1] if len(parts) > 1 else base)
        # regroup properly
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        tmp = []
        for _, r in v6.iterrows():
            m = str(r["model"])
            if m.startswith("r") and "_" in m:
                base = m.split("_", 1)[1]
            else:
                base = m
            tmp.append((r.get("ratio"), base, r["nasa_r2"]))
        tdf = pd.DataFrame(tmp, columns=["ratio", "base", "nasa_r2"]).dropna()
        for base, g in tdf.groupby("base"):
            g = g.sort_values("ratio")
            ax.plot(g["ratio"], g["nasa_r2"], marker="o", label=base)
        ax.set_xlabel("已观察比例")
        ax.set_ylabel("NASA pooled R²")
        ax.set_title("看到任务的多少之后再预报剩余（V2 协议）")
        ax.axhline(0.0, color="#666", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_observe_ratio.png", dpi=160)
        plt.close(fig)


def write_readme(df: pd.DataFrame) -> None:
    best_line = "训练完成后见 `reports/SELECTED.md`。"
    if df is not None and len(df.dropna(subset=["nasa_r2"])):
        main = df[df["version"].isin(["v1_stage_late_pool", "v2_direct_full", "v3_tile_late_mean", "v4_horizon_windows", "v8_quota27_space"])]
        if len(main.dropna(subset=["nasa_r2"])):
            b = main.sort_values("nasa_r2", ascending=False).iloc[0]
            best_line = (
                f"当前主报 **{b['version']} / {b['model']}**："
                f"NASA R² = {b['nasa_r2']:+.3f}，S R² = {b['s_r2']:+.3f}。"
            )
    text = f"""# 下一阶段人因原始指标预报，再走定额 XGB 合成 S

{best_line}

老师需求对应的实现：**先预报下一阶段的 66/264 维人因指标，再送进与正式口径相同的 27 维定额 XGB 算 NASA，最后按 0.70/0.30 合成 S**。不直接把 S 当回归目标。

- 方法与泄漏约定：[PROTOCOL.md](PROTOCOL.md)
- 数据诊断：`reports/00_diagnose/report.md`
- 全量对照：`reports/COMPARISON.md`
- 验收选用：`reports/SELECTED.md`
- 每个版本的预测明细：`reports/<version>/models/<model>/predictions.csv`

## 版本一览

| 版本 | 数据处理 | 目录 |
|---|---|---|
| V1 | 前段 264 → 后段 264，矩合并 | `reports/v1_stage_late_pool/` |
| V2 | 前段 264 → 全任务 264 | `reports/v2_direct_full/` |
| V3 | 预报后段 66 均值，铺窗再聚合 | `reports/v3_tile_late_mean/` |
| V4 | 逐窗条件预报后段 | `reports/v4_horizon_windows/` |
| V5 | 窗级自回归滚动（hop=1 / hop=6） | `reports/v5_ar_rollout_*/` |
| V6 | 观察比例 25–75% | `reports/v6_observe_ratio/` |
| V7 | 下一任务（对照，非主方案） | `reports/v7_next_task/` |
| V8 | 只预报定额 27 列 | `reports/v8_quota27_space/` |

## 复现

```bash
cd 工作目录/03_建模/forecast_next_stage
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \\
    python diagnose_data.py
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \\
    python run_matrix.py
```

窗口缓存写在 `cache/`（可删，脚本会重建）。对照结果全部留在 `reports/`，不互相覆盖。
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names_264, raw = load_feature_names()
    task = load_task_arrays()
    samples = align_samples_to_task_order(load_samples(raw), task["samples"])
    print(f"[{_now()}] samples={len(samples)} windows={sum(len(s.W) for s in samples)}")

    all_rows: list[dict] = []
    runners = [
        ("v1-v3", lambda: run_v1_v2_v3(samples, task, names_264)),
        ("v4", lambda: run_v4(samples, task, names_264)),
        ("v5", lambda: run_v5(samples, task, names_264)),
        ("v6", lambda: run_v6(samples, task, names_264)),
        ("v7", lambda: run_v7(samples, task, names_264)),
        ("v8", lambda: run_v8(samples, task, names_264)),
    ]
    errors = []
    for tag, fn in runners:
        try:
            all_rows.extend(fn())
        except Exception as exc:
            errors.append({"block": tag, "error": str(exc), "trace": traceback.format_exc()})
            print(f"[FAIL] {tag}: {exc}")
            traceback.print_exc()

    df = flatten_leaderboard(all_rows)
    write_comparison(df)
    try:
        make_figures(df)
    except Exception as exc:
        print("[figures] skip", exc)
    write_readme(df)
    save_json(
        REPORTS / "run_log.json",
        {"finished": _now(), "seconds": time.time() - t0, "n_rows": len(all_rows), "errors": errors},
    )
    print(f"\n[{_now()}] done in {time.time() - t0:.0f}s  rows={len(all_rows)}  errors={len(errors)}")
    print("leaderboard:", REPORTS / "leaderboard.csv")


if __name__ == "__main__":
    main()
