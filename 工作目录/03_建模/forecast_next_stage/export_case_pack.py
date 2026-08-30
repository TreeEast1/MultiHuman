#!/usr/bin/env python3
"""把示范案例 subject_02_task_5_6 整理成 2:1 高清图 + 数据 + 说明。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common_stage import (  # noqa: E402
    N_SPLITS,
    STEP_W,
    XGB_NASA_CFG,
    aggregate_windows,
    build_mod_idx,
    eligible_mask,
    enable_xgboost,
    json_ready,
    load_feature_names,
    load_samples,
    load_task_arrays,
    mix_s,
    select_quota,
    split_index,
)
from exp_lstm_tf import TF_H  # noqa: E402
from exp_tf_tune import compose, train_one  # noqa: E402
from exp_trend_shape import SERIES, fill_nan, setup_font  # noqa: E402
from plot_ui_mockup import ridge_overall  # noqa: E402

CASE_ID = "subject_02_task_5_6"
OFFICIAL = {2, 7, 12, 16, 23}
RATIO = 0.50
S_THR = 0.51
OUT = HERE / "case_subject_02_task_5_6"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "data"

OBS = "#1A1D23"
TF = "#E86B2A"
RD = "#1F4E79"
NOW = "#8A8F99"

LABELS: dict[str, tuple[str, str]] = {
    "hr_mean": ("心率均值", "bpm"),
    "hr_std": ("心率波动", "bpm"),
    "hr_max": ("最高心率", "bpm"),
    "eye_pupil_filtered_mean": ("瞳孔直径", "mm"),
    "eye_aoi_coverage_ratio": ("AOI覆盖比例", "比例"),
    "eye_aoi_unique_hit_n": ("点到不同AOI数", "个"),
    "eye_aoi_interval_n": ("AOI区间条数", "条"),
    "eye_aoi_max_share": ("最主要AOI占比", "比例"),
    "log_action_density_win": ("操作密度", "密度"),
    "log_action_count_win": ("操作次数", "次"),
    "log_correct_action_count_win": ("正确操作次数", "次"),
    "log_extra_action_count_win": ("多余操作次数", "次"),
    "log_extra_rate_win": ("多余操作比例", "比例"),
    "log_unique_step_count_win": ("步骤种数", "种"),
    "log_unique_device_count_win": ("设备种数", "种"),
    "eeg_frontal_theta_alpha_z_within_subject": ("额区θ/α", "z"),
    "eeg_frontal_alpha_power_z_within_subject": ("额区α功率", "z"),
    "eeg_frontal_gamma_power_z_within_subject": ("额区γ功率", "z"),
    "eeg_central_alpha_power_z_within_subject": ("中央区α功率", "z"),
    "eeg_parietal_theta_alpha_z_within_subject": ("顶区θ/α", "z"),
    "eeg_parietal_theta_power_z_within_subject": ("顶区θ功率", "z"),
    "blink_rate_per_min": ("眨眼频率", "次/分"),
}


def minutes(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * 5.0 / 60.0


def draw_indicator(path: Path, title: str, ylab: str, t, y, cut, y_tf, y_ridge) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    last = float(y[cut - 1])
    t_f = np.concatenate([[t[cut - 1]], t[cut:]])
    ax.plot(t[:cut], y[:cut], color=OBS, lw=2.0, solid_capstyle="round", label="已观察（原始窗级人因）")
    ax.axvspan(t[cut - 1], t[-1], color="#FFF4EC", alpha=0.55, zorder=0)
    ax.axvline(t[cut - 1], color=NOW, lw=1.0, ls="--", zorder=1)
    ax.plot(
        t_f,
        np.concatenate([[last], y_tf]),
        color=TF,
        lw=2.0,
        ls=(0, (5.0, 2.4)),
        label="Transformer 瞬时人因（预测）",
        zorder=2,
    )
    ax.plot(
        t_f,
        np.concatenate([[last], y_ridge]),
        color=RD,
        lw=2.3,
        ls=(0, (1.4, 1.5)),
        label="Ridge 整体走势（预测）",
        zorder=3,
    )
    ymax = max(np.nanmax(y[:cut]), np.nanmax(y_tf), np.nanmax(y_ridge))
    ymin = min(np.nanmin(y[:cut]), np.nanmin(y_tf), np.nanmin(y_ridge))
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.text(t[cut - 1], ymax + pad * 0.15, "  现在", color=NOW, fontsize=11, va="bottom")
    ax.set_xlim(t[0], t[-1])
    ax.set_title(title, loc="left", fontsize=14, color=OBS, pad=10)
    ax.set_xlabel("任务时间（分钟）", fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C5CAD3")
    ax.tick_params(labelsize=10, colors="#4B5563")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_s_card(path: Path, s_true: float, s_hat: float) -> None:
    del s_true
    setup_font()
    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=300)
    fig.patch.set_facecolor("#F2F3F5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.10, 0.16),
        0.80,
        0.68,
        boxstyle="round,pad=0.02,rounding_size=0.024",
        facecolor="#E7F6EC",
        edgecolor="#B7E0C4",
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(0.16, 0.74, "人员状态", fontsize=15, color="#6B7280", transform=ax.transAxes)
    ax.text(0.16, 0.56, "正常", fontsize=40, color="#1F8A4C", fontweight="medium", transform=ax.transAxes)
    ax.text(0.16, 0.40, f"预测绩效 S    {s_hat:.3f}", fontsize=18, color="#1A1D23", transform=ax.transAxes)
    ax.text(0.16, 0.30, f"预警阈值      {S_THR:.2f}", fontsize=14, color="#4B5563", transform=ax.transAxes)
    ax.text(
        0.16,
        0.21,
        f"预测 S ≥ {S_THR:.2f} 为正常",
        fontsize=13,
        color="#6B7280",
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def write_readme(meta: dict, rows: list[dict]) -> None:
    lines = [
        "# 案例：被试 2 · 任务 5_6 的趋势预测图件\n\n",
        "本文件夹是给软件「趋势预测与预警」用的**固定示范案例**。",
        "图的画法已定：左边已观察，右边预测；预测里同时画 **Transformer 瞬时人因** 和 **Ridge 整体走势**。\n\n",
        "## 1. 这个人、这场任务\n\n",
        f"- 样本编号：`{CASE_ID}`\n",
        "- 验证组：被试 2、7、12、16、23 中的一条（被试 2，任务 5_6）\n",
        f"- 已观察：前 50% 窗口（约 {meta['t_now_min']:.1f} 分钟）\n",
        f"- 预测段：后 50% 窗口（约 {meta['t_late_min']:.1f} 分钟）\n",
        f"- 窗口：30 秒窗长、5 秒一步，共 {meta['n_win']} 窗\n\n",
        "## 2. 右侧人员状态怎么写\n\n",
        "界面右侧**只写预测 S**，不要把 Transformer 虚线积成 S。\n\n",
        f"| 项 | 写法 |\n|---|---|\n",
        f"| 人员状态 | **正常** |\n",
        f"| 预测绩效 S | **{meta['S_ridge']:.3f}** |\n",
        f"| 真值 S（报告对照，可不进界面） | {meta['S_true']:.3f} |\n",
        f"| 预警阈值 | {S_THR:.2f}（低分位） |\n",
        f"| 判定 | 预测 S {meta['S_ridge']:.3f} ≥ {S_THR:.2f} → 正常 |\n\n",
        "S 的算法：已观察 27 维 → 标准化 Ridge(α=10) 补成整场 27 维 → 冻结浅树 XGB 得 NASA → ",
        f"S = {STEP_W:.2f} × 真实步骤 + {1 - STEP_W:.2f} × (1 − NASA/10)。",
        "本条真值步骤与 NASA 见 `data/s.json`。\n\n",
        "## 3. 每张图怎么读\n\n",
        "- **实线（黑）**：已观察的窗级人因，即 27 维之前的原料（每 5 秒一个点）。\n",
        "- **橙色虚线**：Transformer 对后半段**瞬时细节**的预报。\n",
        "- **蓝色点线**：Ridge 折出来的**整体走势**（有斜率列就用预报斜率，从「现在」连出去；没有斜率列就画已观察均值的水平线）。这不是逐窗细节。\n",
        "- 竖虚线 = 现在。右侧浅底 = 预测段。\n",
        "- 尺寸：宽∶高 = **2∶1**，PNG，300 dpi。\n\n",
        "这些曲线**不是**表 B1 的 27 个汇总数字。27 维是整段的 mean / std / median / slope，一场各一个数，见 `data/ridge27.csv`。\n\n",
        "## 4. 图清单\n\n",
        "| 文件 | 指标 | 纵轴 | 说明 |\n|---|---|---|---|\n",
        f"| `figures/00_人员状态_S.png` | 绩效 S | — | 右侧卡片：正常，预测 S={meta['S_ridge']:.3f} |\n",
    ]
    for r in rows:
        lines.append(
            f"| `figures/{r['file']}` | {r['label']} | {r['unit'] or '—'} | 窗级人因；橙=Transformer，蓝=Ridge |\n"
        )
    lines += [
        "\n## 5. 数据子文件夹 `data/`\n\n",
        "- `s.json`：真值 / 预测 S、NASA、步骤、阈值、状态。\n",
        "- `windows.csv`：每个窗口、每个指标的已观察、未来真值、Transformer、Ridge 走势。\n",
        "- `ridge27.csv`：本折定额 27 列的已观察、Ridge 预报整场、真值整场。\n",
        "- `meta.json`：样本、切分、指标列表。\n",
        "- `series_long.csv`：与 `windows.csv` 相同，长表便于画图复查。\n\n",
        "## 6. 复现\n\n",
        "```bash\n",
        "cd 工作目录/03_建模/forecast_next_stage\n",
        "uv run --with pandas --with numpy --with scikit-learn --with xgboost --with pyarrow --with matplotlib \\\n",
        "    python export_case_pack.py\n",
        "```\n",
    ]
    (OUT / "README.md").write_text("".join(lines), encoding="utf-8")
    (OUT / "案例说明.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    names_264, raw = load_feature_names()
    col = {n: i for i, n in enumerate(raw)}
    samples = load_samples(raw)
    task = load_task_arrays()
    by_id = {s.sample_id: s for s in samples}
    samples = [by_id[sid] for sid in task["samples"]]
    mask = eligible_mask(samples, RATIO, 4)
    filled = [np.column_stack([fill_nan(s.W[:, j]) for j in range(s.W.shape[1])]) for s in samples]
    y, step, groups = task["y"], task["step"], task["groups"]
    X_true = task["X"]
    X_early = np.zeros_like(X_true)
    for i, W in enumerate(filled):
        X_early[i] = aggregate_windows(W[: split_index(len(W), RATIO)] if mask[i] else W)

    fold_of = np.full(len(samples), -1)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for f, (_, te) in enumerate(gkf.split(X_true, y, groups)):
        fold_of[te] = f
    official_idx = [i for i in range(len(samples)) if int(groups[i]) in OFFICIAL and mask[i]]
    f_off = int(fold_of[official_idx[0]])
    tr = np.array([i for i in range(len(samples)) if fold_of[i] != f_off])
    te = np.array(official_idx)

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_true[tr])
    Xtr_e = imp.fit_transform(X_early[tr])
    Xte_e = imp.transform(X_early[te])
    Xte_true = imp.transform(X_true[te])
    top = select_quota(Xtr, y[tr], build_mod_idx(names_264))
    ridge = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    ridge.fit(Xtr_e[:, top], Xtr[:, top])
    hat27 = ridge.predict(Xte_e[:, top])
    enable_xgboost()
    from xgboost import XGBRegressor

    xgb = XGBRegressor(**XGB_NASA_CFG)
    xgb.fit(Xtr[:, top], y[tr])
    nasa_hat = xgb.predict(hat27)
    s_hat_all = mix_s(step[te], nasa_hat)
    s_true_all = mix_s(step[te], y[te])
    te_ids = [task["samples"][i] for i in te]
    k = te_ids.index(CASE_ID)
    i_case = int(te[k])
    s_true = float(s_true_all[k])
    s_ridge = float(s_hat_all[k])
    nasa_true = float(y[te][k])
    nasa_pred = float(nasa_hat[k])
    step_v = float(step[te][k])

    bases_27 = []
    for j in top:
        b = names_264[int(j)].rsplit("__", 1)[0]
        if b not in bases_27:
            bases_27.append(b)
    feat_names = []
    for n, _ in SERIES:
        if n not in feat_names:
            feat_names.append(n)
    for b in bases_27:
        if b not in feat_names:
            feat_names.append(b)
    idx = [col[n] for n in feat_names]
    slope_of = {}
    for j, col27 in enumerate(top):
        name = names_264[int(col27)]
        if name.endswith("__slope"):
            slope_of[name[: -len("__slope")]] = float(hat27[k, j])

    tr_items = [ii for ii in range(len(samples)) if fold_of[ii] != f_off and mask[ii]]
    subj = sorted({int(groups[ii]) for ii in tr_items})
    rng = np.random.RandomState(0)
    rng.shuffle(subj)
    val_s = set(subj[: max(2, len(subj) // 6)])
    fit_i = [ii for ii in tr_items if int(groups[ii]) not in val_s]
    val_i = [ii for ii in tr_items if int(groups[ii]) in val_s] or fit_i[:2]

    def slc(ii):
        W = filled[ii][:, idx]
        cut = split_index(len(W), RATIO)
        return W[:cut], W[cut:]

    stack = np.vstack([slc(ii)[0] for ii in fit_i])
    mu, sd = stack.mean(0), np.where(stack.std(0) < 1e-6, 1.0, stack.std(0))

    def pack(ids):
        return [((slc(ii)[0] - mu) / sd, (slc(ii)[1] - mu) / sd) for ii in ids]

    print("[case] train Transformer direct on", len(feat_names), "series")
    model = train_one(pack(fit_i), pack(val_i), "direct")
    e, l = slc(i_case)
    ez = (e - mu) / sd
    yhat = compose("direct_anchor", ez, model.forward(ez, TF_H).data, len(l)) * sd + mu
    cut = split_index(len(filled[i_case]), RATIO)
    y_all = np.vstack([e, l])
    t = minutes(len(y_all))

    ridge27_rows = []
    for j, col27 in enumerate(top):
        ridge27_rows.append(
            {
                "col": names_264[int(col27)],
                "early": float(Xte_e[k, col27]),
                "ridge_full": float(hat27[k, j]),
                "true_full": float(Xte_true[k, col27]),
            }
        )
    pd.DataFrame(ridge27_rows).to_csv(DATA_DIR / "ridge27.csv", index=False)

    long_rows = []
    fig_rows = []
    for fi, name in enumerate(feat_names):
        zh, unit = LABELS.get(name, (name, ""))
        y = y_all[:, fi]
        y_tf = yhat[:, fi]
        y_rd = ridge_overall(e[:, fi], len(l), slope_of.get(name))
        stem = f"{fi + 1:02d}_{name}"
        png = f"{stem}.png"
        draw_indicator(
            FIG_DIR / png,
            f"{zh}    {CASE_ID}",
            unit or "取值",
            t,
            y,
            cut,
            y_tf,
            y_rd,
        )
        fig_rows.append({"file": png, "feature": name, "label": zh, "unit": unit})
        print("  wrote", png)
        for wi in range(len(y)):
            long_rows.append(
                {
                    "sample_id": CASE_ID,
                    "feature": name,
                    "label": zh,
                    "window": wi,
                    "minutes": float(t[wi]),
                    "split": "observed" if wi < cut else "future",
                    "y_true": float(y[wi]),
                    "y_tf": float(y[wi] if wi < cut else y_tf[wi - cut]),
                    "y_ridge": float(y[wi] if wi < cut else y_rd[wi - cut]),
                }
            )

    draw_s_card(FIG_DIR / "00_人员状态_S.png", s_true, s_ridge)
    pd.DataFrame(long_rows).to_csv(DATA_DIR / "windows.csv", index=False)
    pd.DataFrame(long_rows).to_csv(DATA_DIR / "series_long.csv", index=False)
    meta = {
        "sample_id": CASE_ID,
        "subject": 2,
        "task": "5_6",
        "n_win": int(len(y_all)),
        "cut": int(cut),
        "t_now_min": float(t[cut - 1]),
        "t_late_min": float(t[-1] - t[cut - 1]),
        "S_true": s_true,
        "S_ridge": s_ridge,
        "S_threshold": S_THR,
        "status": "正常",
        "features": fig_rows,
    }
    (DATA_DIR / "s.json").write_text(
        json.dumps(
            json_ready(
                {
                    "sample_id": CASE_ID,
                    "status": "正常",
                    "S_true": s_true,
                    "S_pred": s_ridge,
                    "S_threshold": S_THR,
                    "step_true": step_v,
                    "NASA_true": nasa_true,
                    "NASA_pred": nasa_pred,
                    "formula": f"S = {STEP_W:.2f}*step + {1 - STEP_W:.2f}*(1-NASA/10)",
                    "note": "界面右侧写预测 S 与正常；真值 S 仅报告对照。",
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (DATA_DIR / "meta.json").write_text(
        json.dumps(json_ready(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(meta, fig_rows)
    print("[case] S_true", s_true, "S_ridge", s_ridge, "->", OUT)


if __name__ == "__main__":
    main()
