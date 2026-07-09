#!/usr/bin/env python3
"""P6 MI 实验结果可视化与详细整理。

生成内容：
  figures/  — 8 张可视化图表
  data/     — 4 个结构化 CSV
  MI_DETAILED_REPORT.md — 详细整理报告
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 120,
    "savefig.bbox": None,  # 关闭 tight bbox, 避免中文标签导致的尺寸爆炸
    "savefig.pad_inches": 0.1,
})

# 中文字体
for font in ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "SimHei"]:
    try:
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(family=font)
        if fp.get_name() != font:
            continue
        plt.rcParams["font.family"] = font
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import RANDOM_STATE  # noqa: E402

DATA_DIR = HERE / "dataset"
REPORT_DIR = HERE / "reports_exp6"
FIG_DIR = REPORT_DIR / "figures"
CSV_DIR = REPORT_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

MOD_COLORS = {"眼动": "#2196F3", "脑电": "#FF9800", "心率": "#4CAF50", "行为": "#9C27B0"}


def build_modalities(fnames):
    mods = {"眼动": [], "脑电": [], "心率": [], "行为": []}
    for i, n in enumerate(fnames):
        if n.startswith(("eye_aoi", "eye_", "blink_")):
            mods["眼动"].append(i)
        elif n.startswith("eeg_"):
            mods["脑电"].append(i)
        elif n.startswith("hr_"):
            mods["心率"].append(i)
        elif n.startswith("log_"):
            mods["行为"].append(i)
    return mods


def load_data():
    X = np.load(DATA_DIR / "X_cls.npy")
    y_int = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    with open(DATA_DIR / "feature_names_cls.json", encoding="utf-8") as f:
        fnames = json.load(f)
    return X, y_int, groups, fnames


def compute_mi_spectrum(X, y_int, groups, fnames, mods):
    """折内 MI 全谱 (5 折 × 5 种子)."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mi_folds = []
    for tr, te in sgkf.split(X, y_int, groups):
        imp = SimpleImputer(strategy="median")
        Xi = imp.fit_transform(X[tr])
        mi_avg = np.zeros(X.shape[1])
        for s in range(5):
            mi_avg += mutual_info_classif(Xi, y_int[tr], random_state=RANDOM_STATE + s, n_neighbors=3)
        mi_avg /= 5
        mi_folds.append(mi_avg)

    mi_mean = np.mean(mi_folds, axis=0)
    mi_std = np.std(mi_folds, axis=0)
    return mi_mean, mi_std


# ============================================================ #
#  Figure 1: MI 全谱 — 各模态 MI 降序条形图
# ============================================================ #
def fig_mi_spectrum(mi_mean, mi_std, fnames, mods):
    fig = plt.figure(figsize=(11, 13))
    from exp6_mi_multimodal_cls import find_knee_point
    for i, (mod, idx_list) in enumerate(mods.items()):
        ax = fig.add_subplot(4, 1, i + 1)
        mi_vals = mi_mean[idx_list]
        mi_errs = mi_std[idx_list]
        order = np.argsort(-mi_vals)
        ranked_idx = [idx_list[i_] for i_ in order]
        ranked_mi = mi_vals[order]
        ranked_err = mi_errs[order]

        n_show = min(12, len(ranked_idx))
        colors = [MOD_COLORS[mod]] * n_show
        ax.barh(range(n_show - 1, -1, -1), ranked_mi[:n_show],
                xerr=ranked_err[:n_show], color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.5, capsize=2)

        knee = find_knee_point(ranked_mi)
        if knee < n_show:
            ax.axhline(y=n_show - knee, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.text(n_show - knee + 0.3, ranked_mi[knee - 1] + 0.005,
                    f"MI knee K={knee}", color="red", fontsize=8, va="bottom")

        short_names = [fnames[ii].replace("_within_subject", "").replace("_z", "")
                        .replace("eye_aoi_", "").replace("eye_pupil_", "pup_")
                        .replace("log_", "").replace("eeg_", "").replace("hr_", "")
                        .replace("_power", "").replace("_win", "")
                        .replace("_filtered", "").replace("_action", "act").replace("_count", "n")
                        .replace("unique_step", "ustep").replace("action_density", "actden")
                        .replace("error_rate", "err").replace("correct_action", "ok")
                        .replace("unique_device", "udev")
                       [:18] for ii in ranked_idx[:n_show]]
        ax.set_yticks(range(n_show - 1, -1, -1))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("MI", fontsize=10)
        ax.set_title(f"{mod} ({len(idx_list)} features, top-{n_show})", fontsize=11, loc="left")
        ax.invert_yaxis()
        ax.set_xlim(0, max(ranked_mi) * 1.2)

    fig.suptitle("Stage 1 · MI Spectrum per Modality (red dashed=MI knee)",
                  fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.05, hspace=0.35)
    fig.savefig(FIG_DIR / "01_mi_spectrum_per_modality.png", dpi=100)
    plt.close(fig)
    print("  ✓ 01_mi_spectrum_per_modality.png")


# ============================================================ #
#  Figure 2: MI 拐点分析 — 6 种方法推导 K 对比
# ============================================================ #
def fig_knee_analysis(mi_mean, fnames, mods):
    methods = ["拐点", "Top25%", "Top50%", "超阈值", "超中位", "统计检验"]
    k_data = {}
    for mod, idx_list in mods.items():
        mi_sorted = sorted(mi_mean[idx_list], reverse=True)
        n = len(mi_sorted)
        global_all = sorted(mi_mean, reverse=True)
        floor = np.percentile(mi_mean, 10)
        median_g = np.median(mi_mean)

        from exp6_mi_multimodal_cls import find_knee_point
        knee = find_knee_point(mi_sorted)
        k_data[mod] = {
            "拐点": knee,
            "Top25%": max(1, int(np.ceil(n * 0.25))),
            "Top50%": max(1, int(np.ceil(n * 0.50))),
            "超阈值": sum(1 for v in mi_sorted if v > floor + 0.005),
            "超中位": sum(1 for v in mi_sorted if v > median_g + 0.005),
            "统计检验": sum(1 for v in mi_sorted if v > floor + 1.96 * 0.02),
        }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.18
    for i, (mod, ks) in enumerate(k_data.items()):
        vals = [ks[m] for m in methods]
        bars = ax.bar(x + i * width, vals, width, label=mod, color=MOD_COLORS[mod], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=8)

    # 经验最优 K 参考线
    empirical = {"眼动": 6, "脑电": 5, "心率": 4, "行为": 12}
    for i, (mod, k) in enumerate(empirical.items()):
        ax.axhline(y=k, color=MOD_COLORS[mod], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods)
    ax.set_ylabel("推荐 K 值")
    ax.set_title("Stage 1 · 6 种信息论方法推导 K 对比（虚线=经验最优 K）")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(v for v in ks.values()) for ks in k_data.values()) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_mi_knee_analysis.png")
    plt.close(fig)
    print("  ✓ 02_mi_knee_analysis.png")


# ============================================================ #
#  Figure 3: K 验证曲线 — 各模态 K vs F1
# ============================================================ #
def fig_k_validation(results_json):
    stage2 = results_json["stage2_results"]
    mi_derived_k = results_json["stage1_mi_derived_k"]
    empirical_k = results_json["stage2_best_k_per_mod"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    mods_order = ["眼动", "脑电", "心率", "行为"]

    for ax, mod in zip(axes, mods_order):
        sub = sorted([r for r in stage2 if r["target_mod"] == mod], key=lambda x: x["k_target"])
        ks = [r["k_target"] for r in sub]
        f1s = [r["pooled_macro_f1"] for r in sub]

        ax.plot(ks, f1s, "o-", color=MOD_COLORS[mod], linewidth=2, markersize=8)
        for k, f1 in zip(ks, f1s):
            ax.annotate(f"{f1:.3f}", (k, f1), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

        # 标注 MI 推导 K 和经验最优 K
        mi_k = mi_derived_k[mod]
        emp_k = empirical_k[mod]
        ax.axvline(x=mi_k, color="red", linestyle="--", alpha=0.6, label=f"MI推导K={mi_k}")
        ax.axvline(x=emp_k, color="green", linestyle="--", alpha=0.6, label=f"经验最优K={emp_k}")

        # 标注最佳点
        best_idx = np.argmax(f1s)
        ax.scatter([ks[best_idx]], [f1s[best_idx]], s=150, color="gold",
                    edgecolor="black", zorder=5, label=f"最佳K={ks[best_idx]}")

        ax.set_xlabel("K（特征数）")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{mod} 模态 K 验证曲线")
        ax.legend(fontsize=8)
        ax.set_ylim(min(f1s) - 0.02, max(f1s) + 0.03)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Stage 2 · 各模态 K 合理性验证（XGB_shallow）", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_k_validation_curves.png")
    plt.close(fig)
    print("  ✓ 03_k_validation_curves.png")


# ============================================================ #
#  Figure 4: 算法对比 — 14 算法 F1 条形图
# ============================================================ #
def fig_algorithm_comparison(results_json):
    algo = results_json["stage3_all_algo_results"]
    algo_sorted = sorted(algo, key=lambda x: -x["pooled_macro_f1"])

    fig, ax = plt.subplots(figsize=(14, 7))
    names = [f"{r['model']}\n({r['k_set_name'][:5]})" for r in algo_sorted]
    f1s = [r["pooled_macro_f1"] for r in algo_sorted]
    accs = [r["pooled_acc"] for r in algo_sorted]

    colors = []
    for r in algo_sorted:
        if r["model"] == "XGB_shallow" and r["k_set_name"] == "stage2_empirical":
            colors.append("#FF5722")
        elif r["k_set_name"] == "stage2_empirical":
            colors.append("#42A5F5")
        else:
            colors.append("#BDBDBD")

    x = np.arange(len(names))
    w = 0.38
    bars1 = ax.bar(x - w / 2, f1s, w, label="Macro-F1", color=colors, edgecolor="white")
    bars2 = ax.bar(x + w / 2, accs, w, label="Accuracy", color=colors, alpha=0.5, edgecolor="white")

    for bar, f1 in zip(bars1, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=7)

    # 最佳标注
    ax.annotate("★ 最佳", (0 - w / 2, f1s[0]), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=10, color="#FF5722", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Stage 3 · 4 模态 14 算法对比（结果一）\n橙色=最佳 | 蓝色=经验K | 灰色=MI推导K")
    ax.legend()
    ax.set_ylim(0.3, 0.82)
    ax.axhline(y=0.774, color="#FF5722", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_algorithm_comparison.png")
    plt.close(fig)
    print("  ✓ 04_algorithm_comparison.png")


# ============================================================ #
#  Figure 5: 模态消融 — 按模态数分组条形图
# ============================================================ #
def fig_modality_ablation(results_json):
    abl = results_json["stage4_ablation_results"]
    abl_sorted = sorted(abl, key=lambda x: -x["pooled_macro_f1"])

    fig, ax = plt.subplots(figsize=(14, 6))
    names = ["+".join(r["modalities"]) for r in abl_sorted]
    f1s = [r["pooled_macro_f1"] for r in abl_sorted]
    n_mods = [r["n_modalities"] for r in abl_sorted]

    mod_n_colors = {1: "#FFC107", 2: "#FF9800", 3: "#2196F3", 4: "#4CAF50"}
    colors = [mod_n_colors[n] for n in n_mods]

    bars = ax.barh(range(len(names) - 1, -1, -1), f1s, color=colors, edgecolor="white", height=0.7)
    for bar, f1, n_feat in zip(bars, f1s, [r["n_features"] for r in abl_sorted]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f} ({n_feat}feat)", ha="left", va="center", fontsize=8)

    ax.set_yticks(range(len(names) - 1, -1, -1))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Stage 4 · 模态消融（结果二，最佳算法 XGB_shallow）\n绿=4模态 蓝=3模态 橙=2模态 黄=1模态")
    ax.set_xlim(0.38, 0.82)
    ax.axvline(x=0.774, color="green", linestyle="--", alpha=0.4)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f"{n}模态") for n, c in sorted(mod_n_colors.items())]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_modality_ablation.png")
    plt.close(fig)
    print("  ✓ 05_modality_ablation.png")


# ============================================================ #
#  Figure 6: 模态贡献热力图
# ============================================================ #
def fig_ablation_heatmap(results_json):
    abl = results_json["stage4_ablation_results"]
    mod_names = ["眼动", "脑电", "心率", "行为"]

    # 构建矩阵: 行=模态组合, 列=是否包含该模态
    matrix = []
    labels = []
    f1s = []
    for r in sorted(abl, key=lambda x: -x["pooled_macro_f1"]):
        row = [1 if m in r["modalities"] else 0 for m in mod_names]
        matrix.append(row)
        labels.append("+".join(r["modalities"]))
        f1s.append(r["pooled_macro_f1"])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 9))
    # 用 F1 作为颜色
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=min(f1s), vmax=max(f1s))
    colors = [cmap(norm(f)) for f in f1s]

    for i, (row, f1) in enumerate(zip(matrix, f1s)):
        for j, val in enumerate(row):
            color = colors[i] if val else "#E0E0E0"
            ax.add_patch(plt.Rectangle((j, len(f1s) - 1 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            if val:
                ax.text(j + 0.5, len(f1s) - 1 - i + 0.5, "✓", ha="center", va="center", fontsize=14, fontweight="bold")
            else:
                ax.text(j + 0.5, len(f1s) - 1 - i + 0.5, "—", ha="center", va="center", fontsize=10, color="gray")

    ax.set_xticks([j + 0.5 for j in range(4)])
    ax.set_xticklabels(mod_names, fontsize=12)
    ax.set_yticks([len(f1s) - 1 - i + 0.5 for i in range(len(f1s))])
    ax.set_yticklabels([f"{l}  ({f:.3f})" for l, f in zip(labels, f1s)], fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(f1s))
    ax.set_aspect("equal")
    ax.set_title("Stage 4 · 模态消融热力图\n（颜色=Macro-F1, 绿高红低）", fontsize=13)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label="Macro-F1")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_modality_ablation_heatmap.png")
    plt.close(fig)
    print("  ✓ 06_modality_ablation_heatmap.png")


# ============================================================ #
#  Figure 7: 眼动 K 精细网格
# ============================================================ #
def fig_eye_k_fine_grid(results_json):
    fine = results_json["stage6_fine_k_grid"]
    fine_sorted = sorted(fine, key=lambda x: x["k_eye"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ks = [r["k_eye"] for r in fine_sorted]
    f1s = [r["pooled_macro_f1"] for r in fine_sorted]
    n_feats = [r["n_features"] for r in fine_sorted]

    ax.plot(ks, f1s, "o-", color="#2196F3", linewidth=2.5, markersize=10, zorder=5)
    ax.fill_between(ks, [f - 0.099 for f in f1s], [f + 0.099 for f in f1s],
                     alpha=0.1, color="#2196F3", label="±1σ (0.099)")

    for k, f1, nf in zip(ks, f1s, n_feats):
        ax.annotate(f"{f1:.3f}\n({nf}feat)", (k, f1), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    # 标注最佳
    best_idx = np.argmax(f1s)
    ax.scatter([ks[best_idx]], [f1s[best_idx]], s=200, color="gold",
                edgecolor="black", zorder=6, label=f"最佳 K={ks[best_idx]}")

    # MI 拐点参考
    ax.axvline(x=4, color="red", linestyle="--", alpha=0.5, label="MI拐点 K=4")

    ax.set_xlabel("眼动模态 K（特征数）")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Stage 6 · 4 模态下眼动 K 精细网格\n（固定 脑电:5 心率:4 行为:12）")
    ax.legend()
    ax.set_ylim(0.72, 0.79)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_eye_k_fine_grid.png")
    plt.close(fig)
    print("  ✓ 07_eye_k_fine_grid.png")


# ============================================================ #
#  Figure 8: 特征稳定性 — 5 折选中频次
# ============================================================ #
def fig_feature_stability(X, y_int, groups, fnames, mods):
    K = {"眼动": 6, "脑电": 5, "心率": 4, "行为": 12}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (mod, idx_list) in zip(axes, mods.items()):
        k = K[mod]
        cnt = Counter()
        for tr, te in sgkf.split(X, y_int, groups):
            imp = SimpleImputer(strategy="median")
            Xi = imp.fit_transform(X[tr])
            mi_avg = np.zeros(len(idx_list))
            for s in range(5):
                mi_avg += mutual_info_classif(Xi[:, idx_list], y_int[tr],
                                               random_state=RANDOM_STATE + s, n_neighbors=3)
            mi_avg /= 5
            rk = np.argsort(-mi_avg)
            for i in rk[:k]:
                cnt[idx_list[i]] += 1

        # 排序
        sorted_items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
        names = [fnames[idx].replace("_within_subject", "").replace("_z", "")
                  .replace("eye_aoi_", "").replace("log_", "").replace("eeg_", "")
                  .replace("hr_", "").replace("_power", "").replace("_win", "")[:22]
                 for idx, _ in sorted_items]
        hits = [c for _, c in sorted_items]
        colors = ["#4CAF50" if h == 5 else ("#2196F3" if h >= 3 else "#BDBDBD") for h in hits]

        bars = ax.barh(range(len(names) - 1, -1, -1), hits, color=colors, edgecolor="white")
        for bar, h in zip(bars, hits):
            star = " ⭐" if h == 5 else ""
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{h}/5{star}", ha="left", va="center", fontsize=8)

        ax.set_yticks(range(len(names) - 1, -1, -1))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("命中折数")
        ax.set_title(f"{mod} 模态 (K={k}, {len(cnt)} 个不同特征被选)")
        ax.set_xlim(0, 6)
        ax.invert_yaxis()

    fig.suptitle("特征稳定性 · 5 折 MI 选中频次（绿=5/5稳定, 蓝=≥3/5, 灰=<3/5）",
                  fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_stability.png")
    plt.close(fig)
    print("  ✓ 08_feature_stability.png")


# ============================================================ #
#  CSV 数据导出
# ============================================================ #
def export_csvs(results_json, mi_mean, mi_std, fnames, mods):
    # 1. MI 全谱
    rows = []
    for mod, idx_list in mods.items():
        for idx in idx_list:
            rows.append({
                "模态": mod,
                "特征名": fnames[idx],
                "特征序号": idx,
                "MI_mean": round(float(mi_mean[idx]), 6),
                "MI_std": round(float(mi_std[idx]), 6),
            })
    df = pd.DataFrame(rows).sort_values(["模态", "MI_mean"], ascending=[True, False])
    df.to_csv(CSV_DIR / "mi_spectrum.csv", index=False, encoding="utf-8-sig")
    print("  ✓ data/mi_spectrum.csv")

    # 2. K 验证
    df = pd.DataFrame(results_json["stage2_results"])
    df = df[["target_mod", "k_target", "n_features", "pooled_acc", "pooled_macro_f1",
             "fold_f1_mean", "fold_f1_std"]]
    df.columns = ["模态", "K", "总特征数", "Acc", "Macro-F1", "fold_F1均值", "fold_F1标准差"]
    df.to_csv(CSV_DIR / "k_validation.csv", index=False, encoding="utf-8-sig")
    print("  ✓ data/k_validation.csv")

    # 3. 算法对比
    df = pd.DataFrame(results_json["stage3_all_algo_results"])
    df = df[["k_set_name", "model", "n_features", "pooled_acc", "pooled_macro_f1",
             "fold_f1_mean", "fold_f1_std"]]
    df.columns = ["K集", "模型", "总特征数", "Acc", "Macro-F1", "fold_F1均值", "fold_F1标准差"]
    df = df.sort_values("Macro-F1", ascending=False)
    df.to_csv(CSV_DIR / "algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    print("  ✓ data/algorithm_comparison.csv")

    # 4. 模态消融
    df = pd.DataFrame(results_json["stage4_ablation_results"])
    df["模态组合"] = df["modalities"].apply(lambda x: "+".join(x))
    df = df[["n_modalities", "模态组合", "n_features", "pooled_acc", "pooled_macro_f1",
             "fold_f1_mean", "fold_f1_std"]]
    df.columns = ["模态数", "模态组合", "总特征数", "Acc", "Macro-F1", "fold_F1均值", "fold_F1标准差"]
    df = df.sort_values("Macro-F1", ascending=False)
    df.to_csv(CSV_DIR / "modality_ablation.csv", index=False, encoding="utf-8-sig")
    print("  ✓ data/modality_ablation.csv")

    # 5. Stage 5 精调
    if results_json.get("stage5_fine_tune_results"):
        df = pd.DataFrame(results_json["stage5_fine_tune_results"])
        df["基础组合"] = df["base_combo"].apply(lambda x: "+".join(x))
        df = df[["基础组合", "delta_target", "delta", "new_k", "n_features", "pooled_macro_f1"]]
        df.columns = ["基础组合", "精调模态", "ΔK", "新K", "总特征数", "Macro-F1"]
        df = df.sort_values("Macro-F1", ascending=False)
        df.to_csv(CSV_DIR / "stage5_fine_tune.csv", index=False, encoding="utf-8-sig")
        print("  ✓ data/stage5_fine_tune.csv")

    # 6. Stage 6 精细网格
    if results_json.get("stage6_fine_k_grid"):
        df = pd.DataFrame(results_json["stage6_fine_k_grid"])
        df = df[["k_eye", "n_features", "pooled_acc", "pooled_macro_f1"]]
        df.columns = ["眼动K", "总特征数", "Acc", "Macro-F1"]
        df = df.sort_values("眼动K")
        df.to_csv(CSV_DIR / "stage6_eye_k_grid.csv", index=False, encoding="utf-8-sig")
        print("  ✓ data/stage6_eye_k_grid.csv")


# ============================================================ #
#  详细报告
# ============================================================ #
def write_detailed_report(results_json, mi_mean, mi_std, fnames, mods):
    mi_derived_k = results_json["stage1_mi_derived_k"]
    stage2_best = results_json["stage2_best_k_per_mod"]
    best_algo = results_json["stage3_best_algo"]
    ablation = results_json["stage4_ablation_results"]

    lines = []
    lines.append("# P6 MI 特征筛选实验 · 详细结果与可视化报告\n\n")
    lines.append(f"> 生成时间：2026-07-09 | 实验目录：`reports_exp6/`\n\n")
    lines.append("---\n\n")

    lines.append("## 文件夹结构\n\n")
    lines.append("```\n")
    lines.append("reports_exp6/\n")
    lines.append("├── MI_DETAILED_REPORT.md          ← 本文件（详细整理报告）\n")
    lines.append("├── report.md                       ← 自动生成报告\n")
    lines.append("├── results.json                    ← 完整实验数据\n")
    lines.append("├── run.log                         ← 运行日志\n")
    lines.append("├── figures/                        ← 可视化图表\n")
    lines.append("│   ├── 01_mi_spectrum_per_modality.png  ← 各模态MI全谱\n")
    lines.append("│   ├── 02_mi_knee_analysis.png          ← MI拐点K推导对比\n")
    lines.append("│   ├── 03_k_validation_curves.png       ← K验证曲线\n")
    lines.append("│   ├── 04_algorithm_comparison.png      ← 14算法对比\n")
    lines.append("│   ├── 05_modality_ablation.png         ← 模态消融条形图\n")
    lines.append("│   ├── 06_modality_ablation_heatmap.png ← 模态消融热力图\n")
    lines.append("│   ├── 07_eye_k_fine_grid.png           ← 眼动K精细网格\n")
    lines.append("│   └── 08_feature_stability.png         ← 特征稳定性\n")
    lines.append("└── data/                          ← 结构化CSV\n")
    lines.append("    ├── mi_spectrum.csv                  ← MI全谱数据\n")
    lines.append("    ├── k_validation.csv                 ← K验证结果\n")
    lines.append("    ├── algorithm_comparison.csv         ← 算法对比\n")
    lines.append("    ├── modality_ablation.csv            ← 模态消融\n")
    lines.append("    ├── stage5_fine_tune.csv             ← Stage5精调\n")
    lines.append("    └── stage6_eye_k_grid.csv            ← Stage6精细网格\n")
    lines.append("```\n\n---\n\n")

    # Stage 1
    lines.append("## 1. MI 全谱分析\n\n")
    lines.append("![MI全谱](figures/01_mi_spectrum_per_modality.png)\n\n")
    lines.append("![MI拐点分析](figures/02_mi_knee_analysis.png)\n\n")

    lines.append("### 各模态 MI Top-10\n\n")
    for mod, idx_list in mods.items():
        mi_sorted = sorted([(i, mi_mean[i]) for i in idx_list], key=lambda x: -x[1])
        lines.append(f"**{mod}模态**（{len(idx_list)} 特征，MI推导K={mi_derived_k[mod]}）\n\n")
        lines.append("| 排名 | 特征名 | MI | ±σ |\n|---:|---|---:|---:|\n")
        for rank, (idx, mi) in enumerate(mi_sorted[:10], 1):
            lines.append(f"| {rank} | `{fnames[idx]}` | {mi:.4f} | {mi_std[idx]:.4f} |\n")
        lines.append("\n")

    # Stage 2
    lines.append("## 2. K 合理性验证\n\n")
    lines.append("![K验证曲线](figures/03_k_validation_curves.png)\n\n")
    lines.append("| 模态 | MI推导K | 经验最优K | 差值 | 评价 |\n|---|---:|---:|---:|---|\n")
    for mod in mods:
        mi_k = mi_derived_k[mod]
        emp_k = stage2_best[mod]
        delta = emp_k - mi_k
        if abs(delta) <= 2:
            eval_str = "✅ 吻合"
        elif abs(delta) <= 5:
            eval_str = "⚠️ 接近"
        else:
            eval_str = "❌ 偏大"
        lines.append(f"| {mod} | {mi_k} | {emp_k} | {delta:+d} | {eval_str} |\n")
    lines.append("\n")

    # Stage 3
    lines.append("## 3. 算法对比（结果一）\n\n")
    lines.append("![算法对比](figures/04_algorithm_comparison.png)\n\n")
    lines.append(f"**★ 最佳算法 = {best_algo['model']} (K集={best_algo['k_set_name']})，"
                 f"F1 = {best_algo['pooled_macro_f1']:.3f}**\n\n")
    lines.append("Top-5 算法：\n\n")
    lines.append("| 排名 | K集 | 模型 | 特征数 | F1 | fold F1 μ±σ |\n|---:|---|---|---:|---:|---:|\n")
    for i, r in enumerate(sorted(results_json["stage3_all_algo_results"],
                                  key=lambda x: -x["pooled_macro_f1"])[:5], 1):
        lines.append(f"| {i} | {r['k_set_name']} | {r['model']} | {r['n_features']} | "
                     f"**{r['pooled_macro_f1']:.3f}** | {r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n")
    lines.append("\n")

    # Stage 4
    lines.append("## 4. 模态消融（结果二）\n\n")
    lines.append("![模态消融](figures/05_modality_ablation.png)\n\n")
    lines.append("![消融热力图](figures/06_modality_ablation_heatmap.png)\n\n")
    lines.append("| 排名 | 模态数 | 组合 | 特征数 | F1 | fold F1 μ±σ |\n|---:|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(sorted(ablation, key=lambda x: -x["pooled_macro_f1"]), 1):
        lines.append(f"| {i} | {r['n_modalities']} | {'+'.join(r['modalities'])} | "
                     f"{r['n_features']} | **{r['pooled_macro_f1']:.3f}** | "
                     f"{r['fold_f1_mean']:.3f}±{r['fold_f1_std']:.3f} |\n")
    lines.append("\n")

    # Stage 5/6
    lines.append("## 5. 精调与反馈\n\n")
    lines.append("![眼动K精细网格](figures/07_eye_k_fine_grid.png)\n\n")

    if results_json.get("stage5_fine_tune_results"):
        s5_sorted = sorted(results_json["stage5_fine_tune_results"],
                           key=lambda x: -x["pooled_macro_f1"])
        lines.append("### Stage 5 Top-5 精调结果\n\n")
        lines.append("| 排名 | 基础组合 | 精调模态 | ΔK | 新K | 特征数 | F1 |\n|---:|---|---|---:|---:|---:|---:|\n")
        for i, r in enumerate(s5_sorted[:5], 1):
            lines.append(f"| {i} | {'+'.join(r['base_combo'])} | {r['delta_target']} | "
                         f"{r['delta']:+d} | {r['new_k']} | {r['n_features']} | "
                         f"**{r['pooled_macro_f1']:.3f}** |\n")
        lines.append("\n")

    # 特征稳定性
    lines.append("## 6. 特征稳定性\n\n")
    lines.append("![特征稳定性](figures/08_feature_stability.png)\n\n")

    # 最终对比
    lines.append("## 7. 最终对比\n\n")
    lines.append("| 版本 | 特征数 | F1 | 方法 | 多模态 |\n|---:|---:|---:|---|:---:|\n")
    lines.append("| P4b 稳定15 | 15 | 0.810 | 经验选 | ❌ |\n")
    lines.append("| P5-19 | 19 | 0.787 | 经验网格 | ✅ |\n")
    lines.append(f"| **P6 (4模态)** | {best_algo['k']} | **{best_algo['pooled_macro_f1']:.3f}** | MI前向推导 | ✅ |\n\n")

    (REPORT_DIR / "MI_DETAILED_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("  ✓ MI_DETAILED_REPORT.md")


# ============================================================ #
#  主流程
# ============================================================ #
def main():
    print("=" * 60)
    print("P6 MI 实验结果可视化与详细整理")
    print("=" * 60)

    X, y_int, groups, fnames = load_data()
    mods = build_modalities(fnames)
    print(f"  数据: X={X.shape}, 模态: {[(m, len(v)) for m, v in mods.items()]}")

    with open(REPORT_DIR / "results.json", encoding="utf-8") as f:
        results_json = json.load(f)

    print("\n[1/4] 计算 MI 全谱...")
    mi_mean, mi_std = compute_mi_spectrum(X, y_int, groups, fnames, mods)

    print("\n[2/4] 生成可视化图表...")
    fig_mi_spectrum(mi_mean, mi_std, fnames, mods)
    fig_knee_analysis(mi_mean, fnames, mods)
    fig_k_validation(results_json)
    fig_algorithm_comparison(results_json)
    fig_modality_ablation(results_json)
    fig_ablation_heatmap(results_json)
    fig_eye_k_fine_grid(results_json)
    fig_feature_stability(X, y_int, groups, fnames, mods)

    print("\n[3/4] 导出 CSV 数据...")
    export_csvs(results_json, mi_mean, mi_std, fnames, mods)

    print("\n[4/4] 写详细报告...")
    write_detailed_report(results_json, mi_mean, mi_std, fnames, mods)

    print("\n" + "=" * 60)
    print("完成！输出目录：", REPORT_DIR)
    print("  figures/ — 8 张图表")
    print("  data/    — 6 个 CSV")
    print("  MI_DETAILED_REPORT.md — 详细报告")
    print("=" * 60)


if __name__ == "__main__":
    main()
