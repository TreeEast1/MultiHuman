#!/usr/bin/env python3
"""整理 NASA 三分类五折 27 维输入列名，按论文体例出 Word。

协议与 P6 正式分类实验一致：
  StratifiedGroupKFold(5, shuffle=True, random_state=0)
  折内互信息定额：眼动 6 + 脑电 5 + 心率 4 + 行为 12
  mutual_info_classif，5 种子平均，n_neighbors=3
  标签为 NASA-TLX 三分位（低 / 中 / 高），不是绩效 S
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forecast_next_stage"))
from report_fmt import (  # noqa: E402
    add_body,
    add_caption,
    add_h,
    add_note,
    add_title,
    add_toc_heading,
    add_toc_line,
    make_table,
    new_doc,
)

DATA_DIR = HERE / "dataset"
OUT_DIR = HERE / "reports_fold_features"
OUT_DOCX = HERE / "NASA三分类_五折27维输入列名.docx"
DESK = Path("/Users/licochen/Desktop")
RANDOM_STATE = 0
N_SPLITS = 5
N_SEEDS = 5
QUOTA = {"眼动": 6, "脑电": 5, "心率": 4, "行为": 12}

STAT_CN = {"mean": "平均", "std": "波动", "median": "中位数", "slope": "走势"}
REGION_CN = {
    "frontal": "额区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
}
BAND_CN = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}
BASE_CN = {
    "eye_aoi_unique_hit_n": "点到几个不同兴趣区",
    "eye_aoi_interval_n": "兴趣区区间条数",
    "eye_aoi_coverage_ratio": "兴趣区注视覆盖比例",
    "eye_aoi_max_share": "最主要兴趣区占比",
    "eye_aoi_entropy": "兴趣区注视份额熵",
    "eye_aoi_fixation_n": "兴趣区注视次数",
    "eye_aoi_total_fix_ms": "兴趣区注视总时长",
    "eye_aoi_fixation_density_per_sec": "每秒兴趣区注视密度",
    "eye_aoi_pupil_weighted_mean": "兴趣区加权瞳孔直径",
    "eye_pupil_filtered_mean": "滤波瞳孔直径",
    "eye_pupil_filtered_std": "滤波瞳孔直径波动",
    "eye_valid_ratio": "双眼有效采样比例",
    "eye_fixation_ratio": "注视时间比例",
    "eye_saccade_ratio": "扫视时间比例",
    "eye_eyes_not_found_ratio": "找不到眼睛的时间比例",
    "blink_count": "疑似眨眼次数",
    "blink_rate_per_min": "眨眼频率",
    "blink_duration_mean_ms": "眨眼时长均值",
    "blink_duration_std_ms": "眨眼时长波动",
    "blink_duration_median_ms": "眨眼时长中位数",
    "blink_total_duration_ratio": "眨眼总时长占比",
    "hr_mean": "心率均值",
    "hr_std": "心率波动",
    "hr_min": "最低心率",
    "hr_max": "最高心率",
    "hr_slope_bpm_per_min": "窗内心率斜率",
    "log_action_count_win": "操作次数",
    "log_action_density_win": "操作密度",
    "log_unique_device_count_win": "设备种数",
    "log_unique_step_count_win": "步骤种数",
    "log_correct_action_count_win": "正确操作次数",
    "log_error_action_count_win": "错误操作次数",
    "log_duplicate_action_count_win": "重复操作次数",
    "log_extra_action_count_win": "多余操作次数",
    "log_disallowed_action_count_win": "不合规次数合计",
    "log_error_rate_win": "错误操作比例",
    "log_duplicate_rate_win": "重复操作比例",
    "log_extra_rate_win": "多余操作比例",
}


def modality_of(name: str) -> str:
    if name.startswith(("eye_aoi", "eye_", "blink_")):
        return "眼动"
    if name.startswith("eeg_"):
        return "脑电"
    if name.startswith("hr_"):
        return "心率"
    if name.startswith("log_"):
        return "行为"
    raise ValueError(f"未知模态: {name}")


def gloss_eeg(base: str) -> str:
    s = base.replace("eeg_", "").replace("_z_within_subject", "")
    parts = s.split("_")
    region = REGION_CN.get(parts[0], parts[0])
    rest = parts[1:]
    if rest[-1:] == ["power"] and rest[0] in BAND_CN:
        return f"{region} {BAND_CN[rest[0]]} 功率"
    if rest == ["theta", "alpha"]:
        return f"{region} θ/α"
    if rest == ["beta", "alpha"]:
        return f"{region} β/α"
    return base


def gloss(col: str) -> str:
    if "__" in col:
        base, stat = col.rsplit("__", 1)
        tail = STAT_CN.get(stat, stat)
    else:
        base, tail = col, ""
    if base in BASE_CN:
        stem = BASE_CN[base]
    elif base.startswith("eeg_"):
        stem = gloss_eeg(base)
    else:
        stem = base
    return f"{stem} · {tail}" if tail else stem


def build_mod_idx(names: list[str]) -> dict[str, np.ndarray]:
    mods = {"眼动": [], "脑电": [], "心率": [], "行为": []}
    for i, n in enumerate(names):
        mods[modality_of(n)].append(i)
    return {k: np.array(v, dtype=int) for k, v in mods.items()}


def rank_mi(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    mi_avg = np.zeros(X_tr.shape[1])
    for s in range(N_SEEDS):
        mi_avg += mutual_info_classif(
            X_tr, y_tr, random_state=RANDOM_STATE + s, n_neighbors=3,
        )
    mi_avg /= N_SEEDS
    return np.argsort(-mi_avg)


def select_quota(X_tr: np.ndarray, y_tr: np.ndarray, mod_idx: dict[str, np.ndarray]) -> list[int]:
    picked: list[int] = []
    for mod, idx in mod_idx.items():
        k = QUOTA[mod]
        order = rank_mi(X_tr[:, idx], y_tr)[:k]
        picked.extend(idx[order].tolist())
    return picked


def fmt_subjects(ids: list[int]) -> str:
    return "、".join(str(int(x)) for x in ids)


def collect_folds() -> list[dict]:
    X = np.load(DATA_DIR / "X_cls.npy")
    y = np.load(DATA_DIR / "y_cls_int.npy")
    groups = np.load(DATA_DIR / "groups_cls.npy")
    names = json.loads((DATA_DIR / "feature_names_cls.json").read_text(encoding="utf-8"))
    y_lab = np.load(DATA_DIR / "y_cls.npy", allow_pickle=True).astype(str)
    mod_idx = build_mod_idx(names)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_i, (tr, te) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr = SimpleImputer(strategy="median").fit_transform(X[tr])
        picked = select_quota(X_tr, y[tr], mod_idx)
        test_subj = sorted({int(g) for g in groups[te]})
        train_subj = sorted({int(g) for g in groups[tr]})
        lab_counts = Counter(str(v) for v in y_lab[te])
        rows = []
        for j in picked:
            col = names[j]
            rows.append({
                "modality": modality_of(col),
                "name": col,
                "gloss": gloss(col),
            })
        folds.append({
            "fold": fold_i,
            "n_test": int(len(te)),
            "n_train": int(len(tr)),
            "n_test_subj": len(test_subj),
            "test_subjects": test_subj,
            "train_subjects": train_subj,
            "test_label": {
                "低": int(lab_counts.get("低", 0)),
                "中": int(lab_counts.get("中", 0)),
                "高": int(lab_counts.get("高", 0)),
            },
            "columns": rows,
        })
        assert len(rows) == 27, (fold_i, len(rows))
        counts = Counter(r["modality"] for r in rows)
        assert counts == QUOTA, (fold_i, dict(counts))
    return folds


def stable_columns(folds: list[dict]) -> list[dict]:
    counts: Counter[str] = Counter()
    first: dict[str, dict] = {}
    for fold in folds:
        for row in fold["columns"]:
            counts[row["name"]] += 1
            first.setdefault(row["name"], row)
    return [first[n] for n, c in counts.items() if c == N_SPLITS]


def build_doc(folds: list[dict]) -> Path:
    doc = new_doc("NASA 三分类  五折 27 维输入列名")
    add_title(doc, "NASA-TLX 三分类五折输入特征列表", "各折 27 维列名（眼动 6 + 脑电 5 + 心率 4 + 行为 12）")

    add_toc_heading(doc)
    add_toc_line(doc, "1  口径", 1)
    add_toc_line(doc, "2  五折划分", 1)
    add_toc_line(doc, "3  各折入选列", 1)
    for fold in folds:
        add_toc_line(doc, f"3.{fold['fold']}  第 {fold['fold']} 折（被试 {fmt_subjects(fold['test_subjects'])}）", 2)
    add_toc_line(doc, "4  五折均入选的列", 1)

    add_h(doc, "1  口径", 1)
    add_body(
        doc,
        "本表给出 NASA-TLX 加权总分三分类实验中，按被试五折交叉验证时每一折实际送入模型的输入列。标签为问卷加权总分按 33.3% / 66.7% 分位数切成的低、中、高三档，不是绩效 S，也不是连续负荷回归。",
    )
    add_body(
        doc,
        "原始任务级特征为 84×264。每一折只在训练被试上计算互信息，按模态定额录取：眼动 6、脑电 5、心率 4、行为 12，合计 27 维。测试折只用已经定好的这 27 个列名做推理，不重新筛选。27 个名字随训练堆变化，不是全实验锁死一张名单。缺失值用该折训练堆中位数填充。",
    )
    add_caption(doc, "表 1  筛选协议")
    make_table(doc, ["项", "内容"], [
        ["样本", "26 名被试、84 条被试–任务"],
        ["标签", "NASA-TLX 加权总分三分位（低 ≤ 4.267，中 (4.267, 5.733]，高 > 5.733）"],
        ["划分", "StratifiedGroupKFold，5 折，按被试分组，shuffle＝True，random_state＝0（scikit-learn 1.5.2）"],
        ["互信息", "sklearn.feature_selection.mutual_info_classif，对三档标签；5 个随机种子平均，n_neighbors＝3"],
        ["定额", "眼动 6、脑电 5、心率 4、行为 12，合计 27 维"],
        ["填充", "SimpleImputer(median)，只在训练堆上拟合"],
    ], [3.2, 11.6], left_cols={0, 1})
    add_note(doc, "说明：绩效 S 那套附录用的是 GroupKFold，且互信息目标是连续 NASA。两套折的考试被试和 27 个列名都不相同，不能混用。")

    add_h(doc, "2  五折划分", 1)
    add_body(doc, "同一被试的全部任务只出现在同一折。下表考试被试为该折测试堆。")
    add_caption(doc, "表 2  五折考试被试与样本数")
    make_table(
        doc,
        ["折", "考试人数", "考试条数", "考试被试", "考试标签（低/中/高）"],
        [
            [
                str(f["fold"]),
                str(f["n_test_subj"]),
                str(f["n_test"]),
                fmt_subjects(f["test_subjects"]),
                f"{f['test_label']['低']}/{f['test_label']['中']}/{f['test_label']['高']}",
            ]
            for f in folds
        ],
        [1.6, 2.2, 2.2, 5.4, 3.4],
        left_cols={3},
    )

    add_h(doc, "3  各折入选列", 1)
    add_body(doc, "下列为各折训练被试上互信息定额的结果。测试折只用这些列做推理。列顺序为该折互信息从高到低，先眼动、再脑电、心率、行为。")

    for f in folds:
        subj = fmt_subjects(f["test_subjects"])
        add_h(doc, f"3.{f['fold']}  第 {f['fold']} 折（被试 {subj}）", 2)
        add_caption(doc, f"表 B{f['fold']}  第 {f['fold']} 折（被试 {subj}）入选列")
        make_table(
            doc,
            ["模态", "列名", "含义"],
            [[r["modality"], r["name"], r["gloss"]] for r in f["columns"]],
            [1.8, 7.0, 6.0],
            left_cols={1, 2},
            size=9,
        )

    stable = stable_columns(folds)
    add_h(doc, "4  五折均入选的列", 1)
    if stable:
        add_body(
            doc,
            f"下列 {len(stable)} 列在五折训练堆中都进入 27 维，相对稳定。其余列随折变化。",
        )
        add_caption(doc, "表 3  五折均入选的列")
        make_table(
            doc,
            ["模态", "列名", "含义"],
            [[r["modality"], r["name"], r["gloss"]] for r in stable],
            [1.8, 7.0, 6.0],
            left_cols={1, 2},
            size=9,
        )
    else:
        add_body(doc, "没有列在全部五折中都进入 27 维。各折名单随训练被试变化。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    desk = DESK / OUT_DOCX.name
    shutil.copy2(OUT_DOCX, desk)
    return OUT_DOCX


def main() -> None:
    folds = collect_folds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "quota": QUOTA,
        "protocol": {
            "split": "StratifiedGroupKFold",
            "sklearn": "1.5.2",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "mi": "mutual_info_classif",
            "n_seeds": N_SEEDS,
            "n_neighbors": 3,
            "imputer": "median",
            "target": "NASA-TLX tertile (y_cls_int)",
        },
        "folds": folds,
        "stable_all_folds": [r["name"] for r in stable_columns(folds)],
    }
    (OUT_DIR / "fold_features.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    path = build_doc(folds)
    print("wrote", path)
    print("copy ", DESK / path.name)
    for f in folds:
        print(
            f"fold {f['fold']}: test subjects {f['test_subjects']} "
            f"n={f['n_test']} labels={f['test_label']}"
        )


if __name__ == "__main__":
    main()
