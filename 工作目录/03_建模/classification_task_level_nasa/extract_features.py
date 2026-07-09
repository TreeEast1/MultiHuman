#!/usr/bin/env python3
"""提取 Stage3 最佳配置 (stable15+eeg2+hr2) 的5折实际选中的EEG/HR特征。"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from cls_utils import RANDOM_STATE

DATA_DIR = HERE / "dataset"

X = np.load(DATA_DIR / "X_cls.npy")
y_int = np.load(DATA_DIR / "y_cls_int.npy")
groups = np.load(DATA_DIR / "groups_cls.npy")
with open(DATA_DIR / "feature_names_cls.json") as f:
    feature_names = json.load(f)

p4_stable_15 = [
    "eye_aoi_unique_hit_n__std", "eye_aoi_interval_n__std",
    "eye_aoi_interval_n__mean", "eye_aoi_entropy__median",
    "eye_aoi_entropy__mean", "eye_aoi_unique_hit_n__mean",
    "eye_aoi_fixation_n__std", "eye_aoi_fixation_n__slope",
    "eye_aoi_max_share__mean", "eye_aoi_coverage_ratio__slope",
    "log_action_count_win__mean", "log_action_density_win__mean",
    "log_error_rate_win__std", "eye_aoi_coverage_ratio__median",
    "eye_aoi_total_fix_ms__median",
]
fname_to_idx = {n: i for i, n in enumerate(feature_names)}
stable_15_idx = [fname_to_idx[n] for n in p4_stable_15 if n in fname_to_idx]

eeg_idx = [i for i, n in enumerate(feature_names) if n.startswith("eeg_")]
hr_idx = [i for i, n in enumerate(feature_names) if n.startswith("hr_")]
eeg_idx_arr = np.array(eeg_idx)
hr_idx_arr = np.array(hr_idx)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
eeg_counter = Counter()
hr_counter = Counter()
for tr, te in sgkf.split(X, y_int, groups):
    X_tr = X[tr]
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    mi = mutual_info_classif(X_tr_imp[:, eeg_idx_arr], y_int[tr], random_state=RANDOM_STATE)
    eeg_rk = np.argsort(-mi)
    for i in eeg_rk[:2]:
        eeg_counter[eeg_idx_arr[i]] += 1
    mi = mutual_info_classif(X_tr_imp[:, hr_idx_arr], y_int[tr], random_state=RANDOM_STATE)
    hr_rk = np.argsort(-mi)
    for i in hr_rk[:2]:
        hr_counter[hr_idx_arr[i]] += 1

print("=" * 60)
print("5折实际选中的 EEG 特征 (Top-2 每折)")
print("=" * 60)
for idx, cnt in eeg_counter.most_common():
    print(f"  {feature_names[idx]:60s}  命中{cnt}/5折")

print()
print("=" * 60)
print("5折实际选中的 HR 特征 (Top-2 每折)")
print("=" * 60)
for idx, cnt in hr_counter.most_common():
    print(f"  {feature_names[idx]:60s}  命中{cnt}/5折")

print()
print("=" * 60)
print("最终推荐 19 特征（多模态结构完整）")
print("=" * 60)
print("\n【眼动】11 AOI 特征（P4 稳定）：")
for n in p4_stable_15:
    if "eye_aoi" in n:
        print(f"  - {n}")
print("\n【行为】4 Log 特征（P4 稳定）：")
for n in p4_stable_15:
    if "log_" in n:
        print(f"  - {n}")
print("\n【脑电】2 EEG 特征（折内MI选 Top-2，每折可能不同，但以下最稳定）：")
top_eeg = [feature_names[idx] for idx, _ in eeg_counter.most_common(2)]
for n in top_eeg:
    print(f"  - {n}")
print("\n【心率】2 HR 特征（折内MI选 Top-2）：")
top_hr = [feature_names[idx] for idx, _ in hr_counter.most_common(2)]
for n in top_hr:
    print(f"  - {n}")
