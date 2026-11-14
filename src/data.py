"""데이터 로드 · 상수 · 다운샘플링."""
import glob
import os
import sys

import pandas as pd

KEY, T, Y = "experiment_id", "relative_time_s", "label"
SENSORS = [
    "meas_v", "meas_t1", "meas_t2", "meas_p",
    "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
    "meas_n2o", "meas_no", "meas_no2", "meas_so2",
    "atmp_t", "thermal_t", "indu_t",
]
# 사용하지 않는 칼럼: meas_a(항상 0), indu_t.1(Test 전부 결측), meas_t1_plus_t2(t1+t2 중복), image_path
COND = ["bat_class", "bat_cc", "soc_status", "tr_method"]

# 학습용 다운샘플 버전 (k, offset) → 5초*k 간격. Test는 5/10/15초 간격이 섞여 있음.
TRAIN_VERSIONS = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
EVAL_VERSIONS = [(1, 0), (2, 0), (3, 0)]

PATTERNS = {
    "train": "train*.csv",
    "valid": "valid*.csv",
    "test": "test*.csv",
    "sub": "sample_submission*.csv",
}


def find_csv(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        sys.exit(f"[오류] {root} 아래에서 {pattern} 를 찾지 못했습니다.")
    return hits[0]


def load(root, splits=("train", "valid", "test", "sub")):
    out = {}
    for s in splits:
        p = find_csv(root, PATTERNS[s])
        print(f"  {s:6s} ← {p}")
        out[s] = pd.read_csv(p)
    return out


def subsample(df, k, offset):
    """각 실험을 5초*k 간격으로 솎아내고, 첫 관측을 0초로 다시 맞춘다 (Test 정의와 동일).
    원래 실험 id는 grp, 버전은 ver 칼럼에 남긴다."""
    df = df.sort_values([KEY, T], kind="stable")
    pos = df.groupby(KEY).cumcount()
    s = df[(pos >= offset) & ((pos - offset) % k == 0)].copy()
    s[T] = s[T] - s.groupby(KEY)[T].transform("min")
    s["grp"] = s[KEY]
    s["ver"] = f"k{k}o{offset}"
    s[KEY] = s[KEY] + f"__k{k}o{offset}"
    return s
