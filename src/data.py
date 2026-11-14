"""데이터 로드 · 상수 · 다운샘플링."""
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
# 후처리 선택 점수의 간격별 가중치 = Test 실험 구성 (5초 9개, 10초 1개, 15초 1개)
VER_W = {"k1o0": 9, "k2o0": 1, "k3o0": 1}

# 파일명(소문자)에 포함되면 매칭. 앞쪽 키워드가 우선.
PATTERNS = {
    "train": ["train"],
    "valid": ["validation", "valid", "val"],
    "test": ["test"],
    "sub": ["sample_submission", "submission"],
}


def _all_csvs(root):
    out = []
    for d, _, files in os.walk(root, followlinks=True):  # 심볼릭 링크도 따라감
        out += [os.path.join(d, f) for f in files if f.lower().endswith(".csv")]
    return sorted(out)


def find_csv(root, split):
    csvs = _all_csvs(root)
    for kw in PATTERNS[split]:
        hits = [p for p in csvs if kw in os.path.basename(p).lower()
                and (split == "sub" or "submission" not in os.path.basename(p).lower())]
        if hits:
            return min(hits, key=len)
    msg = [f"[오류] '{root}' 아래에서 {split} CSV를 찾지 못했습니다.",
           f"  현재 작업 폴더: {os.getcwd()}",
           f"  '{root}' 존재 여부: {os.path.exists(root)} (깨진 링크면 False)"]
    msg.append(f"  발견된 CSV {len(csvs)}개: {csvs[:20]}" if csvs else "  CSV가 하나도 없습니다 → --data 경로를 확인하세요.")
    sys.exit("\n".join(msg))


def load(root, splits=("train", "valid", "test", "sub")):
    out = {}
    for s in splits:
        p = find_csv(root, s)
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
