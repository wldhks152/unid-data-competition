"""EDA — 모델링 방향을 정하는 핵심 질문에 답한다.
    python eda.py --data_dir <battery_data_light 경로>
"""
import argparse

import numpy as np
import pandas as pd

from src.data import COND, KEY, T, Y, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "--data", dest="data", default="data")
    a = ap.parse_args()
    d = load(a.data)
    tr, va, te = d["train"], d["valid"], d["test"]

    print("\n=== 크기 ===")
    for n, df in [("train", tr), ("valid", va), ("test", te)]:
        print(f"  {n:6s} rows={len(df):>6,}  experiments={df[KEY].nunique():>3}")

    print("\n=== 실험별 측정 간격(초) 분포 ===")
    for n, df in [("train", tr), ("valid", va), ("test", te)]:
        dt = df.sort_values([KEY, T]).groupby(KEY)[T].apply(lambda s: s.diff().median())
        print(f"  {n:6s}", dt.value_counts().sort_index().to_dict())

    print("\n=== label이 실험 안에서 단조 증가(1→6)인가? → 맞으면 cummax 후처리 후보 ===")
    for n, df in [("train", tr), ("valid", va)]:
        s = df.sort_values([KEY, T]).groupby(KEY)[Y]
        mono = s.apply(lambda x: bool((x.diff().dropna() >= 0).all()))
        skip = s.apply(lambda x: bool((x.diff().dropna() > 1).any()))
        print(f"  {n:6s} 단조증가 {mono.mean():.1%} ({mono.sum()}/{len(mono)}), 단계 건너뜀 실험 {skip.sum()}")
        for e in mono[~mono].index[:3]:
            seq = df[df[KEY] == e].sort_values(T)[Y].values
            print(f"    비단조 예시 {e[:10]}…: {seq[np.r_[True, seq[1:] != seq[:-1]]][:20].tolist()}")

    print("\n=== 실험별 등장 단계 (모든 실험이 1~6을 다 거치는가?) ===")
    df = pd.concat([tr, va])
    print("  ", df.groupby(KEY)[Y].apply(lambda x: "".join(map(str, sorted(x.unique()))))
          .value_counts().head(10).to_dict())

    print("\n=== 실험 조건 분포 (실험 수) ===")
    for n, x in [("train", tr), ("valid", va), ("test", te)]:
        ex = x.groupby(KEY)[COND].first()
        print(f"  {n:6s} tr_method={ex.tr_method.value_counts().to_dict()} "
              f"bat_class={ex.bat_class.value_counts().to_dict()} bat_cc={ex.bat_cc.value_counts().to_dict()} "
              f"soc={ex.soc_status.value_counts().to_dict()}")

    print("\n=== 단계별 경과시간(초) ===")
    print(df.groupby(Y)[T].describe()[["mean", "25%", "50%", "75%"]].round(0).to_string())

    print("\n=== 단계별 주요 센서 중앙값 ===")
    cols = ["meas_v", "meas_t1", "meas_t2", "thermal_t", "meas_p", "meas_co", "meas_co2", "meas_hf"]
    print(df.groupby(Y)[cols].median().round(2).to_string())

    print("\n=== tr_method × label (행) ===")
    print(pd.crosstab(df["tr_method"], df[Y]).to_string())


if __name__ == "__main__":
    main()
