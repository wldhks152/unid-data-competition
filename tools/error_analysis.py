"""조건 그룹별 오류 분석 (학습 없음, 집계값만 출력) — 최종 블렌드 OOF(Train 5-fold + Validation 홀드아웃, cummax 후).

    python tools/error_analysis.py --stack_dir runs/stack_trv --data_dir <battery_data_light>

1) 조건(tr_method, bat_class, soc_status, bat_cc) 값별 실험 수, 9:1:1 점수, 5초 단계별 F1
2) 가장 약한 그룹: 5초 혼동행렬과 오류가 몰리는 단계 쌍 상위 5개 (그 그룹 / 나머지 비교)
3) 정답 단계 구간의 센서 통계 그룹별 평균 (실험별로 구한 뒤 평균): 지속시간, dT1/dt, dT2/dt, 전압 변화, 압력 변화
4) lgb 피처 목록에 조건 열이 들어 있는지
실험 ID·행·개별 예측은 출력하지 않는다.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import COND, KEY, T, Y, load  # noqa: E402
from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def stage_f1(y, p):
    out = []
    for c in CLASSES:
        tp = ((y == c) & (p == c)).sum()
        den = (y == c).sum() + (p == c).sum()
        out.append(2 * tp / den if den else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--members", nargs="+", default=["lgb", "dino", "seq_gru"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.4, 0.4, 0.2])
    ap.add_argument("--min_exp", type=int, default=3, help="이보다 실험이 적은 그룹은 표에서 뺀다")
    ap.add_argument("--group", nargs="*", default=None,
                    help="[2][3]을 볼 그룹 (예: bat_cc=73 tr_method=2). 없으면 점수가 가장 낮은 그룹 하나")
    ap.add_argument("--skip_table", action="store_true", help="[1] 조건별 표를 건너뛴다")
    a = ap.parse_args()

    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in a.members})
    lab = cummax_pp(o, CLASSES[np.einsum("mnc,m->nc", Ps, np.asarray(a.weights)).argmax(1)])
    d = load(a.data_dir, ("train", "valid"))
    raw = pd.concat([d["train"], d["valid"]], ignore_index=True)
    cond = raw.groupby(KEY)[COND].first()
    for c in COND:
        o[c] = o.grp.map(cond[c]).to_numpy()
    m5 = (o.ver == "k1o0").to_numpy()
    y = o.y.to_numpy()

    if not a.skip_table:
        print(f"[1] 조건별 점수 (실험 {o.grp.nunique()}개, 9:1:1 가중 / 5초 단계별 F1)")
    worst = None
    for c in (COND if not a.skip_table else []):
        vals = o[c].dropna().unique()
        if len(vals) > 8:  # 연속값(용량 등)은 4분위로
            q = pd.qcut(o.groupby("grp")[c].first(), 4, duplicates="drop")
            o[c + "_bin"] = o.grp.map(q).astype(str)
            col = c + "_bin"
        else:
            col = c
        print(f"\n  {col}")
        print(f"  {'값':<22s} 실험수  점수    " + " ".join(f"F1_{k}" for k in CLASSES))
        for v in sorted(o[col].dropna().unique(), key=str):
            g = (o[col] == v).to_numpy()
            ne = o.grp[g].nunique()
            if ne < a.min_exp:
                continue
            fr = o[g].reset_index(drop=True)
            s = score_fn(fr, lab[g], require_all=False)
            f = stage_f1(y[g & m5], lab[g & m5])
            print(f"  {str(v)[:22]:<22s} {ne:5d}  {s:.4f}  " + " ".join("  -  " if np.isnan(x) else f"{x:.3f}" for x in f))
            if worst is None or s < worst[2]:
                worst = (col, v, s)

    targets = []
    for spec in (a.group or []):
        c, val = spec.split("=", 1)
        vals = o[c].dropna().unique()
        match = [x for x in vals if str(x) == val or (isinstance(x, (int, float, np.number)) and _num(val) == float(x))]
        assert match, f"{c}에 {val} 값이 없어요 (있는 값 {sorted(map(str, vals))})"
        targets.append((c, match[0]))
    if not targets:
        targets = [worst[:2]]
    for col, v in targets:
        analyze(o, raw, lab, y, m5, col, v)


def _num(x):
    try:
        return float(x)
    except ValueError:
        return None


def analyze(o, raw, lab, y, m5, col, v):
    g = (o[col] == v).to_numpy()
    s = score_fn(o[g].reset_index(drop=True), lab[g], require_all=False)
    print(f"\n[2] 그룹: {col} = {v} (실험 {o.grp[g].nunique()}개, 점수 {s:.4f}) — 5초 혼동행렬 (행=정답 1~6, 열=예측 1~6)")
    for name, mask in [("그 그룹", g), ("나머지", ~g)]:
        mm = mask & m5
        cm = pd.crosstab(pd.Categorical(y[mm], CLASSES), pd.Categorical(lab[mm], CLASSES), dropna=False).to_numpy()
        print(f"  {name} (행 {mm.sum():,})")
        for k, row in zip(CLASSES, cm):
            print(f"    {k}: " + " ".join(f"{x:6d}" for x in row))
        off = [(cm[i, j], i + 1, j + 1) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
        tot = max(1, sum(x for x, _, _ in off))
        print("    오류 상위: " + ", ".join(f"{i}→{j} {x}행({x / tot:.0%})" for x, i, j in sorted(off, reverse=True)[:5]))

    print(f"\n[3] 정답 단계 구간의 센서 통계 (실험별 계산 후 평균) — {col} = {v} vs 나머지")
    raw = raw.sort_values([KEY, T]).copy()
    if col.endswith("_bin"):
        gset = set(o.grp[g].unique())
        raw["_g"] = raw[KEY].isin(gset)
    else:
        raw["_g"] = raw[col] == v
    rows = []
    for (e, k), s_ in raw.groupby([KEY, Y], sort=False):
        t = s_[T].to_numpy(float)
        dur = t[-1] - t[0]
        r = {"grp": s_["_g"].iloc[0], "stage": k, "dur": dur}
        for c, nm in [("meas_t1", "dT1/dt"), ("meas_t2", "dT2/dt")]:
            x = s_[c].to_numpy(float)
            r[nm] = (x[-1] - x[0]) / dur if dur > 0 else np.nan
        for c, nm in [("meas_v", "ΔV"), ("meas_p", "ΔP"), ("meas_t1", "T1 진입값")]:
            x = s_[c].to_numpy(float)
            r[nm] = x[0] if nm == "T1 진입값" else x[-1] - x[0]
        rows.append(r)
    st = pd.DataFrame(rows)
    stat_cols = ["dur", "dT1/dt", "dT2/dt", "ΔV", "ΔP", "T1 진입값"]
    tab = st.groupby(["stage", "grp"])[stat_cols].mean().round(3)
    tab.index = tab.index.set_names(["단계", "그룹"])
    tab = tab.rename(index={True: "그 그룹", False: "나머지"}, level=1)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(tab.to_string())



if __name__ == "__main__":
    main()
