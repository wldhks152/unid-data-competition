"""피처 생성 — 모든 피처는 같은 실험의 '현재 및 과거' 관측만 사용한다.

* 누적 통계(cummax/cummin), 첫 관측 대비 변화, 과거 방향 시간창(초 단위)만 사용
* 시간창은 행 수가 아닌 '초' 단위 → 5/10/15초 간격에서 같은 의미를 갖는다
* causality_check()로 '중간에서 잘라도 과거 피처가 변하지 않음'을 검증
"""
import sys

import numpy as np
import pandas as pd

from .data import COND, KEY, SENSORS, T, Y

PEAK_COLS = ["meas_t1", "meas_t2", "thermal_t", "meas_p"]  # 최고점 이후 경과시간 대상
WINDOWS = [30, 60, 120]  # 초
META = ["sample_id", KEY, Y, "grp", "ver"]


def _group_feats(t, X):
    """t: (n,) 정렬된 시간, X: (n, k). 행 i의 출력은 0..i 행만 사용."""
    n, k = X.shape
    feats = {}
    feats["d0"] = X - X[0]
    with np.errstate(invalid="ignore"):
        cmax = np.fmax.accumulate(X, axis=0)
        cmin = np.fmin.accumulate(X, axis=0)
    feats["cmax"] = cmax
    feats["cmin"] = cmin
    feats["drop_from_max"] = cmax - X
    feats["rise_from_min"] = X - cmin

    valid = ~np.isnan(X)
    cs = np.vstack([np.zeros((1, k)), np.cumsum(np.where(valid, X, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, k)), np.cumsum(valid, axis=0)])
    i = np.arange(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in WINDOWS:
            j = np.searchsorted(t, t - w, side="left")  # 창 안의 가장 오래된(과거) 관측
            dt = (t - t[j])[:, None]
            feats[f"rate{w}"] = np.where(dt > 0, (X - X[j]) / np.where(dt > 0, dt, 1), np.nan)
            feats[f"dev{w}"] = X - (cs[i + 1] - cs[j]) / (cn[i + 1] - cn[j])
    return feats


def _peak_feats(t, X):
    n = len(t)
    with np.errstate(invalid="ignore"):
        is_new = X >= np.fmax.accumulate(X, axis=0)
    last = np.maximum.accumulate(np.where(is_new, np.arange(n)[:, None], 0), axis=0)
    return t[:, None] - t[last]


def make_features(df):
    """(실험, 시간) 순으로 정렬된 피처 DataFrame을 반환. sample_id로 원래 행과 매칭."""
    df = df.sort_values([KEY, T], kind="stable").reset_index(drop=True)
    sens = [c for c in SENSORS if c in df.columns]
    peak = [c for c in PEAK_COLS if c in df.columns]
    blocks = []
    for _, g in df.groupby(KEY, sort=False):
        t = g[T].to_numpy(float)
        X = g[sens].to_numpy(float)
        cols = {c: X[:, a] for a, c in enumerate(sens)}
        for name, M in _group_feats(t, X).items():
            for a, c in enumerate(sens):
                cols[f"{c}__{name}"] = M[:, a]
        P = _peak_feats(t, g[peak].to_numpy(float))
        for a, c in enumerate(peak):
            cols[f"{c}__t_since_peak"] = P[:, a]
        cols["t1_minus_t2"] = g["meas_t1"].to_numpy(float) - g["meas_t2"].to_numpy(float)
        cols["t1_minus_amb"] = g["meas_t1"].to_numpy(float) - g["atmp_t"].to_numpy(float)
        blocks.append(pd.DataFrame(cols, index=g.index))
    F = pd.concat(blocks).loc[df.index]
    meta = [c for c in META if c in df.columns] + [T] + COND
    return pd.concat([df[meta], F], axis=1)


def feature_columns(F):
    return [c for c in F.columns if c not in META]


def causality_check(df, n_exp=5, seed=0):
    """시계열을 임의 지점에서 잘라 다시 계산해도 앞부분 피처가 같은지 확인. 다르면 종료."""
    rng = np.random.default_rng(seed)
    exps = rng.choice(df[KEY].unique(), size=min(n_exp, df[KEY].nunique()), replace=False)
    for e in exps:
        g = df[df[KEY] == e].sort_values(T)
        cut = int(rng.integers(2, max(3, len(g))))
        full = make_features(g).set_index("sample_id")
        part = make_features(g.iloc[:cut]).set_index("sample_id")
        cols = feature_columns(part.reset_index())
        bad = [c for c in cols
               if not np.allclose(full.loc[part.index, c], part[c], equal_nan=True)]
        if bad:
            sys.exit(f"[누수!] 미래 정보에 의존하는 피처: {bad[:10]}")
    print(f"  피처 누수 검사 OK — {len(exps)}개 실험을 임의 지점에서 잘라도 과거 피처 동일")
