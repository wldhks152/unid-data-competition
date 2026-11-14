"""피처 생성 — 모든 피처는 같은 실험의 '현재 및 과거' 관측만 사용한다.

* 누적 통계(cummax/cummin), 첫 관측 대비 변화, 과거 방향 시간창(초 단위)만 사용
* 시간창은 행 수가 아닌 '초' 단위 → 5/10/15초 간격에서 같은 의미를 갖는다
* causality_check()로 '중간에서 잘라도 과거 피처가 변하지 않음'을 검증
"""
import sys

import numpy as np
import pandas as pd

from .data import COND, KEY, SENSORS, T, Y
from .images import IMG_COLS

PEAK_COLS = ["meas_t1", "meas_t2", "thermal_t", "meas_p", "meas_v"]  # 최고점 이후 경과시간 대상
WINDOWS = [30, 60, 120]  # 초
META = ["sample_id", KEY, Y, "grp", "ver"]
GASES = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
         "meas_n2o", "meas_no", "meas_no2", "meas_so2"]

# 열폭주 '사건' 정의 — 단계별 센서 중앙값(EDA)과 열폭주 문헌의 단계 구분에 근거한 고정 임계값.
#   1 정상 → 2 벤팅(압력 급등) → 3 가열 지속 → 4 내부단락(전압 0V) → 5 열화상 온도 급락 → 6 열폭주(고온)
# 각 사건에 대해 '지금까지 발생했나'와 '처음 발생 후 경과시간'을 만든다 (과거 정보만 사용).
# 단계 전환 규칙 분석(train+valid, '처음 임계값 도달 시점 = 전환 시점' 일치율)으로 찾은 값을 포함:
#   1→2: t1 ≥ 45°C (100%) · 3→4: V ≤ 2.45 / 최고전압 대비 하락 ≥ 0.78 (가열 96~97%), ≥ 1.0 (과충전 92%)
#   4→5(과충전): t1 ≥ 146.5 (92%) · 5→6(과충전): t2 ≥ 242 (54%)
EVENTS = (
    [(f"p_jump{x}", "p_d0", ">", x) for x in (5, 10, 20, 35)]          # 벤팅: 압력 급등
    + [(f"v_below{x}", "v", "<", x) for x in (3.9, 3.5, 3.08, 2.45, 2.0, 0.91, 0.5)]  # 전압 저하 → 내부단락
    + [(f"v_drop{x}", "v_dropmax", ">", x) for x in (0.2, 0.5, 0.78, 1.0, 1.5, 2.45, 5.0)]  # 최고 전압 대비 하락
    + [(f"v_over{x}", "v", ">", x) for x in (4.3, 4.6, 5.0)]           # 과충전: 정격 초과
    + [(f"t1_above{x}", "t1", ">=", x) for x in (45, 50, 80, 110, 146.5, 250)]
    + [(f"t2_above{x}", "t2", ">=", x) for x in (60, 66.4, 90, 122, 135.7, 200, 242)]
    + [(f"t1_rate{x}", "t1_rate30", ">", x) for x in (0.2, 0.5, 1.0, 3.0)]  # °C/s, 열폭주 트리거
    + [(f"th_drop{x}", "th_dropmax", ">", x) for x in (15, 30, 60)]   # 열화상 온도 급락 (5단계)
    + [(f"gas{x}", "gas_d0", ">", x) for x in (1, 5, 20)]              # 가스 첫 검출
)


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


def _event_feats(t, g):
    """EVENTS 각각에 대해 발생 여부(0/1)와 첫 발생 후 경과시간(미발생이면 NaN). 과거만 사용."""
    col = lambda c: g[c].to_numpy(float) if c in g.columns else np.full(len(t), np.nan)
    v, t1, t2, p, th = col("meas_v"), col("meas_t1"), col("meas_t2"), col("meas_p"), col("thermal_t")
    gas = g[[c for c in GASES if c in g.columns]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        j = np.searchsorted(t, t - 30, side="left")
        dt = t - t[j]
        sig = {
            "v": v, "t1": t1, "t2": t2,
            "p_d0": p - p[0],
            "v_dropmax": np.fmax.accumulate(v) - v,
            "th_dropmax": np.fmax.accumulate(th) - th,
            "t1_rate30": np.where(dt > 0, (t1 - t1[j]) / np.where(dt > 0, dt, 1), np.nan),
            "gas_d0": np.nansum(gas - gas[0], axis=1) if gas.size else np.full(len(t), np.nan),
        }
    out, n_occ = {}, np.zeros(len(t))
    idx = np.arange(len(t))
    for name, s, op, x in EVENTS:
        with np.errstate(invalid="ignore"):
            hit = {">": sig[s] > x, ">=": sig[s] >= x, "<": sig[s] < x}[op]
        occ = np.maximum.accumulate(np.nan_to_num(hit, nan=0).astype(bool))
        first = np.where(occ.any(), np.argmax(occ), len(t))
        out[f"ev_{name}"] = occ.astype(float)
        out[f"ev_{name}__t_since"] = np.where(idx >= first, t - t[min(first, len(t) - 1)], np.nan)
        n_occ += occ
    out["ev_count"] = n_occ
    out["th_minus_t1"] = th - t1  # 5단계: 열화상 온도가 표면 온도보다 크게 낮아짐
    return out


X_IMG = ["img_pt_max", "img_pt_p99", "img_pt_p90", "img_pt_mean", "img_frac_red", "img_frac_white", "img_frac_yellow"]
X_VOL = ["meas_p", "meas_t1", "meas_t2", "meas_v", "thermal_t"]


def _extra_feats(t, g):
    """추가 피처(x_ 접두사, lgbx 멤버만 사용). 모두 현재·과거 행만 사용.
    1) 이미지 동역학: 이미지 통계의 30/60/120초 기울기, 누적 최대, 최대 대비 하락, 최대 갱신 후 경과시간
    2) 변동성: 최근 30/60초 표준편차 (벤팅 직후 압력·온도 요동 vs 3단계의 꾸준한 승온 구분)
    3) 전압-온도 결합: 60초 dV/dt ÷ dT1/dt, 시작 대비 전압 변화 ÷ 온도 변화"""
    out = {}
    n = len(t)
    idx = np.arange(n)
    imgs = [c for c in X_IMG if c in g.columns]
    if imgs:
        I = g[imgs].to_numpy(float)
        F = _group_feats(t, I)
        P = _peak_feats(t, I)
        for a, c in enumerate(imgs):
            for w in WINDOWS:
                out[f"x_img__{c}__rate{w}"] = F[f"rate{w}"][:, a]
            out[f"x_img__{c}__cmax"] = F["cmax"][:, a]
            out[f"x_img__{c}__drop_from_max"] = F["drop_from_max"][:, a]
            out[f"x_img__{c}__t_since_peak"] = P[:, a]
        if "img_pt_max" in imgs and "meas_t1" in g.columns:
            t1 = g["meas_t1"].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["x_img__ptmax_d0_per_t1_d0"] = (I[:, imgs.index("img_pt_max")] - I[0, imgs.index("img_pt_max")]) / (np.abs(t1 - t1[0]) + 1.0)
    vol = [c for c in X_VOL if c in g.columns]
    if vol:
        X = g[vol].to_numpy(float)
        Xz = np.where(np.isnan(X), 0.0, X)
        valid = (~np.isnan(X)).astype(float)
        cs = np.vstack([np.zeros((1, len(vol))), np.cumsum(Xz, 0)])
        cs2 = np.vstack([np.zeros((1, len(vol))), np.cumsum(Xz ** 2, 0)])
        cn = np.vstack([np.zeros((1, len(vol))), np.cumsum(valid, 0)])
        for w in (30, 60):
            j = np.searchsorted(t, t - w, side="left")
            with np.errstate(invalid="ignore", divide="ignore"):
                m = (cs[idx + 1] - cs[j]) / (cn[idx + 1] - cn[j])
                v2 = (cs2[idx + 1] - cs2[j]) / (cn[idx + 1] - cn[j]) - m ** 2
            sd = np.sqrt(np.clip(v2, 0, None))
            for a, c in enumerate(vol):
                out[f"x_vol__{c}__std{w}"] = sd[:, a]
    if "meas_v" in g.columns and "meas_t1" in g.columns:
        v, t1 = g["meas_v"].to_numpy(float), g["meas_t1"].to_numpy(float)
        j = np.searchsorted(t, t - 60, side="left")
        dt = t - t[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            rv = np.where(dt > 0, (v - v[j]) / np.where(dt > 0, dt, 1), np.nan)
            rt = np.where(dt > 0, (t1 - t1[j]) / np.where(dt > 0, dt, 1), np.nan)
            out["x_vt__dv_per_dt1_60"] = np.clip(rv / (np.abs(rt) + 0.01), -50, 50)
            out["x_vt__v_d0_per_t1_d0"] = (v - v[0]) / (np.abs(t1 - t1[0]) + 1.0)
    return out


F_SENS = ["meas_t1", "meas_t2", "meas_v", "meas_p", "thermal_t", "indu_t"]


def _fast_feats(t, g):
    """짧은 창 피처(f_ 접두사, lgbf 멤버만 사용). 39Ah 셀처럼 4·5단계가 수십 초밖에 안 되는 실험에서
    30초 이상 창의 기울기는 급변을 늦게 잡는다. 직전 관측 대비 변화율과 15초 기울기를 더한다. 과거 행만 사용."""
    out = {}
    n = len(t)
    jp = np.r_[0, np.arange(n - 1)]  # 직전 관측
    j15 = np.searchsorted(t, t - 15, side="left")
    for c in F_SENS:
        if c not in g.columns:
            continue
        x = g[c].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            for name, j in [("rate_last", jp), ("rate15", j15)]:
                dt = t - t[j]
                out[f"f_{c}__{name}"] = np.where(dt > 0, (x - x[j]) / np.where(dt > 0, dt, 1), np.nan)
    return out


def _since_vent_feats(t, g):
    """벤팅(t1 ≥ 45°C 첫 도달) 기준 전압·압력 변화 (s_ 접두사, 과거만). 오류 분석에서 과충전(tr_method=2)은
    2·3단계에 전압이 크게 오르고(ΔV +2.5, +7.5) 73Ah는 3단계 압력 상승이 커서, 벤팅 이후 누적 변화와 그 속도를 넣는다."""
    out = {}
    if "meas_t1" not in g.columns:
        return out
    t1 = g["meas_t1"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        hit = np.maximum.accumulate(np.nan_to_num(t1 >= 45, nan=0).astype(bool))
    iv = int(np.argmax(hit)) if hit.any() else len(t)
    after = np.arange(len(t)) >= iv
    tsv = np.where(after, t - t[min(iv, len(t) - 1)], np.nan)
    for c, nm in [("meas_v", "v"), ("meas_p", "p")]:
        if c not in g.columns:
            continue
        x = g[c].to_numpy(float)
        x0 = x[iv] if iv < len(t) else np.nan
        d = np.where(after, x - x0, np.nan)
        out[f"s_{nm}_since_vent"] = d
        out[f"s_{nm}_since_vent_rate"] = d / np.maximum(tsv, 5.0)
        with np.errstate(invalid="ignore"):
            out[f"s_{nm}_since_vent_max"] = np.where(after, np.fmax.accumulate(np.where(after, d, -np.inf)), np.nan)
    return out


Z_SENS = ["meas_t1", "meas_t2", "meas_v", "meas_p", "thermal_t"]


def _z_feats(t, g):
    """복합 이탈 피처 (z_ 접두사, 과거만). 그룹마다 2→3 신호가 온도·전압·압력으로 달라서, 개별 신호 대신
    "어느 신호든 벤팅 이후 자기 기준선에서 벗어났나"를 잡는다.
    - 신호별 z: 벤팅(t1 ≥ 45°C 첫 도달) 이후 지금까지의 평균·표준편차(누적, 과거만) 기준 현재값 z와 30초 기울기 z
    - 복합: 신호별 |z|의 최대·합, 최대 z가 2·3을 처음 넘은 뒤 경과시간, 최대 z 초과분의 누적 적분,
            가장 먼저 2를 넘은 신호 번호(-1 = 아직 없음)"""
    out = {}
    n = len(t)
    if "meas_t1" not in g.columns or n == 0:
        return out
    t1 = g["meas_t1"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        hit = np.maximum.accumulate(np.nan_to_num(t1 >= 45, nan=0).astype(bool))
    iv = int(np.argmax(hit)) if hit.any() else n
    after = np.arange(n) >= iv
    j30 = np.searchsorted(t, t - 30, side="left")
    dt30 = t - t[j30]
    Zl, Zr = [], []
    for c in Z_SENS:
        x = g[c].to_numpy(float) if c in g.columns else np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(dt30 > 0, (x - x[j30]) / np.where(dt30 > 0, dt30, 1), np.nan)
        for arr, store in [(x, Zl), (r, Zr)]:
            v = np.where(after & np.isfinite(arr), arr, 0.0)
            k = np.cumsum(after & np.isfinite(arr))
            s1, s2 = np.cumsum(v), np.cumsum(v * v)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = s1 / k
                sd = np.sqrt(np.clip(s2 / k - mu * mu, 0, None))
                z = np.where(after & (k >= 3), (arr - mu) / (sd + 1e-3 * (np.abs(mu) + 1)), np.nan)
            store.append(np.clip(z, -20, 20))
    Zl, Zr = np.stack(Zl, 1), np.stack(Zr, 1)
    for a, c in enumerate(Z_SENS):
        out[f"z_{c}__lvl"] = Zl[:, a]
        out[f"z_{c}__rate"] = Zr[:, a]
    with np.errstate(invalid="ignore"):
        A = np.abs(np.concatenate([Zl, Zr], 1))
        zmax = np.where(after, np.nanmax(np.where(np.isfinite(A), A, -1), 1), np.nan)
        zmax = np.where(zmax < 0, np.nan, zmax)
        out["z_max"] = zmax
        out["z_sum"] = np.where(after, np.nansum(A, 1), np.nan)
        out["z_max_rate"] = np.where(after, np.nanmax(np.where(np.isfinite(np.abs(Zr)), np.abs(Zr), -1), 1), np.nan)
        dt = np.diff(t, prepend=t[0])
        out["z_excess_int"] = np.cumsum(np.nan_to_num(np.clip(zmax - 1, 0, None)) * dt)
        for th in (2, 3):
            occ = np.maximum.accumulate(np.nan_to_num(zmax > th, nan=0).astype(bool))
            first = int(np.argmax(occ)) if occ.any() else n
            out[f"z_t_since_max{th}"] = np.where(np.arange(n) >= first, t - t[min(first, n - 1)], np.nan)
        firsts = []
        for a in range(len(Z_SENS)):
            col = np.nan_to_num(np.maximum(np.abs(Zl[:, a]), np.abs(Zr[:, a])) > 2, nan=0).astype(bool)
            firsts.append(int(np.argmax(col)) if col.any() else n)
        firsts = np.array(firsts)
        lead = np.full(n, -1.0)
        order = np.argsort(firsts, kind="stable")
        if firsts[order[0]] < n:
            lead[firsts[order[0]]:] = order[0]
        out["z_first_signal"] = lead
    return out


def make_features(df):
    """(실험, 시간) 순으로 정렬된 피처 DataFrame을 반환. sample_id로 원래 행과 매칭."""
    df = df.sort_values([KEY, T], kind="stable").reset_index(drop=True)
    sens = [c for c in SENSORS if c in df.columns]
    imgs = [c for c in IMG_COLS if c in df.columns]  # 이미지는 가볍게: 현재값 + 첫 프레임 대비 변화만
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
        cols.update(_event_feats(t, g))
        cols.update(_extra_feats(t, g))
        cols.update(_fast_feats(t, g))
        cols.update(_since_vent_feats(t, g))
        cols.update(_z_feats(t, g))
        if imgs:
            I = g[imgs].to_numpy(float)
            for a, c in enumerate(imgs):
                cols[c] = I[:, a]
                cols[f"{c}__d0"] = I[:, a] - I[0, a]
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
