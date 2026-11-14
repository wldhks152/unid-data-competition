"""벤팅 기준 물리 피처 — 2↔3 단계 경계용 (research/bms_feature_ideas.md A1~A9 + 조건 정규화).

모든 피처는 같은 실험의 '현재 및 과거' 관측만 사용한다.
* 기준점(벤팅) = t1이 처음 45°C 이상이 된 시점. 이 데이터에서 1→2 전환과 100% 일치하는 규칙이라
  '2단계 시작'을 과거 정보만으로 정확히 잡는다. 벤팅 전 행에서는 벤팅 기준 피처가 전부 NaN.
* 벤팅 시점 값은 그 시점이 이미 지난 행에서만 쓴다 (과거 값 고정).
* 시간창은 초 단위 → 5/10/15초 간격에서 같은 의미. 누적량은 시간 적분(Δt 가중)이라 간격 무관.

그룹 (피처 이름 접두어 ph_<그룹>__):
  vent  : 벤팅 기준 상대 변화, 벤팅 후 경과시간, 진행률 (A4)
  dip   : 벤팅 후 냉각 딥과 회복, 기울기 부호 전환 (A1, A2)
  press : 벤팅 후 압력 정점과 해소 비율 (A3)
  self  : 히터 대비 자기발열, 내부-표면 역전 (A5, A6)
  acc   : 열 가속도(기울기의 변화) (A7)
  gas   : 가스별 첫 검출, 가스 비율, 벤팅 후 가스 적분 (A8, A9)
  norm  : 용량(bat_cc) 정규화 (조건 정규화)
"""
import numpy as np

GROUPS = ["vent", "dip", "press", "self", "acc", "gas", "norm"]
VENT_T1 = 45.0
GAS_DETECT = 1.0  # 가스 첫 검출: 첫 관측 대비 증가량 (기존 ev_gas1과 같은 기준)
GASES = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
         "meas_n2o", "meas_no", "meas_no2", "meas_so2"]


def parse_groups(physics):
    """None/""/"none" → [], "all" → 전체, "vent,dip" 또는 리스트 → 그 그룹들."""
    if physics is None or physics in ("", "none"):
        return []
    if isinstance(physics, str):
        physics = GROUPS if physics == "all" else [s.strip() for s in physics.split(",") if s.strip()]
    bad = [g for g in physics if g not in GROUPS]
    if bad:
        raise ValueError(f"알 수 없는 물리 피처 그룹 {bad}. 가능: {GROUPS}")
    return [g for g in GROUPS if g in physics]


def group_of(col):
    """피처 이름 → 그룹 (물리 피처가 아니면 None)."""
    return col.split("__")[0][3:] if col.startswith("ph_") else None


def _rate(t, x, w):
    """과거 w초 구간 기울기 (초당). 행 i는 t[i]-w 이후의 가장 오래된 관측과 비교."""
    j = np.searchsorted(t, t - w, side="left")
    dt = t - t[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dt > 0, (x - x[j]) / np.where(dt > 0, dt, 1), np.nan), j, dt


def _first(mask):
    """누적 발생 여부와 첫 발생 인덱스 (없으면 n)."""
    occ = np.maximum.accumulate(np.asarray(mask, dtype=bool)) if len(mask) else np.zeros(0, bool)
    return occ, (int(np.argmax(occ)) if occ.any() else len(mask))


def _since(t, first):
    """첫 발생 후 경과시간 (발생 전 NaN)."""
    idx = np.arange(len(t))
    return np.where(idx >= first, t - t[min(first, len(t) - 1)], np.nan)


def _last_new_extreme(t, xs, kind):
    """xs(벤팅 전 NaN)의 누적 최대/최소가 마지막으로 갱신된 뒤 경과시간."""
    acc = np.fmax.accumulate if kind == "max" else np.fmin.accumulate
    with np.errstate(invalid="ignore"):
        ext = acc(xs)
        is_new = (xs >= ext) if kind == "max" else (xs <= ext)
    last = np.maximum.accumulate(np.where(is_new, np.arange(len(t)), -1))
    return ext, np.where(last >= 0, t - t[np.maximum(last, 0)], np.nan)


def _cum_integral(t, x):
    """∫x dt (사다리꼴, 과거만). NaN 구간은 0으로."""
    x = np.nan_to_num(x, nan=0.0)
    seg = np.diff(t) * (x[1:] + x[:-1]) / 2
    return np.concatenate([[0.0], np.cumsum(seg)])


def physics_features(t, g, groups):
    """t: (n,) 정렬된 시간, g: 한 실험의 DataFrame(같은 순서). 행 i의 출력은 0..i 행만 사용."""
    groups = parse_groups(groups)
    if not groups:
        return {}
    n = len(t)
    nan = np.full(n, np.nan)
    col = lambda c: g[c].to_numpy(float) if c in g.columns else nan.copy()
    t1, t2, p, th, indu = col("meas_t1"), col("meas_t2"), col("meas_p"), col("thermal_t"), col("indu_t")
    cc = col("bat_cc")
    idx = np.arange(n)

    # 벤팅 기준점 (과거만): t1이 처음 45°C 이상
    with np.errstate(invalid="ignore"):
        occ, iv = _first(np.nan_to_num(t1, nan=-np.inf) >= VENT_T1)
    after = idx >= iv
    ivc = min(iv, n - 1)
    at_vent = lambda x: np.where(after, x[ivc], np.nan)
    since_vent = lambda x: np.where(after, x, np.nan)
    t_sv = _since(t, iv)
    t_vent = np.where(after, t[ivc] - t[0], np.nan)  # 시작→벤팅에 걸린 시간 (벤팅 후에만 알 수 있음)

    r30_t1, j30, dt30 = _rate(t, t1, 30)
    r60_t1, _, _ = _rate(t, t1, 60)
    r60_t2, _, _ = _rate(t, t2, 60)

    gas_cols = [c for c in GASES if c in g.columns]
    G = g[gas_cols].to_numpy(float) if gas_cols else np.full((n, 0), np.nan)
    Gd = np.clip(G - G[:1], 0, None) if gas_cols else G  # 첫 관측 대비 증가량 (음수는 0)
    gsum = np.nansum(Gd, axis=1) if gas_cols else nan.copy()

    out = {}
    if "vent" in groups:
        out["ph_vent__occ"] = occ.astype(float)
        out["ph_vent__t_since"] = t_sv
        out["ph_vent__t_vent"] = t_vent
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ph_vent__progress"] = t_sv / np.maximum(t_vent, 1.0)  # 벤팅까지 걸린 시간 대비 진행률
        for name, x in [("t1", t1), ("t2", t2), ("p", p), ("th", th), ("indu", indu), ("gas", gsum)]:
            out[f"ph_vent__{name}_rel"] = x - at_vent(x)
        out["ph_vent__r60_t1_rel"] = r60_t1 - at_vent(r60_t1)  # 벤팅 순간 대비 승온률 변화

    if "dip" in groups:
        for name, x, r in [("t1", t1, r30_t1), ("t2", t2, _rate(t, t2, 30)[0])]:
            xs = since_vent(x)
            mn, t_min = _last_new_extreme(t, xs, "min")
            mx, _ = _last_new_extreme(t, xs, "max")
            out[f"ph_dip__{name}_rebound"] = x - mn                 # 벤팅 후 최저점 대비 회복
            out[f"ph_dip__{name}_drop_from_vmax"] = mx - x          # 벤팅 후 최고점 대비 하락
            out[f"ph_dip__{name}_depth"] = at_vent(x) - mn          # 벤팅 시점 대비 딥 깊이
            out[f"ph_dip__{name}_t_since_min"] = t_min
            rs = since_vent(r)
            with np.errstate(invalid="ignore"):
                out[f"ph_dip__{name}_r30_min_sv"] = np.fmin.accumulate(rs)
                cool, ic = _first(after & (np.nan_to_num(r, nan=1.0) <= 0))      # 벤팅 후 첫 냉각(기울기≤0)
                _, ir = _first(cool & (np.nan_to_num(r, nan=-1.0) > 0))          # 그 뒤 첫 재상승
            out[f"ph_dip__{name}_cooled"] = cool.astype(float)
            out[f"ph_dip__{name}_t_since_cool"] = _since(t, ic)
            out[f"ph_dip__{name}_t_since_rewarm"] = _since(t, ir)

    if "press" in groups:
        ps = since_vent(p)
        pmax, t_pk = _last_new_extreme(t, ps, "max")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ph_press__max_sv_rel"] = pmax - at_vent(p)
            out["ph_press__relax_sv"] = np.clip((pmax - p) / (pmax - at_vent(p) + 1.0), -5, 5)
            out["ph_press__t_since_peak_sv"] = t_pk
            out["ph_press__peak_frac"] = t_pk / np.maximum(t_sv, 1.0)  # 벤팅 후 경과 중 정점 이후 비율
            gmax = np.fmax.accumulate(p)
            out["ph_press__relax_all"] = np.clip((gmax - p) / (gmax - p[0] + 1.0), -5, 5)
            r30_p, _, _ = _rate(t, p, 30)
            out["ph_press__r30_min_sv"] = np.fmin.accumulate(since_vent(r30_p))

    if "self" in groups:
        gap = indu - t1
        r60_in, _, _ = _rate(t, indu, 60)
        out["ph_self__gap_heater"] = gap
        out["ph_self__gap_heater_r60"] = _rate(t, gap, 60)[0]
        out["ph_self__excess_r60"] = r60_t1 - r60_in          # κ=1: 히터보다 빨리 오르면 양수
        # 벤팅 전 구간에서 dT1/dt ≈ κ·(히터−셀) 을 실험별 누적 최소제곱으로 추정 (과거만)
        ok = (~after) & np.isfinite(r60_t1) & np.isfinite(gap)
        srg = np.cumsum(np.where(ok, r60_t1 * gap, 0.0))
        sgg = np.cumsum(np.where(ok, gap * gap, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa = np.where(sgg > 1e-6, srg / np.where(sgg > 1e-6, sgg, 1), np.nan)
            out["ph_self__kappa"] = kappa
            out["ph_self__q_resid"] = r60_t1 - kappa * gap      # 히터로 설명 안 되는 승온 = 자기발열
            out["ph_self__rise_ratio"] = (t1 - t1[0]) / (np.abs(indu - indu[0]) + 1.0)
        d21 = t2 - t1
        out["ph_self__t2_minus_t1_r60"] = r60_t2 - r60_t1
        out["ph_self__t2_minus_t1_rel"] = d21 - at_vent(d21)
        with np.errstate(invalid="ignore"):
            _, ix = _first(np.nan_to_num(d21, nan=-np.inf) > 0)
        out["ph_self__t_since_t2_over_t1"] = _since(t, ix)

    if "acc" in groups:
        for name, x in [("t1", t1), ("t2", t2), ("th", th)]:
            r, j, dt = _rate(t, x, 30)
            with np.errstate(invalid="ignore", divide="ignore"):
                a = np.where(dt > 0, (r - r[j]) / np.where(dt > 0, dt, 1), np.nan)  # °C/s²
            out[f"ph_acc__{name}"] = a
            out[f"ph_acc__{name}_r30_r60"] = r - _rate(t, x, 60)[0]                    # 단기-장기 기울기
            last_nonpos = np.maximum.accumulate(np.where(np.nan_to_num(a, nan=-1) <= 0, idx, -1))
            start = np.maximum(last_nonpos + 1, iv)  # 벤팅 후 가속(a>0) 연속 지속시간
            dur = np.where(start <= idx, t - t[np.clip(start - 1, 0, n - 1)], 0.0)
            out[f"ph_acc__{name}_pos_dur_sv"] = np.where(after, dur, np.nan)

    if "gas" in groups:
        for c, k in zip(gas_cols, range(len(gas_cols))):
            _, ig = _first(np.nan_to_num(Gd[:, k], nan=0) > GAS_DETECT)
            out[f"ph_gas__{c[5:]}_t_since_detect"] = _since(t, ig)
        gi = {c: Gd[:, k] for k, c in enumerate(gas_cols)}
        z = np.zeros(n)
        co, co2 = gi.get("meas_co", nan), gi.get("meas_co2", nan)
        out["ph_gas__log_co_co2"] = np.log1p(co) - np.log1p(co2)
        out["ph_gas__acid_frac"] = (gi.get("meas_hf", z) + gi.get("meas_hcl", z)) / (gsum + 1.0)
        out["ph_gas__co_frac"] = co / (gsum + 1.0)
        out["ph_gas__log_sum"] = np.log1p(gsum)
        out["ph_gas__n_detected"] = (np.fmax.accumulate(np.nan_to_num(Gd, nan=0), axis=0) > GAS_DETECT).sum(1) \
            if gas_cols else nan.copy()
        integ = _cum_integral(t, gsum)
        out["ph_gas__integral_sv"] = np.where(after, integ - integ[ivc], np.nan)
        out["ph_gas__log_integral_sv"] = np.log1p(np.clip(out["ph_gas__integral_sv"], 0, None))
        r30_g, _, _ = _rate(t, gsum, 30)
        _, t_gpk = _last_new_extreme(t, since_vent(r30_g), "max")
        out["ph_gas__t_since_rate_peak_sv"] = t_gpk

    if "norm" in groups:
        s = np.sqrt(np.maximum(cc, 1e-3))
        out["ph_norm__t_since_vent_sqcc"] = t_sv / s
        out["ph_norm__t_since_vent_cc"] = t_sv / np.maximum(cc, 1e-3)
        out["ph_norm__t_vent_sqcc"] = t_vent / s
        out["ph_norm__t_sqcc"] = (t - t[0]) / s
        out["ph_norm__r60_t1_sqcc"] = r60_t1 * s
        out["ph_norm__t1_rel_sqcc"] = (t1 - at_vent(t1)) * s
    return out
