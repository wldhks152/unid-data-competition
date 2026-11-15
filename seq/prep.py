"""시퀀스 입력 · fold 분할 · 스케일러. 모든 입력은 같은 실험의 현재·과거 행만으로 계산한다.

채널 구성 (행 t, 센서 x마다 3개):
  slog(x_t)                      현재값 (부호 유지 log1p — 가스 ppm처럼 범위가 큰 값 안정화)
  slog(x_t) - slog(x_0)          첫 관측 대비 변화
  (slog(x_t) - slog(x_{t-1}))·5/Δt   직전 관측 대비 변화율 (5초당) → 5/10/15초 간격에서 같은 의미
공통 채널: log1p(Δt), 경과시간(분), 실험 조건 4종
결측은 실험 안에서 forward-fill(과거값)만 하고, 남은 결측은 표준화 후 0(=학습 평균)으로 채운다.
표준화 통계는 해당 fold의 학습 실험(5초 원본)으로만 계산한다.
"""
import numpy as np
import pandas as pd

from src.data import COND, KEY, SENSORS, T, Y, subsample

N_PER_SENSOR = 3


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _cond_numeric(df):
    out = {}
    for c in COND:
        v = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
        if v.isna().all() and c in df.columns:  # 문자열 범주면 고정 사전 순서로 코드화
            v = pd.Series(pd.Categorical(df[c].astype(str), categories=sorted(df[c].astype(str).unique())).codes,
                          index=df.index, dtype=float)
        out[c] = np.log1p(v.astype(float).clip(lower=0)) if c == "bat_cc" else v.astype(float)
    return out


def channel_names(sens):
    names = []
    for c in sens:
        names += [f"{c}", f"{c}__d0", f"{c}__rate"]
    return names + ["log_dt", "elapsed_min"] + COND


EXTRA_SENS = ["meas_t1", "meas_t2", "meas_v", "meas_p", "thermal_t"]


def _extra_channels(t, g):
    """짧은·중간 창 기울기 채널 (extra=True일 때만): 핵심 센서 5개의 15초·60초 기울기(slog 공간, 5초당). 과거 행만 사용."""
    cols = []
    for c in EXTRA_SENS:
        x = slog(pd.to_numeric(g[c], errors="coerce").ffill().to_numpy(float)) if c in g.columns else np.full(len(t), np.nan)
        for w in (15, 60):
            j = np.searchsorted(t, t - w, side="left")
            dt = t - t[j]
            with np.errstate(invalid="ignore", divide="ignore"):
                cols.append(np.where(dt > 0, (x - x[j]) * 5.0 / np.where(dt > 0, dt, 1), 0.0))
    return np.stack(cols, 1).astype(np.float32)


def make_img(emb, n_comp=24, fit_ids=None):
    """이미지 채널 준비물. emb: sample_id 인덱스의 DINOv2 임베딩 DataFrame.
    PCA(평균·축)는 fit_ids(Train 이미지)로만 구한다 — 라벨을 쓰지 않는 전처리 통계."""
    E = emb.to_numpy(np.float32)
    ids = np.asarray(emb.index).astype(str)
    pos = {s: i for i, s in enumerate(ids)}
    sel = np.array([pos[str(s)] for s in fit_ids]) if fit_ids is not None else np.arange(len(E))
    X = E[sel].astype(np.float64)
    mean = X.mean(0)
    w, V = np.linalg.eigh(np.cov(X - mean, rowvar=False))
    W = V[:, np.argsort(w)[::-1][:n_comp]]
    return {"E": E, "pos": pos, "mean": mean, "W": W}


def _img_channels(g, t, img):
    """이미지 채널 (img가 있을 때만): 임베딩 PCA 성분의 현재값 · 첫 프레임 대비 변화 + 시간 변화 피처(src.rn18.emb_dynamics).
    모두 같은 실험의 현재·과거 프레임만 사용. 행렬곱은 einsum으로 행마다 따로 계산한다."""
    from src.rn18 import DYN_COLS, emb_dynamics
    E = img["E"][[img["pos"][str(s)] for s in g["sample_id"].to_numpy()]]
    Z = np.einsum("ij,jk->ik", E.astype(np.float64) - img["mean"], img["W"])
    dyn = emb_dynamics(pd.DataFrame({KEY: "e", T: t}), E)[list(DYN_COLS)].to_numpy(np.float64)
    return np.concatenate([Z, Z - Z[0], dyn], axis=1).astype(np.float32)


def build_sequences(df, extra=False, img=None):
    """df(여러 실험, 임의 간격) → 실험별 dict 리스트. extra=True면 _extra_channels를, img가 있으면 _img_channels를 뒤에 붙인다.
    각 dict: key, X (L, C) float32 (표준화 전), y (L,) 또는 None, sample_id (L,), t (L,), ver, grp"""
    df = df.sort_values([KEY, T], kind="stable")
    sens = [c for c in SENSORS if c in df.columns]
    conds = _cond_numeric(df)
    out = []
    for e, g in df.groupby(KEY, sort=False):
        t = g[T].to_numpy(float)
        R = g[sens].apply(pd.to_numeric, errors="coerce").ffill().to_numpy(float)  # 과거값으로만 채움
        S = slog(R)
        # ffill 했으므로 S는 '처음 관측 이전'에만 NaN → 첫 유효값 대비 변화도 그 이후 행에만 생긴다 (과거 정보)
        fi = np.argmax(~np.isnan(S), axis=0)
        d0 = S - S[fi, np.arange(S.shape[1])]
        dt = np.diff(t, prepend=t[0])
        rate = np.vstack([np.full((1, S.shape[1]), 0.0), np.diff(S, axis=0)]) * 5.0 / np.maximum(dt, 1e-6)[:, None]
        rate[0] = 0.0
        X = np.empty((len(t), N_PER_SENSOR * len(sens) + 2 + len(COND)), dtype=np.float32)
        X[:, 0:3 * len(sens):3] = S
        X[:, 1:3 * len(sens):3] = d0
        X[:, 2:3 * len(sens):3] = rate
        X[:, 3 * len(sens)] = np.log1p(dt)
        X[:, 3 * len(sens) + 1] = t / 60.0
        for i, c in enumerate(COND):
            X[:, 3 * len(sens) + 2 + i] = conds[c].loc[g.index].to_numpy()
        if extra:
            X = np.concatenate([X, _extra_channels(t, g)], axis=1)
        if img is not None:
            X = np.concatenate([X, _img_channels(g, t, img)], axis=1)
        out.append({
            "key": e, "X": X, "t": t,
            "y": g[Y].to_numpy(int) if Y in g.columns else None,
            "sample_id": g["sample_id"].to_numpy() if "sample_id" in g.columns else np.arange(len(g)),
            "ver": g["ver"].iloc[0] if "ver" in g.columns else "k1o0",
            "grp": g["grp"].iloc[0] if "grp" in g.columns else e,
        })
    return out


class Scaler:
    def fit(self, seqs):
        X = np.concatenate([s["X"] for s in seqs])
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd < 1e-6)] = 1.0
        self.mu[~np.isfinite(self.mu)] = 0.0
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(Z, -10, 10).astype(np.float32)

    def state(self):
        return {"mu": self.mu, "sd": self.sd}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.mu, s.sd = st["mu"], st["sd"]
        return s


def versions(tv, vers):
    """(k, offset) 목록 → subsample 결과를 합친 DataFrame (KEY에 __k{k}o{o} 접미사, grp/ver 칼럼)."""
    return pd.concat([subsample(tv, k, o) for k, o in vers], ignore_index=True)
