"""단조 단계 구조를 이용한 인과(causal) 후처리 — 좌→우 HMM forward 필터.

EDA: label은 모든 실험에서 단조 증가(1→6), 일부 실험은 단계를 건너뜀(예: 3→5).
cummax(argmax)는 한 번 튄 예측이 영구히 고정되는 반면, forward 필터는 확률을 누적하므로
일시적인 오탐에 덜 민감하다. 각 시점의 출력은 '현재까지'의 관측만 사용한다 (smoothing 아님).

  alpha_t ∝ (alpha_{t-1} @ A^s) * (P_t / prior)^tau,   s = round(Δt / 5초)
"""
import numpy as np

from .data import KEY, T

STEP = 5.0


def estimate_transitions(y_seqs, K, fwd_smooth=0.5, back=0.0):
    """5초 간격 label 시퀀스(0-index)로 전이 행렬 A와 초기 분포를 추정.
    fwd_smooth: 앞으로 가는 전이(건너뛰기 포함)에 더하는 가상 카운트.
    back: 뒤로 가는 전이 확률(0이면 완전 단조). 오탐 복구 여지를 주는 안전장치."""
    C = np.zeros((K, K))
    init = np.full(K, 0.5)
    for y in y_seqs:
        init[y[0]] += 1
        np.add.at(C, (y[:-1], y[1:]), 1)
    A = np.triu(C, 0) + fwd_smooth * np.triu(np.ones((K, K)), 1)
    A[np.arange(K), np.arange(K)] += 1.0
    A /= A.sum(1, keepdims=True)
    if back > 0:
        for i in range(1, K):
            A[i] *= 1 - back
            A[i, :i] += back / i
    return A, init / init.sum()


def _filter_one(P, t, A, init, prior, tau, cache):
    E = (P / prior) ** tau
    out = np.empty_like(P)
    a = init.copy()
    for i in range(len(P)):
        if i > 0:
            s = max(1, int(round((t[i] - t[i - 1]) / STEP)))
            if s not in cache:
                cache[s] = np.linalg.matrix_power(A, s)
            a = a @ cache[s]
        a = a * E[i]
        z = a.sum()
        a = a / z if z > 0 else np.full(len(a), 1.0 / len(a))
        out[i] = a
    return out


def hmm_filter(df, P, params):
    """df(KEY, T 칼럼; 행 순서 = P 행 순서)의 실험별로 시간순 forward 필터를 적용한 확률을 반환."""
    A, init = np.asarray(params["A"]), np.asarray(params["init"])
    prior, tau = np.asarray(params["prior"]), float(params["tau"])
    tt = df[T].to_numpy(float)
    out = np.empty_like(P, dtype=float)
    cache = {}
    for ix in df.groupby(KEY, sort=False).indices.values():
        ix = ix[np.argsort(tt[ix], kind="stable")]
        out[ix] = _filter_one(P[ix], tt[ix], A, init, prior, tau, cache)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 인과 HSMM forward 필터 (단계 체류시간 사전분포 + 건너뛰기 허용)
#
# 상태 = (단계 k, 그 단계에 머문 5초 칸 수 d). 5초마다
#   - 확률 h_k(d)로 단계를 떠나 다음 단계 j>k 로 이동 (j는 nxt[k, j], 건너뛰기 포함), d=0
#   - 아니면 같은 단계에 머물고 d+1 (d가 D-1에 닿으면 그대로 → 그 뒤는 상수 위험률)
# h_k(d)는 Train label의 단계 체류시간(오른쪽 중도절단 포함)으로 이산 Kaplan-Meier 추정.
# 관측 간격이 10·15초면 관측 없이 전이만 2·3번 적용한다. 출력은 '현재까지' 관측만 쓴 filtered posterior.
# ─────────────────────────────────────────────────────────────────────────────

def stage_runs(y_seq, t_seq):
    """시간순 0-index label·시간 → [(단계, 체류 5초칸 수, 끝났는지)] 목록. 마지막 구간은 중도절단."""
    y_seq, t_seq = np.asarray(y_seq), np.asarray(t_seq, float)
    cut = np.flatnonzero(np.diff(y_seq) != 0) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(y_seq)]
    runs = []
    for s, e in zip(starts, ends):
        t_end = t_seq[e] if e < len(y_seq) else t_seq[-1] + STEP
        runs.append((int(y_seq[s]), max(1, int(round((t_end - t_seq[s]) / STEP))), e < len(y_seq)))
    return runs


def estimate_hsmm(seqs, K, dmax_s=600.0, shrink=10.0, skip_smooth=0.5, geometric=False):
    """seqs: [(0-index label 배열, 시간 배열)] (5초 원본 간격 권장).
    shrink: 경험 위험률을 상수(기하분포) 위험률 쪽으로 당기는 가상 표본 수. 클수록 HMM에 가깝다.
    geometric=True면 체류시간 사전분포 없이 상수 위험률(= 건너뛰기 허용 좌→우 HMM).
    반환: {"haz": (K, D), "nxt": (K, K), "init": (K,)}"""
    D = max(2, int(round(dmax_s / STEP)) + 1)
    ev, risk = np.zeros((K, D)), np.zeros((K, D))
    nx = np.zeros((K, K))
    init = np.full(K, 0.5)
    for y, t in seqs:
        runs = stage_runs(y, t)
        init[runs[0][0]] += 1
        for i, (k, n, done) in enumerate(runs):
            # 떠난 구간: 칸 0..n-1 에서 전이 관측 (n-1 칸에서 떠남). 중도절단: 칸 0..n-2 에서 머묾만 관측
            x = n if done else n - 1
            risk[k, :min(x, D)] += 1
            if x > D:  # 꼬리: D-1 칸(상한)에 남은 노출을 몰아 넣음
                risk[k, D - 1] += x - D
            if done:
                ev[k, min(n - 1, D - 1)] += 1
                nx[k, runs[i + 1][0]] += 1
    h0 = ev.sum(1) / np.maximum(risk.sum(1), 1e-9)  # 단계별 상수 위험률 (MLE)
    if geometric:
        haz = np.repeat(h0[:, None], D, 1)
    else:
        haz = (ev + shrink * h0[:, None]) / (risk + shrink)
        haz[:, D - 1] = (ev[:, D - 1] + shrink * h0) / (risk[:, D - 1] + shrink)
    haz = np.clip(haz, 1e-6, 1 - 1e-6)
    haz[K - 1] = 0.0  # 마지막 단계는 흡수 상태
    nxt = np.triu(nx, 1) + skip_smooth * np.triu(np.ones((K, K)), 1)
    nxt[K - 1] = 0.0
    nxt[:K - 1] /= nxt[:K - 1].sum(1, keepdims=True)
    return {"haz": haz, "nxt": nxt, "init": init / init.sum()}


def _hsmm_step(a, haz, nxt):
    """a: (E, K, D) → 5초 한 칸 전이 후."""
    leave = a * haz
    stay = a - leave
    out = np.zeros_like(a)
    out[:, :, 1:] = stay[:, :, :-1]
    out[:, :, -1] += stay[:, :, -1]
    out[:, :, 0] += leave.sum(2) @ nxt
    return out


def hsmm_filter(df, P, params, max_gap_steps=720):
    """df(KEY, T 칼럼; 행 순서 = P 행 순서)의 실험별 인과 forward 필터 → 단계 사후확률 (행 순서 그대로).
    params: estimate_hsmm 결과 + {"tau", "eps", "prior"}.
      emission ∝ ((1-eps)·P + eps/K) / prior) ** tau   (tau: 모델 확률을 얼마나 믿을지)"""
    haz, nxt, init = (np.asarray(params[k], float) for k in ("haz", "nxt", "init"))
    K, D = haz.shape
    tau, eps = float(params.get("tau", 1.0)), float(params.get("eps", 0.02))
    prior = np.asarray(params.get("prior", np.full(K, 1.0 / K)), float)
    P = np.asarray(P, float)
    Em = (((1 - eps) * P + eps / K) / prior) ** tau
    Em /= Em.sum(1, keepdims=True)

    tt = df[T].to_numpy(float)
    groups = [ix[np.argsort(tt[ix], kind="stable")] for ix in df.groupby(KEY, sort=False).indices.values()]
    E, L = len(groups), max(len(g) for g in groups)
    idx = np.full((E, L), -1)
    steps = np.zeros((E, L), int)
    for e, g in enumerate(groups):
        idx[e, :len(g)] = g
        if len(g) > 1:
            steps[e, 1:len(g)] = np.clip(np.rint(np.diff(tt[g]) / STEP), 1, max_gap_steps)

    a = np.zeros((E, K, D))
    a[:, :, 0] = init
    out = np.empty((len(P), K))
    for i in range(L):
        live = np.flatnonzero(idx[:, i] >= 0)
        if i > 0:
            s = steps[live, i]
            for r in range(1, s.max() + 1):
                m = live[s >= r]
                a[m] = _hsmm_step(a[m], haz, nxt)
        rows = idx[live, i]
        a[live] *= Em[rows][:, :, None]
        z = a[live].sum((1, 2), keepdims=True)
        a[live] /= np.where(z > 0, z, 1.0)
        out[rows] = a[live].sum(2)
    return out


def decode(post, df, how="argmax", w=None, thr=0.5):
    """filtered posterior → 0-index 단계. 모두 과거·현재만 사용.
    argmax: argmax(post·w) / cummax: 그 결과의 실험별 누적최대
    median: P(S ≥ k) ≥ thr 인 가장 큰 k 의 누적최대 (w 미사용)"""
    post = np.asarray(post)
    if how == "median":
        tail = np.cumsum(post[:, ::-1], 1)[:, ::-1]  # P(S >= k)
        pred = (tail >= thr).sum(1) - 1
        return cummax_rows(df, pred)
    w = np.ones(post.shape[1]) if w is None else np.asarray(w)
    pred = (post * w).argmax(1)
    return cummax_rows(df, pred) if how == "cummax" else pred


def cummax_rows(df, pred):
    """numpy 실험별 시간순 누적최대 (행 순서 유지). pred는 0 이상 정수."""
    pred = np.asarray(pred)
    g = df[KEY].astype("category").cat.codes.to_numpy().astype(np.int64)
    order = np.lexsort((df[T].to_numpy(float), g))
    big = int(pred.max()) + 1 if len(pred) else 1
    v = pred[order] + big * g[order]
    out = np.empty_like(pred)
    out[order] = np.maximum.accumulate(v) - big * g[order]
    return out
