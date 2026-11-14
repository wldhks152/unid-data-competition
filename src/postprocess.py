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
