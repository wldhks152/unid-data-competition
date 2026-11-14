"""'재확인 후 상승' 후처리 — 팀원 제출 #34(team27_late)의 방식을 매개변수로 일반화한 것. 과거 행만 사용.

행 i에서 단계 k 이상으로 올라가려면 (k ≥ kmin일 때만 적용)
  * 꼬리 확률 P(단계 ≥ k) ≥ strong 이거나
  * 직전 need개 관측(간격 gap초 이내)의 argmax도 모두 k 이상이어야 한다.
kmin 미만 단계와 실험의 첫 관측은 argmax 그대로. 마지막에 실험별 누적 최대(cummax).
팀원 #34 = kmin 4, strong 0.75, need 1, gap 15.

bundle["post"] = {"method": "<POSTS 이름>", "module": "src.late", ...} 로 test.py에서 쓴다.
"""
import numpy as np
import pandas as pd

from .data import KEY, T

CLASSES = np.arange(1, 7)


def confirm_labels(F, P, w=None, strong=0.75, gap=15.0, kmin=4, need=1):
    """F: 행 프레임(KEY, T), P: (n, 6) 확률. F 행 순서의 라벨(1~6)을 돌려준다."""
    P = np.asarray(P, dtype=float)
    if w is not None:
        P = P * np.asarray(w, dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    codes = pd.factorize(F[KEY].to_numpy())[0]
    t = pd.to_numeric(F[T], errors="coerce").to_numpy(float)
    order = np.lexsort((t, codes))
    c, tt, p = codes[order], t[order], P[order]
    raw = CLASSES[p.argmax(axis=1)]
    sup = raw[:, None] >= CLASSES[None, :]
    tails = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    first = np.r_[True, c[1:] != c[:-1]]
    ok = np.zeros(len(c), bool)  # 직전 관측이 같은 실험이고 gap초 이내
    ok[1:] = (~first[1:]) & (np.diff(tt) > 0) & (np.diff(tt) <= gap)
    prev = np.ones_like(sup) if need > 0 else np.zeros_like(sup)
    link = np.ones(len(c), bool)
    for j in range(1, need + 1):
        link = link & np.r_[np.zeros(j, bool), ok[j:] if j == 1 else np.ones(len(c) - j, bool)]
        if j > 1:  # i-1 … i-j 사이가 모두 이어져 있어야 한다
            link = link & np.r_[np.zeros(j, bool), ok[1:len(c) - j + 1]]
        prev = prev & np.r_[np.zeros((j, sup.shape[1]), bool), sup[:-j]] & link[:, None]
    ready = sup & ((tails >= strong) | prev)
    ready[:, :kmin - 1] = sup[:, :kmin - 1]
    ready[first] = sup[first]
    lab = (ready * CLASSES).max(axis=1)
    off = c.astype(np.int64) * 1000
    lab = np.maximum.accumulate(lab + off) - off
    out = np.empty(len(c), dtype=np.int64)
    out[order] = lab
    return out


def _preset(**kw):
    return lambda F, P, w=None: confirm_labels(F, P, w, **kw)


POSTS = {
    "late": _preset(kmin=4, strong=0.75, need=1),          # 팀원 #34와 같은 규칙
    "late_k3": _preset(kmin=3, strong=0.75, need=1),
    "late_k2": _preset(kmin=2, strong=0.75, need=1),
    "late_k4_s90": _preset(kmin=4, strong=0.90, need=1),
    "late_k4_s60": _preset(kmin=4, strong=0.60, need=1),
    "late_k4_n2": _preset(kmin=4, strong=0.75, need=2),
    "late_k5": _preset(kmin=5, strong=0.75, need=1),
}
