"""2↔3단계 후처리 비교 (재학습 없음) — 최종 블렌드 OOF 위에서 Train 5-fold / Validation 홀드아웃 / 최저. 집계값만 출력.

    python tools/eval_23.py --stack_dir runs/stack_trv

(1) bias : p3에 배율 b를 곱한 뒤 argmax → cummax
(2) cusum: 단계는 내려가지 않으므로 2→3 전환을 실험마다 한 번만 정한다. 과거 행만 쓰는 온라인 방식:
           cummax 단계가 2 이상이 된 뒤부터 g_t = max(0, g_{t-1} + log(b·p3_t) − log(p2_t)),
           g_t > h가 처음 되는 시점부터 3 (그 뒤로 계속 3). cummax 단계가 2·3이 아닌 행은 cummax 그대로.
           미래 행은 쓰지 않는다 (실험 전체를 보고 전환점을 고르지 않음).
파라미터는 Train fold 점수로만 고른다 (★). Validation은 확인용.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import KEY, T  # noqa: E402
from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def cusum23(F, P, L, b, h):
    """F: 행 프레임, P: (n,6) 확률, L: cummax 라벨. 과거 행만 사용."""
    out = L.copy()
    d = np.log(np.clip(P[:, 2] * b, 1e-9, None)) - np.log(np.clip(P[:, 1], 1e-9, None))
    codes = pd.factorize(F[KEY].to_numpy())[0]
    order = np.lexsort((F[T].to_numpy(), codes))
    c_sorted = codes[order]
    starts = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        g, trig = 0.0, False
        for i in order[s:e]:
            if L[i] >= 2 and not trig:
                g = max(0.0, g + d[i])
                trig = g > h
            if L[i] in (2, 3):
                out[i] = 3 if trig else 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--members", nargs="+", default=["lgb", "dino", "seq_gru"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.4, 0.4, 0.2])
    ap.add_argument("--holdout_fold", type=int, default=5)
    a = ap.parse_args()
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in a.members})
    P = np.einsum("mnc,m->nc", Ps, np.asarray(a.weights))
    tr = (o.fold != a.holdout_fold).to_numpy()
    va = ~tr
    ot, ov = o[tr].reset_index(drop=True), o[va].reset_index(drop=True)
    Pt, Pv = P[tr], P[va]

    def both(fn):
        return score_fn(ot, fn(ot, Pt)), score_fn(ov, fn(ov, Pv))

    def bias_fn(b):
        def f(F, Q):
            Q = Q.copy()
            Q[:, 2] *= b
            return cummax_pp(F, CLASSES[Q.argmax(1)])
        return f

    def cusum_fn(b, h):
        def f(F, Q):
            return cusum23(F, Q, cummax_pp(F, CLASSES[Q.argmax(1)]), b, h)
        return f

    rows = []
    for b in [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4]:
        rows.append(("bias", f"b={b:g}", *both(bias_fn(b))))
    for b in [0.8, 1.0, 1.25]:
        for h in [0.5, 1, 2, 4, 8]:
            rows.append(("cusum", f"b={b:g} h={h:g}", *both(cusum_fn(b, h))))
            print(f"  cusum b={b:g} h={h:g} 완료", flush=True)
    base = [r for r in rows if r[0] == "bias" and r[1] == "b=1"][0]
    print(f"\n{'방식':<7s}{'파라미터':<14s}Train-fold  Validation  최저")
    for kind in ["bias", "cusum"]:
        part = [r for r in rows if r[0] == kind]
        best = max(part, key=lambda r: r[2])  # Train 점수로만 선택
        for r in part:
            mark = " ★" if r is best else ("  (현재 최종)" if r is base else "")
            print(f"{r[0]:<7s}{r[1]:<14s}{r[2]:.4f}      {r[3]:.4f}      {min(r[2], r[3]):.4f}{mark}")
    print("★ = Train fold 점수로 고른 값")


if __name__ == "__main__":
    main()
