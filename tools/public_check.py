"""Public 점수 패턴이 '우연'으로 설명되는지 확인 (학습 없음, 집계값만 출력).

    python tools/public_check.py --stack_dir runs/stack_trv

Public에서 본 것: 이미지 CatBoost 단독(v2)이 최종 블렌드보다 +0.039 높고, lgb 단독은 최종보다 0.064 낮았다.
Test가 Train/Validation과 같은 분포라면, 우리 OOF(Train 5-fold + Validation 홀드아웃)의 실험에서 n개를 무작위로 뽑았을 때도
이런 차이가 가끔 나와야 한다. 그 빈도를 센다 (5초 간격 행만, 뽑힌 실험의 행을 합쳐 Macro F1).
  [1] Public 크기(n=3,4,5)에서: P(alt − final ≥ gap_alt), P(final − weak ≥ gap_weak), 둘 다일 확률
  [2] Private 크기(n=7,8)에서: alt가 final을 이기는 비율, 차이의 평균과 5%·95%, 크게 지는(−0.05 이하) 비율
빈도가 아주 낮으면 '분포가 다르다'는 쪽, 흔하면 'Public은 잡음'이라는 쪽의 근거다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import CLASSES, align_oofs, read_oof  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def parse(r):
    return dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in r.split(","))


def f1_from_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    den = cm.sum(-1) + cm.sum(-2)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    present = den > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--final", default="lgb=.4,dino=.4,seq_gru=.2")
    ap.add_argument("--alt", default="rn18s=1", help="Public에서 더 높았던 모델 (팀원 v2에 해당)")
    ap.add_argument("--weak", default="lgb=1", help="Public에서 낮았던 모델")
    ap.add_argument("--alt_raw", action="store_true", help="alt는 cummax 없이 (팀원 제출과 같게)")
    ap.add_argument("--gap_alt", type=float, default=0.039)
    ap.add_argument("--gap_weak", type=float, default=0.064)
    ap.add_argument("--n_public", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--n_private", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--n_sim", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    combos = {"final": parse(a.final), "alt": parse(a.alt), "weak": parse(a.weak)}
    names = sorted({n for c in combos.values() for n in c})
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in names})
    idx = {n: i for i, n in enumerate(names)}
    m5 = (o.ver == "k1o0").to_numpy()
    grps = np.array(sorted(o.grp[m5].unique()))
    gi = pd.Index(grps).get_indexer(o.grp[m5])
    y = o.y.to_numpy().astype(int)[m5] - 1
    CM = {}
    for k, wts in combos.items():
        w = np.zeros(len(names))
        for n, x in wts.items():
            w[idx[n]] = x
        lab = CLASSES[np.einsum("mnc,m->nc", Ps, w).argmax(1)]
        if not (k == "alt" and a.alt_raw):
            lab = cummax_pp(o, lab)
        cm = np.zeros((len(grps), 6, 6), np.int64)
        np.add.at(cm, (gi, y, lab.astype(int)[m5] - 1), 1)
        CM[k] = cm
    print(f"실험 {len(grps)}개 (5초 행만). final={a.final}  alt={a.alt}{' (raw)' if a.alt_raw else ''}  weak={a.weak}")
    print("전체 실험 점수: " + "  ".join(f"{k} {f1_from_cm(cm.sum(0)):.4f}" for k, cm in CM.items()))

    rng = np.random.default_rng(a.seed)
    print(f"\n[1] Public 크기에서 본 패턴이 나올 빈도 ({a.n_sim}회)")
    print(f"  n   P(alt−final ≥ {a.gap_alt})  P(final−weak ≥ {a.gap_weak})  둘 다     alt−final 평균 [5%, 95%]")
    for n in a.n_public:
        picks = np.stack([rng.choice(len(grps), n, replace=False) for _ in range(a.n_sim)])
        S = {k: f1_from_cm(cm[picks].sum(1)) for k, cm in CM.items()}
        d1, d2 = S["alt"] - S["final"], S["final"] - S["weak"]
        e1, e2 = d1 >= a.gap_alt, d2 >= a.gap_weak
        print(f"  {n}   {e1.mean():>8.1%}               {e2.mean():>8.1%}               {(e1 & e2).mean():>6.1%}   "
              f"{d1.mean():+.4f} [{np.quantile(d1, .05):+.4f}, {np.quantile(d1, .95):+.4f}]")

    print(f"\n[2] Private 크기에서 alt 대 final ({a.n_sim}회)")
    print("  n   alt가 이긴 비율  차이 평균   [5%, 95%]            alt가 0.05 이상 짐  alt가 0.02 이상 이김")
    for n in a.n_private:
        picks = np.stack([rng.choice(len(grps), n, replace=False) for _ in range(a.n_sim)])
        S = {k: f1_from_cm(cm[picks].sum(1)) for k, cm in CM.items()}
        d = S["alt"] - S["final"]
        print(f"  {n}   {(d > 0).mean():>8.1%}       {d.mean():+.4f}   [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]   "
              f"{(d <= -0.05).mean():>8.1%}            {(d >= 0.02).mean():>8.1%}")


if __name__ == "__main__":
    main()
