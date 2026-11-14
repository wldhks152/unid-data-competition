"""블렌드 견고성 검증 (학습 없음, 집계값만 출력) — Private처럼 '실험 몇 개만 뽑힌 상황'에서 점수가 얼마나 흔들리나.

    python tools/robustness.py --stack_dir runs/stack_trv --data_dir <battery_data_light>

1) Private 시뮬레이션: Train OOF + Validation 홀드아웃의 실험(grp)에서 n개를 무작위로 n_sim번 뽑아 9:1:1 점수 분포
   (평균, 표준편차, 하위 10%·5%). 모든 조합이 같은 표본을 쓴다 (짝지은 비교). 현재 최종 대비 이긴 비율도 출력.
   Macro F1은 표본 안의 정답·예측에 나온 단계만으로 계산 (sklearn 기본과 같음).
2) fold별 점수 (fold 0~4 = Train, fold 5 = Validation): 표준편차와 최악 fold.
3) 실험 조건별 점수 (tr_method, bat_class): 조건 값마다 실험 수와 점수. --data_dir가 있을 때만.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import VER_W  # noqa: E402
from src.evaluation import CLASSES, align_oofs, read_oof  # noqa: E402
from src.model import cummax_pp  # noqa: E402

COMBOS = {
    "최종 lgb.4+dino.4+gru.2": {"lgb": .4, "dino": .4, "seq_gru": .2},
    "lgb.5+v2.5": {"lgb": .5, "rn18s": .5},
    "lgb.3+dino.7": {"lgb": .3, "dino": .7},
    "lgb.5+dino.3+gru.2": {"lgb": .5, "dino": .3, "seq_gru": .2},
    "lgbx.5+dino.3+gru.2": {"lgbx": .5, "dino": .3, "seq_gru": .2},
}
VERS = list(VER_W)


def f1_from_cm(cm):
    """cm: (..., 6, 6). 정답·예측에 한 번이라도 나온 단계만 평균 (sklearn 기본 labels)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    rs, cs = cm.sum(-1), cm.sum(-2)
    den = rs + cs
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    present = den > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def weighted(cm_v):
    """cm_v: (..., 3, 6, 6) 버전별 혼동행렬 → 9:1:1 가중 점수."""
    w = np.array([VER_W[v] for v in VERS], float)
    return (f1_from_cm(cm_v) * w).sum(-1) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_exp", type=int, nargs="+", default=[8, 11])
    ap.add_argument("--n_sim", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    names = sorted({n for c in COMBOS.values() for n in c})
    have = [n for n in names if os.path.exists(os.path.join(a.stack_dir, f"oof_{n}.csv"))]
    combos = {k: v for k, v in COMBOS.items() if all(n in have for n in v)}
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in have})
    idx = {n: i for i, n in enumerate(have)}
    grps = np.array(sorted(o.grp.unique()))
    gi = pd.Index(grps).get_indexer(o.grp)
    vi = pd.Index(VERS).get_indexer(o.ver)
    y = o.y.to_numpy().astype(int) - 1
    fold_of_g = o.groupby("grp").fold.first().reindex(grps).to_numpy()
    print(f"실험 {len(grps)}개 (Train {int((fold_of_g < 5).sum())} + Validation {int((fold_of_g == 5).sum())}), 조합 {len(combos)}개")

    # 실험 × 버전별 혼동행렬 (cummax 후)
    CM = {}
    for k, wts in combos.items():
        w = np.zeros(len(have))
        for n, x in wts.items():
            w[idx[n]] = x
        lab = cummax_pp(o, CLASSES[np.einsum("mnc,m->nc", Ps, w).argmax(1)]).astype(int) - 1
        cm = np.zeros((len(grps), len(VERS), 6, 6), np.int64)
        np.add.at(cm, (gi, vi, y, lab), 1)
        CM[k] = cm

    rng = np.random.default_rng(a.seed)
    base = list(combos)[0]
    for n in a.n_exp:
        picks = np.stack([rng.choice(len(grps), n, replace=False) for _ in range(a.n_sim)])
        S = {k: weighted(cm[picks].sum(1)) for k, cm in CM.items()}
        print(f"\n[1] Private 시뮬레이션: 실험 {n}개 × {a.n_sim}회")
        print(f"  {'조합':<26s} 평균    표준편차  하위10%  하위5%   최종 대비 이긴 비율")
        for k, s in S.items():
            win = "" if k == base else f"{(s > S[base]).mean():.0%} (비김 {(s == S[base]).mean():.0%})"
            print(f"  {k:<26s} {s.mean():.4f}  {s.std():.4f}    {np.quantile(s, .1):.4f}   {np.quantile(s, .05):.4f}   {win}")

    print("\n[2] fold별 점수 (0~4 Train, 5 Validation)")
    folds = sorted(set(fold_of_g))
    print(f"  {'조합':<26s} " + " ".join(f"f{f}    " for f in folds) + " 표준편차  최악")
    for k, cm in CM.items():
        sc = np.array([weighted(cm[fold_of_g == f].sum(0)) for f in folds])
        print(f"  {k:<26s} " + " ".join(f"{x:.4f}" for x in sc) + f"  {sc.std():.4f}    {sc.min():.4f}")

    if a.data_dir:
        from src.data import load
        d = load(a.data_dir, ("train", "valid"))
        raw = pd.concat([d["train"], d["valid"]], ignore_index=True)
        key = "experiment_id"
        for c in ["tr_method", "bat_class"]:
            if c not in raw:
                continue
            cond = raw.groupby(key)[c].first()
            cg = pd.Series(grps).map(lambda g: cond.get(g, np.nan)).to_numpy()
            print(f"\n[3] 조건별 점수: {c}")
            vals = [v for v in pd.unique(cg) if pd.notna(v)]
            print(f"  {'조합':<26s} " + " ".join(f"{str(v)[:10]:>10s}" for v in vals))
            print(f"  {'(실험 수)':<26s} " + " ".join(f"{int((cg == v).sum()):>10d}" for v in vals))
            for k, cm in CM.items():
                print(f"  {k:<26s} " + " ".join(f"{weighted(cm[cg == v].sum(0)):>10.4f}" for v in vals))


if __name__ == "__main__":
    main()
