"""'재확인 후 상승' 후처리(src/late.py) 매개변수 비교 — 블렌드 OOF 위에서, 학습 없음, 집계값만 출력.

    python tools/eval_late.py --stack_dir runs/stack_trv --rows lgb=.4,dino=.4,seq_gru=.2

기준은 cummax. 각 설정마다 Train 5-fold / Validation / 최저(9:1:1 가중), cummax와 다른 행 수(5초),
실험 8개·11개를 무작위로 뽑았을 때 cummax를 이긴 비율(짝지은 비교)을 낸다. ★ = Train-fold 점수로만 고른 값.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness import VERS, weighted  # noqa: E402
from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.late import confirm_labels  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--rows", default="lgb=.4,dino=.4,seq_gru=.2")
    ap.add_argument("--holdout_fold", type=int, default=5)
    ap.add_argument("--kmin", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--strong", type=float, nargs="+", default=[0.6, 0.75, 0.9, 1.01])
    ap.add_argument("--need", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n_sim", type=int, default=3000)
    a = ap.parse_args()
    wts = dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in a.rows.split(","))
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in wts})
    P = np.einsum("mnc,m->nc", Ps, np.array([wts[m] for m in wts]))
    tr = (o.fold != a.holdout_fold).to_numpy()
    ot, ov = o[tr].reset_index(drop=True), o[~tr].reset_index(drop=True)
    grps = np.array(sorted(o.grp.unique()))
    gi, vi = pd.Index(grps).get_indexer(o.grp), pd.Index(VERS).get_indexer(o.ver)
    y = o.y.to_numpy().astype(int) - 1
    m5 = (o.ver == "k1o0").to_numpy()
    rng = np.random.default_rng(0)
    picks = {n: np.stack([rng.choice(len(grps), n, replace=False) for _ in range(a.n_sim)]) for n in (8, 11)}

    def cms(lab):
        cm = np.zeros((len(grps), len(VERS), 6, 6), np.int64)
        np.add.at(cm, (gi, vi, y, lab.astype(int) - 1), 1)
        return cm

    def row(name, lab, base=None):
        s1, s2 = score_fn(ot, lab[tr]), score_fn(ov, lab[~tr])
        r = {"name": name, "tr": s1, "va": s2, "lab": lab}
        if base is not None:
            cm, cb = cms(lab), base["cm"]
            r["chg"] = int((lab[m5] != base["lab"][m5]).sum())
            r["win"] = [((weighted(cm[p].sum(1)) > weighted(cb[p].sum(1))).mean(),
                         (weighted(cm[p].sum(1)) < weighted(cb[p].sum(1))).mean()) for p in picks.values()]
        else:
            r["cm"] = cms(lab)
        return r

    base = row("cummax (현재)", cummax_pp(o, CLASSES[P.argmax(1)]))
    rows = []
    for k in a.kmin:
        for s in a.strong:
            for n in a.need:
                rows.append(row(f"kmin={k} strong={s:g} need={n}", confirm_labels(o, P, None, strong=s, kmin=k, need=n), base))
    best = max(rows, key=lambda r: r["tr"])
    print(f"블렌드 {a.rows}   (5초 행 {int(m5.sum()):,})")
    print(f"{'설정':<28s}Train-fold  Validation  최저    바뀐 행  8개 승/패    11개 승/패")
    print(f"{base['name']:<28s}{base['tr']:.4f}      {base['va']:.4f}      {min(base['tr'], base['va']):.4f}")
    for r in rows:
        mark = " ★" if r is best else ("  ← 팀원 #34" if r["name"] == "kmin=4 strong=0.75 need=1" else "")
        w8, w11 = r["win"]
        print(f"{r['name']:<28s}{r['tr']:.4f}      {r['va']:.4f}      {min(r['tr'], r['va']):.4f}  {r['chg']:5d}   "
              f"{w8[0]:.0%}/{w8[1]:.0%}     {w11[0]:.0%}/{w11[1]:.0%}{mark}")
    print("★ = Train-fold 점수로 고른 값. 승/패 = 무작위 실험 묶음에서 cummax보다 높은/낮은 비율 (나머지는 동점).")


if __name__ == "__main__":
    main()
