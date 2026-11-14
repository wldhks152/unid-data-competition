"""2↔3 전용 모델(s23) 효과 표 — 학습 없이 OOF로 계산. 집계값만 출력.

    python tools/eval_refine.py --stack_dir runs/stack_trv --members lgb rn18s --weights 0.5 0.5
Train-fold(fold<holdout)와 Validation(fold==holdout)을 따로, cummax 후 점수와 5초 2·3단계 F1을 보여 준다.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score  # noqa: E402

from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.members import refine23  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--members", nargs="+", default=["lgb", "rn18s"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.5])
    ap.add_argument("--holdout_fold", type=int, default=5)
    a = ap.parse_args()
    names = a.members + ["s23"]
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in names})
    base = np.einsum("mnc,m->nc", Ps[:-1], np.asarray(a.weights))
    q = Ps[-1][:, 2]
    parts = [("Train-fold", (o.fold != a.holdout_fold).to_numpy()), ("Validation", (o.fold == a.holdout_fold).to_numpy())]
    print("반영비율   " + "   ".join(f"{n} 점수 / 2단계F1 / 3단계F1" for n, _ in parts))
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P = refine23(base, q, alpha)
        cells = []
        for _, m in parts:
            fr = o[m].reset_index(drop=True)
            lab = cummax_pp(fr, CLASSES[P[m].argmax(1)])
            m5 = (fr.ver == "k1o0").to_numpy()
            f = f1_score(fr.y.to_numpy()[m5], lab[m5], average=None, labels=CLASSES)
            cells.append(f"{score_fn(fr, lab):.4f} / {f[1]:.3f} / {f[2]:.3f}")
        print(f"  {alpha:.2f}     " + "        ".join(cells))


if __name__ == "__main__":
    main()
