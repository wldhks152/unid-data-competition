"""블렌드 비교표 — 가중 평균 조합과 메타 모델 스태킹을 Train 5-fold / Validation 홀드아웃 / 최저로 비교한다. 집계값만 출력.

    python tools/compare_blends.py --rows lgb=.4,dino=.4,seq_gru=.2 lgb=.4,r50=.4,seq_gru=.2 \\
                                   --meta lgb,dino,seq_gru lgb,dino,seq_gru,xgb

* 가중 평균: 멤버 확률 × 가중치 → argmax → cummax.
* 메타(--meta): 멤버 확률의 log를 이어 붙여 다항 로지스틱 회귀(클래스 균형). 누수 없이 평가한다:
  Train fold f 점수 = fold f를 뺀 Train fold OOF로 학습한 메타 모델로 fold f 예측,
  Validation 점수 = Train OOF 전체로 학습한 메타 모델로 Validation(fold 5) 예측.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--rows", nargs="*", default=[], help="name=w,name=w ... (가중 평균 조합)")
    ap.add_argument("--meta", nargs="*", default=[], help="name,name ... (로지스틱 회귀 메타 스태킹)")
    ap.add_argument("--C", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--holdout_fold", type=int, default=5)
    a = ap.parse_args()

    rows = [dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in r.split(",")) for r in a.rows]
    metas = [m.split(",") for m in a.meta]
    names = sorted({n for r in rows for n in r} | {n for m in metas for n in m})
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in names})
    idx = {n: i for i, n in enumerate(names)}
    fold = o.fold.to_numpy()
    tr, va = fold != a.holdout_fold, fold == a.holdout_fold
    ot, ov = o[tr].reset_index(drop=True), o[va].reset_index(drop=True)
    print(f"멤버 {len(names)}개 정렬 완료, Train 실험 {ot.grp.nunique()} / Validation 실험 {ov.grp.nunique()}")

    def score(lab):
        s1 = score_fn(ot, cummax_pp(ot, lab[tr]))
        s2 = score_fn(ov, cummax_pp(ov, lab[va]))
        return s1, s2

    print(f"\n{'조합':<44s} Train-fold  Validation  최저")
    for r in rows:
        w = np.zeros(len(names))
        for n, x in r.items():
            w[idx[n]] = x
        s1, s2 = score(CLASSES[np.einsum("mnc,m->nc", Ps, w).argmax(1)])
        print(f"{' + '.join(f'{n} {x:g}' for n, x in r.items()):<44s} {s1:.4f}      {s2:.4f}      {min(s1, s2):.4f}", flush=True)

    for m in metas:
        X = np.concatenate([np.log(np.clip(Ps[idx[n]], 1e-6, 1)) for n in m], axis=1)
        y = o.y.to_numpy()
        for C in a.C:
            lab = np.empty(len(o), dtype=CLASSES.dtype)
            for f in sorted(set(fold[tr])):
                fit_m, pred_m = tr & (fold != f), fold == f
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X[fit_m], y[fit_m])
                lab[pred_m] = clf.predict(X[pred_m])
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
            lab[va] = clf.predict(X[va])
            s1, s2 = score(lab)
            print(f"{'메타 LR(C=' + f'{C:g}) ' + '+'.join(m):<44s} {s1:.4f}      {s2:.4f}      {min(s1, s2):.4f}", flush=True)


if __name__ == "__main__":
    main()
