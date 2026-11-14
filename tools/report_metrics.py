"""발표용 집계 지표 — 최종 블렌드의 단계별 F1 · 혼동행렬 (Train 5-fold OOF / Validation 홀드아웃).

    python tools/report_metrics.py --stack_dir runs/stack_trv --members lgb rn18s --weights 0.5 0.5 --post cummax
집계값만 출력·저장한다 (reports/metrics.json). 행·ID·개별 예측은 저장하지 않는다.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.evaluation import CLASSES, align_oofs, read_oof, score_fn  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", default="runs/stack_trv")
    ap.add_argument("--members", nargs="+", default=["lgb", "rn18s"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.5])
    ap.add_argument("--post", default="cummax", choices=["raw", "cummax"])
    ap.add_argument("--holdout_fold", type=int, default=5, help="Validation이 들어 있는 fold 번호")
    ap.add_argument("--out", default="reports/metrics.json")
    a = ap.parse_args()
    o, Ps, _ = align_oofs({m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in a.members})
    P = np.einsum("mnc,m->nc", Ps, np.asarray(a.weights))
    out = {"members": dict(zip(a.members, a.weights)), "post": a.post}
    parts = {"train_5fold_oof": (o.fold != a.holdout_fold).to_numpy(), "validation_holdout": (o.fold == a.holdout_fold).to_numpy()}
    for name, m in parts.items():
        if not m.any():
            continue
        fr = o[m].reset_index(drop=True)
        lab = CLASSES[P[m].argmax(1)]
        if a.post == "cummax":
            lab = cummax_pp(fr, lab)
        r = {"experiments": int(fr.grp.nunique()), "weighted_9_1_1": round(score_fn(fr, lab), 4)}
        for v in ["k1o0", "k2o0", "k3o0"]:
            mv = (fr.ver == v).to_numpy()
            y, p = fr.y.to_numpy()[mv], lab[mv]
            r[v] = {"macro_f1": round(float(f1_score(y, p, average="macro", labels=CLASSES)), 4),
                    "stage_f1": dict(zip(map(int, CLASSES), np.round(f1_score(y, p, average=None, labels=CLASSES), 4).tolist())),
                    "confusion": confusion_matrix(y, p, labels=CLASSES).tolist()}
        out[name] = r
        print(f"\n[{name}] 실험 {r['experiments']}개  가중 점수(9:1:1) {r['weighted_9_1_1']:.4f}")
        print(f"  5초 Macro F1 {r['k1o0']['macro_f1']:.4f}  단계별 F1 {list(r['k1o0']['stage_f1'].values())}")
        print("  혼동행렬(5초, 행=정답 1~6, 열=예측 1~6)")
        for c, row in zip(CLASSES, r["k1o0"]["confusion"]):
            print(f"    {c}: " + " ".join(f"{x:5d}" for x in row))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장 → {a.out}")


if __name__ == "__main__":
    main()
