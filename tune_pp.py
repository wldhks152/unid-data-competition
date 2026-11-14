"""후처리 튜닝 — train.py가 저장한 OOF 확률만으로 후처리 방식을 비교·선택한다 (재학습 불필요).
    python tune_pp.py --run lgb_v1      → outputs/lgb_v1/pp.json  (predict.py --pp hmm 이 사용)

비교: raw / +w / cummax / hmm / hmm+w
선택 점수: Test 구성(5초 9개, 10초 1개, 15초 1개 실험)에 맞춘 가중 Macro-F1
누수 방지: 전이 행렬 A는 fold별로 '평가 fold를 제외한' 실험의 label로만 추정해서 평가한다.
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import KEY, T
from src.model import cummax_pp, tune_class_weights
from src.postprocess import estimate_transitions, hmm_filter

VER_W = {"k1o0": 9, "k2o0": 1, "k3o0": 1}  # Test 실험 간격 구성
TAUS = [0.3, 0.5, 0.75, 1.0]
BACKS = [0.0, 1e-3, 1e-2]


def seqs(df, cidx):
    return [g.sort_values(T).y.map(cidx).to_numpy() for _, g in df.groupby(KEY)]


def score(oof, pred):
    s = {v: f1_score(oof.y[oof.ver == v], pred[(oof.ver == v).to_numpy()], average="macro") for v in VER_W}
    s["weighted"] = sum(VER_W[v] * s[v] for v in VER_W) / sum(VER_W.values())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lgb_v1")
    a = ap.parse_args()
    run = os.path.join("outputs", a.run)
    oof = pd.read_csv(os.path.join(run, "oof.csv"))
    oof = oof[oof.ver.isin(VER_W)].reset_index(drop=True)
    pcols = [c for c in oof.columns if c.startswith("p") and c[1:].isdigit()]
    classes = np.array([int(c[1:]) for c in pcols])
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    P = oof[pcols].to_numpy()
    base = oof[oof.ver == "k1o0"]
    prior = base[pcols].to_numpy().mean(0)  # 모델이 평균적으로 내는 클래스 분포

    def hmm_oof(tau, back):
        """fold별로 평가 fold를 뺀 label로 A를 추정해 필터링 (CV 누수 방지)."""
        out = np.empty_like(P)
        for f in sorted(oof.fold.unique()):
            A, init = estimate_transitions(seqs(base[base.fold != f], cidx), K, back=back)
            m = (oof.fold == f).to_numpy()
            out[m] = hmm_filter(oof[m].reset_index(drop=True), P[m],
                                {"A": A, "init": init, "prior": prior, "tau": tau})
        return out

    rows = []
    w, _ = tune_class_weights(P, oof.y.to_numpy(), classes)
    p_raw, p_w = classes[P.argmax(1)], classes[(P * w).argmax(1)]
    rows.append(("raw", score(oof, p_raw)))
    rows.append(("+w", score(oof, p_w)))
    rows.append(("cummax", score(oof, cummax_pp(oof, p_raw))))
    rows.append(("+w+cummax", score(oof, cummax_pp(oof, p_w))))

    print(f"HMM 그리드 탐색: tau={TAUS}, back={BACKS}")
    best = None
    for tau, back in itertools.product(TAUS, BACKS):
        H = hmm_oof(tau, back)
        s = score(oof, classes[H.argmax(1)])
        print(f"  tau={tau:<4} back={back:<6} weighted={s['weighted']:.4f}  5s={s['k1o0']:.4f}")
        if best is None or s["weighted"] > best[2]["weighted"]:
            best = (tau, back, s, H)
    tau, back, s_h, H = best
    rows.append((f"hmm(tau={tau},back={back})", s_h))
    w_h, _ = tune_class_weights(H, oof.y.to_numpy(), classes)
    rows.append(("hmm+w", score(oof, classes[(H * w_h).argmax(1)])))

    print("\n=== 후처리 비교 (OOF Macro-F1) ===")
    print(f"  {'방식':<28s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
    for name, s in rows:
        print(f"  {name:<28s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}")
    win = max(rows, key=lambda r: r[1]["weighted"])[0]
    print(f"\n  최고: {win}")
    print("  ※ w·tau는 OOF에 맞춘 값이라 약간 낙관적. 차이가 0.005 미만이면 단순한 쪽을 고르세요.")

    # 최종 파라미터: 전체 label로 A 재추정
    A, init = estimate_transitions(seqs(base, cidx), K, back=back)
    pp = {"A": A.round(6).tolist(), "init": init.tolist(), "prior": prior.tolist(), "tau": tau, "back": back,
          "w": w_h.tolist(), "classes": classes.tolist(),
          "oof": {n: {k: round(v, 4) for k, v in s.items()} for n, s in rows}}
    with open(os.path.join(run, "pp.json"), "w", encoding="utf-8") as f:
        json.dump(pp, f, ensure_ascii=False, indent=1)
    print(f"\n저장 → {run}/pp.json   (적용: python predict.py --run {a.run} --pp hmm)")


if __name__ == "__main__":
    main()
