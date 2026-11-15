"""후처리 튜닝 (선택) — train_v1.py가 저장한 OOF 확률로 HMM 후처리까지 비교한다 (재학습 불필요).
    python tune_pp.py --model model.joblib            # 비교만
    python tune_pp.py --model model.joblib --apply    # 최고 방식을 model.joblib에 기록

비교: raw / +w / cummax / +w+cummax / hmm / hmm+w
선택 점수: Test 구성(5초 9개, 10초 1개, 15초 1개 실험)에 맞춘 가중 Macro-F1
누수 방지: 전이 행렬 A는 fold별로 '평가 fold를 제외한' 실험의 label로만 추정해서 평가한다.
"""
import argparse
import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import KEY, T, VER_W
from src.model import cummax_pp, tune_class_weights
from src.postprocess import estimate_transitions, hmm_filter

TAUS = [0.5, 1.0, 1.5, 2.0, 3.0]  # 클수록 모델 확률을 더 믿음 (단계 전환이 빨라짐)
BACKS = [0.0, 1e-3, 1e-2]


def seqs(df, cidx):
    return [g.sort_values(T).y.map(cidx).to_numpy() for _, g in df.groupby(KEY)]


def score(oof, pred):
    s = {v: f1_score(oof.y[oof.ver == v], pred[(oof.ver == v).to_numpy()], average="macro") for v in VER_W}
    s["weighted"] = sum(VER_W[v] * s[v] for v in VER_W) / sum(VER_W.values())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--apply", action="store_true", help="최고 방식을 model.joblib의 후처리 설정으로 저장")
    a = ap.parse_args()
    stem = os.path.splitext(a.model)[0]
    bundle = joblib.load(a.model)
    oof = pd.read_csv(f"{stem}_oof.csv")
    oof = oof[oof.ver.isin(VER_W)].reset_index(drop=True)
    classes = np.asarray(bundle["classes"])
    pcols = [f"p{c}" for c in classes]
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    P = oof[pcols].to_numpy()
    base = oof[oof.ver == "k1o0"]
    prior = base[pcols].to_numpy().mean(0)  # 모델이 평균적으로 내는 클래스 분포

    def hmm_oof(tau, back):
        out = np.empty_like(P)
        for f in sorted(oof.fold.unique()):
            A, init = estimate_transitions(seqs(base[base.fold != f], cidx), K, back=back)
            m = (oof.fold == f).to_numpy()
            out[m] = hmm_filter(oof[m].reset_index(drop=True), P[m],
                                {"A": A, "init": init, "prior": prior, "tau": tau})
        return out

    w = np.asarray(bundle["w"])
    p_raw, p_w = classes[P.argmax(1)], classes[(P * w).argmax(1)]
    rows = {
        ("none", False): score(oof, p_raw), ("none", True): score(oof, p_w),
        ("cummax", False): score(oof, cummax_pp(oof, p_raw)), ("cummax", True): score(oof, cummax_pp(oof, p_w)),
    }

    print(f"HMM 그리드 탐색: tau={TAUS}, back={BACKS}")
    best = None
    for tau, back in itertools.product(TAUS, BACKS):
        H = hmm_oof(tau, back)
        s = score(oof, classes[H.argmax(1)])
        print(f"  tau={tau:<4} back={back:<6} weighted={s['weighted']:.4f}  5s={s['k1o0']:.4f}")
        if best is None or s["weighted"] > best[2]["weighted"]:
            best = (tau, back, s, H)
    tau, back, s_h, H = best
    w_h, _ = tune_class_weights(H, oof.y.to_numpy(), classes)
    rows[("hmm", False)] = s_h
    rows[("hmm", True)] = score(oof, classes[(H * w_h).argmax(1)])

    names = {("none", False): "raw", ("none", True): "+w", ("cummax", False): "cummax",
             ("cummax", True): "+w+cummax", ("hmm", False): f"hmm(tau={tau},back={back})", ("hmm", True): "hmm+w"}
    cur = (bundle["post"]["method"], bundle["post"]["use_w"])
    print("\n=== 후처리 비교 (OOF Macro-F1) ===")
    print(f"  {'방식':<28s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
    for k, s in rows.items():
        mark = "  ← 현재 model.joblib" if k == cur else ""
        print(f"  {names[k]:<28s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}{mark}")
    win = max(rows, key=lambda k: rows[k]["weighted"])
    gain = rows[win]["weighted"] - rows[cur]["weighted"]
    print(f"\n  최고: {names[win]}  (현재 대비 {gain:+.4f})")
    print("  ※ OOF에 맞춘 값이라 약간 낙관적. 차이가 0.005 미만이면 현재(단순한 쪽) 유지를 권장.")

    if a.apply and win != cur:
        hmm = None
        if win[0] == "hmm":
            A, init = estimate_transitions(seqs(base, cidx), K, back=back)  # 전체 label로 재추정
            hmm = {"A": A.tolist(), "init": init.tolist(), "prior": prior.tolist(), "tau": tau,
                   "w": (w_h if win[1] else np.ones(K)).tolist()}
        bundle["post"] = {"method": win[0], "use_w": bool(win[1]), "hmm": hmm}
        joblib.dump(bundle, a.model, compress=3)
        print(f"  → {a.model} 후처리를 '{names[win]}'로 갱신")
    elif a.apply:
        print("  → 이미 최고 방식이라 변경 없음")


if __name__ == "__main__":
    main()
