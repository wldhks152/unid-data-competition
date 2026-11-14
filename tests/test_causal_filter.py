"""합성 데이터로 인과 HSMM 필터 검증 (실데이터 불필요).
    python tests/test_causal_filter.py           # 단위 검사 + 합성 OOF 생성 (runs/synth_oof.csv)
그 뒤: python tune_filter.py --oof runs/synth_oof.csv --quick
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import KEY, T, VER_W  # noqa: E402
from src.postprocess import cummax_rows, decode, estimate_hsmm, hsmm_filter, stage_runs  # noqa: E402

K = 6
MEAN_S = [600, 300, 400, 200, 150, 400]  # 단계별 평균 체류(초)


def synth(n_exp=60, seed=0, spike=0.004, noise=1.0):
    """단조 label(가끔 건너뜀: 3→5, 4→6) + 감마 체류시간 + 잡음 확률 + 이른 '가짜 점프' 스파이크."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_exp):
        path = [0, 1, 2, 3, 4, 5]
        if rng.random() < 0.25:
            path.remove(4 if rng.random() < 0.5 else 3)
        y = []
        for k in path:
            n = max(2, int(rng.gamma(4.0, MEAN_S[k] / 4.0) / 5))  # 감마(4) → 최소 체류 효과
            y += [k] * n
        y = np.array(y)
        logit = rng.normal(0, noise, (len(y), K))
        logit[np.arange(len(y)), y] += 2.0
        sp = rng.random(len(y)) < spike  # 가짜 점프: 2~3 단계 위로 강하게
        tgt = np.minimum(y + rng.integers(2, 4, len(y)), K - 1)
        logit[sp, tgt[sp]] += 5.0
        P = np.exp(logit - logit.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        Q = np.roll(P, 0, 1) * 0.7 + 0.3 / K
        for k_, o_ in [(1, 0), (2, 0), (3, 0)]:
            sel = np.arange(o_, len(y), k_)
            d = pd.DataFrame(P[sel], columns=[f"p{c}" for c in range(1, K + 1)])
            for c in range(K):
                d[f"q{c + 1}"] = Q[sel, c]
            d["y"] = y[sel] + 1
            d[KEY] = f"E{e:03d}__k{k_}o{o_}"
            d[T] = np.arange(len(sel)) * 5.0 * k_
            d["ver"], d["fold"] = f"k{k_}o{o_}", e % 5
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def check():
    oof = synth()
    y0 = oof.y.to_numpy() - 1
    k1 = oof[oof.ver == "k1o0"]
    seqs = [(g.y.to_numpy() - 1, g[T].to_numpy()) for _, g in k1.groupby(KEY)]

    # 1) stage_runs: 마지막 구간은 중도절단
    r = stage_runs(np.array([0, 0, 1, 1, 1, 3]), np.arange(6) * 5.0)
    assert r == [(0, 2, True), (1, 3, True), (3, 1, False)], r

    # 2) 추정: 건너뛰기 허용, 마지막 단계 흡수, 확률 합 1
    p = estimate_hsmm(seqs, K, shrink=10)
    assert np.allclose(p["nxt"][:K - 1].sum(1), 1) and (np.tril(p["nxt"]) == 0).all()
    assert (p["haz"][K - 1] == 0).all() and p["nxt"][2, 4] > 0.05, p["nxt"][2]
    # 감마(4) 체류 → 위험률이 처음엔 낮고 나중에 높다 (HMM의 상수 위험률과 다름)
    assert p["haz"][0, :10].mean() < p["haz"][0, 100:120].mean()
    p.update(tau=1.0, eps=0.02)

    P = oof[[f"p{c}" for c in range(1, K + 1)]].to_numpy()
    post = hsmm_filter(oof, P, p)
    assert np.allclose(post.sum(1), 1)

    # 3) 인과성: 실험을 임의 지점에서 잘라도 남은 행의 posterior가 같다
    rng = np.random.default_rng(1)
    keep = np.ones(len(oof), bool)
    for key, ix in oof.groupby(KEY).indices.items():
        cut = rng.integers(1, len(ix))
        keep[ix[np.argsort(oof[T].to_numpy()[ix])][cut:]] = False
    sub = oof[keep].reset_index(drop=True)
    post_s = hsmm_filter(sub, P[keep], p)
    assert np.allclose(post_s, post[keep]), "미래 정보 누수!"
    # 행 순서를 섞어도 같은 결과
    perm = rng.permutation(len(oof))
    assert np.allclose(hsmm_filter(oof.iloc[perm].reset_index(drop=True), P[perm], p), post[perm])

    # 4) cummax_rows == pandas cummax
    pr = P.argmax(1)
    ref = pd.Series(pr).groupby(oof[KEY]).cummax().to_numpy()
    s = oof.sort_values([KEY, T]).index
    ref = pd.Series(pr[s], index=s).groupby(oof[KEY].to_numpy()[s]).cummax().sort_index().to_numpy()
    assert (cummax_rows(oof, pr) == ref).all()

    # 5) 품질: 가짜 점프가 있는 합성 데이터에서 필터가 cummax보다 낫다 (5초)
    from sklearn.metrics import f1_score
    m = (oof.ver == "k1o0").to_numpy()
    f = {n: f1_score(y0[m], v[m], average="macro") for n, v in
         [("raw", pr), ("cummax", cummax_rows(oof, pr)),
          ("hsmm", decode(post, oof, "argmax")), ("hsmm+cummax", decode(post, oof, "cummax")),
          ("hsmm median", decode(post, oof, "median"))]}
    print("  합성 5초 Macro-F1:", {k: round(v, 4) for k, v in f.items()})
    assert f["hsmm+cummax"] > f["cummax"] + 0.02
    print("  모든 검사 통과")
    os.makedirs("runs", exist_ok=True)
    oof.to_csv("runs/synth_oof.csv", index=False)
    print("  → runs/synth_oof.csv 저장 (tune_filter.py 연습용)")


if __name__ == "__main__":
    check()
