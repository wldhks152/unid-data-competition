"""블렌딩 하네스 — 여러 브랜치의 OOF를 받아 멤버 비율 · 클래스 배율 · 후처리를 고르고,
'메타 교차검증(base 학습 미분리)' 점수를 함께 보고한 뒤 model.joblib 하나로 묶는다.

    python blend.py --members lgb cat team seq=runs/seq --stack_dir runs/stack \
                    [--post_module src.causal_filter] --model_out model_stack.joblib

멤버 지정: 이름 또는 이름=폴더. 폴더에 oof_<이름>.csv 와 <이름>.joblib 이 있어야 한다 (VALIDATION.md 계약).
선택 대상
  * 멤버 비율: 단계 그룹 A(1~3) · B(4~6)마다 따로, 합 1, --step 간격. 단독 멤버(꼭짓점)도 후보에 포함
  * 클래스 배율 w: --w_lo~--w_hi, 공통 점수(score_fn)로 튜닝, --w_min_gain 이상 오를 때만 사용
  * 후처리: raw, cummax, cummax2, cummax3 + --post_module의 POSTS
보고
  * dev 점수  : 전체 OOF에서 고르고 같은 OOF로 잰 점수 (낙관적, 참고용)
  * outer 점수: fold마다 다른 4개 fold로 고르고 남은 fold에 적용한 점수
                (메타 교차검증, base 학습 미분리: 독립 성능 추정이 아님. dev보다는 덜 낙관적이라 비교 기준으로 씀)
"""
import argparse
import importlib
import itertools
import json
import os

import joblib
import numpy as np

from src.evaluation import CLASSES, align_oofs, outer_meta_eval, read_oof, score_fn, tune_w
from src.model import cummax_pp, stable_cummax_pp

GROUPS = {"A": [0, 1, 2], "B": [3, 4, 5]}


def builtin_posts():
    def lab(P, w):
        return CLASSES[(P * w).argmax(1)]
    return {
        "raw": lambda fr, P, w: lab(P, w),
        "cummax": lambda fr, P, w: cummax_pp(fr, lab(P, w)),
        "cummax2": lambda fr, P, w: stable_cummax_pp(fr, lab(P, w), 2),
        "cummax3": lambda fr, P, w: stable_cummax_pp(fr, lab(P, w), 3),
    }


def simplex(n, step):
    k = int(round(1 / step))
    return [np.array(c, dtype=float) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def class_weights(WA, WB):
    W = np.zeros((len(WA), 6))
    W[:, GROUPS["A"]] = np.asarray(WA)[:, None]
    W[:, GROUPS["B"]] = np.asarray(WB)[:, None]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=["lgb", "cat", "team"])
    ap.add_argument("--stack_dir", default="runs/stack", help="폴더를 안 적은 멤버의 기본 폴더")
    ap.add_argument("--model_out", default="model_stack.joblib")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--search_post", nargs="+", default=["raw", "cummax"], help="비율 탐색 때 쓸 후처리 (각각 탐색)")
    ap.add_argument("--posts", nargs="+", default=None, help="비교할 후처리 (기본: 전부)")
    ap.add_argument("--post_module", default=None, help="POSTS = {이름: fn(frame, P, w) -> 라벨} 을 가진 모듈")
    ap.add_argument("--w_lo", type=float, default=0.67)
    ap.add_argument("--w_hi", type=float, default=1.5)
    ap.add_argument("--w_min_gain", type=float, default=0.003)
    ap.add_argument("--no_outer", action="store_true", help="메타 교차검증 생략 (빠름)")
    ap.add_argument("--no_bundle", action="store_true", help="표만 출력, model.joblib 만들지 않음")
    a = ap.parse_args()

    names, dirs = [], {}
    for m in a.members:
        n, _, d = m.partition("=")
        names.append(n)
        dirs[n] = d or a.stack_dir
    oofs = {n: read_oof(os.path.join(dirs[n], f"oof_{n}.csv")) for n in names}
    o, Ps, dropped = align_oofs(oofs)
    if any(dropped.values()):
        print(f"  [안내] 공통 행만 사용 (멤버별 제외 행 수 {dropped})")
    M = len(names)

    posts = builtin_posts()
    if a.post_module:
        posts.update(importlib.import_module(a.post_module).POSTS)
    post_names = a.posts or list(posts)
    grid = simplex(M, a.step)
    ones = np.ones(6)

    def predict(cfg, mask):
        fr = o[mask].reset_index(drop=True)
        P = np.einsum("mnc,mc->nc", Ps[:, mask], class_weights(cfg["WA"], cfg["WB"]))
        return posts[cfg["post"]](fr, P, np.asarray(cfg["w"]))

    def select(mask, verbose=False):
        fr = o[mask].reset_index(drop=True)
        Pm = Ps[:, mask]
        searched = []
        for spn in a.search_post:
            sp = posts[spn]
            ev = lambda WA, WB: score_fn(fr, sp(fr, np.einsum("mnc,mc->nc", Pm, class_weights(WA, WB)), ones))
            starts = [np.full(M, 1 / M)] + [np.eye(M)[i] for i in range(M)]  # 균등 + 단독 멤버에서 시작
            best = max(((ev(s0, s0), s0, s0) for s0 in starts), key=lambda r: r[0])
            for _ in range(2):
                for g in ("A", "B"):
                    for wv in grid:
                        WA, WB = (wv, best[2]) if g == "A" else (best[1], wv)
                        s = ev(WA, WB)
                        if s > best[0] + 1e-6:
                            best = (s, WA, WB)
            searched.append(best)
        cands = []
        for _, WA, WB in searched:
            P = np.einsum("mnc,mc->nc", Pm, class_weights(WA, WB))
            w_t, _ = tune_w(P, fr, lo=a.w_lo, hi=a.w_hi)
            for pn in post_names:
                for uw, ww in [(False, ones), (True, w_t)]:
                    cands.append((score_fn(fr, posts[pn](fr, P, ww)), pn, uw, WA, WB, w_t))
        no_w = max((c for c in cands if not c[2]), key=lambda c: c[0])
        with_w = max((c for c in cands if c[2]), key=lambda c: c[0])
        pick = with_w if with_w[0] >= no_w[0] + a.w_min_gain else no_w
        _, _, _, WA, WB, w_t = pick
        cfg = {"WA": np.asarray(WA).tolist(), "WB": np.asarray(WB).tolist(), "post": pick[1], "use_w": pick[2],
               "w": (w_t if pick[2] else ones).tolist(), "dev": pick[0]}
        if verbose:
            print(f"  비율 A(1~3단계) {dict(zip(names, np.round(WA, 2).tolist()))}  B(4~6단계) {dict(zip(names, np.round(WB, 2).tolist()))}")
            seen = set()
            for c in sorted(cands, key=lambda c: -c[0]):
                key = (c[1], c[2])
                if key in seen:
                    continue
                seen.add(key)
                print(f"    {c[1]:<10s} {'+w' if c[2] else '  '}  {c[0]:.4f}")
            print(f"  w 후보 {np.round(w_t, 2).tolist()} → 선택 {pick[1]} {'+w' if pick[2] else '(w 없음)'}")
        return cfg

    print("=== 멤버 단독 (공통 점수, dev) ===")
    for i, n in enumerate(names):
        print(f"  {n:<10s} " + "  ".join(f"{pn} {score_fn(o, posts[pn](o, Ps[i], ones)):.4f}"
                                         for pn in ["raw", "cummax"]))

    print("\n=== 전체 OOF로 선택 (dev) ===")
    allm = np.ones(len(o), bool)
    cfg = select(allm, verbose=True)
    dev = score_fn(o, predict(cfg, allm), detail=True)
    print(f"  dev 점수 {dev['weighted']:.4f}  fold(5s) {np.round(list(dev['fold_5s'].values()), 3).tolist()}")
    print(f"  클래스별 F1(5s) {np.round(list(dev['class_5s'].values()), 3).tolist()}")

    outer = None
    if not a.no_outer:
        print("\n=== 메타 교차검증, base 학습 미분리 (fold마다 나머지 fold로 메타 선택 → 그 fold에 적용) ===")
        outer, cfgs = outer_meta_eval(o, select, predict)
        for f, c in cfgs.items():
            print(f"  fold{f}: A {np.round(c['WA'], 2).tolist()} B {np.round(c['WB'], 2).tolist()} "
                  f"post {c['post']} w {'O' if c['use_w'] else 'X'}")
        print(f"  outer 점수 {outer['weighted']:.4f}  (dev {dev['weighted']:.4f}, 차이 = 메타 선택의 낙관 정도. base 학습은 미분리)")
        print(f"  fold(5s) {np.round(list(outer['fold_5s'].values()), 3).tolist()}  "
              f"클래스별(5s) {np.round(list(outer['class_5s'].values()), 3).tolist()}")

    W = class_weights(cfg["WA"], cfg["WB"])
    report = {"members": {n: dirs[n] for n in names}, "blend": {n: W[i].round(4).tolist() for i, n in enumerate(names)},
              "post": cfg["post"], "use_w": cfg["use_w"], "w": np.round(cfg["w"], 4).tolist(),
              "post_module": a.post_module, "dev": round(dev["weighted"], 4),
              "outer": None if outer is None else round(outer["weighted"], 4)}
    rpath = os.path.splitext(a.model_out)[0] + "_blend.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n설정 저장 → {rpath}")
    if a.no_bundle:
        return

    members = {n: joblib.load(os.path.join(dirs[n], f"{n}.joblib")) for n in names}
    feats = sorted({c for mem in members.values() for c in (mem.get("feats") or [])})
    bundle = {"members": members, "blend": {n: W[i] for i, n in enumerate(names)}, "classes": CLASSES,
              "feats": feats, "w": np.asarray(cfg["w"]), "use_images": False,
              "post": {"method": cfg["post"], "use_w": cfg["use_w"], "hmm": None, "gate": "none",
                       "module": a.post_module if cfg["post"] not in builtin_posts() else None}}
    joblib.dump(bundle, a.model_out, compress=3)
    print(f"모델 저장 → {a.model_out} ({os.path.getsize(a.model_out) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
