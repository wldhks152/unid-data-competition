"""cummax 대체 후보: 인과 HSMM forward 필터를 앙상블 OOF로 평가 (재학습 불필요, 실험 단위 nested CV).

    python tune_filter.py --oof runs/ens_oof_oof.csv                       # 비교만
    python tune_filter.py --oof runs/ens_oof_oof.csv --quick               # 작은 그리드로 빠르게
    python tune_filter.py --oof runs/ens_oof_oof.csv --model runs/ens_oof.joblib --save runs/ens_hsmm.joblib
                                                                           # 최종 설정을 새 bundle로 저장

입력 OOF 칼럼: p1..p6 (LightGBM), q1..q6 (CatBoost, 있으면 α로 섞음), y, ver, fold, experiment_id, relative_time_s
점수: 버전별 Macro-F1을 VER_W(5초 9 : 10초 1 : 15초 1)로 가중 평균

nested CV (누수 방지):
  바깥 fold f 마다, 나머지 fold들만으로
    · 안쪽 fold g 마다 체류시간·건너뛰기 사전분포를 (f, g 제외) 실험 label로 추정 → g에 필터 적용
    · 그 안쪽 점수로 (tau, 체류시간 강도, prior, decode, w)를 고름 (w 튜닝도 안쪽 점수로만)
  고른 설정 + (f 제외) 사전분포로 f에 적용 → f의 점수는 선택에 한 번도 쓰이지 않음
  비교군(raw / cummax / cummax2)의 w도 같은 방식으로 바깥 fold 제외 데이터에서만 튜닝
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

from src.data import KEY, T, VER_W
from src.model import stable_cummax_pp
from src.postprocess import decode, estimate_hsmm, hsmm_filter

GRID = {"tau": [0.5, 1.0, 1.5, 2.0], "dur": ["geo", 50.0, 10.0, 2.0], "prior": ["uniform", "mean"]}
QUICK = {"tau": [1.0, 2.0], "dur": ["geo", 10.0], "prior": ["mean"]}
DECODERS = [("argmax", False), ("argmax", True), ("cummax", False), ("cummax", True), ("median", 0.5), ("median", 0.65)]
EPS, DMAX_S = 0.02, 600.0


# ── 빠른 점수 계산 ───────────────────────────────────────────────────────────
class Scorer:
    def __init__(self, df, y0, K):
        self.K, self.y = K, np.asarray(y0)
        self.ver = df.ver.to_numpy()
        self.masks = {v: self.ver == v for v in VER_W}

    def mf1(self, y, p):
        K = self.K
        cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
        tp = np.diag(cm).astype(float)
        r, c = cm.sum(1), cm.sum(0)
        return (2 * tp / np.maximum(r + c, 1)).mean()  # = f1_score(..., average="macro", labels=1..6) (score_fn와 동일)

    def score(self, pred, rows=None):
        """rows: 이 점수에 쓸 행(bool). 버전별 Macro-F1 + 가중."""
        rows = np.ones(len(self.y), bool) if rows is None else rows
        s = {}
        for v, m in self.masks.items():
            mm = m & rows
            s[v] = self.mf1(self.y[mm], pred[mm]) if mm.any() else np.nan
        s["weighted"] = sum(VER_W[v] * s[v] for v in VER_W) / sum(VER_W.values())
        return s


def tune_w(fn, K, rounds=2, lo=0.67, hi=1.5):  # VALIDATION.md 공통 범위
    """fn(w) → 가중 점수. tune_class_weights와 같은 좌표 탐색 (첫 클래스 1 고정)."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 21))
    w = np.ones(K)
    best = fn(w)
    for _ in range(rounds):
        for c in range(1, K):
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = fn(w2)
                if s > best + 1e-6:
                    best, w = s, w2
    return w, best


# ── 데이터 ──────────────────────────────────────────────────────────────────
def load_oof(path, alpha):
    oof = pd.read_csv(path)
    oof = oof[oof.ver.isin(VER_W)].reset_index(drop=True)
    pc = sorted([c for c in oof.columns if c[0] == "p" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    classes = np.array([int(c[1:]) for c in pc])
    P = oof[pc].to_numpy(float)
    qc = [f"q{c}" for c in classes]
    if all(c in oof.columns for c in qc) and alpha < 1:
        P = alpha * P + (1 - alpha) * oof[qc].to_numpy(float)
        print(f"  확률 = {alpha:g}·LightGBM + {1 - alpha:g}·CatBoost")
    else:
        print("  확률 = LightGBM만 (q 칼럼 없음 또는 alpha=1)")
    if "grp" not in oof.columns:
        oof["grp"] = oof[KEY].astype(str).str.split("__").str[0]
    cidx = {c: i for i, c in enumerate(classes)}
    y0 = oof.y.map(cidx).to_numpy()
    return oof, P, classes, y0


def label_seqs(oof, y0, grps):
    """5초 원본(k1o0) 행으로 실험별 (label, 시간) 시퀀스."""
    b = oof[(oof.ver == "k1o0") & oof.grp.isin(grps)]
    out = []
    for _, ix in b.groupby("grp").indices.items():
        ix = b.index[ix]
        o = np.argsort(oof.loc[ix, T].to_numpy(), kind="stable")
        out.append((y0[ix][o], oof.loc[ix, T].to_numpy(float)[o]))
    return out


def make_params(cfg, seqs, K, prior_mean):
    tau, dur, prior = cfg
    p = estimate_hsmm(seqs, K, dmax_s=DMAX_S, shrink=0.0 if dur == "geo" else dur, geometric=dur == "geo")
    p.update(tau=tau, eps=EPS, prior=prior_mean if prior == "mean" else np.full(K, 1.0 / K))
    return p


def cfg_name(cfg, dec):
    tau, dur, prior = cfg
    d = "HMM" if dur == "geo" else f"HSMM(s={dur:g})"
    how, x = dec
    dd = f"median{x}" if how == "median" else how + ("+w" if x else "")
    return f"{d} tau={tau:g} prior={prior} → {dd}"


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="runs/ens_oof_oof.csv")
    ap.add_argument("--alpha", type=float, default=None, help="LightGBM 비율 (기본: --model의 alpha, 없으면 0.5)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--model", default=None, help="--save할 때 기반 bundle")
    ap.add_argument("--out_pred", default=None, help="nested 예측 CSV (sample_id, experiment_id, ver, fold, grp, y, 방식별 예측 1..6)")
    ap.add_argument("--export_params", default=None,
                    help="최종 설정 + 전체 label 사전분포를 JSON으로 저장 (src/causal_filter.py 플러그인이 읽음, 예: src/causal_filter_params.json)")
    ap.add_argument("--save", default=None, help="최종 HSMM 설정을 넣은 새 bundle 경로 (원본은 그대로)")
    a = ap.parse_args()

    bundle = None
    if a.model:
        import joblib
        bundle = joblib.load(a.model)
    alpha = a.alpha if a.alpha is not None else float(np.mean(bundle.get("alpha", 0.5))) if bundle else 0.5
    oof, P, classes, y0 = load_oof(a.oof, alpha)
    K = len(classes)
    sc = Scorer(oof, y0, K)
    folds = sorted(oof.fold.unique())
    fold = oof.fold.to_numpy()
    grp_of_fold = {f: set(oof.grp[oof.fold == f]) for f in folds}
    grid = QUICK if a.quick else GRID
    cfgs = list(itertools.product(grid["tau"], grid["dur"], grid["prior"]))
    print(f"  행 {len(oof):,} · 실험 {oof.grp.nunique()} · fold {len(folds)} · 설정 {len(cfgs)} × decode {len(DECODERS)}")

    # 1) 필요한 posterior 계산: '제외 fold 집합' X (크기 1 또는 2) 마다 사전분포 추정, 그 안의 각 fold에 적용
    #    X={g}: g 에 대한 바깥 적용(=최종 선택용 안쪽)   X={f,g}: 바깥 f 의 안쪽 fold g
    excl = [frozenset([g]) for g in folds] + [frozenset(c) for c in itertools.combinations(folds, 2)]

    def run(X, cfg):
        keep = set().union(*(grp_of_fold[f] for f in folds if f not in X))
        trm = (oof.ver == "k1o0").to_numpy() & oof.grp.isin(keep).to_numpy()
        params = make_params(cfg, label_seqs(oof, y0, keep), K, P[trm].mean(0))
        res = {}
        for g in X:
            m = fold == g
            res[g] = hsmm_filter(oof[m].reset_index(drop=True), P[m], params)
        return X, cfg, res

    t0 = time.time()
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=a.n_jobs, verbose=0)(delayed(run)(X, c) for X in excl for c in cfgs)
    post = {}  # (X, cfg, g) → posterior rows of fold g
    for X, c, res in out:
        for g, v in res.items():
            post[(X, c, g)] = v
    print(f"  필터 계산 {len(out)}회 ({time.time() - t0:.0f}s)")

    def assemble(cfg, outer):
        """outer=None: 각 fold g에 X={g} posterior (최종 선택용). outer=f: f 제외 fold들의 X={f,g} posterior."""
        A = np.full((len(oof), K), 1.0 / K)  # 선택에 안 쓰는 행은 균등 (점수에서 제외됨)
        for g in folds:
            if g == outer:
                continue
            X = frozenset([g]) if outer is None else frozenset([outer, g])
            A[fold == g] = post[(X, cfg, g)]
        return A

    def best_on(rows, outer):
        """rows(바깥 fold 제외)에서 (cfg, decoder, w) 선택."""
        best = None
        for cfg in cfgs:
            A = assemble(cfg, outer)
            for how, x in DECODERS:
                if how == "median":
                    w, s = None, sc.score(decode(A, oof, "median", thr=x), rows)["weighted"]
                elif x:
                    w, s = tune_w(lambda w_: sc.score(decode(A, oof, how, w_), rows)["weighted"], K)
                else:
                    w, s = None, sc.score(decode(A, oof, how), rows)["weighted"]
                if best is None or s > best[0]:
                    best = (s, cfg, (how, x), w)
        return best

    # 2) 비교군 + HSMM nested
    preds = {k: np.zeros(len(oof), int) for k in
             ["raw", "+w", "cummax", "+w+cummax", "cummax2", "+w+cummax2", "HSMM 필터 (outer)"]}
    chosen = []
    print("\n[바깥 fold별 선택]")
    for f in folds:
        te, tr = fold == f, fold != f
        w, _ = tune_w(lambda w_: sc.score((P * w_).argmax(1), tr)["weighted"], K)  # 레포 방식: raw argmax로 w 튜닝
        for nm, ww in [("", np.ones(K)), ("+w+", w)]:
            p0 = (P * ww).argmax(1)
            preds["raw" if not nm else "+w"][te] = p0[te]
            preds[nm + "cummax"][te] = decode(P * ww, oof, "cummax")[te]
            c2 = stable_cummax_pp(oof, classes[p0], 2)
            preds[nm + "cummax2"][te] = np.searchsorted(classes, c2)[te]
        s, cfg, dec, wf = best_on(tr, f)
        A = post[(frozenset([f]), cfg, f)]
        how, x = dec
        pr = decode(A, oof[te].reset_index(drop=True), how, wf, thr=x if how == "median" else 0.5)
        preds["HSMM 필터 (outer)"][te] = pr
        chosen.append((f, cfg, dec, s, sc.score(preds["HSMM 필터 (outer)"], te)["weighted"]))
        print(f"  fold{f}: {cfg_name(cfg, dec)}  안쪽 {s:.4f} → 바깥 {chosen[-1][-1]:.4f}")

    # 3) 결과 표
    k1 = (oof.ver == "k1o0").to_numpy()
    print("\n=== 후처리 비교 (앙상블 OOF, 모든 튜닝은 바깥 fold 밖에서만) ===")
    print(f"  {'방식':<22s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s} {'과대예측행':>9s} {'과대예측실험':>10s}")
    for nm, p in preds.items():
        s = sc.score(p)
        over = (p > y0) & k1
        ov_exp = pd.Series(over[k1]).groupby(oof.grp[k1].to_numpy()).any().mean()
        print(f"  {nm:<22s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}"
              f" {over[k1].mean():9.3%} {ov_exp:10.1%}")
    print("  과대예측행 = 5초 행 중 예측 단계 > 정답 단계 비율, 과대예측실험 = 그런 행이 한 번이라도 있는 실험 비율")
    print("  (cummax는 한 번 과대예측하면 그 뒤로 계속 과대예측 → 이 두 칸이 '이른 점프 고정'의 크기)")

    if a.out_pred:
        keys = [c for c in ["sample_id", KEY, "ver", "fold", "grp", T, "y"] if c in oof.columns]
        dfp = oof[keys].copy()
        for nm, p in preds.items():
            dfp[nm] = classes[p]
        dfp.to_csv(a.out_pred, index=False)
        print(f"  → nested 예측 저장: {a.out_pred}")

    # 4) 최종 설정 (전체 fold를 안쪽으로 써서 선택) — 제출용
    s, cfg, dec, wf = best_on(np.ones(len(oof), bool), None)
    print(f"\n[최종 선택 = dev (전체 5 fold 안쪽 점수)] {cfg_name(cfg, dec)}  점수 {s:.4f}"
          + (f"  w={np.round(wf, 2).tolist()}" if wf is not None else ""))
    print("  ※ 이 점수는 선택에 쓴 데이터라 낙관적. 비교·판단은 위 표의 outer 값으로 (VALIDATION.md 6절).")

    if a.export_params:
        import json
        k1all = (oof.ver == "k1o0").to_numpy()
        params = make_params(cfg, label_seqs(oof, y0, set(oof.grp)), K, P[k1all].mean(0))
        how, x = dec
        js = {k: np.asarray(v).round(8).tolist() for k, v in params.items()}
        js.update(decode=how, thr=float(x) if how == "median" else 0.5, source=os.path.basename(a.oof),
                  alpha=alpha, config=cfg_name(cfg, dec),
                  nested_weighted=round(float(sc.score(preds["HSMM 필터 (outer)"])["weighted"]), 4))
        with open(a.export_params, "w", encoding="utf-8") as fh:
            json.dump(js, fh)
        print(f"  → 플러그인 파라미터 저장: {a.export_params}  (blend.py --post_module src.causal_filter)")

    if a.save:
        if bundle is None:
            sys.exit("--save에는 --model이 필요합니다")
        k1all = (oof.ver == "k1o0").to_numpy()
        params = make_params(cfg, label_seqs(oof, y0, set(oof.grp)), K, P[k1all].mean(0))
        how, x = dec
        hs = {k: np.asarray(v).tolist() for k, v in params.items()}
        hs.update(decode=how, thr=float(x) if how == "median" else 0.5)
        bundle = dict(bundle)
        if a.alpha is not None:
            bundle["alpha"] = a.alpha
        bundle["post"] = {"method": "hsmm", "use_w": wf is not None, "hmm": None, "gate": "none",
                          "hsmm": hs, "hsmm_w": (wf if wf is not None else np.ones(K)).tolist()}
        import joblib
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        joblib.dump(bundle, a.save, compress=3)
        print(f"  → {a.save} 저장 (원본 {a.model}은 그대로). 추론: python test.py --model {a.save} ...")


if __name__ == "__main__":
    main()
