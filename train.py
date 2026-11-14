"""학습 — 실험 단위 K-fold CV로 모델을 학습하고, 결과를 outputs/<run>/ 에 저장한다.
    python train.py --data data --run lgb_v1

저장물
  outputs/<run>/bundle.joblib  fold 모델들 + 피처 목록 + 클래스 배율 w (predict.py가 사용)
  outputs/<run>/oof.csv        OOF 확률 (앙상블/분석용)
  outputs/<run>/cv.json        CV 점수 · 설정 (EXPERIMENTS.md에 옮겨 적기)
"""
import argparse
import json
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.data import COND, EVAL_VERSIONS, KEY, T, TRAIN_VERSIONS, Y, load, subsample
from src.features import causality_check, feature_columns, make_features
from src.model import cummax_pp, make_model, tune_class_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--run", default="lgb_v1", help="outputs/<run>/ 에 저장")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-balance", action="store_true", help="class_weight=balanced 끄기")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = os.path.join("outputs", a.run)
    os.makedirs(out, exist_ok=True)

    print("[1] 데이터 로드")
    d = load(a.data, ("train", "valid"))
    tv = pd.concat([d["train"], d["valid"]], ignore_index=True)

    print("\n[2] 피처 누수 검사")
    causality_check(tv)

    print("\n[3] 피처 생성 (5/10/15초 다운샘플 버전 포함)")
    t0 = time.time()
    F = pd.concat([make_features(subsample(tv, k, o)) for k, o in TRAIN_VERSIONS], ignore_index=True)
    feats = feature_columns(F)
    print(f"  {len(F):,} rows × {len(feats)} features ({time.time() - t0:.0f}s)")

    print(f"\n[4] {a.folds}-fold CV (실험 단위 분할)")
    ex = F[F.ver == "k1o0"].groupby("grp")[COND].first()
    strata = ex.tr_method.astype(str) + "_" + ex.bat_class.astype(str)
    if strata.value_counts().min() < a.folds:
        strata = ex.tr_method.astype(str)
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    models, oof = [], []
    for fold, (tr_i, va_i) in enumerate(skf.split(ex.index, strata)):
        tr_g, va_g = set(ex.index[tr_i]), set(ex.index[va_i])
        trn = F[F.grp.isin(tr_g)]
        t0 = time.time()
        m = make_model(a.seed + fold, balanced=not a.no_balance)
        m.fit(trn[feats], trn[Y])
        models.append(m)
        msg = []
        for k, o in EVAL_VERSIONS:
            v = F[F.grp.isin(va_g) & (F.ver == f"k{k}o{o}")]
            P = m.predict_proba(v[feats])
            dd = pd.DataFrame(P, columns=[f"p{c}" for c in m.classes_], index=v.index)
            dd["sample_id"], dd["y"], dd[KEY], dd[T] = v.sample_id.values, v[Y].values, v[KEY].values, v[T].values
            dd["ver"], dd["fold"] = f"k{k}o{o}", fold
            oof.append(dd)
            msg.append(f"{5 * k}s F1={f1_score(v[Y], m.classes_[P.argmax(1)], average='macro'):.3f}")
        print(f"  fold{fold}  " + "  ".join(msg) + f"  ({time.time() - t0:.0f}s)")

    classes = models[0].classes_
    pcols = [f"p{c}" for c in classes]
    oof = pd.concat(oof)
    w, _ = tune_class_weights(oof[pcols].to_numpy(), oof.y.to_numpy(), classes)
    print("\n  클래스 배율 w (Macro-F1 튜닝):", dict(zip(classes.tolist(), np.round(w, 2).tolist())))

    print("\n=== CV Macro-F1 ===")
    print(f"  {'간격':>4s} | {'raw':>6s} {'+w':>6s} {'+cummax':>8s} {'+w+cummax':>10s} | {'Acc(+w)':>7s}")
    cv = {}
    for k, o in EVAL_VERSIONS:
        dd = oof[oof.ver == f"k{k}o{o}"]
        P = dd[pcols].to_numpy()
        p_raw, p_w = classes[P.argmax(1)], classes[(P * w).argmax(1)]
        r = [f1_score(dd.y, p, average="macro") for p in (p_raw, p_w, cummax_pp(dd, p_raw), cummax_pp(dd, p_w))]
        cv[f"{5 * k}s"] = dict(zip(["raw", "w", "cummax", "w_cummax"], np.round(r, 4).tolist()))
        print(f"  {5 * k:>3d}s | {r[0]:6.4f} {r[1]:6.4f} {r[2]:8.4f} {r[3]:10.4f} | {accuracy_score(dd.y, p_w):7.4f}")
    dd = oof[oof.ver == "k1o0"]
    per_class = f1_score(dd.y, classes[(dd[pcols].to_numpy() * w).argmax(1)], average=None, labels=classes)
    print("\n  클래스별 F1 (5초, +w):", dict(zip(classes.tolist(), np.round(per_class, 3).tolist())))
    print("  ※ w는 OOF에 맞춘 값이라 약간 낙관적. +w 이득이 0.01 미만이면 predict.py --no-tune 고려")

    if hasattr(models[0], "feature_importances_"):
        imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=feats)
        imp.sort_values(ascending=False).to_csv(os.path.join(out, "importance.csv"), header=["gain"])
        print("\n=== 중요 피처 Top 20 ===")
        print(imp.sort_values(ascending=False).head(20).round(1).to_string())

    joblib.dump({"models": models, "feats": feats, "classes": classes, "w": w}, os.path.join(out, "bundle.joblib"))
    oof.to_csv(os.path.join(out, "oof.csv"), index=False)
    with open(os.path.join(out, "cv.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "model": type(models[0]).__name__, "n_features": len(feats),
                   "w": dict(zip(map(str, classes.tolist()), np.round(w, 3).tolist())),
                   "cv_macro_f1": cv, "per_class_f1_5s": dict(zip(map(str, classes.tolist()),
                                                                  np.round(per_class, 4).tolist()))},
                  f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료 → {out}/ (bundle.joblib, oof.csv, cv.json)")


if __name__ == "__main__":
    main()
