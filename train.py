"""학습 — Train+Validation을 합쳐 실험 단위 K-fold로 학습하고, 단일 가중치 파일을 저장한다.
    python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

model.joblib 하나에 fold 모델 · 피처 목록 · 클래스 배율 w · 후처리 설정이 모두 들어 있어
test.py만으로 재학습 없이 결과를 재현할 수 있다.
부가 산출물 (model_out과 같은 폴더): <이름>_oof.csv, <이름>_cv.json, <이름>_importance.csv
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

from src.data import COND, EVAL_VERSIONS, KEY, T, TRAIN_VERSIONS, VER_W, Y, find_csv, load, subsample
from src.features import causality_check, feature_columns, make_features
from src.images import add_image_features
from src.model import cummax_pp, make_model, predict_labels, tune_class_weights


def weighted(scores):
    return sum(VER_W[v] * scores[v] for v in VER_W) / sum(VER_W.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "--data", dest="data_dir", required=True, help="battery_data_light 경로")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_balance", action="store_true", help="class_weight=balanced 끄기")
    ap.add_argument("--no_images", action="store_true", help="이미지 통계 피처 끄기")
    ap.add_argument("--cache_dir", default="cache", help="이미지 통계 캐시 폴더")
    ap.add_argument("--holdout_valid", action="store_true",
                    help="비교용: Train만으로 학습하고 제공 Validation(19개 실험)으로 평가. 제출용 아님")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    stem = os.path.splitext(a.model_out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.model_out)), exist_ok=True)

    print("[1] 데이터 로드 (Train + Validation)")
    d = load(a.data_dir, ("train", "valid"))
    if not a.no_images:
        print("  이미지 통계 계산 (처음 1회만 오래 걸리고 이후 캐시 사용)")
        for s in ("train", "valid"):
            root = os.path.dirname(find_csv(a.data_dir, s))
            d[s] = add_image_features(d[s], root, cache_dir=a.cache_dir, tag=s)
    if a.holdout_valid:
        print("  ※ holdout 모드: Train만 학습, Validation은 마지막 평가에만 사용")
        tv = d["train"].copy()
    else:
        tv = pd.concat([d["train"], d["valid"]], ignore_index=True)

    print("\n[2] 피처 누수 검사")
    causality_check(tv)

    print("\n[3] 피처 생성 (5/10/15초 다운샘플 버전 포함)")
    t0 = time.time()
    F = pd.concat([make_features(subsample(tv, k, o)) for k, o in TRAIN_VERSIONS], ignore_index=True)
    feats = feature_columns(F)
    const = [c for c in feats if F[c].nunique(dropna=True) <= 1]  # 상수·전부 결측 피처 제외
    feats = [c for c in feats if c not in const]
    print(f"  {len(F):,} rows × {len(feats)} features ({time.time() - t0:.0f}s), 상수 피처 {len(const)}개 제외")

    print(f"\n[4] {a.folds}-fold CV (실험 단위 분할) — fold 모델들을 최종 앙상블로 사용")
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
    oof = pd.concat(oof).reset_index(drop=True)
    P = oof[pcols].to_numpy()
    w, _ = tune_class_weights(P, oof.y.to_numpy(), classes)
    print("\n  클래스 배율 w (Macro-F1 튜닝):", dict(zip(classes.tolist(), np.round(w, 2).tolist())))

    print("\n[5] 후처리 선택 (OOF Macro-F1, 가중 = Test 간격 구성 9:1:1)")
    p_raw, p_w = classes[P.argmax(1)], classes[(P * w).argmax(1)]
    cands = {
        ("none", False): p_raw, ("none", True): p_w,
        ("cummax", False): cummax_pp(oof, p_raw), ("cummax", True): cummax_pp(oof, p_w),
    }
    table = {}
    print(f"  {'방식':<14s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
    for (meth, uw), pred in cands.items():
        s = {v: f1_score(oof.y[oof.ver == v], pred[(oof.ver == v).to_numpy()], average="macro") for v in VER_W}
        s["weighted"] = weighted(s)
        name = ("+w+" if uw else "") + meth if meth != "none" else ("+w" if uw else "raw")
        table[name] = {k: round(x, 4) for k, x in s.items()}
        print(f"  {name:<14s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}")
    (meth, uw) = max(cands, key=lambda c: table[
        (("+w+" if c[1] else "") + c[0]) if c[0] != "none" else ("+w" if c[1] else "raw")]["weighted"])
    post = {"method": meth, "use_w": bool(uw), "hmm": None}
    print(f"  → 선택: method={meth}, use_w={uw}  (HMM 비교는 tune_pp.py)")

    dd = oof[oof.ver == "k1o0"]
    best_pred = cands[(meth, uw)][(oof.ver == "k1o0").to_numpy()]
    per_class = f1_score(dd.y, best_pred, average=None, labels=classes)
    print("\n  클래스별 F1 (5초, 선택된 방식):", dict(zip(classes.tolist(), np.round(per_class, 3).tolist())))
    print(f"  Acc (5초): {accuracy_score(dd.y, best_pred):.4f}")

    if hasattr(models[0], "feature_importances_"):
        imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=feats)
        imp.sort_values(ascending=False).to_csv(f"{stem}_importance.csv", header=["gain"])
        print("\n=== 중요 피처 Top 15 ===")
        print(imp.sort_values(ascending=False).head(15).round(1).to_string())

    bundle = {"models": models, "feats": feats, "classes": classes, "w": w, "post": post,
              "use_images": not a.no_images}

    holdout = None
    if a.holdout_valid:
        print("\n[6] Holdout 평가 — 제공 Validation (학습에 전혀 쓰지 않은 실험)")
        va = d["valid"]
        print(f"  {'방식':<14s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
        holdout = {}
        for meth, uw in [("none", False), ("none", True), ("cummax", False), ("cummax", True)]:
            name = ("+w+" if uw else "") + meth if meth != "none" else ("+w" if uw else "raw")
            s = {}
            for k, o in EVAL_VERSIONS:
                Fv = make_features(subsample(va, k, o))
                pv, _ = predict_labels(bundle, Fv, post={"method": meth, "use_w": uw, "hmm": None})
                s[f"k{k}o{o}"] = f1_score(Fv[Y], pv, average="macro")
            s["weighted"] = weighted(s)
            holdout[name] = {k: round(x, 4) for k, x in s.items()}
            mark = "  ← 선택된 방식" if (meth, uw) == (post["method"], post["use_w"]) else ""
            print(f"  {name:<14s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}{mark}")
        Fv = make_features(va)
        pv, _ = predict_labels(bundle, Fv)
        print("  클래스별 F1 (선택된 방식, 5초):",
              dict(zip(classes.tolist(), np.round(f1_score(Fv[Y], pv, average=None, labels=classes), 3).tolist())))
        print("  ※ 팀원 CatBoost v1의 Validation Macro F1 = 0.7505 와 비교하세요 (같은 조건: Train만 학습)")

    joblib.dump(bundle, a.model_out, compress=3)
    oof.to_csv(f"{stem}_oof.csv", index=False)
    with open(f"{stem}_cv.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "model": type(models[0]).__name__, "n_features": len(feats),
                   "w": dict(zip(map(str, classes.tolist()), np.round(w, 3).tolist())),
                   "post": {k: v for k, v in post.items() if k != "hmm"}, "oof_macro_f1": table,
                   "per_class_f1_5s": dict(zip(map(str, classes.tolist()), np.round(per_class, 4).tolist())),
                   "holdout_valid": holdout},
                  f, ensure_ascii=False, indent=2)
    size = os.path.getsize(a.model_out) / 1e6
    print(f"\n저장 → {a.model_out} ({size:.1f}MB), {stem}_oof.csv, {stem}_cv.json")


if __name__ == "__main__":
    main()
