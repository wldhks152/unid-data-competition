"""학습 — Train+Validation을 합쳐 실험 단위 K-fold로 학습하고, 단일 가중치 파일을 저장한다.
    python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

model.joblib 하나에 fold 모델 · 피처 목록 · 클래스 배율 w · 후처리 설정 · 팀원 CatBoost(앙상블)가 모두 들어 있어
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

from src.ensemble import blend_proba, fit_team, team_image_features
from src.data import COND, EVAL_VERSIONS, KEY, T, TRAIN_VERSIONS, VER_W, Y, find_csv, load, subsample
from src.features import causality_check, feature_columns, make_features
from src.images import add_image_features
from src.model import cummax_pp, make_model, predict_labels, rule_gate, tune_class_weights

GATE_COLS = ["ev_t1_above45", "ev_v_below2.45", "ev_v_drop0.78", "ev_v_drop1.0", "tr_method"]
GATES = ["none", "12", "1234"]


def pp_name(meth, uw, gate="none"):
    n = ("+w" if uw else "") + ("+" + meth if meth != "none" else "")
    n = n[1:] if n.startswith("+cummax") else (n or "raw")
    return n + ("" if gate == "none" else f"+gate{gate}")


def weighted(scores):
    return sum(VER_W[v] * scores[v] for v in VER_W) / sum(VER_W.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "--data", dest="data_dir", required=True, help="battery_data_light 경로")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_balance", action="store_true", help="class_weight=balanced 끄기")
    ap.add_argument("--images", action="store_true", help="이미지 팔레트 통계 피처 켜기 (기본 꺼짐: Validation에서 이득 없음)")
    ap.add_argument("--no_images", action="store_true", help="(기본값과 같음, 하위 호환용)")
    ap.add_argument("--cache_dir", default="cache", help="이미지 통계 캐시 폴더")
    ap.add_argument("--alpha", type=float, default=0.5, help="앙상블에서 우리 LightGBM 비율 (나머지는 팀원 CatBoost)")
    ap.add_argument("--no_catboost", action="store_true", help="팀원 CatBoost 앙상블 끄기 (LightGBM 단독)")
    ap.add_argument("--holdout_valid", action="store_true",
                    help="비교용: Train만으로 학습하고 제공 Validation(19개 실험)으로 평가. 제출용 아님")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    stem = os.path.splitext(a.model_out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.model_out)), exist_ok=True)

    print("[1] 데이터 로드 (Train + Validation)")
    d = load(a.data_dir, ("train", "valid"))
    if a.images:
        print("  이미지 통계 계산 (처음 1회만 오래 걸리고 이후 캐시 사용)")
        for s in ("train", "valid"):
            root = os.path.dirname(find_csv(a.data_dir, s))
            d[s] = add_image_features(d[s], root, cache_dir=a.cache_dir, tag=s)
    timg = None
    if not a.no_catboost:
        print("  팀원 CatBoost용 이미지 통계 계산 (캐시 사용)")
        timg = pd.concat([team_image_features(d[s], os.path.dirname(find_csv(a.data_dir, s)),
                                              cache_dir=a.cache_dir, tag=s) for s in ("train", "valid")])
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
            for c in GATE_COLS:
                dd[c] = v[c].values
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
    cands = {}
    for uw, p0 in [(False, p_raw), (True, p_w)]:
        for gate in GATES:
            pg = rule_gate(oof, p0, gate)
            cands[("none", uw, gate)] = pg
            cands[("cummax", uw, gate)] = cummax_pp(oof, pg)
    table = {}
    print(f"  {'방식':<22s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
    for (meth, uw, gate), pred in cands.items():
        s = {v: f1_score(oof.y[oof.ver == v], pred[(oof.ver == v).to_numpy()], average="macro") for v in VER_W}
        s["weighted"] = weighted(s)
        name = pp_name(meth, uw, gate)
        table[name] = {k: round(x, 4) for k, x in s.items()}
        print(f"  {name:<22s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}")
    (meth, uw, gate) = max(cands, key=lambda c: table[pp_name(*c)]["weighted"])
    post = {"method": meth, "use_w": bool(uw), "hmm": None, "gate": gate}
    print(f"  → 선택: method={meth}, use_w={uw}, gate={gate}  (HMM 비교는 tune_pp.py)")

    dd = oof[oof.ver == "k1o0"]
    best_pred = cands[(meth, uw, gate)][(oof.ver == "k1o0").to_numpy()]
    per_class = f1_score(dd.y, best_pred, average=None, labels=classes)
    print("\n  클래스별 F1 (5초, 선택된 방식):", dict(zip(classes.tolist(), np.round(per_class, 3).tolist())))
    print(f"  Acc (5초): {accuracy_score(dd.y, best_pred):.4f}")

    if hasattr(models[0], "feature_importances_"):
        imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=feats)
        imp.sort_values(ascending=False).to_csv(f"{stem}_importance.csv", header=["gain"])
        print("\n=== 중요 피처 Top 15 ===")
        print(imp.sort_values(ascending=False).head(15).round(1).to_string())

    bundle = {"models": models, "feats": feats, "classes": classes, "w": w, "post": post,
              "use_images": a.images}

    if not a.no_catboost:
        print(f"\n[5b] 팀원 CatBoost v1 학습 (같은 데이터 {tv[KEY].nunique()}개 실험, 5초 원본 행) → 앙상블 α={a.alpha}")
        t0 = time.time()
        bundle["team"] = fit_team(tv, timg)
        bundle["alpha"] = a.alpha
        # 앙상블 후처리는 Validation 비교(ensemble_eval.py)에서 가장 좋았던 +w+cummax로 고정
        bundle["post"] = post = {"method": "cummax", "use_w": True, "hmm": None, "gate": "none"}
        print(f"  완료 ({time.time() - t0:.0f}s) · 앙상블 후처리 = +w+cummax (Validation 비교 결과로 고정)")

    holdout = None
    if a.holdout_valid:
        print("\n[6] Holdout 평가 — 제공 Validation (학습에 전혀 쓰지 않은 실험)")
        va = d["valid"]
        vers = {}
        for k, o in EVAL_VERSIONS:
            vk = subsample(va, k, o)
            Fv = make_features(vk).reset_index(drop=True)
            vers[f"k{k}o{o}"] = (Fv, blend_proba(dict(bundle, team=None), Fv),
                                 blend_proba(bundle, Fv, vk, timg) if "team" in bundle else None)
        holdout = {}
        for title, idx in [("LightGBM 단독", 1), ("앙상블 (α=%.2f)" % a.alpha, 2)]:
            if idx == 2 and "team" not in bundle:
                continue
            print(f"  --- {title} ---")
            print(f"  {'방식':<22s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
            for meth, uw, gate in [(m_, u_, g_) for g_ in GATES for u_ in (False, True) for m_ in ("none", "cummax")]:
                name = pp_name(meth, uw, gate)
                s = {}
                for v, tup in vers.items():
                    pv, _ = predict_labels(bundle, tup[0], post={"method": meth, "use_w": uw, "hmm": None, "gate": gate}, P=tup[idx])
                    s[v] = f1_score(tup[0][Y], pv, average="macro")
                s["weighted"] = weighted(s)
                holdout[f"{title}|{name}"] = {k: round(x, 4) for k, x in s.items()}
                mark = "  ← 저장된 설정" if (meth, uw, gate) == (post["method"], post["use_w"], post.get("gate", "none")) and (
                    idx == 2 or "team" not in bundle) else ""
                print(f"  {name:<22s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}{mark}")
        Fv, Po, Pe = vers["k1o0"]
        pv, _ = predict_labels(bundle, Fv, P=Pe if Pe is not None else Po)
        print("  클래스별 F1 (저장된 설정, 5초):",
              dict(zip(classes.tolist(), np.round(f1_score(Fv[Y], pv, average=None, labels=classes), 3).tolist())))
        print("  ※ 참고: 팀원 CatBoost v1 단독 Validation Macro F1 = 0.7505 (raw)")

    joblib.dump(bundle, a.model_out, compress=3)
    oof.to_csv(f"{stem}_oof.csv", index=False)
    with open(f"{stem}_cv.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "model": type(models[0]).__name__, "n_features": len(feats),
                   "w": dict(zip(map(str, classes.tolist()), np.round(w, 3).tolist())),
                   "post": {k: v for k, v in post.items() if k != "hmm"}, "oof_macro_f1": table,
                   "per_class_f1_5s": dict(zip(map(str, classes.tolist()), np.round(per_class, 4).tolist())),
                   "holdout_valid": holdout,
                   "ensemble": None if "team" not in bundle else {"alpha": a.alpha, "partner": "team CatBoost v1"}},
                  f, ensure_ascii=False, indent=2)
    size = os.path.getsize(a.model_out) / 1e6
    print(f"\n저장 → {a.model_out} ({size:.1f}MB), {stem}_oof.csv, {stem}_cv.json")


if __name__ == "__main__":
    main()
