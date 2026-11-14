"""스태킹 1단계 — 멤버 모델을 같은 실험 단위 5-fold로 학습하고 OOF 확률과 fold 모델을 저장한다.
    python stack.py --data_dir <battery_data_light> --out_dir runs/stack --members lgb cat team

산출물 (out_dir)
  oof_<멤버>.csv   : sample_id, ver(k1o0/k2o0/k3o0), grp, y, experiment_id, relative_time_s, 규칙 사건 열, p1..p6
  <멤버>.joblib    : {"kind", "models"(fold 5개), "feats"}
  folds.json       : fold별 검증 실험 (멤버끼리 같은 분할)
이미 있는 멤버 파일은 건너뛴다(--force로 다시 학습). 다음 단계는 blend.py.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import EVAL_VERSIONS, KEY, TRAIN_VERSIONS, find_csv, load, subsample
from src.ensemble import team_image_features
from src.features import causality_check, feature_columns, make_features
from src.evaluation import CLASSES, PCOLS, experiment_folds, make_oof, save_folds, write_oof
from src.members import fit_member, predict_member

GATE_COLS = ["ev_t1_above45", "ev_v_below2.45", "ev_v_drop0.78", "ev_v_drop1.0", "tr_method"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs/stack")
    ap.add_argument("--members", nargs="+", default=["lgb", "cat", "team"],
                    help="lgb, cat, team. 시드 변형은 이름 뒤에 _s<번호> (예: lgb_s1)")
    ap.add_argument("--seed", type=int, default=42, help="모델 시드 (fold 분할은 src/evaluation.py에 고정)")
    ap.add_argument("--cat_iters", type=int, default=1500)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--gpu", action="store_true", help="cat 멤버를 GPU로 학습")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("[1] 데이터 로드 (Train + Validation)")
    d = load(a.data_dir, ("train", "valid"))
    tv = pd.concat([d["train"], d["valid"]], ignore_index=True)
    need_team = any(m.split("_")[0] == "team" for m in a.members)
    timg = None
    if need_team:
        timg = pd.concat([team_image_features(d[s], os.path.dirname(find_csv(a.data_dir, s)),
                                              cache_dir=a.cache_dir, tag=s) for s in ("train", "valid")])
    causality_check(tv)

    print("[2] 피처 생성 (5/10/15초 버전)")
    t0 = time.time()
    F = pd.concat([make_features(subsample(tv, k, o)) for k, o in TRAIN_VERSIONS], ignore_index=True)
    feats = [c for c in feature_columns(F) if F[c].nunique(dropna=True) > 1]
    raw_ver = {f"k{k}o{o}": subsample(tv, k, o) for k, o in EVAL_VERSIONS}
    print(f"  {len(F):,} rows × {len(feats)} features ({time.time() - t0:.0f}s)")

    splits = experiment_folds(tv)
    save_folds(splits, os.path.join(a.out_dir, "folds.json"))

    for name in a.members:
        kind = name.split("_")[0]
        seed_off = int(name.split("_s")[1]) * 100 if "_s" in name else 0
        path = os.path.join(a.out_dir, f"{name}.joblib")
        if os.path.exists(path) and not a.force:
            print(f"\n[{name}] 이미 있음 → 건너뜀 (--force로 다시 학습)")
            continue
        print(f"\n[{name}] 학습 ({kind}, {len(splits)}-fold)")
        models, oof = [], []
        for fold, (tr_g, va_g) in enumerate(splits):
            t0 = time.time()
            m = fit_member(kind, a.seed + fold + seed_off, trn_F=F[F.grp.isin(tr_g)], feats=feats,
                           trn_raw=tv[tv[KEY].isin(tr_g)], timg=timg, gpu=a.gpu, cat_iters=a.cat_iters)
            models.append(m)
            msg = []
            for k, o in EVAL_VERSIONS:
                ver = f"k{k}o{o}"
                v = F[F.grp.isin(va_g) & (F.ver == ver)]
                rv = raw_ver[ver]
                P = predict_member(kind, m, v, feats, rv[rv.grp.isin(va_g)], timg)
                dd = make_oof(P, v, ver, fold, extra_cols=GATE_COLS)
                oof.append(dd)
                msg.append(f"{5 * k}s {f1_score(dd.y, CLASSES[P.argmax(1)], average='macro'):.3f}")
            print(f"  fold{fold}  " + "  ".join(msg) + f"  ({time.time() - t0:.0f}s)")
        oof = pd.concat(oof, ignore_index=True)
        write_oof(oof, os.path.join(a.out_dir, f"oof_{name}.csv"))
        joblib.dump({"kind": kind, "models": models, "feats": feats if kind != "team" else None}, path, compress=3)
        m5 = oof.ver == "k1o0"
        print(f"  OOF raw Macro F1 (5s) {f1_score(oof.y[m5], CLASSES[oof.loc[m5, PCOLS].to_numpy().argmax(1)], average='macro'):.4f}"
              f"  → {path}")


if __name__ == "__main__":
    main()
