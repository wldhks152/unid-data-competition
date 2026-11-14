"""스태킹 멤버 — 같은 실험 단위 fold로 학습하고, Test 예측은 fold 모델 평균을 쓴다.

kind
  custom: 다른 브랜치 모델. joblib에 {"kind": "custom", "module": "src.xxx", ...}을 두고,
          그 모듈의 predict_proba(member, F, df, timg) -> (n, 6) 확률(열 순서 1~6)을 호출한다.
  lgb  : LightGBM, 우리 피처(make_features), 5/10/15초 증강 행으로 학습
  cat  : CatBoost, 우리 피처, 5/10/15초 증강 행, 클래스 균형 가중
  team : 팀원 CatBoost v1 (원본 5초 행 + 팀원 피처 33개)
  rn18s: rn18 + 현재 행 센서 13개 (팀원 v2)
  s23  : 2↔3단계 전용 이진 LightGBM (2·3단계 행만 학습, P(3 | 2 또는 3)). OOF는 [0, 1-q, q, 0, 0, 0].
         블렌드 멤버가 아니라 bundle["refine23"]로 붙여, 블렌드의 2·3단계 확률 합을 q로 다시 나눈다.
  rn18 : 팀원 CatBoost v1 피처 33개 + ImageNet ResNet18 임베딩 512개 (src/rn18.py). 저장 시 custom 멤버가 된다.
모든 멤버의 확률 열 순서는 CLASSES(1~6)로 맞춘다.
"""
import os

import numpy as np

from . import team_catboost as TM
from .ensemble import fit_team, team_proba
from .model import make_model

CLASSES = np.arange(1, 7)


def make_cat(seed, iterations=1500, depth=7, lr=0.05, gpu=False):
    from catboost import CatBoostClassifier
    kw = dict(task_type="GPU", devices="0") if gpu else dict(thread_count=os.cpu_count() or 4)
    return CatBoostClassifier(
        loss_function="MultiClass", iterations=iterations, depth=depth, learning_rate=lr,
        l2_leaf_reg=5.0, auto_class_weights="Balanced", random_seed=seed,
        allow_writing_files=False, verbose=False, **kw)


def fit_member(kind, seed, trn_F=None, feats=None, trn_raw=None, timg=None, gpu=False, cat_iters=1500, emb=None,
               seed_variant=False):
    if kind == "lgb":
        m = make_model(seed)
        m.fit(trn_F[feats], trn_F["label"])
        return m
    if kind == "cat":
        m = make_cat(seed, iterations=cat_iters, gpu=gpu)
        m.fit(trn_F[feats].to_numpy(np.float32), trn_F["label"].to_numpy())
        return m
    if kind == "team":
        return fit_team(trn_raw, timg)
    if kind == "s23":
        m = make_model(seed)
        sel = trn_F["label"].isin([2, 3]).to_numpy()
        m.fit(trn_F.loc[sel, feats], (trn_F.loc[sel, "label"] == 3).astype(int))
        return m
    if kind in ("rn18", "rn18s"):
        from . import rn18
        return rn18.fit(trn_raw, timg, emb, gpu=gpu, sensors=kind == "rn18s", seed=seed if seed_variant else None)
    raise ValueError(kind)


def _reorder(P, classes):
    classes = [int(c) for c in classes]
    return P[:, [classes.index(int(c)) for c in CLASSES]]


def predict_member(kind, model, F=None, feats=None, raw=None, timg=None, emb=None):
    """F 행 순서대로 (n, 6) 확률. team은 raw(F와 같은 sample_id를 가진 원본 행)와 이미지 통계가 필요."""
    if kind == "lgb":
        return _reorder(model.predict_proba(F[feats]), model.classes_)
    if kind == "cat":
        return _reorder(model.predict_proba(F[feats].to_numpy(np.float32)), model.classes_)
    if kind == "team":
        return team_proba(model, raw, timg, F.sample_id.values, list(CLASSES))
    if kind == "s23":
        q = model.predict_proba(F[feats])[:, list(model.classes_).index(1)]
        P = np.zeros((len(F), len(CLASSES)))
        P[:, 1], P[:, 2] = 1 - q, q
        return P
    if kind in ("rn18", "rn18s"):
        from . import rn18
        return rn18.proba(model, raw, timg, emb, F.sample_id.values)
    raise ValueError(kind)


def stack_proba(bundle, F, df=None, timg=None):
    """bundle["members"] = {name: {"kind", "models", "feats"}}, bundle["blend"] = {name: (6,) 가중치}.
    클래스(열)마다 멤버 가중 평균. 가중치는 클래스마다 합이 1."""
    import importlib
    out = 0.0
    for name, mem in bundle["members"].items():
        wgt = np.asarray(bundle["blend"][name], dtype=float)
        if not np.any(wgt > 0):
            continue
        if mem["kind"] == "custom":
            P = np.asarray(importlib.import_module(mem["module"]).predict_proba(mem, F, df, timg), dtype=float)
        else:
            P = np.mean([predict_member(mem["kind"], m, F, mem.get("feats"), df, timg) for m in mem["models"]], axis=0)
        out = out + P * wgt
    ref = bundle.get("refine23")
    if ref:
        out = refine23(out, np.mean([predict_member("s23", m, F, ref["feats"]) for m in ref["models"]], axis=0)[:, 2],
                       ref["alpha"])
    return out


def refine23(P, q, alpha):
    """블렌드 확률 P의 2·3단계 합을 유지한 채, 3단계 몫을 alpha*q + (1-alpha)*원래 몫으로 바꾼다."""
    P = np.array(P, dtype=float, copy=True)
    s = P[:, 1] + P[:, 2]
    own = np.divide(P[:, 2], s, out=np.full(len(s), 0.5), where=s > 0)
    share = alpha * np.asarray(q) + (1 - alpha) * own
    P[:, 1], P[:, 2] = s * (1 - share), s * share
    return P
