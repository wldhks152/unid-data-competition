"""스태킹 멤버 — 같은 실험 단위 fold로 학습하고, Test 예측은 fold 모델 평균을 쓴다.

kind
  custom: 다른 브랜치 모델. joblib에 {"kind": "custom", "module": "src.xxx", ...}을 두고,
          그 모듈의 predict_proba(member, F, df, timg) -> (n, 6) 확률(열 순서 1~6)을 호출한다.
  lgb  : LightGBM, 우리 피처(make_features), 5/10/15초 증강 행으로 학습
  lgbf : lgb + 짧은 창 피처(f_: 직전 관측 대비·15초 변화율, 급한 3→4·4→5 전환용)
  lgbx : lgb와 같되 추가 피처(src/features.py _extra_feats, x_ 접두사)까지 사용
  cat  : CatBoost, 우리 피처, 5/10/15초 증강 행, 클래스 균형 가중
  ord  : 순서형 LightGBM ("단계 ≥ k" 이진 5개 → 단조 보정 → 6클래스), 우리 피처
  ordx : ord와 같되 이진 분류기를 XGBoost(GPU, CUDA)로
  tabnet: pytorch-tabnet (GPU), 우리 피처 306개 (표준화 + 결측 표시)
  et   : ExtraTrees (sklearn, CPU), 우리 피처 (lgb와 같은 306개)
  xgb  : XGBoost (GPU), 우리 피처, 5/10/15초 증강 행, 클래스 균형 가중
  team : 팀원 CatBoost v1 (원본 5초 행 + 팀원 피처 33개)
  rn18s: rn18 + 현재 행 센서 13개 (팀원 v2)
  dino : rn18s와 같되 이미지 특징을 DINOv2-small 384차원으로 (src/dino.py)
  r50  : rn18s와 같되 이미지 특징을 ImageNet ResNet50 2048차원으로 (src/r50.py)
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


class OrdinalLGB:
    """순서형 LightGBM: "단계 ≥ k" 이진 분류기 5개(k=2..6). P(≥k)를 k에 대해 단조 감소로 보정(누적 최소)한 뒤
    P(1)=1−P(≥2), P(k)=P(≥k)−P(≥k+1), P(6)=P(≥6)로 6클래스 확률을 만든다. 각 이진 분류기는 클래스 균형 가중."""

    def __init__(self, seed, backend="lgb", gpu=True):
        self.seed, self.backend, self.gpu = seed, backend, gpu
        self.models = []
        self.classes_ = CLASSES

    def _X(self, X):
        return X.to_numpy(np.float32) if self.backend == "xgb" and hasattr(X, "to_numpy") else X

    def fit(self, X, y):
        import time
        from sklearn.utils.class_weight import compute_sample_weight
        y = np.asarray(y)
        X = self._X(X)
        self.models = []
        for k in CLASSES[1:]:
            t0 = time.time()
            yk = (y >= k).astype(int)
            if self.backend == "xgb":
                m = make_xgb(self.seed + int(k), gpu=self.gpu)
                m.set_params(objective="binary:logistic", eval_metric="logloss")
                m.fit(X, yk, sample_weight=compute_sample_weight("balanced", yk))
            else:
                m = make_model(self.seed + int(k))
                m.fit(X, yk)
            self.models.append(m)
            print(f"    ord 단계≥{k} 완료 ({time.time() - t0:.0f}s)", flush=True)
        return self

    def predict_proba(self, X):
        X = self._X(X)
        q = np.stack([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in self.models], axis=1)
        q = np.minimum.accumulate(q, axis=1)
        P = np.concatenate([1 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]], axis=1)
        return np.clip(P, 0, None)


class ExtraTreesMember:
    """sklearn ExtraTrees (CPU 전용: sklearn에는 GPU 구현이 없음). NaN·inf는 -1e6으로 바꿔 트리가 따로 가르게 한다.
    학습 행이 많아 fold마다 최대 max_rows행을 무작위 추출해 학습한다 (클래스 균형 가중)."""

    def __init__(self, seed, n_estimators=300, max_rows=300000):
        self.seed, self.n_estimators, self.max_rows = seed, n_estimators, max_rows
        self.classes_ = CLASSES

    @staticmethod
    def _X(X):
        X = np.asarray(X, dtype=np.float32)
        return np.where(np.isfinite(X), X, np.float32(-1e6))

    def fit(self, X, y):
        import time
        from sklearn.ensemble import ExtraTreesClassifier
        t0 = time.time()
        X, y = self._X(X), np.asarray(y)
        if len(y) > self.max_rows:
            ix = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X[ix], y[ix]
        self.m = ExtraTreesClassifier(n_estimators=self.n_estimators, min_samples_leaf=10, max_features="sqrt",
                                      class_weight="balanced", n_jobs=-1, random_state=self.seed).fit(X, y)
        print(f"    ExtraTrees {len(y):,}행 학습 ({time.time() - t0:.0f}s)", flush=True)
        return self

    def predict_proba(self, X):
        P = self.m.predict_proba(self._X(X))
        cls = [int(c) for c in self.m.classes_]
        out = np.zeros((len(P), len(CLASSES)))
        for j, c in enumerate(cls):
            out[:, list(CLASSES).index(c)] = P[:, j]
        return out


class TabNetMember:
    """pytorch-tabnet (GPU). 입력은 fold 학습 행의 평균·표준편차로 표준화 후 ±5로 자르고 NaN·inf는 0(=학습 평균).
    결측 여부도 정보라 NaN 비율이 있는 열은 결측 표시(0/1) 열을 함께 넣는다. 조기 종료 없이 고정 epoch (검증 정보 미사용).
    학습 행이 많아 max_rows행을 무작위 추출, 클래스 균형 샘플링(weights=1)."""

    def __init__(self, seed, max_rows=150000, epochs=20, gpu=True):
        self.seed, self.max_rows, self.epochs, self.gpu = seed, max_rows, epochs, gpu
        self.classes_ = CLASSES

    def _X(self, X):
        X = np.asarray(X, dtype=np.float32)
        miss = ~np.isfinite(X[:, self.miss_cols])
        Z = np.where(np.isfinite(X), (X - self.mu) / self.sd, 0.0)
        return np.concatenate([np.clip(Z, -5, 5), miss], axis=1).astype(np.float32)

    def fit(self, X, y):
        import time
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        t0 = time.time()
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
        if len(y) > self.max_rows:
            ix = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X[ix], y[ix]
        fin = np.isfinite(X)
        Xn = np.where(fin, X, np.nan)
        self.mu = np.nan_to_num(np.nanmean(Xn, 0)).astype(np.float32)
        self.sd = np.nan_to_num(np.nanstd(Xn, 0), nan=1.0).astype(np.float32)
        self.sd[self.sd < 1e-6] = 1.0
        self.miss_cols = np.flatnonzero((~fin).mean(0) > 0.01)
        dev = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        self.m = TabNetClassifier(n_d=32, n_a=32, n_steps=4, gamma=1.5, lambda_sparse=1e-4,
                                  optimizer_params=dict(lr=2e-2), seed=self.seed, device_name=dev, verbose=5,
                                  scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params=dict(step_size=8, gamma=0.5))
        self.m.fit(self._X(X), y - 1, max_epochs=self.epochs, patience=0, batch_size=8192, virtual_batch_size=512,
                   weights=1, drop_last=True)
        print(f"    TabNet {len(y):,}행 × {self._X(X[:1]).shape[1]}열, {self.epochs} epoch ({dev}, {time.time() - t0:.0f}s)", flush=True)
        return self

    def predict_proba(self, X):
        P = self.m.predict_proba(self._X(X))
        out = np.zeros((len(P), len(CLASSES)))
        for j, c in enumerate(self.m.classes_):
            out[:, int(c)] = P[:, j]
        return out


def make_xgb(seed, gpu=True):
    """XGBoost (GPU: device=cuda, tree_method=hist). 클래스 균형은 sample_weight로."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist", device="cuda" if gpu else "cpu",
        objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=os.cpu_count() or 4)


def fit_member(kind, seed, trn_F=None, feats=None, trn_raw=None, timg=None, gpu=False, cat_iters=1500, emb=None,
               seed_variant=False):
    if kind in ("lgb", "lgbx", "lgbf"):  # lgbx/lgbf: 추가 피처(x_/f_)까지 쓰는 LightGBM
        m = make_model(seed)
        m.fit(trn_F[feats], trn_F["label"])
        return m
    if kind == "cat":
        m = make_cat(seed, iterations=cat_iters, gpu=gpu)
        m.fit(trn_F[feats].to_numpy(np.float32), trn_F["label"].to_numpy())
        return m
    if kind == "tabnet":
        return TabNetMember(seed, gpu=gpu).fit(trn_F[feats], trn_F["label"])
    if kind == "et":
        return ExtraTreesMember(seed).fit(trn_F[feats], trn_F["label"])
    if kind in ("ord", "ordx"):  # ordx: 같은 순서형을 XGBoost(CUDA)로
        return OrdinalLGB(seed, backend="xgb" if kind == "ordx" else "lgb", gpu=gpu).fit(trn_F[feats], trn_F["label"])
    if kind in ("xgb", "xgbf"):  # 라벨 1..6 → 0..5 (xgbf: + 짧은 창 피처 f_)
        from sklearn.utils.class_weight import compute_sample_weight
        y = trn_F["label"].to_numpy().astype(int) - 1
        m = make_xgb(seed, gpu=gpu)
        m.fit(trn_F[feats].to_numpy(np.float32), y, sample_weight=compute_sample_weight("balanced", y))
        return m
    if kind == "team":
        return fit_team(trn_raw, timg)
    if kind == "s23":
        m = make_model(seed)
        sel = trn_F["label"].isin([2, 3]).to_numpy()
        m.fit(trn_F.loc[sel, feats], (trn_F.loc[sel, "label"] == 3).astype(int))
        return m
    if kind in ("rn18", "rn18s", "dino", "r50"):
        from . import rn18
        return rn18.fit(trn_raw, timg, emb, gpu=gpu, sensors=kind != "rn18", seed=seed if seed_variant else None)
    raise ValueError(kind)


def _reorder(P, classes):
    classes = [int(c) for c in classes]
    return P[:, [classes.index(int(c)) for c in CLASSES]]


def predict_member(kind, model, F=None, feats=None, raw=None, timg=None, emb=None):
    """F 행 순서대로 (n, 6) 확률. team은 raw(F와 같은 sample_id를 가진 원본 행)와 이미지 통계가 필요."""
    if kind in ("lgb", "lgbx", "lgbf"):
        return _reorder(model.predict_proba(F[feats]), model.classes_)
    if kind == "cat":
        return _reorder(model.predict_proba(F[feats].to_numpy(np.float32)), model.classes_)
    if kind in ("ord", "ordx", "et", "tabnet"):
        return model.predict_proba(F[feats])
    if kind in ("xgb", "xgbf"):
        return _reorder(model.predict_proba(F[feats].to_numpy(np.float32)), model.classes_ + 1)
    if kind == "team":
        return team_proba(model, raw, timg, F.sample_id.values, list(CLASSES))
    if kind == "s23":
        q = model.predict_proba(F[feats])[:, list(model.classes_).index(1)]
        P = np.zeros((len(F), len(CLASSES)))
        P[:, 1], P[:, 2] = 1 - q, q
        return P
    if kind in ("rn18", "rn18s", "dino", "r50"):
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
