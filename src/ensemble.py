"""앙상블 — 우리 LightGBM(fold 평균)과 팀원 CatBoost v1의 확률을 α : (1-α)로 섞는다.

Validation(학습에 안 쓴 19개 실험) 비교 결과 (ensemble_eval.py):
  우리 단독 +w 0.897 · CatBoost 단독 raw 0.750 · α=0.25~0.5 + w + cummax 0.903~0.905 → α=0.5 채택
두 모델 모두 같은 실험의 현재·과거 행만 쓰므로 섞어도 미래 정보가 들어가지 않는다.
"""
import os

import numpy as np
import pandas as pd

from . import team_catboost as TM


def team_image_features(df, image_root, cache_dir=None, tag=None, n_jobs=4):
    """팀원 방식(회색조 밝기 통계)의 이미지 피처. 행별 독립 계산이라 캐시해도 안전."""
    cache = os.path.join(cache_dir, f"team_img_{tag}.csv") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        c = pd.read_csv(cache, index_col="sample_id")
        if set(df.sample_id) <= set(c.index):
            return c.loc[df.sample_id, TM.IMAGE_STATS]
    from pathlib import Path
    root = Path(image_root)
    paths = TM.resolve_image_paths(df, [root, root.parent])
    out = TM.extract_image_features(df, paths, n_jobs=n_jobs)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_csv(cache)
    return out


def catboost_to_gpu(pipe):
    """팀원 파이프라인의 CatBoost를 GPU 학습으로 바꾼다 (src.model.DEVICE가 cpu면 그대로). 추론은 어느 쪽이든 CPU."""
    from .model import DEVICE
    clf = pipe.named_steps.get("classifier")
    if DEVICE["prefer"] != "cpu" and type(clf).__name__ == "CatBoostClassifier":
        pipe.set_params(classifier__task_type="GPU", classifier__devices="0")
    return pipe


def fit_team(df, img, thread_count=None):
    """팀원 CatBoost 파이프라인을 df의 행으로 학습 (GPU 기본)."""
    X = TM.build_features(df, img)
    pipe = catboost_to_gpu(TM.build_model(thread_count=thread_count or os.cpu_count() or 4))
    pipe.fit(X, df["label"].astype(int).to_numpy())
    return {"features": list(X.columns), "pipeline": pipe}


def team_proba(team, df, img, sample_ids, classes):
    """df 행으로 CatBoost 확률을 계산해 sample_ids 순서 · classes 열 순서로 반환."""
    frame = df.set_index("sample_id").loc[list(sample_ids)].reset_index()
    X = TM.build_features(frame, img)[team["features"]]
    P = team["pipeline"].predict_proba(X)
    tc = list(team["pipeline"].classes_)
    return P[:, [tc.index(c) for c in classes]]


def blend_proba(bundle, F, df=None, img=None):
    """F(우리 피처, make_features 결과)의 행 순서대로 앙상블 확률을 반환.
    bundle에 'team'이 없으면 우리 모델 확률만 반환. 'members'가 있으면 스태킹 모델(blend.py)."""
    if "members" in bundle:
        from .members import stack_proba
        return stack_proba(bundle, F, df, img)
    P = np.mean([m.predict_proba(F[bundle["feats"]]) for m in bundle["models"]], axis=0)
    team = bundle.get("team")
    if team is None:
        return P
    a = np.asarray(bundle.get("alpha", 0.5), dtype=float)  # 스칼라 또는 클래스별 벡터
    Pt = team_proba(team, df, img, F.sample_id.values, list(bundle["classes"]))
    return a * P + (1 - a) * Pt
