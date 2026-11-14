"""모델 · Macro-F1 튜닝 · 후처리."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import KEY, T


def make_model(seed, balanced=True):
    """lightgbm이 있으면 LGBM, 없으면 sklearn HistGradientBoosting (동작 확인용)."""
    cw = "balanced" if balanced else None
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
            class_weight=cw, random_state=seed, deterministic=True, force_col_wise=True,
            verbose=-1, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, class_weight=cw, random_state=seed,
        )


def tune_class_weights(P, y, classes, rounds=3):
    """Macro-F1 직접 최대화: argmax(P * w)의 클래스별 배율 w를 좌표 탐색으로 찾는다.
    OOF 확률(실험 단위 분할)로만 튜닝 → Test 정보 미사용. 범위를 좁혀 과적합 억제."""
    grid = np.exp(np.linspace(np.log(0.4), np.log(3.0), 21))
    w = np.ones(len(classes))
    best = f1_score(y, classes[(P * w).argmax(1)], average="macro")
    for _ in range(rounds):
        for c in range(1, len(classes)):  # 첫 클래스는 1로 고정 (기준)
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = f1_score(y, classes[(P * w2).argmax(1)], average="macro")
                if s > best + 1e-6:
                    best, w = s, w2
    return w, best


def cummax_pp(F, pred):
    """실험별 시간순 누적최대 — 과거 예측만 쓰므로 규칙 위반 아님. label이 단조증가일 때만 의미."""
    s = pd.Series(np.asarray(pred), index=F.index)
    order = F.sort_values([KEY, T]).index
    return s.loc[order].groupby(F.loc[order, KEY].values).cummax().loc[F.index].to_numpy()


def predict_labels(bundle, F, post=None, P=None):
    """저장된 bundle로 F의 라벨을 예측. 후처리 설정은 bundle["post"]에 내장되어 있다.
    post = {"method": "none"|"cummax"|"hmm", "use_w": bool, "hmm": {...} | None}
    P: 미리 계산한 확률(앙상블 등). 없으면 우리 fold 모델 평균."""
    post = post or bundle["post"]
    if P is None:
        P = np.mean([m.predict_proba(F[bundle["feats"]]) for m in bundle["models"]], axis=0)
    classes = bundle["classes"]
    w = np.asarray(bundle["w"])
    if post["method"] == "hmm":
        from .postprocess import hmm_filter
        P = hmm_filter(F.reset_index(drop=True), P, post["hmm"])
        w = np.asarray(post["hmm"]["w"])
    if not post["use_w"]:
        w = np.ones(len(classes))
    pred = classes[(P * w).argmax(1)]
    if post["method"] == "cummax":
        pred = cummax_pp(F, pred)
    return pred.astype(int), P
