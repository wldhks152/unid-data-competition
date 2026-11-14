"""검증 프레임워크 — 모든 브랜치가 같은 fold · 같은 점수 · 같은 OOF 형식을 쓰도록 하는 공통 모듈.

1) 고정 fold      : experiment_folds()  — 실험 단위 StratifiedKFold(tr_method×bat_class), seed 42, 5-fold
2) 공통 점수      : score_fn()          — 5/10/15초 버전별 Macro F1을 9:1:1로 가중 (Test 간격 구성)
3) OOF 형식       : make_oof() / write_oof() / read_oof() — 계약은 VALIDATION.md
4) 공정한 메타 선택: outer_meta_eval()  — 블렌드 비율·w·후처리를 '다른 fold의 OOF'로 고르고
                                         '남은 fold'에 적용해 점수를 낸다 (선택 편향 제거)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .data import COND, KEY, T, VER_W

CLASSES = np.arange(1, 7)
PCOLS = [f"p{c}" for c in CLASSES]
OOF_COLS = ["sample_id", "ver", "grp", "fold", "y", KEY, T]
FOLD_SEED, N_FOLDS = 42, 5


# ---------------------------------------------------------------------------
# 1. 고정 fold
# ---------------------------------------------------------------------------
def experiment_folds(tv, n_folds=N_FOLDS, seed=FOLD_SEED):
    """tv: Train+Validation 원본(5초) DataFrame. 반환: [(train 실험 id 집합, 검증 실험 id 집합)] × n_folds.
    train.py · stack.py와 같은 규칙(실험 단위, tr_method×bat_class 층화, 실험 id 정렬 후 분할)."""
    ex = tv.groupby(KEY)[COND].first().sort_index()
    strata = ex.tr_method.astype(str) + "_" + ex.bat_class.astype(str)
    if strata.value_counts().min() < n_folds:
        strata = ex.tr_method.astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(set(ex.index[i]), set(ex.index[j])) for i, j in skf.split(ex.index, strata)]


def save_folds(splits, path):
    with open(path, "w") as f:
        json.dump([sorted(v) for _, v in splits], f)


def load_folds(path):
    with open(path) as f:
        vals = [set(v) for v in json.load(f)]
    allx = set().union(*vals)
    return [(allx - v, v) for v in vals]


# ---------------------------------------------------------------------------
# 2. 공통 점수
# ---------------------------------------------------------------------------
def score_fn(o, pred, detail=False):
    """o: OOF 프레임(ver 열 필요), pred: o 행 순서의 예측 라벨. 버전별 Macro F1을 VER_W(9:1:1)로 가중."""
    pred = np.asarray(pred)
    y = o["y"].to_numpy()
    ver = o["ver"].to_numpy()
    s = {}
    for v in VER_W:
        m = ver == v
        if m.any():
            s[v] = f1_score(y[m], pred[m], average="macro", labels=CLASSES)
    tot = sum(VER_W[v] * s[v] for v in s) / sum(VER_W[v] for v in s)
    if not detail:
        return tot
    m5 = ver == "k1o0"
    folds = {int(f): f1_score(y[m5 & (o["fold"].to_numpy() == f)], pred[m5 & (o["fold"].to_numpy() == f)],
                              average="macro", labels=CLASSES) for f in sorted(o["fold"].unique())}
    per_class = f1_score(y[m5], pred[m5], average=None, labels=CLASSES)
    return {"weighted": tot, **s, "fold_5s": folds, "class_5s": dict(zip(CLASSES.tolist(), per_class))}


def tune_w(P, o, lo=0.67, hi=1.5, rounds=3, n_grid=21, post=None):
    """클래스 배율 w를 score_fn(가중 목적함수)으로 좌표 탐색. post가 있으면 후처리 후 점수로 평가(느림)."""
    post = post or (lambda p: p)
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    w = np.ones(len(CLASSES))
    best = score_fn(o, post(CLASSES[(P * w).argmax(1)]))
    for _ in range(rounds):
        changed = False
        for c in range(1, len(CLASSES)):  # 1단계는 기준(1)으로 고정
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = score_fn(o, post(CLASSES[(P * w2).argmax(1)]))
                if s > best + 1e-6:
                    best, w, changed = s, w2, True
        if not changed:
            break
    return w, best


# ---------------------------------------------------------------------------
# 3. OOF 형식
# ---------------------------------------------------------------------------
def make_oof(P, F, ver, fold, extra_cols=()):
    """P: (n, 6) 확률(열 순서 CLASSES), F: 같은 행 순서의 피처/원본 프레임(sample_id, grp, label, KEY, T 필요)."""
    o = pd.DataFrame(np.asarray(P, dtype=float), columns=PCOLS)
    o["sample_id"] = F["sample_id"].to_numpy()
    o["ver"], o["fold"] = ver, fold
    o["grp"] = F["grp"].to_numpy() if "grp" in F else F[KEY].to_numpy()
    o["y"] = F["label"].to_numpy()
    o[KEY], o[T] = F[KEY].to_numpy(), F[T].to_numpy()
    for c in extra_cols:
        o[c] = F[c].to_numpy()
    return o


def check_oof(o, name="oof"):
    miss = [c for c in OOF_COLS + PCOLS if c not in o.columns]
    assert not miss, f"{name}: OOF 열 없음 {miss}"
    assert set(o.ver.unique()) <= set(VER_W), f"{name}: ver는 {list(VER_W)} 중 하나"
    s = o[PCOLS].sum(1)
    assert np.allclose(s, 1, atol=1e-3), f"{name}: 확률 합이 1이 아님 (min {s.min():.3f}, max {s.max():.3f})"
    assert not o.duplicated(["sample_id", "ver"]).any(), f"{name}: (sample_id, ver) 중복"
    return o


def write_oof(o, path):
    check_oof(o, os.path.basename(path))
    o.to_csv(path, index=False)


def read_oof(path):
    return check_oof(pd.read_csv(path), os.path.basename(path))


def align_oofs(oofs):
    """여러 멤버의 OOF를 (sample_id, ver) 기준으로 같은 행 순서로 맞춘다. 공통 행만 남기고, fold가 다르면 오류."""
    names = list(oofs)
    base = oofs[names[0]][OOF_COLS + [c for c in oofs[names[0]].columns if c.startswith("ev_") or c == "tr_method"]]
    keys = set(map(tuple, base[["sample_id", "ver"]].to_numpy()))
    for n in names[1:]:
        keys &= set(map(tuple, oofs[n][["sample_id", "ver"]].to_numpy()))
    base = base[[tuple(x) in keys for x in base[["sample_id", "ver"]].to_numpy()]].reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(base[["sample_id", "ver"]])
    Ps = []
    for n in names:
        x = oofs[n].set_index(["sample_id", "ver"]).loc[idx]
        assert (x["fold"].to_numpy() == base["fold"].to_numpy()).all(), f"{n}: fold 분할이 다릅니다 (folds.json 공유 필요)"
        Ps.append(x[PCOLS].to_numpy())
    dropped = {n: len(oofs[n]) - len(base) for n in names}
    return base, np.stack(Ps), dropped


# ---------------------------------------------------------------------------
# 4. 공정한 메타 선택
# ---------------------------------------------------------------------------
def outer_meta_eval(o, select, apply):
    """select(mask) -> cfg : o[mask] 행(다른 fold)만 보고 비율·w·후처리 등을 고른다.
    apply(cfg, mask) -> o[mask] 행의 예측 라벨.
    fold f마다 다른 fold로 고르고 fold f에 적용한 예측을 모아 score_fn으로 평가한다.
    주의: 다른 fold의 OOF를 만든 base 모델은 fold f 실험으로도 학습됐으므로 완전한 nested CV는 아니다.
    그래도 '같은 OOF에서 고르고 같은 OOF로 보고'하는 선택 편향은 제거된다."""
    pred = np.zeros(len(o), dtype=int)
    cfgs = {}
    fold = o["fold"].to_numpy()
    for f in sorted(np.unique(fold)):
        cfg = select(fold != f)
        cfgs[int(f)] = cfg
        pred[fold == f] = apply(cfg, fold == f)
    return score_fn(o, pred, detail=True), cfgs
