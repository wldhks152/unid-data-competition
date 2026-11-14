"""검증 프레임워크 — 모든 브랜치가 같은 fold · 같은 점수 · 같은 OOF 형식을 쓰도록 하는 공통 모듈.

1) 고정 fold      : experiment_folds()  — 실험 단위 StratifiedKFold(tr_method×bat_class), seed 42, 5-fold
2) 공통 점수      : score_fn()          — 5/10/15초 버전별 Macro F1을 9:1:1로 가중 (Test 간격 구성)
3) OOF 형식       : make_oof() / write_oof() / read_oof() — 계약은 VALIDATION.md
4) 메타 교차검증 : outer_meta_eval()  — 블렌드 비율·w·후처리를 '다른 fold의 OOF'로 고르고
                                         '남은 fold'에 적용해 점수를 낸다 (메타 교차검증, base 학습 미분리)
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
def macro_f1(y, pred, labels=CLASSES):
    """sklearn f1_score(average="macro", labels=1..6, zero_division=0)와 같은 값, numpy로 빠르게."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    k = int(max(labels.max(), y.max(initial=0), pred.max(initial=0))) + 1
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = cm[labels, labels].astype(float)
    fp = cm[:, labels].sum(0) - tp
    fn = cm[labels, :].sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return float(f1.mean())


def score_fn(o, pred, detail=False, require_all=True):
    """o: OOF 프레임(ver 열 필요), pred: o 행 순서의 예측 라벨. 버전별 Macro F1을 VER_W(9:1:1)로 가중.
    기본은 3개 버전이 모두 있어야 한다 (빠진 버전을 조용히 재가중하지 않음)."""
    pred = np.asarray(pred)
    if len(pred) != len(o):
        raise ValueError("예측 길이가 OOF 행 수와 다릅니다")
    y = o["y"].to_numpy()
    ver = o["ver"].to_numpy()
    if require_all and set(np.unique(ver)) != set(VER_W):
        raise ValueError(f"score_fn: 버전 {list(VER_W)}가 모두 필요합니다 (현재 {sorted(set(np.unique(ver)))})")
    s = {}
    for v in VER_W:
        m = ver == v
        if m.any():
            s[v] = macro_f1(y[m], pred[m])
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


def _audit_module():
    """tools/audit_oof.py (독립 검사기)를 불러온다. 없으면 None."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "audit_oof.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("audit_oof", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_oof(o, name="oof"):
    """OOF 계약 검사. 기본 검사 + tools/audit_oof.py의 독립 검사(fold 혼합, NaN/음수 확률, 버전 누락 등)."""
    miss = [c for c in OOF_COLS + PCOLS if c not in o.columns]
    assert not miss, f"{name}: OOF 열 없음 {miss}"
    P = o[PCOLS].to_numpy(dtype=float)
    assert np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all(), f"{name}: 확률에 NaN/음수/1 초과 값"
    s = P.sum(1)
    assert np.allclose(s, 1, atol=1e-3), f"{name}: 확률 합이 1이 아님 (min {s.min():.3f}, max {s.max():.3f})"
    assert set(o.ver.unique()) == set(VER_W), f"{name}: 버전 {list(VER_W)}가 모두 필요 (현재 {sorted(o.ver.unique())})"
    assert not o.duplicated(["sample_id", "ver"]).any(), f"{name}: (sample_id, ver) 중복"
    span = o.groupby("grp")["fold"].nunique()
    assert (span == 1).all(), f"{name}: 원래 실험이 여러 fold에 걸침 {span[span > 1].index.tolist()[:5]}"
    mod = _audit_module()
    if mod is not None:
        rep = mod.audit_oof(o[[c for c in o.columns if c in OOF_COLS + PCOLS]])
        assert rep["ok"], f"{name}: audit_oof 실패 {rep['errors']}"
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
    메타 교차검증, base 학습 미분리: 다른 fold의 OOF를 만든 base 모델은 fold f 실험으로도 학습됐다.
    그래서 독립 성능 추정이나 선택 편향 제거를 보장하지 않는다. 메타 선택(비율·w·후처리)만 교차검증한 값이다."""
    pred = np.zeros(len(o), dtype=int)
    cfgs = {}
    fold = o["fold"].to_numpy()
    for f in sorted(np.unique(fold)):
        cfg = select(fold != f)
        cfgs[int(f)] = cfg
        pred[fold == f] = apply(cfg, fold == f)
    return score_fn(o, pred, detail=True), cfgs
