"""Uni-D 배터리 열폭주 6단계 분류 — 베이스라인 v7 공통 모듈.

센서 5개 + 열화상 이미지 통계 + 과거 방향(인과) 시계열 피처 + CatBoost (분류기만 교체한 비교 실험).
train.py와 test.py가 이 모듈을 사용합니다. 특징 함수는 제공 베이스라인 그대로입니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LABELS = list(range(1, 7))
SENSORS = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']
IMAGE_STATS = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio']
# 시계열 피처를 만들 신호: 센서 5개 + 이미지 고온 통계 2개
WINDOW_SIGNALS = SENSORS + ['img_max', 'img_p99']
WINDOW_S = 30.0          # 과거 윈도우 길이(초). 행 수가 아니라 relative_time_s 기준입니다.
HOT_THRESHOLD = 0.8      # 밝기 0~1 기준 고온 픽셀 임계값(고정값, 데이터로 학습하지 않음)


# ---------------------------------------------------------------------------
# 1. 이미지 경로 연결
# ---------------------------------------------------------------------------
def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    """CSV의 image_path를 실제 파일 경로로 바꿉니다.

    roots를 순서대로 기준 폴더로 시도하고, 모두 실패하면 파일명으로 찾습니다.
    """
    rel = frame['image_path'].astype(str)
    for root in roots:
        candidate = rel.map(lambda p: Path(root) / p)
        if candidate.map(Path.is_file).all():
            return candidate
    # 경로 규칙이 맞지 않으면 파일명 기준으로 검색합니다.
    index = {}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file():
                index.setdefault(path.name, path)
    candidate = rel.map(lambda p: index.get(Path(p).name))
    missing = candidate.isna().sum()
    assert missing == 0, f'이미지 {missing}개를 찾지 못했습니다. image_root를 확인하십시오. 예: {rel[candidate.isna()].iloc[0]}'
    return candidate


# ---------------------------------------------------------------------------
# 2. 이미지 통계 피처 (샘플마다 독립 계산 → 다른 행 정보 사용 없음)
# ---------------------------------------------------------------------------
def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.max()),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
        float((arr >= HOT_THRESHOLD).mean()),
    ]


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """sample_id를 인덱스로 하는 이미지 통계 표를 만듭니다."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id'))


# ---------------------------------------------------------------------------
# 3. 과거 방향 시계열 피처
#    같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다.
#    실험 전체 통계(평균·최댓값·길이·순위 등)는 사용하지 않습니다.
# ---------------------------------------------------------------------------
def _causal_one_experiment(t: np.ndarray, x: np.ndarray, window_s: float):
    """한 실험(시간순 정렬)에 대해 행마다 (t-window, t] 평균·최댓값과 window 이전 값과의 차이를 계산합니다."""
    n = len(t)
    mean = np.full(n, np.nan)
    vmax = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    start = np.searchsorted(t, t - window_s, side='right')   # 윈도우 시작 위치
    end = np.searchsorted(t, t, side='right')                 # 같은 시각 관측까지 포함(<= t)
    lag = np.searchsorted(t, t - window_s, side='right') - 1  # t-window 이하의 마지막 관측
    for i in range(n):
        w = x[start[i]:end[i]]
        w = w[~np.isnan(w)]
        if len(w):
            mean[i] = w.mean()
            vmax[i] = w.max()
        if lag[i] >= 0:
            delta[i] = x[i] - x[lag[i]]
    return mean, vmax, delta


def add_causal_features(base: pd.DataFrame, signals: list[str], window_s: float = WINDOW_S) -> pd.DataFrame:
    """실험마다 따로 계산하므로 다른 실험·미래 관측이 입력에 있든 없든 결과가 같습니다."""
    w = int(window_s)
    cols = [f'{s}_{k}{w}s' for s in signals for k in ('mean', 'max', 'delta')]
    out = pd.DataFrame(np.nan, index=base.index, columns=cols)
    times = base['relative_time_s'].to_numpy(dtype=float)
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        t = times[idx]
        for s in signals:
            m, mx, d = _causal_one_experiment(t, base[s].to_numpy(dtype=float)[idx], window_s)
            out.iloc[idx, out.columns.get_loc(f'{s}_mean{w}s')] = m
            out.iloc[idx, out.columns.get_loc(f'{s}_max{w}s')] = mx
            out.iloc[idx, out.columns.get_loc(f'{s}_delta{w}s')] = d
    return out


# ---------------------------------------------------------------------------
# 4. 전체 피처 조립
# ---------------------------------------------------------------------------
def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 피처를 만듭니다. 다른 split·다른 실험의 행은 쓰지 않습니다."""
    base = frame[['sample_id', 'experiment_id']].copy()
    base['relative_time_s'] = pd.to_numeric(frame['relative_time_s'], errors='coerce')
    base[SENSORS] = numeric(frame, SENSORS)
    img = image_features.reindex(frame['sample_id'].values)
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = img.values
    causal = add_causal_features(base, WINDOW_SIGNALS)
    return pd.concat([base[['relative_time_s'] + SENSORS + IMAGE_STATS], causal], axis=1)


def build_model(*, iterations=600, depth=6, learning_rate=0.05, thread_count=4) -> Pipeline:
    """Keep baseline preprocessing and replace only the unweighted classifier."""
    from catboost import CatBoostClassifier

    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', CatBoostClassifier(
            loss_function='MultiClass',
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            task_type='CPU',
            thread_count=thread_count,
            allow_writing_files=False,
            verbose=False,
        )),
    ])
