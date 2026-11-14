"""Keep the original 33 features and add 512 frozen ResNet18 features."""
from __future__ import annotations

import numpy as np
import pandas as pd

import baseline_features as B
from encoder import EMBEDDING_COLUMNS

SEED = B.SEED
LABELS = B.LABELS
SENSORS = B.SENSORS
BASE_FEATURE_COUNT = 33
FEATURE_COUNT = BASE_FEATURE_COUNT + len(EMBEDDING_COLUMNS)
resolve_image_paths = B.resolve_image_paths
extract_image_features = B.extract_image_features


def validate_frame(frame: pd.DataFrame, *, require_labels=False):
    required = {'sample_id', 'experiment_id', 'relative_time_s', 'image_path', *SENSORS}
    if require_labels:
        required.add('label')
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'입력 CSV에 필요한 열이 없습니다: {missing}')
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('sample_id가 누락되었거나 중복되었습니다.')
    if frame.experiment_id.isna().any():
        raise ValueError('experiment_id가 누락되었습니다.')
    times = pd.to_numeric(frame['relative_time_s'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError('relative_time_s는 유한한 숫자여야 합니다.')
    if require_labels:
        labels = pd.to_numeric(frame['label'], errors='raise')
        if frame.empty or not labels.isin(LABELS).all():
            raise ValueError('학습 CSV에는 1~6 정수 label이 있는 행이 필요합니다.')


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame, embeddings: np.ndarray):
    original = B.build_features(frame, image_features)
    if original.shape[1] != BASE_FEATURE_COUNT:
        raise ValueError('기존 특징이 33개가 아닙니다.')
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(frame), len(EMBEDDING_COLUMNS)):
        raise ValueError(f'ResNet18 특징 크기가 올바르지 않습니다: {embeddings.shape}')
    if not np.isfinite(embeddings).all():
        raise ValueError('ResNet18 특징에 유효하지 않은 값이 있습니다.')
    image_vectors = pd.DataFrame(embeddings, columns=EMBEDDING_COLUMNS, index=frame.index)
    # Embeddings are extracted in precisely the frame's current row order.
    return pd.concat([original, image_vectors], axis=1)


def build_model(**parameters):
    return B.build_model(**parameters)


def predict_labels(pipeline, features):
    labels = np.asarray(pipeline.predict(features)).reshape(-1)
    if len(labels) != len(features) or not np.isin(labels, LABELS).all():
        raise ValueError('예측 행 수 또는 라벨 범위가 올바르지 않습니다.')
    return labels.astype(int)
