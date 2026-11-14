"""LightGBM v7과 ResNet18 CatBoost v2의 50:50 확률 평균."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

import lightgbm_features as lightgbm
from resnet18_v2 import model as resnet
from resnet18_v2.encoder import TRANSFORM_SPEC, FrozenResNet18

LABELS = np.array(resnet.LABELS)
WEIGHTS_DIR = Path(__file__).resolve().parent / 'weights'
MODEL_NAMES = ('LightGBM v7', 'ResNet18 CatBoost v2', '50:50')


def load_models(weights_dir):
    v7_path = weights_dir / 'lightgbm_v7.joblib'
    v2_path = weights_dir / 'resnet18_v2' / 'model.joblib'
    v7, v2 = (joblib.load(path) for path in (v7_path, v2_path))
    if not v7.get('extended') or not set(lightgbm.HISTORY_FEATURES).issubset(v7['features']):
        raise ValueError(f'120초 센서 피처를 쓰는 LightGBM v7 가중치가 필요합니다. {v7_path}')
    if (v2.get('experiment') != resnet.EXPERIMENT
            or v2.get('feature_count') != resnet.FEATURE_COUNT
            or v2.get('added_sensors') != resnet.ADDED_SENSORS
            or v2.get('encoder_transform') != TRANSFORM_SPEC):
        raise ValueError(f'558개 피처를 쓰는 ResNet18 CatBoost v2 가중치가 필요합니다. {v2_path}')
    manifest = Path(v2['encoder_manifest'])
    if manifest.is_absolute() or '..' in manifest.parts:
        raise ValueError('ResNet18 encoder manifest는 가중치 폴더 안의 상대 경로여야 합니다.')
    encoder = FrozenResNet18(
        v2_path.parent / manifest,
        device=v2['encoder_device'],
        expected_sha256=v2['encoder_sha256'],
    )
    v7_pipeline = Pipeline([
        ('preprocessor', v7['preprocessor']),
        ('classifier', v7['classifier']),
    ])
    return (v7_pipeline, v2['pipeline']), (v7['features'], v2['features']), encoder


def build_features(frame, image_roots, columns, encoder, *, image_jobs, split):
    paths = resnet.resolve_image_paths(frame, image_roots)
    image_stats = resnet.extract_image_features(frame, paths, n_jobs=image_jobs)
    v7_features = lightgbm.build_features(frame, image_stats, extended=True)
    embeddings = encoder.extract(paths.tolist(), progress_label=split)
    v2_features = resnet.build_features(frame, image_stats, embeddings)
    for name, features, expected in zip(MODEL_NAMES, (v7_features, v2_features), columns):
        if list(features.columns) != list(expected):
            raise ValueError(f'{name}의 피처 열 또는 순서가 가중치와 다릅니다.')
    return v7_features, v2_features


def stage_probabilities(model, features):
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    order = np.argsort(classes)
    if not np.array_equal(classes[order], LABELS):
        raise ValueError('두 모델의 클래스는 각각 1~6이어야 합니다.')
    if probabilities.shape != (len(features), len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError('예측 확률의 크기 또는 값이 올바르지 않습니다.')
    return probabilities[:, order]


def predict_all(models, features):
    v7, v2 = (stage_probabilities(model, values) for model, values in zip(models, features))
    ensemble = (v7 + v2) / 2
    return tuple(LABELS[probabilities.argmax(axis=1)] for probabilities in (v7, v2, ensemble))
