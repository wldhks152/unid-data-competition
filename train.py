"""Train the first controlled CatBoost comparison on the provided server."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import model as M

HERE = Path(__file__).resolve().parent


def read_training_split(data_dir: Path, split: str, image_jobs: int):
    print(f'[{split}] CSV와 이미지 특징을 읽습니다...', flush=True)
    frame = pd.read_csv(
        data_dir / split / f'{split}.csv',
        dtype={'sample_id': 'string', 'experiment_id': 'string'},
    )
    if frame.empty or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError(f'{split}: 비어 있거나 sample_id가 누락/중복되었습니다.')
    if frame.experiment_id.isna().any():
        raise ValueError(f'{split}: experiment_id가 누락되었습니다.')
    times = pd.to_numeric(frame['relative_time_s'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError(f'{split}: 유효하지 않은 relative_time_s가 있습니다.')
    numeric_y = pd.to_numeric(frame['label'], errors='raise')
    if not numeric_y.isin(M.LABELS).all():
        raise ValueError(f'{split}: label은 1~6 정수여야 합니다.')
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    image_features = M.extract_image_features(frame, paths, n_jobs=image_jobs)
    features = M.build_features(frame, image_features)
    if features.shape[1] != 33:
        raise ValueError(f'기존 33개 특징을 예상했지만 {features.shape[1]}개입니다.')
    print(
        f'[{split}] 완료: {len(frame):,}행 / '
        f'{frame.experiment_id.nunique()}개 실험 / {features.shape[1]}개 특징',
        flush=True,
    )
    return frame, features, numeric_y.astype(int)


def predict_labels(pipeline, features):
    # CatBoost can return (N, 1), while a submission needs one label per row.
    values = np.asarray(pipeline.predict(features)).reshape(-1)
    if len(values) != len(features) or not np.isin(values, M.LABELS).all():
        raise ValueError('모델 예측의 길이 또는 라벨 범위가 올바르지 않습니다.')
    return values.astype(int)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model_out', type=Path, default=HERE / 'weights' / 'model.joblib')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--image_jobs', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.image_jobs < 1:
        parser.error('--threads와 --image_jobs는 1 이상이어야 합니다.')

    np.random.seed(M.SEED)
    started = time.monotonic()
    train, x_train, y_train = read_training_split(args.data_dir, 'train', args.image_jobs)
    validation, x_val, y_val = read_training_split(args.data_dir, 'validation', args.image_jobs)
    if not set(train.experiment_id).isdisjoint(validation.experiment_id):
        raise ValueError('Train과 Validation에 중복 실험이 있습니다.')
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError('Train과 Validation의 특징 열이 다릅니다.')

    n_splits = min(5, train.experiment_id.nunique())
    if n_splits < 2:
        raise ValueError('실험 단위 교차검증에는 학습 실험이 2개 이상 필요합니다.')
    parameters = dict(iterations=600, depth=6, learning_rate=0.05, thread_count=args.threads)
    print('설정: class_weight=none / CatBoost CPU / 600 trees / depth=6 / learning_rate=0.05', flush=True)
    print('기존 median imputer와 StandardScaler를 각 fold의 학습 부분에만 fit합니다.', flush=True)
    scores = []
    folds = GroupKFold(n_splits=n_splits).split(x_train, y_train, groups=train.experiment_id)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f'[CV {fold}/{n_splits}] 학습 중...', flush=True)
        fold_model = M.build_model(**parameters)
        fold_model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        score = macro_f1(y_train.iloc[va_idx], predict_labels(fold_model, x_train.iloc[va_idx]))
        scores.append(score)
        print(f'[CV {fold}/{n_splits}] Macro F1 {score:.4f}', flush=True)
        del fold_model

    print('[최종 학습] Train 전체만 사용합니다. Validation은 학습하지 않습니다.', flush=True)
    final_model = M.build_model(**parameters)
    final_model.fit(x_train, y_train)
    validation_pred = predict_labels(final_model, x_val)
    validation_score = macro_f1(y_val, validation_pred)

    bundle = {
        'pipeline': final_model,
        'features': list(x_train.columns),
        'class_weight': None,
        'experiment': 'catboost_v1_same_33_features_unweighted',
        'parameters': parameters,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    report_path = args.model_out.with_suffix('.metrics.json')
    report = {
        'experiment': bundle['experiment'],
        'train_rows': len(train),
        'train_experiments': int(train.experiment_id.nunique()),
        'validation_rows': len(validation),
        'validation_experiments': int(validation.experiment_id.nunique()),
        'feature_count': len(bundle['features']),
        'features': bundle['features'],
        'seed': M.SEED,
        'class_weight': None,
        'parameters': parameters,
        'selection_policy': 'Fixed unweighted configuration; no validation-based configuration selection in this run.',
        'cv_folds': n_splits,
        'cv_scores': scores,
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores)),
        'validation_macro_f1': validation_score,
        'validation_classification_report': classification_report(
            y_val, validation_pred, labels=M.LABELS, output_dict=True, zero_division=0,
        ),
        'validation_confusion_matrix': confusion_matrix(y_val, validation_pred, labels=M.LABELS).tolist(),
        'elapsed_seconds': time.monotonic() - started,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'package_versions': {
            name: importlib.metadata.version(name)
            for name in ('numpy', 'pandas', 'scikit-learn', 'pillow', 'joblib', 'catboost')
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f'class_weight=none: Train GroupKFold Macro F1 {report["cv_mean"]:.4f} '
        f'± {report["cv_std"]:.4f} | Validation Macro F1 {validation_score:.4f}',
        flush=True,
    )
    print('모델 저장:', args.model_out, flush=True)
    print('평가 요약 저장:', report_path, flush=True)


if __name__ == '__main__':
    main()
