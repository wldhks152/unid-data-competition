"""Add 13 current sensor values to the frozen ResNet18 + CatBoost v1 recipe."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import model as M
from encoder import FrozenResNet18, TRANSFORM_SPEC, prepare_encoder, resolve_device, runtime_info, verify_encoder

HERE = Path(__file__).resolve().parent


def prepare_encoder_for_run(weights_dir: Path, pretrained_dir: Path | None):
    """Read verified official weights from v1 and copy an independent encoder.

    Nothing in pretrained_dir is modified. The complete new package can infer
    offline without the old model folder. Inference never calls this function.
    """
    weights_dir = Path(weights_dir)
    if pretrained_dir is None:
        return prepare_encoder(weights_dir)
    source_manifest = Path(pretrained_dir).resolve() / 'encoder' / 'manifest.json'
    metadata = verify_encoder(source_manifest)
    target_dir = weights_dir / 'encoder'
    target_manifest = target_dir / 'manifest.json'
    if target_manifest.exists():
        return verify_encoder(target_manifest, expected_sha256=metadata['full_sha256'])
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f'공식 사전학습 encoder를 검증·복사합니다: {source_manifest.parent}', flush=True)
    with tempfile.TemporaryDirectory(prefix='.encoder_copy_', dir=weights_dir) as temporary:
        staging = Path(temporary)
        for part in metadata['parts']:
            shutil.copyfile(source_manifest.parent / part['name'], staging / part['name'])
        shutil.copyfile(source_manifest, staging / 'manifest.json')
        verify_encoder(staging / 'manifest.json', expected_sha256=metadata['full_sha256'])
        target_dir.mkdir(parents=True, exist_ok=True)
        # Install the manifest last so incomplete copies never look complete.
        for part in metadata['parts']:
            (staging / part['name']).replace(target_dir / part['name'])
        (staging / 'manifest.json').replace(target_manifest)
    return verify_encoder(target_manifest, expected_sha256=metadata['full_sha256'])


def read_training_split(data_dir: Path, split: str, image_jobs: int, encoder):
    print(f'[{split}] CSV와 기존 이미지 통계를 읽습니다...', flush=True)
    frame = pd.read_csv(
        data_dir / split / f'{split}.csv',
        dtype={'sample_id': 'string', 'experiment_id': 'string'},
    )
    M.validate_frame(frame, require_labels=True)
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    image_features = M.extract_image_features(frame, paths, n_jobs=image_jobs)
    print(f'[{split}] 고정 ResNet18 이미지 특징을 추출합니다...', flush=True)
    embeddings = encoder.extract(paths.tolist(), progress_label=split)
    features = M.build_features(frame, image_features, embeddings)
    if features.shape[1] != M.FEATURE_COUNT:
        raise ValueError(f'{M.FEATURE_COUNT}개 특징을 예상했습니다.')
    print(
        f'[{split}] 완료: {len(frame):,}행 / {frame.experiment_id.nunique()}개 실험 / '
        f'{features.shape[1]}개 특징 (기존 33 + ResNet18 512 + 추가 센서 13)', flush=True,
    )
    return frame, features, pd.to_numeric(frame['label'], errors='raise').astype(int)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model_out', type=Path, default=HERE / 'weights' / 'model.joblib')
    parser.add_argument('--pretrained_dir', type=Path, default=None,
                        help='기존 ResNet18 CatBoost의 weights 폴더. 공식 encoder를 읽기만 합니다.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--image_jobs', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.image_jobs < 1:
        parser.error('--threads와 --image_jobs는 1 이상이어야 합니다.')
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f'데이터 폴더가 없습니다: {args.data_dir}')
    if args.model_out.exists():
        raise FileExistsError(f'이미 학습된 모델입니다. 새 --model_out 경로를 지정하세요: {args.model_out}')

    np.random.seed(M.SEED)
    started = time.monotonic()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f'ResNet18 특징 추출 장치: {device}, batch=1 (동결 모델)', flush=True)
    print('추가 입력 13개:', ', '.join(M.ADDED_SENSORS), flush=True)
    print('기존 545개 특징·이미지 전처리·분류기·검증 방법을 유지합니다.', flush=True)
    if args.pretrained_dir is None:
        print('공식 ImageNet 사전학습 가중치를 준비합니다. 없으면 다운로드합니다.', flush=True)
    encoder_metadata = prepare_encoder_for_run(args.model_out.parent, args.pretrained_dir)
    manifest = args.model_out.parent / 'encoder' / 'manifest.json'
    image_encoder = FrozenResNet18(manifest, device=device, expected_sha256=encoder_metadata['full_sha256'])
    encoder_runtime = runtime_info(device)

    train, x_train, y_train = read_training_split(args.data_dir, 'train', args.image_jobs, image_encoder)
    validation, x_val, y_val = read_training_split(args.data_dir, 'validation', args.image_jobs, image_encoder)
    if not set(train.experiment_id).isdisjoint(validation.experiment_id):
        raise ValueError('Train과 Validation에 중복 실험이 있습니다.')
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError('Train과 Validation의 특징 열이 다릅니다.')

    # The encoder is fixed before the competition data are read. No encoder
    # fitting or per-input learned normalization occurs during extraction.
    del image_encoder
    gc.collect()
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()
    print('ResNet18 특징 추출 완료. 이제 CatBoost를 CPU에서 학습합니다.', flush=True)

    n_splits = min(5, train.experiment_id.nunique())
    if n_splits < 2:
        raise ValueError('교차검증에 학습 실험이 2개 이상 필요합니다.')
    parameters = dict(iterations=600, depth=6, learning_rate=0.05, thread_count=args.threads)
    print('설정: class_weight=none / CatBoost CPU / 600 trees / depth=6 / learning_rate=0.05', flush=True)
    print('전처리는 각 fold의 학습 부분에만 fit합니다.', flush=True)
    scores = []
    folds = GroupKFold(n_splits=n_splits).split(x_train, y_train, groups=train.experiment_id)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f'[CV {fold}/{n_splits}] 학습 중...', flush=True)
        fold_model = M.build_model(**parameters)
        fold_model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        score = macro_f1(y_train.iloc[va_idx], M.predict_labels(fold_model, x_train.iloc[va_idx]))
        scores.append(score)
        print(f'[CV {fold}/{n_splits}] Macro F1 {score:.4f}', flush=True)
        del fold_model

    print('[최종 학습] Train 전체만 사용합니다. Validation은 학습하지 않습니다.', flush=True)
    final_model = M.build_model(**parameters)
    final_model.fit(x_train, y_train)
    validation_pred = M.predict_labels(final_model, x_val)
    validation_score = macro_f1(y_val, validation_pred)
    bundle = {
        'pipeline': final_model,
        'features': list(x_train.columns),
        'feature_count': M.FEATURE_COUNT,
        'class_weight': None,
        'experiment': M.EXPERIMENT,
        'added_sensors': list(M.ADDED_SENSORS),
        'parameters': parameters,
        'encoder_manifest': 'encoder/manifest.json',
        'encoder_sha256': encoder_metadata['full_sha256'],
        'encoder_device': device,
        'encoder_runtime': encoder_runtime,
        'encoder_transform': TRANSFORM_SPEC,
    }
    joblib.dump(bundle, args.model_out)
    report_path = args.model_out.with_suffix('.metrics.json')
    report = {
        'experiment': bundle['experiment'],
        'train_rows': len(train),
        'train_experiments': int(train.experiment_id.nunique()),
        'validation_rows': len(validation),
        'validation_experiments': int(validation.experiment_id.nunique()),
        'feature_count': M.FEATURE_COUNT,
        'base_feature_count': 33,
        'resnet18_feature_count': 512,
        'original_feature_count': M.ORIGINAL_FEATURE_COUNT,
        'added_sensor_count': len(M.ADDED_SENSORS),
        'added_sensors': list(M.ADDED_SENSORS),
        'features': bundle['features'],
        'seed': M.SEED,
        'class_weight': None,
        'parameters': parameters,
        'selection_policy': 'Fixed unweighted configuration; no validation-based model selection or early stopping in this run.',
        'cv_folds': n_splits,
        'cv_scores': scores,
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores)),
        'validation_macro_f1': validation_score,
        'validation_classification_report': classification_report(
            y_val, validation_pred, labels=M.LABELS, output_dict=True, zero_division=0,
        ),
        'validation_confusion_matrix': confusion_matrix(y_val, validation_pred, labels=M.LABELS).tolist(),
        'pretrained_encoder': encoder_metadata,
        'encoder_runtime': encoder_runtime,
        'elapsed_seconds': time.monotonic() - started,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'package_versions': {
            name: importlib.metadata.version(name)
            for name in ('numpy', 'pandas', 'scikit-learn', 'pillow', 'joblib', 'catboost', 'torch')
        },
        'torchvision_runtime_dependency': False,
    }
    matrix = np.asarray(report['validation_confusion_matrix'])
    report['validation_focus_errors'] = {
        f'actual_{actual}_predicted_{predicted}': int(matrix[actual - 1, predicted - 1])
        for actual, predicted in [(3, 2), (5, 4), (2, 3), (4, 5)]
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f'class_weight=none: Train GroupKFold Macro F1 {report["cv_mean"]:.4f} '
        f'± {report["cv_std"]:.4f} | Validation Macro F1 {validation_score:.4f}', flush=True,
    )
    print('Validation 단계별 성능: label / precision / recall / F1 / support', flush=True)
    for label in M.LABELS:
        row = report['validation_classification_report'][str(label)]
        print(f'  {label}: {row["precision"]:.4f} / {row["recall"]:.4f} / '
              f'{row["f1-score"]:.4f} / {int(row["support"])}', flush=True)
    print('Validation 주요 혼동:', report['validation_focus_errors'], flush=True)
    print('모델 저장:', args.model_out, flush=True)
    print('ResNet18 가중치 저장:', manifest.parent, flush=True)
    print('평가 요약 저장:', report_path, flush=True)


if __name__ == '__main__':
    main()
