"""Offline v2 inference: original 545 features plus 13 current sensor values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import model as M
from encoder import FrozenResNet18, TRANSFORM_SPEC

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=HERE / 'weights' / 'model.joblib')
    parser.add_argument('--image_jobs', type=int, default=4)
    args = parser.parse_args()
    if args.image_jobs < 1:
        parser.error('--image_jobs는 1 이상이어야 합니다.')
    if not args.model.is_file():
        raise FileNotFoundError(f'학습 모델이 없습니다. 먼저 train.py를 실행하세요: {args.model}')

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    M.validate_frame(frame)
    if frame.empty:
        submission = pd.DataFrame({'sample_id': frame.sample_id, 'label': pd.Series(dtype='int64')})
    else:
        bundle = joblib.load(args.model)
        if bundle.get('experiment') != M.EXPERIMENT:
            raise ValueError('ResNet18 + 추가 센서 + CatBoost v2 가중치 파일이 아닙니다.')
        if bundle.get('feature_count') != M.FEATURE_COUNT or bundle.get('added_sensors') != list(M.ADDED_SENSORS):
            raise ValueError('저장된 특징 수·추가 센서 목록과 v2 코드가 다릅니다.')
        canonical = lambda value: json.dumps(value, sort_keys=True)
        if canonical(bundle['encoder_transform']) != canonical(TRANSFORM_SPEC):
            raise ValueError('저장된 모델과 현재 코드의 이미지 전처리가 다릅니다.')
        relative_manifest = Path(bundle['encoder_manifest'])
        if relative_manifest.is_absolute() or '..' in relative_manifest.parts:
            raise ValueError('가중치의 encoder_manifest 상대 경로가 올바르지 않습니다.')
        manifest = args.model.parent / relative_manifest
        # No downloads, fitting or full-Test caches are used here.
        image_encoder = FrozenResNet18(
            manifest, device=bundle['encoder_device'], expected_sha256=bundle['encoder_sha256'],
        )
        print(f'입력 {len(frame):,}행 / v2 558개 특징: 기존 545 + 현재 센서 13', flush=True)
        paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
        image_features = M.extract_image_features(frame, paths, n_jobs=args.image_jobs)
        embeddings = image_encoder.extract(paths.tolist(), progress_label='test')
        features = M.build_features(frame, image_features, embeddings)
        if bundle.get('features') != list(features.columns):
            raise ValueError('저장된 특징 열·순서와 v2 코드가 다릅니다.')
        labels = M.predict_labels(bundle['pipeline'], features[bundle['features']])
        submission = pd.DataFrame({'sample_id': frame.sample_id.to_numpy(), 'label': labels})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission):,}행)', flush=True)


if __name__ == '__main__':
    main()
