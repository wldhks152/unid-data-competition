"""Predict only the supplied CSV rows, using saved preprocessing and weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model as M

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
        raise FileNotFoundError(f'학습한 모델이 없습니다. 먼저 train.py를 실행하세요: {args.model}')

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('입력 CSV의 sample_id가 누락/중복되었습니다.')
    if frame.experiment_id.isna().any():
        raise ValueError('입력 CSV의 experiment_id가 누락되었습니다.')
    times = pd.to_numeric(frame['relative_time_s'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError('유효하지 않은 relative_time_s가 있습니다.')

    if frame.empty:
        submission = pd.DataFrame({'sample_id': frame.sample_id, 'label': pd.Series(dtype='int64')})
    else:
        print(f'입력 {len(frame):,}행의 이미지와 과거 특징을 계산합니다...', flush=True)
        # Recompute solely from this input; no full-Test cache or fitting.
        paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
        image_features = M.extract_image_features(frame, paths, n_jobs=args.image_jobs)
        features = M.build_features(frame, image_features)
        bundle = joblib.load(args.model)
        features = features[bundle['features']]
        labels = np.asarray(bundle['pipeline'].predict(features)).reshape(-1)
        if len(labels) != len(frame) or not np.isin(labels, M.LABELS).all():
            raise ValueError('예측의 행 수 또는 라벨 범위가 올바르지 않습니다.')
        submission = pd.DataFrame({
            'sample_id': frame.sample_id.to_numpy(),
            'label': labels.astype(int),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission):,}행)', flush=True)


if __name__ == '__main__':
    main()
